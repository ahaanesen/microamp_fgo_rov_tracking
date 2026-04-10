#include <gtest/gtest.h>

#include <cmath>

#include <gtsam/base/Matrix.h>
#include <gtsam/base/numericalDerivative.h>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/geometry/Rot3.h>
#include <gtsam/navigation/NavState.h>
#include <gtsam/noiseModel/Isotropic.h>

#include "microamp_fgo_rov_tracking/rov_factors.hpp"

namespace {

template <typename A, typename B>
void ExpectMatrixNear(const A& a, const B& b, double tol) {
    ASSERT_EQ(a.rows(), b.rows());
    ASSERT_EQ(a.cols(), b.cols());
    for (int r = 0; r < a.rows(); ++r) {
        for (int c = 0; c < a.cols(); ++c) {
            EXPECT_NEAR(a(r, c), b(r, c), tol) << "Mismatch at (" << r << "," << c << ")";
        }
    }
}

double Rad2Deg(double rad) { return rad * 180.0 / M_PI; }

TEST(UsblFactorTest, JacobiansMatchNumericalDerivatives) {
    const gtsam::Key asvKey = 1;
    const gtsam::Key rovKey = 2;

    const gtsam::Pose3 asvPose(
            gtsam::Rot3::RzRyRx(0.2, -0.1, 0.3),
            gtsam::Point3(4.0, -2.0, 1.0));
    const gtsam::Vector3 asvVel(0.4, -0.2, 0.1);
    const gtsam::NavState asvState(asvPose, asvVel);

    const gtsam::Point3 rovPoint(12.0, 3.0, -5.0);

    const gtsam::Pose3 body_P_sensor(
            gtsam::Rot3::RzRyRx(0.01, -0.02, 0.03),
            gtsam::Point3(0.5, -0.3, 0.2));

    // Use consistent measurements so residual is near zero.
    const gtsam::Pose3 world_P_sensor = asvPose.compose(body_P_sensor);
    const gtsam::Point3 local = world_P_sensor.transformTo(rovPoint);
    double az = std::atan2(local.y(), local.x());
    if (az < 0.0) az += 2.0 * M_PI;
    const double el = std::atan2(local.z(), std::sqrt(local.x() * local.x() + local.y() * local.y()));

    const auto model = gtsam::noiseModel::Isotropic::Sigma(2, 0.1);
    UsblFactor factor(asvKey, rovKey, Rad2Deg(az), Rad2Deg(el), body_P_sensor, model);

    gtsam::Matrix H_asv_analytic, H_rov_analytic;
    const gtsam::Vector err = factor.evaluateError(asvState, rovPoint, H_asv_analytic, H_rov_analytic);

    EXPECT_NEAR(err(0), 0.0, 1e-9);
    EXPECT_NEAR(err(1), 0.0, 1e-9);

    auto f = [&factor](const gtsam::NavState& a, const gtsam::Point3& r) {
        return factor.evaluateError(a, r);
    };

    const gtsam::Matrix H_asv_num =
            gtsam::numericalDerivative21<gtsam::Vector, gtsam::NavState, gtsam::Point3>(
                    f, asvState, rovPoint, 1e-6);
    const gtsam::Matrix H_rov_num =
            gtsam::numericalDerivative22<gtsam::Vector, gtsam::NavState, gtsam::Point3>(
                    f, asvState, rovPoint, 1e-6);

    ExpectMatrixNear(H_asv_analytic, H_asv_num, 1e-5);
    ExpectMatrixNear(H_rov_analytic, H_rov_num, 1e-6);
}

TEST(DepthFactorTest, JacobianMatchesNumericalDerivative) {
    const gtsam::Key rovKey = 5;
    const double measuredDepth = -10.0;
    const auto model = gtsam::noiseModel::Isotropic::Sigma(1, 0.5);

    DepthFactor factor(rovKey, measuredDepth, model);

    const gtsam::Point3 p(1.2, -3.4, -9.25);

    gtsam::Matrix H_analytic;
    const gtsam::Vector err = factor.evaluateError(p, H_analytic);

    EXPECT_NEAR(err(0), p.z() - measuredDepth, 1e-12);

    auto f = [&factor](const gtsam::Point3& x) { return factor.evaluateError(x); };
    const gtsam::Matrix H_num =
            gtsam::numericalDerivative11<gtsam::Vector, gtsam::Point3>(f, p, 1e-6);

    ExpectMatrixNear(H_analytic, H_num, 1e-9);
}

TEST(ROVConstantVelocityFactorTest, JacobiansMatchNumericalDerivatives) {
    const gtsam::Key p_t_key = 10;
    const gtsam::Key v_t_key = 11;
    const gtsam::Key p_tp1_key = 12;
    const double dt = 0.2;
    const auto model = gtsam::noiseModel::Isotropic::Sigma(3, 0.1);

    ROVConstantVelocityFactor factor(p_t_key, v_t_key, p_tp1_key, dt, model);

    const gtsam::Point3 p_t(2.0, -1.0, 0.5);
    const gtsam::Vector3 v_t(0.3, 0.1, -0.2);
    const gtsam::Point3 p_tp1 = p_t + v_t * dt;  // zero residual setup

    gtsam::Matrix H1_analytic, H2_analytic, H3_analytic;
    const gtsam::Vector err = factor.evaluateError(p_t, v_t, p_tp1, H1_analytic, H2_analytic, H3_analytic);

    EXPECT_NEAR(err.norm(), 0.0, 1e-12);

    auto f = [&factor](const gtsam::Point3& p, const gtsam::Vector3& v, const gtsam::Point3& pn) {
        return factor.evaluateError(p, v, pn);
    };

    const gtsam::Matrix H1_num =
            gtsam::numericalDerivative31<gtsam::Vector, gtsam::Point3, gtsam::Vector3, gtsam::Point3>(
                    f, p_t, v_t, p_tp1, 1e-6);
    const gtsam::Matrix H2_num =
            gtsam::numericalDerivative32<gtsam::Vector, gtsam::Point3, gtsam::Vector3, gtsam::Point3>(
                    f, p_t, v_t, p_tp1, 1e-6);
    const gtsam::Matrix H3_num =
            gtsam::numericalDerivative33<gtsam::Vector, gtsam::Point3, gtsam::Vector3, gtsam::Point3>(
                    f, p_t, v_t, p_tp1, 1e-6);

    ExpectMatrixNear(H1_analytic, H1_num, 1e-9);
    ExpectMatrixNear(H2_analytic, H2_num, 1e-9);
    ExpectMatrixNear(H3_analytic, H3_num, 1e-9);
}

}  // namespace