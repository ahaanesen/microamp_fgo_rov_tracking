#include <gtsam/geometry/Point3.h>
#include <gtsam/inference/Key.h>
#include <gtsam/nonlinear/NonlinearFactor.h>
#include <gtsam/navigation/NavState.h>
#include <cmath>

#include "microamp_fgo_rov_tracking/rov_factors.hpp"

using namespace std;
using namespace gtsam;

inline double wrapToPi(double angle) {
    while (angle > M_PI) angle -= 2.0 * M_PI;
    while (angle < -M_PI) angle += 2.0 * M_PI;
    return angle;
}

UsblFactor::UsblFactor(
    Key asvKey,
    Key rovKey,
    double measuredAzimuth,
    double measuredElevation,
    const gtsam::Pose3& body_P_sensor,
    const SharedNoiseModel& model)
    : NoiseModelFactor2<Pose3, Point3>(model, asvKey, rovKey),
      measuredAzimuth_(measuredAzimuth),
      measuredElevation_(measuredElevation),
      body_P_sensor_(body_P_sensor) {}

Vector UsblFactor::evaluateError(
    const gtsam::Pose3& asvPose,
    const gtsam::Point3& rovPoint,
    boost::optional<gtsam::Matrix&> H_asv,
    boost::optional<gtsam::Matrix&> H_rov) const {
    // USBL azimuth/elevation are reported in world NED, not in the sensor/body frame.
    // Model the measurement directly from the world-frame ASV->ROV vector.
    (void)body_P_sensor_;
    double measAzimuthRad = measuredAzimuth_ * M_PI / 180.0;
    double measElevationRad = measuredElevation_ * M_PI / 180.0;

    gtsam::Vector3 p_world = rovPoint - asvPose.translation();

    double x = p_world.x();
    double y = p_world.y();
    double z = p_world.z();
    double r2 = x * x + y * y;
    double r = std::sqrt(r2);
    double rho2 = r2 + z * z;
    const double epsilon = 1e-6;

    double expAz = std::atan2(y, x);
    if (expAz < 0.0) {
        expAz += 2.0 * M_PI;
    }
    double expEl = std::atan2(z, r);

    double errAz = wrapToPi(expAz - measAzimuthRad);
    double errEl = expEl - measElevationRad;

    gtsam::Matrix23 H_angles_world = gtsam::Matrix23::Zero();
    if (r2 > epsilon && r > epsilon && rho2 > epsilon) {
        H_angles_world << -y / r2, x / r2, 0.0,
                          -x * z / (rho2 * r), -y * z / (rho2 * r), r / rho2;
    }

    if (H_asv) {
        gtsam::Matrix26 H_err_pose = gtsam::Matrix26::Zero();
        H_err_pose.block<2, 3>(0, 3) = -H_angles_world;
        *H_asv = H_err_pose;
    }

    if (H_rov) {
        *H_rov = H_angles_world;
    }

    return (gtsam::Vector(2) << errAz, errEl).finished();
}

gtsam::NonlinearFactor::shared_ptr UsblFactor::clone() const {
    return boost::static_pointer_cast<gtsam::NonlinearFactor>(
        gtsam::NonlinearFactor::shared_ptr(new UsblFactor(*this)));
}

DepthFactor::DepthFactor(
    Key rovKey,
    double measuredDepth,
    const gtsam::SharedNoiseModel& model)
    : gtsam::NoiseModelFactor1<gtsam::Point3>(model, rovKey),
      measuredDepth_(measuredDepth) {}

gtsam::Vector DepthFactor::evaluateError(
    const gtsam::Point3& rovPoint,
    boost::optional<gtsam::Matrix&> H) const {
    if (H) {
        *H = (gtsam::Matrix13() << 0.0, 0.0, 1.0).finished();
    }
    return (gtsam::Vector(1) << rovPoint.z() - measuredDepth_).finished();
}

gtsam::NonlinearFactor::shared_ptr DepthFactor::clone() const {
    return boost::static_pointer_cast<gtsam::NonlinearFactor>(
        gtsam::NonlinearFactor::shared_ptr(new DepthFactor(*this)));
}

PsudoRangeFactor::PsudoRangeFactor(
    Key asvKey,
    Key rovKey,
    double tof,
    double v_sound,
    const gtsam::Pose3& body_P_sensor,
    const gtsam::SharedNoiseModel& model)
    : gtsam::NoiseModelFactor2<gtsam::Pose3, gtsam::Point3>(model, asvKey, rovKey),
      measuredTOF_(tof),
      soundSpeed_(v_sound),
      body_P_sensor_(body_P_sensor) {}

Vector PsudoRangeFactor::evaluateError(
    const gtsam::Pose3& asvPose,
    const gtsam::Point3& rovPoint,
    boost::optional<gtsam::Matrix&> H_asv,
    boost::optional<gtsam::Matrix&> H_rov) const {
    gtsam::Matrix66 H_pose_asv;
    gtsam::Pose3 world_P_sensor = asvPose.compose(body_P_sensor_, H_asv ? &H_pose_asv : nullptr);

    gtsam::Matrix16 H_dist_sensor;
    gtsam::Matrix13 H_dist_rov;
    double dist = world_P_sensor.range(
        rovPoint,
        H_asv ? &H_dist_sensor : nullptr,
        H_rov ? &H_dist_rov : nullptr);

    double expectedTOF = dist / soundSpeed_;

    if (H_asv) {
        *H_asv = (1.0 / soundSpeed_) * H_dist_sensor * H_pose_asv;
    }

    if (H_rov) {
        *H_rov = (1.0 / soundSpeed_) * H_dist_rov;
    }

    return (gtsam::Vector(1) << expectedTOF - measuredTOF_).finished();
}

gtsam::NonlinearFactor::shared_ptr PsudoRangeFactor::clone() const {
    return boost::static_pointer_cast<gtsam::NonlinearFactor>(
        gtsam::NonlinearFactor::shared_ptr(new PsudoRangeFactor(*this)));
}

ConstantVelocityFactor::ConstantVelocityFactor(
    gtsam::Key p_prev,
    gtsam::Key v_prev,
    gtsam::Key p_curr,
    gtsam::Key v_curr,
    double dt,
    const gtsam::SharedNoiseModel& model)
    : NoiseModelFactor4(model, p_prev, v_prev, p_curr, v_curr), dt_(dt) {}

gtsam::Vector ConstantVelocityFactor::evaluateError(
    const gtsam::Point3& p_prev,
    const gtsam::Vector3& v_prev,
    const gtsam::Point3& p_curr,
    const gtsam::Vector3& v_curr,
    boost::optional<gtsam::Matrix&> H1,
    boost::optional<gtsam::Matrix&> H2,
    boost::optional<gtsam::Matrix&> H3,
    boost::optional<gtsam::Matrix&> H4) const {

    gtsam::Matrix63 J_p_prev = gtsam::Matrix63::Zero();
    gtsam::Matrix63 J_v_prev = gtsam::Matrix63::Zero();
    gtsam::Matrix63 J_p_curr = gtsam::Matrix63::Zero();
    gtsam::Matrix63 J_v_curr = gtsam::Matrix63::Zero();

    J_p_prev.block<3, 3>(0, 0) = gtsam::Matrix33::Identity();
    J_v_prev.block<3, 3>(0, 0) = gtsam::Matrix33::Identity() * dt_;
    J_v_prev.block<3, 3>(3, 0) = gtsam::Matrix33::Identity();
    J_p_curr.block<3, 3>(0, 0) = -gtsam::Matrix33::Identity();
    J_v_curr.block<3, 3>(3, 0) = -gtsam::Matrix33::Identity();
    if (H1) {
        *H1 = J_p_prev;

    }
    if (H2) {
        *H2 = J_v_prev;
    }
    if (H3) {
        *H3 = J_p_curr;
    }
    if (H4) {
        *H4 = J_v_curr;
    }

    gtsam::Vector6 error;
    error.head<3>() = (p_prev + v_prev * dt_) - p_curr;
    error.tail<3>() = v_prev - v_curr;
    return error;
}

gtsam::NonlinearFactor::shared_ptr ConstantVelocityFactor::clone() const {
    return boost::static_pointer_cast<gtsam::NonlinearFactor>(
        gtsam::NonlinearFactor::shared_ptr(new ConstantVelocityFactor(*this)));
}
