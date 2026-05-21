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

// =====================================================================
// UsblFactor
// =====================================================================

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
    typename Base::template OptionalMatrixTypeT<gtsam::Pose3> H_asv,
    typename Base::template OptionalMatrixTypeT<gtsam::Point3> H_rov) const {

    const double measAzimuthRad   = measuredAzimuth_   * M_PI / 180.0;
    const double measElevationRad = measuredElevation_ * M_PI / 180.0;

    // Sensor position in world (depends on ASV pose + lever arm)
    gtsam::Matrix36 H_sensor_pos_asv;
    const gtsam::Point3 sensor_world =
        asvPose.transformFrom(body_P_sensor_.translation(),
                              H_asv ? &H_sensor_pos_asv : nullptr);

    const gtsam::Vector3 p_world = rovPoint - sensor_world;
    const double x = p_world.x();
    const double y = p_world.y();
    const double z = p_world.z();

    const double r2   = x * x + y * y;   // horizontal distance squared
    const double rho2 = r2 + z * z;      // 3D distance squared

    // Near-degeneracy guard (ROV essentially overhead or co-located)
    constexpr double kEps = 1e-6;
    if (r2 < kEps || rho2 < kEps) {
        if (H_asv) *H_asv = gtsam::Matrix26::Zero();
        if (H_rov) *H_rov = gtsam::Matrix23::Zero();
        return gtsam::Vector2::Zero();
    }

    const double r = std::sqrt(r2);

    double expAz = std::atan2(y, x);
    if (expAz < 0.0) expAz += 2.0 * M_PI;
    const double expEl = std::atan2(z, r);

    const double errAz = wrapToPi(expAz - measAzimuthRad);
    const double errEl = expEl - measElevationRad;

    // d[az; el] / d p_world
    //   az = atan2(y, x)        -> [-y/r2,  x/r2,  0 ]
    //   el = atan2(z, r)        -> [-xz/(r*rho2), -yz/(r*rho2), r/rho2 ]
    gtsam::Matrix23 H_angles_world;
    H_angles_world <<
        -y / r2,            x / r2,            0.0,
        -x * z / (r * rho2), -y * z / (r * rho2), r / rho2;

    // p_world = rovPoint - sensor_world, so
    //   d p_world / d rovPoint     = +I
    //   d p_world / d sensor_world = -I
    if (H_asv) *H_asv = -H_angles_world * H_sensor_pos_asv;
    if (H_rov) *H_rov =  H_angles_world;

    return (gtsam::Vector(2) << errAz, errEl).finished();
}

gtsam::NonlinearFactor::shared_ptr UsblFactor::clone() const {
    return std::static_pointer_cast<gtsam::NonlinearFactor>(
        std::make_shared<UsblFactor>(*this));
}

// =====================================================================
// DepthFactor
// =====================================================================

DepthFactor::DepthFactor(
    Key rovKey,
    double measuredDepth,
    const gtsam::SharedNoiseModel& model)
    : gtsam::NoiseModelFactor1<gtsam::Point3>(model, rovKey),
      measuredDepth_(measuredDepth) {}

gtsam::Vector DepthFactor::evaluateError(
    const gtsam::Point3& rovPoint,
    typename Base::template OptionalMatrixTypeT<gtsam::Point3> H) const {
    if (H) {
        *H = (gtsam::Matrix13() << 0.0, 0.0, 1.0).finished();
    }
    return (gtsam::Vector(1) << rovPoint.z() - measuredDepth_).finished();
}

gtsam::NonlinearFactor::shared_ptr DepthFactor::clone() const {
    return std::static_pointer_cast<gtsam::NonlinearFactor>(
        std::make_shared<DepthFactor>(*this));
}

// =====================================================================
// PsudoRangeFactor
// =====================================================================

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
    typename Base::template OptionalMatrixTypeT<gtsam::Pose3> H_asv,
    typename Base::template OptionalMatrixTypeT<gtsam::Point3> H_rov) const {
    gtsam::Matrix66 H_pose_asv;
    gtsam::Pose3 world_P_sensor = asvPose.compose(body_P_sensor_, H_asv ? &H_pose_asv : nullptr);

    gtsam::Matrix16 H_dist_sensor;
    gtsam::Matrix13 H_dist_rov;
    double dist = world_P_sensor.range(
        rovPoint,
        H_asv ? &H_dist_sensor : nullptr,
        H_rov ? &H_dist_rov : nullptr);

    double expectedTOF = dist / soundSpeed_;

    if (H_asv) *H_asv = (1.0 / soundSpeed_) * H_dist_sensor * H_pose_asv;
    if (H_rov) *H_rov = (1.0 / soundSpeed_) * H_dist_rov;

    return (gtsam::Vector(1) << expectedTOF - measuredTOF_).finished();
}

gtsam::NonlinearFactor::shared_ptr PsudoRangeFactor::clone() const {
    return std::static_pointer_cast<gtsam::NonlinearFactor>(
        std::make_shared<PsudoRangeFactor>(*this));
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
    typename Base::template OptionalMatrixTypeT<gtsam::Point3> H1,
    typename Base::template OptionalMatrixTypeT<gtsam::Vector3> H2,
    typename Base::template OptionalMatrixTypeT<gtsam::Point3> H3,
    typename Base::template OptionalMatrixTypeT<gtsam::Vector3> H4) const {

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
    return std::static_pointer_cast<gtsam::NonlinearFactor>(
        std::make_shared<ConstantVelocityFactor>(*this));
}


// =====================================================================
// ConstantVelocityIntegrationFactor
//   error = (p_prev + v_prev·dt) − p_curr   (3-D)
// =====================================================================

ConstantVelocityIntegrationFactor::ConstantVelocityIntegrationFactor(
    gtsam::Key p_prev,
    gtsam::Key v_prev,
    gtsam::Key p_curr,
    double dt,
    const gtsam::SharedNoiseModel& model)
    : Base(model, p_prev, v_prev, p_curr), dt_(dt) {}

gtsam::Vector ConstantVelocityIntegrationFactor::evaluateError(
    const gtsam::Point3& p_prev,
    const gtsam::Vector3& v_prev,
    const gtsam::Point3& p_curr,
    typename Base::template OptionalMatrixTypeT<gtsam::Point3> H1,
    typename Base::template OptionalMatrixTypeT<gtsam::Vector3> H2,
    typename Base::template OptionalMatrixTypeT<gtsam::Point3> H3) const {
    if (H1) *H1 =  gtsam::Matrix33::Identity();
    if (H2) *H2 =  gtsam::Matrix33::Identity() * dt_;
    if (H3) *H3 = -gtsam::Matrix33::Identity();
    return (p_prev + v_prev * dt_) - p_curr;
}

gtsam::NonlinearFactor::shared_ptr ConstantVelocityIntegrationFactor::clone() const {
    return std::static_pointer_cast<gtsam::NonlinearFactor>(
        std::make_shared<ConstantVelocityIntegrationFactor>(*this));
}

// =====================================================================
// VelocityRandomWalkFactor
//   error = v_prev − v_curr   (3-D)
// =====================================================================

VelocityRandomWalkFactor::VelocityRandomWalkFactor(
    gtsam::Key v_prev,
    gtsam::Key v_curr,
    const gtsam::SharedNoiseModel& model)
    : Base(model, v_prev, v_curr) {}

gtsam::Vector VelocityRandomWalkFactor::evaluateError(
    const gtsam::Vector3& v_prev,
    const gtsam::Vector3& v_curr,
    typename Base::template OptionalMatrixTypeT<gtsam::Vector3> H1,
    typename Base::template OptionalMatrixTypeT<gtsam::Vector3> H2) const {
    if (H1) *H1 =  gtsam::Matrix33::Identity();
    if (H2) *H2 = -gtsam::Matrix33::Identity();
    return v_prev - v_curr;
}

gtsam::NonlinearFactor::shared_ptr VelocityRandomWalkFactor::clone() const {
    return std::static_pointer_cast<gtsam::NonlinearFactor>(
        std::make_shared<VelocityRandomWalkFactor>(*this));
}

gtsam::Vector RollPitchPriorFactor::evaluateError(
    const gtsam::Pose3& pose,
    typename Base::template OptionalMatrixTypeT<gtsam::Pose3> H) const {
 
    // Extract roll, pitch from the rotation. Using ypr ordering matches GTSAM
    // conventions: ypr returns (yaw, pitch, roll).
    const gtsam::Rot3 R = pose.rotation();
    gtsam::Matrix3 H_ypr_R;
    const gtsam::Vector3 ypr = R.ypr(H ? &H_ypr_R : nullptr);
    // ypr = [yaw, pitch, roll]; we want [roll, pitch] = [ypr(2), ypr(1)]
 
    if (H) {
        // Chain rule: d[roll;pitch]/d Pose3
        // Pose3 tangent ordering: [rot(3); trans(3)]
        // H_ypr_R is 3x3: rows = [yaw, pitch, roll], cols = rotation tangent.
        gtsam::Matrix26 H_pose = gtsam::Matrix26::Zero();
        H_pose.block<1, 3>(0, 0) = H_ypr_R.row(2);  // roll
        H_pose.block<1, 3>(1, 0) = H_ypr_R.row(1);  // pitch
        // translation columns (3..5) stay zero
        *H = H_pose;
    }
 
    return (gtsam::Vector(2) << ypr(2), ypr(1)).finished();
}