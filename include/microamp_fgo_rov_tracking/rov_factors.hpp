#pragma once

#include <gtsam/geometry/Pose3.h>
#include <gtsam/geometry/Point3.h>
#include <gtsam/nonlinear/NonlinearFactor.h>
#include <gtsam/inference/Key.h>

class UsblFactor : public gtsam::NoiseModelFactor2<gtsam::Pose3, gtsam::Point3> {
public:
    using Base = gtsam::NoiseModelFactor2<gtsam::Pose3, gtsam::Point3>;
    using shared_ptr = std::shared_ptr<UsblFactor>;
    UsblFactor(gtsam::Key asvKey,
                gtsam::Key rovKey,
                double measuredAzimuth,
                double measuredElevation,
                const gtsam::Pose3& body_P_sensor,
                const gtsam::SharedNoiseModel& model);
    ~UsblFactor() override = default;
    gtsam::Vector evaluateError(const gtsam::Pose3& asvPose,
                                const gtsam::Point3& rovPoint,
                                typename Base::template OptionalMatrixTypeT<gtsam::Pose3> H_asv = nullptr,
                                typename Base::template OptionalMatrixTypeT<gtsam::Point3> H_rov = nullptr) const override;
    gtsam::NonlinearFactor::shared_ptr clone() const override;
private:
    double measuredAzimuth_;
    double measuredElevation_;
    gtsam::Pose3 body_P_sensor_;
};

class DepthFactor : public gtsam::NoiseModelFactor1<gtsam::Point3> {
public:
    using Base = gtsam::NoiseModelFactor1<gtsam::Point3>;
    using shared_ptr = std::shared_ptr<DepthFactor>;
    DepthFactor(gtsam::Key rovKey,
                double measuredDepth,
                const gtsam::SharedNoiseModel& model);
    ~DepthFactor() override = default;
    gtsam::Vector evaluateError(const gtsam::Point3& rovPoint,
                                typename Base::template OptionalMatrixTypeT<gtsam::Point3> H = nullptr) const override;
    gtsam::NonlinearFactor::shared_ptr clone() const override;
private:
    double measuredDepth_;
};

class PsudoRangeFactor : public gtsam::NoiseModelFactor2<gtsam::Pose3, gtsam::Point3> {
public:
    using Base = gtsam::NoiseModelFactor2<gtsam::Pose3, gtsam::Point3>;
    using shared_ptr = std::shared_ptr<PsudoRangeFactor>;
    PsudoRangeFactor(gtsam::Key asvKey,
                    gtsam::Key rovKey,
                    double tof,
                    double v_sound,
                    const gtsam::Pose3& body_P_sensor,
                    const gtsam::SharedNoiseModel& model);
    ~PsudoRangeFactor() override = default;
    gtsam::Vector evaluateError(const gtsam::Pose3& asvPose,
                                const gtsam::Point3& rovPoint,
                                typename Base::template OptionalMatrixTypeT<gtsam::Pose3> H_asv = nullptr,
                                typename Base::template OptionalMatrixTypeT<gtsam::Point3> H_rov = nullptr) const override;
    gtsam::NonlinearFactor::shared_ptr clone() const override;
private:
    double measuredTOF_;
    double soundSpeed_;
    gtsam::Pose3 body_P_sensor_;
};

class ConstantVelocityFactor : public gtsam::NoiseModelFactor4<gtsam::Point3, gtsam::Vector3, gtsam::Point3, gtsam::Vector3> {
public:
    using Base = gtsam::NoiseModelFactor4<gtsam::Point3, gtsam::Vector3, gtsam::Point3, gtsam::Vector3>;
    using shared_ptr = std::shared_ptr<ConstantVelocityFactor>;
	ConstantVelocityFactor(gtsam::Key p_prev, 
                            gtsam::Key v_prev, 
                            gtsam::Key p_curr,
                            gtsam::Key v_curr,
                            double dt, 
                            const gtsam::SharedNoiseModel& model);
	gtsam::Vector evaluateError(const gtsam::Point3& p_prev, 
                                const gtsam::Vector3& v_prev, 
                                const gtsam::Point3& p_curr,
                                const gtsam::Vector3& v_curr,
                                typename Base::template OptionalMatrixTypeT<gtsam::Point3> H1 = nullptr,
                                typename Base::template OptionalMatrixTypeT<gtsam::Vector3> H2 = nullptr,
                                typename Base::template OptionalMatrixTypeT<gtsam::Point3> H3 = nullptr,
                                typename Base::template OptionalMatrixTypeT<gtsam::Vector3> H4 = nullptr) const override;
	gtsam::NonlinearFactor::shared_ptr clone() const override;
private:
	double dt_;
};


// ---------------------------------------------------------------------------
// Constant-velocity process model, split into two independent factors.
//
// Rationale:
//   The traditional 6-D CV factor with diagonal noise on
//   [p_curr - (p_prev + v_prev·dt); v_curr - v_prev] produces a near-rank-
//   deficient information matrix when the position sigma is tight, because
//   adjacent factors share variables and the position rows are linear
//   combinations of (p_prev, v_prev, p_curr) with deterministic structure.
//   This breaks Cholesky factorisation (which requires SPD) while QR survives
//   via column pivoting.
//
//   Splitting the 6-D residual into two independent 3-D factors keeps the
//   same information content but yields full-rank Jacobian blocks, making
//   the iSAM2 information matrix safely SPD.
//
//   ConstantVelocityIntegrationFactor:
//     error = (p_prev + v_prev·dt) − p_curr,
//     noise σ_p = σ_a · dt² / 2   (double-integrated acceleration noise)
//
//   VelocityRandomWalkFactor:
//     error = v_prev − v_curr,
//     noise σ_v = σ_a · dt        (single-integrated acceleration noise)
// ---------------------------------------------------------------------------

class ConstantVelocityIntegrationFactor
    : public gtsam::NoiseModelFactor3<gtsam::Point3, gtsam::Vector3, gtsam::Point3> {
public:
    using Base = gtsam::NoiseModelFactor3<gtsam::Point3, gtsam::Vector3, gtsam::Point3>;
    using shared_ptr = std::shared_ptr<ConstantVelocityIntegrationFactor>;
    ConstantVelocityIntegrationFactor(gtsam::Key p_prev,
                                      gtsam::Key v_prev,
                                      gtsam::Key p_curr,
                                      double dt,
                                      const gtsam::SharedNoiseModel& model);
    ~ConstantVelocityIntegrationFactor() override = default;
    gtsam::Vector evaluateError(
        const gtsam::Point3& p_prev,
        const gtsam::Vector3& v_prev,
        const gtsam::Point3& p_curr,
        typename Base::template OptionalMatrixTypeT<gtsam::Point3> H1 = nullptr,
        typename Base::template OptionalMatrixTypeT<gtsam::Vector3> H2 = nullptr,
        typename Base::template OptionalMatrixTypeT<gtsam::Point3> H3 = nullptr) const override;
    gtsam::NonlinearFactor::shared_ptr clone() const override;
private:
    double dt_;
};

class VelocityRandomWalkFactor
    : public gtsam::NoiseModelFactor2<gtsam::Vector3, gtsam::Vector3> {
public:
    using Base = gtsam::NoiseModelFactor2<gtsam::Vector3, gtsam::Vector3>;
    using shared_ptr = std::shared_ptr<VelocityRandomWalkFactor>;
    VelocityRandomWalkFactor(gtsam::Key v_prev,
                             gtsam::Key v_curr,
                             const gtsam::SharedNoiseModel& model);
    ~VelocityRandomWalkFactor() override = default;
    gtsam::Vector evaluateError(
        const gtsam::Vector3& v_prev,
        const gtsam::Vector3& v_curr,
        typename Base::template OptionalMatrixTypeT<gtsam::Vector3> H1 = nullptr,
        typename Base::template OptionalMatrixTypeT<gtsam::Vector3> H2 = nullptr) const override;
    gtsam::NonlinearFactor::shared_ptr clone() const override;
};

// Weak prior that roll and pitch are near zero. Useful for surface vehicles
// where the body z-axis stays nearly aligned with gravity. Does not constrain
// yaw or position. Eliminates the unobservable roll/pitch DOF that otherwise
// causes Cholesky factorization failures in long runs with small GPS lever arms.
class RollPitchPriorFactor : public gtsam::NoiseModelFactor1<gtsam::Pose3> {
public:
    using Base = gtsam::NoiseModelFactor1<gtsam::Pose3>;
    using shared_ptr = std::shared_ptr<RollPitchPriorFactor>;
 
    RollPitchPriorFactor(gtsam::Key key, const gtsam::SharedNoiseModel& model)
        : Base(model, key) {}
    ~RollPitchPriorFactor() override = default;
 
    gtsam::Vector evaluateError(
        const gtsam::Pose3& pose,
        typename Base::template OptionalMatrixTypeT<gtsam::Pose3> H = nullptr) const override;
 
    gtsam::NonlinearFactor::shared_ptr clone() const override {
        return std::static_pointer_cast<gtsam::NonlinearFactor>(
            std::make_shared<RollPitchPriorFactor>(*this));
    }
};