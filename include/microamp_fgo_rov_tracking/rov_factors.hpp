
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