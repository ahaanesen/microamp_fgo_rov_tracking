#include <gtsam/geometry/Point3.h>
#include <gtsam/inference/Key.h>
#include <gtsam/nonlinear/NonlinearFactor.h>
#include <gtsam/navigation/NavState.h>

using namespace std;
using namespace gtsam;


class UsblFactor : public NoiseModelFactor2<NavState, Point3> {
private:
    double measuredAzimuth_; // Measured azimuth from USBL (degrees)
    double measuredElevation_; // Measured elevation from USBL (degrees)
    gtsam::Pose3 body_P_sensor_; // The lever arm AND sensor orientation from ASV body frame to USBL sensor frame

public:
    /// shorthand for a smart pointer to a factor
    typedef std::shared_ptr<UsblFactor> shared_ptr;

    UsblFactor(Key asvKey,
                Key rovKey,
                double measuredAzimuth,
                double measuredElevation,
                const gtsam::Pose3& body_P_sensor,
                const SharedNoiseModel& model)
        : NoiseModelFactor2<NavState, Point3>(model, asvKey, rovKey),
          measuredAzimuth_(measuredAzimuth), measuredElevation_(measuredElevation),
          body_P_sensor_(body_P_sensor) {}

    ~UsblFactor() override {}

    Vector evaluateError(
        const gtsam::NavState& asvState,
        const gtsam::Point3& rovPoint,
        boost::optional<gtsam::Matrix&> H_asv = boost::none,
        boost::optional<gtsam::Matrix&> H_rov = boost::none) const override
    {
        // 0. Convert measured angles from degrees to radians for error calculation
        double measAzimuthRad = measuredAzimuth_ * M_PI / 180.0; // [0, 2pi)
        double measElevationRad = measuredElevation_ * M_PI / 180.0; // [-pi/2, pi/2)


        // 1. Get the Sensor Pose in World Frame
        //      Note: H_comp is the Jacobian of the composition
        gtsam::Matrix66 H_pose_asv; 
        gtsam::Pose3 world_P_sensor = asvState.pose().compose(body_P_sensor_, H_asv ? &H_pose_asv : nullptr);

        // 2. Transform ROV point into Sensor Local Frame
        gtsam::Matrix36 H_local_sensor; // d(P_local) / d(SensorPose)
        gtsam::Matrix33 H_local_rov;    // d(P_local) / d(ROV_Point)
        gtsam::Point3 P_local = world_P_sensor.transformTo(rovPoint, 
                                    H_asv ? &H_local_sensor : nullptr, 
                                    H_rov ? &H_local_rov : nullptr);

        // 3. Calculate expected angles from P_local
        double x = P_local.x(), y = P_local.y(), z = P_local.z();
        double r2 = x*x + y*y;
        double r = std::sqrt(r2);
        double rho2 = r2 + z*z;

        double expAz = std::atan2(y, x);
        if (expAz < 0.0) { expAz += 2.0 * M_PI; } // Convert to [0, 2pi) for correct comparison with measured azimuth
        double expEl = std::atan2(z, r);

        // 4. Handle Jacobians via Chain Rule
        if (H_asv) {
            // Jacobian of [az, el] w.r.t P_local
            gtsam::Matrix23 H_angles_local;
            H_angles_local << -y/r2,            x/r2,           0.0,
                              -x*z/(rho2 * r), -y*z/(rho2 * r), r/rho2;

            // d(angles)/d(NavState) = d(angles)/d(P_local) * d(P_local)/d(SensorPose) * d(SensorPose)/d(AsvPose)
            // NavState has 9 DOF: [Rot, Pos, Vel]. Vel derivatives are 0.
            gtsam::Matrix29 H_total = gtsam::Matrix::Zero(2, 9);
            H_total.block<2, 6>(0, 0) = H_angles_local * H_local_sensor * H_pose_asv;
            *H_asv = H_total;
        }

        if (H_rov) {
            gtsam::Matrix23 H_angles_local;
            H_angles_local << -y/r2,            x/r2,           0.0,
                              -x*z/(rho2 * r), -y*z/(rho2 * r), r/rho2;
            
            *H_rov = H_angles_local * H_local_rov;
        }

        return (gtsam::Vector(2) << (expAz - measAzimuthRad), (expEl - measElevationRad)).finished();
    }
};

class DepthFactor : public NoiseModelFactor1<Point3> {
private:
    double measuredDepth_;

public:
    /// shorthand for a smart pointer to a factor
    typedef std::shared_ptr<DepthFactor> shared_ptr;

    DepthFactor(Key rovKey,
                        double measuredDepth,
                        const SharedNoiseModel& model)
        : NoiseModelFactor1<Point3>(model, rovKey),
          measuredDepth_(measuredDepth) {}
    
    ~DepthFactor() override {}

    gtsam::Vector evaluateError(const gtsam::Point3& rovPoint, 
                                boost::optional<gtsam::Matrix&> H = boost::none) const override {
        if (H) {
            // Jacobian of error (z - depth) w.r.t Point3 (x, y, z)
            // Dimensions: 1x3
            *H = (gtsam::Matrix13() << 0.0, 0.0, 1.0).finished();
        }
        return (gtsam::Vector(1) << rovPoint.z() - measuredDepth_).finished();
    }

    // Clone function to allow copying of the factor
    gtsam::NonlinearFactor::shared_ptr clone() const override {
        return std::static_pointer_cast<gtsam::NonlinearFactor>(
            gtsam::NonlinearFactor::shared_ptr(new DepthFactor(*this)));
    }
};


class PsudoRangeFactor : public NoiseModelFactor2<gtam::NavState, gtsm::Point3> {
private:
    double measuredTOF_;
    double soundSpeed_;
    gtsam::Pose3 body_P_sensor_; // The lever arm and sensor orientation from ASV body frame to usbl frame

public:
    /// shorthand for a smart pointer to a factor
    typedef std::shared_ptr<PsudoRangeFactor> shared_ptr;

    PsudoRangeFactor(Key asvKey,
                      Key rovKey,
                      double tof,
                      double v_sound,
                      const SharedNoiseModel& model)
        : NoiseModelFactor2<NavState, Point3>(model, asvKey, rovKey),
          measuredTOF_(tof), soundSpeed_(v_sound) {}

    ~PsudoRangeFactor() override {}

    Vector evaluateError(
        const gtsam::NavState& asvPose,
        const gtsam::Point3& landmark,
        boost::optional<gtsam::Matrix&> H_asv = boost::none,
        boost::optional<gtsam::Matrix&> H_rov = boost::none) const override
    {
        // 1. Position of USBL sensor in world frame
        gtsam::Matrix66 H_pose_asv;
        gtsam::Pose3 world_P_sensor = asvState.pose().compose(body_P_sensor_, H_asv ? &H_pose_asv : nullptr);

        // 2. Distance from sensor to ROV
        gtsam::Matrix16 H_dist_sensor; // d(dist)/d(sensorPose)
        gtsam::Matrix13 H_dist_rov;    // d(dist)/d(rovPoint)
        double dist = world_P_sensor.range(rovPoint, 
                                           H_asv ? &H_dist_sensor : nullptr, 
                                           H_rov ? &H_dist_rov : nullptr);

        double expectedTOF = dist / soundSpeed_;

        // 3. Chain Rule for Jacobians
        if (H_asv) {
            // H_asv must be 1x9. Velocity parts (indices 6-8) are zero.
            gtsam::Matrix19 H_total = gtsam::Matrix::Zero(1, 9);
            H_total.block<1, 6>(0, 0) = (1.0 / soundSpeed_) * H_dist_sensor * H_pose_asv;
            *H_asv = H_total;
        }

        if (H_rov) {
            // H_rov is 1x3
            *H_rov = (1.0 / soundSpeed_) * H_dist_rov;
        }

        return (gtsam::Vector(1) << expectedTOF - measuredTOF_).finished();
    }

};

class ConstantVelocityFactor : public gtsam::NoiseModelFactor3<gtsam::Point3, gtsam::Vector3, gtsam::Point3> {
private:
    double dt_;

public:
    ConstantVelocityFactor(gtsam::Key p_t, gtsam::Key v_t, gtsam::Key p_t_plus_1, 
                              double dt, const gtsam::SharedNoiseModel& model)
        : NoiseModelFactor3(model, p_t, v_t, p_t_plus_1), dt_(dt) {}

    gtsam::Vector evaluateError(const gtsam::Point3& p_t, const gtsam::Vector3& v_t, const gtsam::Point3& p_next,
                                boost::optional<gtsam::Matrix&> H1 = boost::none,
                                boost::optional<gtsam::Matrix&> H2 = boost::none,
                                boost::optional<gtsam::Matrix&> H3 = boost::none) const override {
        
        if (H1) *H1 = gtsam::Matrix33::Identity();           // d_err/d_pt
        if (H2) *H2 = gtsam::Matrix33::Identity() * dt_;     // d_err/d_vt
        if (H3) *H3 = -gtsam::Matrix33::Identity();          // d_err/d_pnext

        return (p_t + v_t * dt_) - p_next;
    }
};
