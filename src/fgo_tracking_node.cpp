#include "microamp_fgo_rov_tracking/fgo_tracking_node.hpp"

#include <gtsam/geometry/Point3.h>
#include <gtsam/nonlinear/NonlinearFactor.h>
#include <gtsam/base/numericalDerivative.h>

// TODO: Move custom factors to separate files for cleanliness
// ============================================================
// CUSTOM FACTOR: USBL Bearing (Azimuth + Elevation)
// ============================================================
// Measures bearing from ASV USBL sensor to ROV position
// State: ASV pose X(k), ROV position R(k)
// Measurement: [azimuth, elevation] in radians

class USBLBearingFactor
    : public gtsam::NoiseModelFactor2<gtsam::Pose3, gtsam::Point3> {
private:
  double measured_azimuth_;   // radians, NED frame (0 = North, π/2 = East)
  double measured_elevation_; // radians (positive = above horizontal)
  gtsam::Point3 usbl_offset_; // USBL sensor offset in ASV body frame

public:
  USBLBearingFactor(gtsam::Key asv_pose_key, gtsam::Key rov_pos_key,
                    double azimuth, double elevation,
                    const gtsam::Point3 &usbl_offset,
                    const gtsam::SharedNoiseModel &model)
      : NoiseModelFactor2<gtsam::Pose3, gtsam::Point3>(model, asv_pose_key,
                                                        rov_pos_key),
        measured_azimuth_(azimuth), measured_elevation_(elevation),
        usbl_offset_(usbl_offset) {}

  // Error function: returns [azimuth_error, elevation_error]
  gtsam::Vector evaluateError(
      const gtsam::Pose3 &asv_pose, const gtsam::Point3 &rov_pos,
      boost::optional<gtsam::Matrix &> H1 = boost::none,
      boost::optional<gtsam::Matrix &> H2 = boost::none) const override {

    // 1. Compute USBL sensor position in NED frame
    gtsam::Point3 usbl_pos_ned = asv_pose.transformFrom(usbl_offset_);

    // 2. Relative vector from USBL to ROV (in NED)
    gtsam::Point3 rel = rov_pos - usbl_pos_ned;
    double n = rel.x();
    double e = rel.y();
    double d = rel.z();

    // 3. Predicted bearing
    double horiz_range = std::sqrt(n * n + e * e);
    double predicted_azimuth = std::atan2(e, n); // NED: 0=North, π/2=East
    double predicted_elevation = std::atan2(-d, horiz_range); // positive = up

    // 4. Angular error (wrap to [-π, π])
    auto wrapAngle = [](double a) {
      while (a > M_PI)
        a -= 2.0 * M_PI;
      while (a < -M_PI)
        a += 2.0 * M_PI;
      return a;
    };

    double az_error = wrapAngle(predicted_azimuth - measured_azimuth_);
    double el_error = wrapAngle(predicted_elevation - measured_elevation_);

    // 5. Numerical derivatives (GTSAM will use these for optimization)
    // TODO: Implement analytical Jacobians (see eskf for reference) for better performance
    if (H1) {
      *H1 = gtsam::numericalDerivative21<gtsam::Vector, gtsam::Pose3,
                                         gtsam::Point3>(
          [this](const gtsam::Pose3 &p, const gtsam::Point3 &r) {
            return this->evaluateError(p, r);
          },
          asv_pose, rov_pos);
    }
    if (H2) {
      *H2 = gtsam::numericalDerivative22<gtsam::Vector, gtsam::Pose3,
                                         gtsam::Point3>(
          [this](const gtsam::Pose3 &p, const gtsam::Point3 &r) {
            return this->evaluateError(p, r);
          },
          asv_pose, rov_pos);
    }

    return (gtsam::Vector(2) << az_error, el_error).finished();
  }
};

// ============================================================
// CUSTOM FACTOR: Acoustic Range
// ============================================================
// Measures distance between ASV USBL and ROV

class AcousticRangeFactor
    : public gtsam::NoiseModelFactor2<gtsam::Pose3, gtsam::Point3> {
private:
  double measured_range_;
  gtsam::Point3 usbl_offset_;

public:
  AcousticRangeFactor(gtsam::Key asv_pose_key, gtsam::Key rov_pos_key,
                      double range, const gtsam::Point3 &usbl_offset,
                      const gtsam::SharedNoiseModel &model)
      : NoiseModelFactor2<gtsam::Pose3, gtsam::Point3>(model, asv_pose_key,
                                                        rov_pos_key),
        measured_range_(range), usbl_offset_(usbl_offset) {}

  gtsam::Vector evaluateError(
      const gtsam::Pose3 &asv_pose, const gtsam::Point3 &rov_pos,
      boost::optional<gtsam::Matrix &> H1 = boost::none,
      boost::optional<gtsam::Matrix &> H2 = boost::none) const override {

    gtsam::Point3 usbl_pos_ned = asv_pose.transformFrom(usbl_offset_);
    gtsam::Point3 rel = rov_pos - usbl_pos_ned;
    double predicted_range = rel.norm();

    if (H1) {
      *H1 = gtsam::numericalDerivative21<gtsam::Vector, gtsam::Pose3,
                                         gtsam::Point3>(
          [this](const gtsam::Pose3 &p, const gtsam::Point3 &r) {
            return this->evaluateError(p, r);
          },
          asv_pose, rov_pos);
    }
    if (H2) {
      *H2 = gtsam::numericalDerivative22<gtsam::Vector, gtsam::Pose3,
                                         gtsam::Point3>(
          [this](const gtsam::Pose3 &p, const gtsam::Point3 &r) {
            return this->evaluateError(p, r);
          },
          asv_pose, rov_pos);
    }

    return (gtsam::Vector(1) << (predicted_range - measured_range_))
        .finished();
  }
};

// ============================================================
// CONSTRUCTOR
// ============================================================

FactorGraphTrackingNode::FactorGraphTrackingNode()
    : Node("microamp_factor_graph_expanded"), isam2_(gtsam::ISAM2Params()),
      gravity_(9.82145996), datum_initialised_(false),
      graph_initialised_(false), rov_initialised_(false), key_(0) {

  // ------------------------------------------------------------------
  // ROS parameters — BOAT
  // ------------------------------------------------------------------
  declare_parameter("accel_noise", 1e-2);
  declare_parameter("gyro_noise", 1e-3);
  declare_parameter("accel_rw", 1e-4);
  declare_parameter("gyro_rw", 1e-5);
  declare_parameter("prior_pose_sigma", 1e-2);
  declare_parameter("prior_vel_sigma", 1e-2);
  declare_parameter("prior_bias_sigma", 1e-3);
  declare_parameter("gps_sigma_floor", 0.5);
  declare_parameter("gps_sigma_max", 50.0);

  accel_noise_ = get_parameter("accel_noise").as_double();
  gyro_noise_ = get_parameter("gyro_noise").as_double();
  accel_rw_ = get_parameter("accel_rw").as_double();
  gyro_rw_ = get_parameter("gyro_rw").as_double();
  prior_pose_sigma_ = get_parameter("prior_pose_sigma").as_double();
  prior_vel_sigma_ = get_parameter("prior_vel_sigma").as_double();
  prior_bias_sigma_ = get_parameter("prior_bias_sigma").as_double();
  gps_sigma_floor_ = get_parameter("gps_sigma_floor").as_double();
  gps_sigma_max_ = get_parameter("gps_sigma_max").as_double();

  // ------------------------------------------------------------------
  // ROS parameters — ROV
  // ------------------------------------------------------------------
  declare_parameter("rov_prior_pos_sigma", 10.0);   // metres
  declare_parameter("rov_prior_vel_sigma", 1.0);    // m/s
  declare_parameter("rov_process_vel_sigma", 0.5);  // constant-vel noise m/s
  declare_parameter("usbl_azimuth_sigma", 0.05);    // radians (~3 deg)
  declare_parameter("usbl_elevation_sigma", 0.05);  // radians
  declare_parameter("acoustic_range_sigma", 1.0);   // metres
  declare_parameter("rov_depth_sigma", 0.2);        // metres

  // USBL sensor offset (x, y, z) in boat body frame
  declare_parameter("usbl_offset_x", 0.0);
  declare_parameter("usbl_offset_y", 0.0);
  declare_parameter("usbl_offset_z", 1.5); // 1.5m below boat IMU

  rov_prior_pos_sigma_ = get_parameter("rov_prior_pos_sigma").as_double();
  rov_prior_vel_sigma_ = get_parameter("rov_prior_vel_sigma").as_double();
  rov_process_vel_sigma_ = get_parameter("rov_process_vel_sigma").as_double();
  usbl_azimuth_sigma_ = get_parameter("usbl_azimuth_sigma").as_double();
  usbl_elevation_sigma_ = get_parameter("usbl_elevation_sigma").as_double();
  acoustic_range_sigma_ = get_parameter("acoustic_range_sigma").as_double();
  rov_depth_sigma_ = get_parameter("rov_depth_sigma").as_double();

  double ux = get_parameter("usbl_offset_x").as_double();
  double uy = get_parameter("usbl_offset_y").as_double();
  double uz = get_parameter("usbl_offset_z").as_double();
  usbl_offset_ = gtsam::Point3(ux, uy, uz);

  // ------------------------------------------------------------------
  // IMU Preintegration (NED)
  // ------------------------------------------------------------------
  auto params =
      gtsam::PreintegratedCombinedMeasurements::Params::MakeSharedD(gravity_);

  params->accelerometerCovariance = accel_noise_ * accel_noise_ * gtsam::I_3x3;
  params->gyroscopeCovariance = gyro_noise_ * gyro_noise_ * gtsam::I_3x3;
  params->integrationCovariance = 1e-8 * gtsam::I_3x3;
  params->biasAccCovariance = accel_rw_ * accel_rw_ * gtsam::I_3x3;
  params->biasOmegaCovariance = gyro_rw_ * gyro_rw_ * gtsam::I_3x3;
  params->biasAccOmegaInt = 1e-3 * gtsam::I_6x6;

  bias_ = gtsam::imuBias::ConstantBias();
  pim_ =
      std::make_unique<gtsam::PreintegratedCombinedMeasurements>(params, bias_);

  // ------------------------------------------------------------------
  // ROS I/O — BOAT
  // ------------------------------------------------------------------
  imu_sub_ = create_subscription<Imu>(
      "/imu/data", rclcpp::SensorDataQoS(),
      std::bind(&FactorGraphTrackingNode::imuCallback, this,
                std::placeholders::_1));

  gnss_sub_ = create_subscription<GNSSNavPvt>(
      "/gnss/navpvt", 10,
      std::bind(&FactorGraphTrackingNode::gnssCallback, this,
                std::placeholders::_1));

  state_pub_ = create_publisher<BoatState>("/state/boat", 10);

  // ------------------------------------------------------------------
  // ROS I/O — ROV
  // ------------------------------------------------------------------
  usbl_sub_ = create_subscription<USBLMeasurement>(
      "/rov/bearing", 10,
      std::bind(&FactorGraphTrackingNode::usblCallback, this,
                std::placeholders::_1));

  range_sub_ = create_subscription<AcousticRange>(
      "/rov/range", 10,
      std::bind(&FactorGraphTrackingNode::rangeCallback, this,
                std::placeholders::_1));

  depth_sub_ = create_subscription<ROVDepth>(
      "/rov/depth", 10,
      std::bind(&FactorGraphTrackingNode::depthCallback, this,
                std::placeholders::_1));

  rov_state_pub_ = create_publisher<ROVState>("/state/rov", 10);

  RCLCPP_INFO(get_logger(), "Expanded factor graph node initialized.");
}

// ============================================================
// BOAT CALLBACKS (same as original)
// ============================================================

void FactorGraphTrackingNode::imuCallback(const Imu::SharedPtr msg) {
  rclcpp::Time stamp = msg->header.stamp;
  last_gyro_z_ = msg->angular_velocity.z;

  if (!imu_initialised_) {
    last_imu_time_ = stamp;
    imu_initialised_ = true;
    return;
  }

  double dt = (stamp - last_imu_time_).seconds();
  last_imu_time_ = stamp;

  if (dt <= 0.0) {
    RCLCPP_WARN(get_logger(), "Non-positive IMU dt=%.6f, skipping", dt);
    return;
  }

  gtsam::Vector3 acc(msg->linear_acceleration.x, msg->linear_acceleration.y,
                     msg->linear_acceleration.z);
  gtsam::Vector3 gyro(msg->angular_velocity.x, msg->angular_velocity.y,
                      msg->angular_velocity.z);

  std::lock_guard<std::mutex> lock(graph_mutex_);
  pim_->integrateMeasurement(acc, gyro, dt);
}

void FactorGraphTrackingNode::initializeDatumFromGNSS(const GNSSNavPvt &msg) {
  ned_.setDatum(msg.lat, msg.lon, msg.height);
  datum_initialised_ = true;
  RCLCPP_INFO(get_logger(), "Datum initialised: lat=%.8f lon=%.8f h=%.2f",
              msg.lat, msg.lon, msg.height);
}

void FactorGraphTrackingNode::gnssCallback(const GNSSNavPvt::SharedPtr msg) {
  constexpr uint8_t FIX_3D = 3;
  if (msg->fix_type < FIX_3D || !msg->gnss_fix_ok) {
    return;
  }

  if (!datum_initialised_) {
    initializeDatumFromGNSS(*msg);
  }

  if (!graph_initialised_) {
    initializeGraph();
    graph_initialised_ = true;
    return;
  }

  std::lock_guard<std::mutex> lock(graph_mutex_);

  double n, e, d;
  ned_.gnssToNED(msg->lat, msg->lon, msg->height, n, e, d);

  double pos_sigma = std::clamp(static_cast<double>(msg->h_acc) / 1000.0,
                                gps_sigma_floor_, gps_sigma_max_);

  uint64_t k_next = key_ + 1;

  // IMU factor (boat state propagation)
  graph_.add(gtsam::CombinedImuFactor(X(key_), V(key_), X(k_next), V(k_next),
                                      B(key_), B(k_next), *pim_));

  // GPS factor
  auto gps_noise = gtsam::noiseModel::Isotropic::Sigma(3, pos_sigma);
  graph_.add(gtsam::GPSFactor(X(k_next), gtsam::Point3(n, e, d), gps_noise));

  // Predict boat state
  gtsam::Values est = updateAndGetEstimate();

  if (!est.exists(X(key_)) || !est.exists(V(key_)) || !est.exists(B(key_))) {
    RCLCPP_ERROR(get_logger(), "Boat keys missing at key=%lu", key_);
    graph_.resize(0);
    values_.clear();
    pim_->resetIntegrationAndSetBias(bias_);
    return;
  }

  auto prev_pose = est.at<gtsam::Pose3>(X(key_));
  auto prev_vel = est.at<gtsam::Vector3>(V(key_));
  auto prev_bias = est.at<gtsam::imuBias::ConstantBias>(B(key_));

  gtsam::NavState prev_state(prev_pose, prev_vel);
  auto predicted = pim_->predict(prev_state, prev_bias);

  values_.insert(X(k_next), predicted.pose());
  values_.insert(V(k_next), predicted.v());
  values_.insert(B(k_next), prev_bias);

  // ------------------------------------------------------------------
  // ROV STATE PROPAGATION (constant velocity)
  // ------------------------------------------------------------------
  if (rov_initialised_ && est.exists(R(key_)) && est.exists(W(key_))) {
    auto prev_rov_pos = est.at<gtsam::Point3>(R(key_));
    auto prev_rov_vel = est.at<gtsam::Vector3>(W(key_));

    // Constant velocity prediction
    double dt_rov = (msg->header.stamp - last_rov_update_time_).seconds();
    if (dt_rov > 0.0) {
      gtsam::Point3 predicted_rov_pos =
          prev_rov_pos + gtsam::Point3(prev_rov_vel * dt_rov);

      values_.insert(R(k_next), predicted_rov_pos);
      values_.insert(W(k_next), prev_rov_vel);

      // Constant velocity process noise (velocity shouldn't change much)
      auto vel_process_noise =
          gtsam::noiseModel::Isotropic::Sigma(3, rov_process_vel_sigma_);
      graph_.add(gtsam::BetweenFactor<gtsam::Vector3>(
          W(key_), W(k_next), gtsam::Vector3::Zero(), vel_process_noise));

      last_rov_update_time_ = msg->header.stamp;
    } else {
      // Fallback: same state
      values_.insert(R(k_next), prev_rov_pos);
      values_.insert(W(k_next), prev_rov_vel);
    }
  }

  // Update iSAM2
  isam2_.update(graph_, values_);
  isam2_.update();

  graph_.resize(0);
  values_.clear();

  bias_ = isam2_.calculateEstimate<gtsam::imuBias::ConstantBias>(B(k_next));
  pim_->resetIntegrationAndSetBias(bias_);

  key_ = k_next;

  gtsam::Values final_est = isam2_.calculateEstimate();
  publishBoatState(final_est);
  if (rov_initialised_) {
    publishROVState(final_est);
  }
}

// ============================================================
// ROV CALLBACKS
// ============================================================

void FactorGraphTrackingNode::usblCallback(
    const USBLMeasurement::SharedPtr msg) {
  if (!graph_initialised_) {
    return;
  }

  // Buffer the measurement
  USBLBuffer buf;
  buf.stamp = msg->header.stamp;
  buf.azimuth = msg->azimuth;           // radians
  buf.elevation = msg->elevation;       // radians
  buf.azimuth_std = msg->azimuth_std;   // radians
  buf.elevation_std = msg->elevation_std; // radians
  pending_usbl_ = buf;

  // Try to add factor if we have boat state at current key
  std::lock_guard<std::mutex> lock(graph_mutex_);

  if (!rov_initialised_) {
    initializeROVState();
  }

  // Add USBL bearing factor
  gtsam::Vector2 sigmas(buf.azimuth_std, buf.elevation_std);
  auto usbl_noise = gtsam::noiseModel::Diagonal::Sigmas(sigmas);

  graph_.add(boost::make_shared<USBLBearingFactor>(
      X(key_), R(key_), buf.azimuth, buf.elevation, usbl_offset_, usbl_noise));

  RCLCPP_DEBUG(get_logger(), "Added USBL factor: az=%.3f el=%.3f", buf.azimuth,
               buf.elevation);
}

void FactorGraphTrackingNode::rangeCallback(
    const AcousticRange::SharedPtr msg) {
  if (!graph_initialised_ || !rov_initialised_) {
    return;
  }

  std::lock_guard<std::mutex> lock(graph_mutex_);

  auto range_noise =
      gtsam::noiseModel::Isotropic::Sigma(1, msg->range_std > 0.0 
                                              ? msg->range_std 
                                              : acoustic_range_sigma_);

  graph_.add(boost::make_shared<AcousticRangeFactor>(
      X(key_), R(key_), msg->range, usbl_offset_, range_noise));

  RCLCPP_DEBUG(get_logger(), "Added range factor: %.2f m", msg->range);
}

void FactorGraphTrackingNode::depthCallback(const ROVDepth::SharedPtr msg) {
  if (!graph_initialised_ || !rov_initialised_) {
    return;
  }

  std::lock_guard<std::mutex> lock(graph_mutex_);

  // Depth is the DOWN component (z) in NED frame
  // We use a prior on just the z-component of ROV position
  // GTSAM doesn't have a 1D position prior, so we'll use a custom approach

  // Option 1: Use a very loose XY sigma, tight Z sigma (hacky but works)
  gtsam::Vector3 measured_pos(
      0.0, 0.0, msg->depth); // We don't know N, E — only care about D

  gtsam::Vector3 sigmas(1e6, 1e6, msg->depth_std > 0.0 ? msg->depth_std 
                                                        : rov_depth_sigma_);
  auto depth_noise = gtsam::noiseModel::Diagonal::Sigmas(sigmas);

  // This is a soft constraint: only the D component is informative
  graph_.add(gtsam::PriorFactor<gtsam::Point3>(R(key_), measured_pos,
                                               depth_noise));

  RCLCPP_DEBUG(get_logger(), "Added depth factor: %.2f m", msg->depth);
}

// ============================================================
// ROV INITIALIZATION
// ============================================================

void FactorGraphTrackingNode::initializeROVState() {
  if (rov_initialised_)
    return;

  // Initialize ROV at an arbitrary position relative to boat
  // (50m ahead, 20m down as initial guess)
  gtsam::Values est = updateAndGetEstimate();

  if (!est.exists(X(key_))) {
    RCLCPP_WARN(get_logger(),
                "Cannot initialize ROV: boat state not available");
    return;
  }

  auto boat_pose = est.at<gtsam::Pose3>(X(key_));

  // Initial guess: 50m North, 0m East, 20m Down relative to boat
  gtsam::Point3 rov_initial_offset(50.0, 0.0, 20.0);
  gtsam::Point3 rov_pos_ned = boat_pose.translation() + rov_initial_offset;

  gtsam::Vector3 rov_vel_initial(0.0, 0.0, 0.0); // stationary initially

  // Add priors for ROV state at current key
  auto rov_pos_noise =
      gtsam::noiseModel::Isotropic::Sigma(3, rov_prior_pos_sigma_);
  auto rov_vel_noise =
      gtsam::noiseModel::Isotropic::Sigma(3, rov_prior_vel_sigma_);

  graph_.add(
      gtsam::PriorFactor<gtsam::Point3>(R(key_), rov_pos_ned, rov_pos_noise));
  graph_.add(gtsam::PriorFactor<gtsam::Vector3>(W(key_), rov_vel_initial,
                                                 rov_vel_noise));

  values_.insert(R(key_), rov_pos_ned);
  values_.insert(W(key_), rov_vel_initial);

  last_rov_update_time_ = now();
  rov_initialised_ = true;

  RCLCPP_INFO(get_logger(), "ROV state initialized at key %lu", key_);
}

// ============================================================
// GRAPH INITIALIZATION
// ============================================================

void FactorGraphTrackingNode::initializeGraph() {
  std::lock_guard<std::mutex> lock(graph_mutex_);

  graph_ = gtsam::NonlinearFactorGraph();
  values_ = gtsam::Values();

  gtsam::Pose3 priorPose;
  gtsam::Vector3 priorVel(0.0, 0.0, 0.0);
  gtsam::imuBias::ConstantBias priorBias;

  auto pose_noise = gtsam::noiseModel::Diagonal::Sigmas(
      (gtsam::Vector(6) << prior_pose_sigma_, prior_pose_sigma_,
       prior_pose_sigma_, prior_pose_sigma_, prior_pose_sigma_,
       prior_pose_sigma_)
          .finished());

  graph_.add(gtsam::PriorFactor<gtsam::Pose3>(X(key_), priorPose, pose_noise));
  graph_.add(gtsam::PriorFactor<gtsam::Vector3>(
      V(key_), priorVel,
      gtsam::noiseModel::Isotropic::Sigma(3, prior_vel_sigma_)));
  graph_.add(gtsam::PriorFactor<gtsam::imuBias::ConstantBias>(
      B(key_), priorBias,
      gtsam::noiseModel::Isotropic::Sigma(6, prior_bias_sigma_)));

  values_.insert(X(key_), priorPose);
  values_.insert(V(key_), priorVel);
  values_.insert(B(key_), priorBias);

  isam2_.update(graph_, values_);
  graph_.resize(0);
  values_.clear();

  RCLCPP_INFO(get_logger(), "Boat factor graph initialized at key 0.");
}

gtsam::Values FactorGraphTrackingNode::updateAndGetEstimate() {
  return isam2_.calculateEstimate();
}

// ============================================================
// PUBLISH STATES
// ============================================================

void FactorGraphTrackingNode::publishBoatState(const gtsam::Values &est) {
  if (!est.exists(X(key_)) || !est.exists(V(key_)) || !est.exists(B(key_))) {
    return;
  }

  auto pose = est.at<gtsam::Pose3>(X(key_));
  auto vel = est.at<gtsam::Vector3>(V(key_));
  auto bias = est.at<gtsam::imuBias::ConstantBias>(B(key_));

  double yaw = std::atan2(pose.rotation().matrix()(1, 0),
                          pose.rotation().matrix()(0, 0));

  BoatState s;
  s.header.stamp = now();
  s.x = pose.translation().x();
  s.y = pose.translation().y();
  s.yaw = yaw;
  s.surge = vel.x();
  s.sway = vel.y();
  s.yaw_r = last_gyro_z_ - bias.gyroscope()(2);

  state_pub_->publish(s);
}

void FactorGraphTrackingNode::publishROVState(const gtsam::Values &est) {
  if (!est.exists(R(key_)) || !est.exists(W(key_))) {
    return;
  }

  auto rov_pos = est.at<gtsam::Point3>(R(key_));
  auto rov_vel = est.at<gtsam::Vector3>(W(key_));

  ROVState rs;
  rs.header.stamp = now();
  rs.x = rov_pos.x(); // North
  rs.y = rov_pos.y(); // East
  rs.z = rov_pos.z(); // Down
  rs.vx = rov_vel.x();
  rs.vy = rov_vel.y();
  rs.vz = rov_vel.z();

  rov_state_pub_->publish(rs);
}

int main(int argc, char **argv) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<FactorGraphTrackingNode>());
  rclcpp::shutdown();
  return 0;
}
