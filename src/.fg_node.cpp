#include "microamp_factor_graph/fg_node.hpp"

// ------------------------------------------------------------
// Constructor
// ------------------------------------------------------------

FactorGraphNode::FactorGraphNode()
    : Node("microamp_factor_graph"), isam2_(gtsam::ISAM2Params()),
      gravity_(9.82145996), datum_initialised_(false),
      graph_initialised_(false), key_(0) {

  // ------------------------------------------------------------------
  // ROS parameters — tune without recompiling
  // ------------------------------------------------------------------
  declare_parameter("accel_noise", 1e-2);
  declare_parameter("gyro_noise", 1e-3);
  declare_parameter("accel_rw", 1e-4);
  declare_parameter("gyro_rw", 1e-5);
  declare_parameter("prior_pose_sigma", 1e-2);
  declare_parameter("prior_vel_sigma", 1e-2);
  declare_parameter("prior_bias_sigma", 1e-3);
  declare_parameter("gps_sigma_floor", 0.5); // metres
  declare_parameter("gps_sigma_max", 50.0);  // metres

  accel_noise_ = get_parameter("accel_noise").as_double();
  gyro_noise_ = get_parameter("gyro_noise").as_double();
  accel_rw_ = get_parameter("accel_rw").as_double();
  gyro_rw_ = get_parameter("gyro_rw").as_double();
  prior_pose_sigma_ = get_parameter("prior_pose_sigma").as_double();
  prior_vel_sigma_ = get_parameter("prior_vel_sigma").as_double();
  prior_bias_sigma_ = get_parameter("prior_bias_sigma").as_double();
  gps_sigma_floor_ = get_parameter("gps_sigma_floor").as_double();
  gps_sigma_max_ = get_parameter("gps_sigma_max").as_double();

  // ---------------------------
  // IMU Preintegration (NED)
  // ---------------------------

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

  // ---------------------------
  // ROS I/O
  // ---------------------------
  imu_sub_ = create_subscription<Imu>(
      "/imu/data", rclcpp::SensorDataQoS(),
      std::bind(&FactorGraphNode::imuCallback, this, std::placeholders::_1));

  gnss_sub_ = create_subscription<GNSSNavPvt>(
      "/gnss/navpvt", 10,
      std::bind(&FactorGraphNode::gnssCallback, this, std::placeholders::_1));

  state_pub_ = create_publisher<BoatState>("/state/boat", 10);
}

// -----------------------
// IMU CALLBACK
// -----------------------
void FactorGraphNode::imuCallback(const Imu::SharedPtr msg) {
  rclcpp::Time stamp = msg->header.stamp;

  // Store raw gyro z for yaw-rate publishing (bias correction applied later)
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

// ------------------------------------------------------------
// Datum initialisation (FIRST GNSS FIX)
// ------------------------------------------------------------
void FactorGraphNode::initializeDatumFromGNSS(const GNSSNavPvt &msg) {
  ned_.setDatum(msg.lat, msg.lon, msg.height);
  datum_initialised_ = true;
  RCLCPP_INFO(get_logger(), "Datum initialised: lat=%.8f lon=%.8f h=%.2f",
              msg.lat, msg.lon, msg.height);
}

// -----------------------
// GNSS CALLBACK
// -----------------------
void FactorGraphNode::gnssCallback(const GNSSNavPvt::SharedPtr msg) {

  // ------------------------------------------------------------------
  // 1. Fix-quality gate
  //    fix_type: 0=no fix, 1=dead reck, 2=2D, 3=3D, 4=GNSS+DR, 5=time
  //    flags bit 0: gnssFixOK (carrier solution valid)
  // ------------------------------------------------------------------
  constexpr uint8_t FIX_3D = 3;
  if (msg->fix_type < FIX_3D) {
    RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 5000,
                         "Waiting for 3D fix (current fix_type=%d)",
                         msg->fix_type);
    return;
  }
  if (!msg->gnss_fix_ok) {
    RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 5000,
                         "gnss_fix_ok not set, skipping");
    return;
  }

  // ------------------------------------------------------------------
  // 2. Datum
  // ------------------------------------------------------------------
  if (!datum_initialised_) {
    initializeDatumFromGNSS(*msg);
  }

  // ------------------------------------------------------------------
  // 3. Graph init
  // ------------------------------------------------------------------
  if (!graph_initialised_) {
    initializeGraph();
    graph_initialised_ = true;
    return; // wait for next fix to add the first factor
  }

  std::lock_guard<std::mutex> lock(graph_mutex_);

  // ------------------------------------------------------------------
  // 4. NED position from GNSS
  // ------------------------------------------------------------------

  double n, e, d;
  ned_.gnssToNED(msg->lat, msg->lon, msg->height, n, e, d);

  // h_acc is in mm — convert to metres and clamp
  double pos_sigma = std::clamp(static_cast<double>(msg->h_acc),
                                gps_sigma_floor_, gps_sigma_max_);

  uint64_t k_next = key_ + 1;

  // ------------------------------------------------------------------
  // 5. IMU factor
  // ------------------------------------------------------------------
  graph_.add(gtsam::CombinedImuFactor(X(key_), V(key_), X(k_next), V(k_next),
                                      B(key_), B(k_next), *pim_));

  // ------------------------------------------------------------------
  // 6. GPS factor
  //
  //    GTSAM's GPSFactor uses a navigation (NED) frame when GTSAM is
  //    built with -DGTSAM_USE_QUATERNIONS=OFF and the PreintegrationParams
  //    are created with MakeSharedD() (D = Down = NED gravity convention).
  //    We pass (north, east, down) directly — see GTSAM_NED note below.
  //
  //    If your GTSAM was built with the default ENU convention
  //    (MakeSharedU), convert here: Point3(e, n, -d).
  // ------------------------------------------------------------------
  auto gps_noise = gtsam::noiseModel::Isotropic::Sigma(3, pos_sigma);
  graph_.add(gtsam::GPSFactor(X(k_next), gtsam::Point3(n, e, d), gps_noise));

  // ------------------------------------------------------------------
  // 7. Initial value prediction via IMU
  // ------------------------------------------------------------------
  gtsam::Values est = updateAndGetEstimate();

  // Guard: keys must exist before we access them
  if (!est.exists(X(key_)) || !est.exists(V(key_)) || !est.exists(B(key_))) {
    RCLCPP_ERROR(get_logger(), "Expected keys missing from estimate at key=%lu",
                 key_);
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
  // 8. iSAM2 update
  // ------------------------------------------------------------------
  isam2_.update(graph_, values_);
  // Extra update pass helps iSAM2 converge on nonlinear factors
  isam2_.update();

  graph_.resize(0);
  values_.clear();

  // ------------------------------------------------------------------
  // 9. Bias reset for next integration window
  // ------------------------------------------------------------------
  bias_ = isam2_.calculateEstimate<gtsam::imuBias::ConstantBias>(B(k_next));
  pim_->resetIntegrationAndSetBias(bias_);

  key_ = k_next;

  // ------------------------------------------------------------------
  // 10. Publish — pass the already-computed estimate to avoid a second
  //     calculateEstimate() call this cycle
  // ------------------------------------------------------------------
  gtsam::Values final_est = isam2_.calculateEstimate();
  publishState(final_est);
}

// -----------------------
// INITIAL GRAPH SETUP  (priors at key 0)
// -----------------------
void FactorGraphNode::initializeGraph() {

  std::lock_guard<std::mutex> lock(graph_mutex_);

  graph_ = gtsam::NonlinearFactorGraph();
  values_ = gtsam::Values();

  // Identity pose at NED origin (= datum point)
  gtsam::Pose3 priorPose;
  gtsam::Vector3 priorVel(0.0, 0.0, 0.0);
  gtsam::imuBias::ConstantBias priorBias;

  // Pose prior — (roll, pitch, yaw, x, y, z) sigmas
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

  RCLCPP_INFO(get_logger(), "Factor graph initialised with priors at key 0.");
}

// ============================================================
// GET CURRENT ESTIMATE  (no side-effects on member state)
// ============================================================
gtsam::Values FactorGraphNode::updateAndGetEstimate() {
  return isam2_.calculateEstimate();
}

// -----------------------
// PUBLISH BoatState
// -----------------------
void FactorGraphNode::publishState(const gtsam::Values &est) {
  if (!est.exists(X(key_)) || !est.exists(V(key_)) || !est.exists(B(key_))) {
    return;
  }
  auto pose = est.at<gtsam::Pose3>(X(key_));
  auto vel = est.at<gtsam::Vector3>(V(key_));
  auto bias = est.at<gtsam::imuBias::ConstantBias>(B(key_));

  // Yaw from rotation matrix (NED: yaw = atan2(R10, R00))
  double yaw = std::atan2(pose.rotation().matrix()(1, 0),
                          pose.rotation().matrix()(0, 0));

  BoatState s;
  s.header.stamp = now();

  // NED position (metres from datum)
  s.x = pose.translation().x(); // North
  s.y = pose.translation().y(); // East
  s.yaw = yaw;

  // Body-frame velocities (approximation: NED ≈ body for small heel/trim)
  s.surge = vel.x(); // North velocity
  s.sway = vel.y();  // East  velocity

  // Yaw rate = raw gyro z  minus  optimised z-gyro bias
  s.yaw_r = last_gyro_z_ - bias.gyroscope()(2);

  state_pub_->publish(s);
}

int main(int argc, char **argv) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<FactorGraphNode>());
  rclcpp::shutdown();
  return 0;
}