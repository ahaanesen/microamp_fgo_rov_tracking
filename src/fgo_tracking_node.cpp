#include "microamp_fgo_rov_tracking/fgo_tracking_node.hpp"
#include "microamp_fgo_rov_tracking/rov_factors.hpp"
#include "microamp_fgo_rov_tracking/config.hpp"

#include <gtsam/geometry/Point3.h>
#include <gtsam/nonlinear/NonlinearFactor.h>
#include <gtsam/base/numericalDerivative.h>

#include <algorithm>
#include <cmath>


// ============================================================
// CONSTRUCTOR, DESTRUCTOR, AND INITIALIZATION
// ============================================================

FactorGraphTrackingNode::FactorGraphTrackingNode()
    : Node("microamp_factor_graph_tracking"), isam2_(gtsam::ISAM2Params()),
      graph_initialised_(false), asv_index_(0), datum_initialised_(false) {
  
  loadConfigurations();
  max_imu_buffer_duration_ = env_config_.imu_buffer_duration_sec;

  // ------------------------------------------------------------------
  // IMU Preintegration (NED)
  // ------------------------------------------------------------------
  pim_params_ =
    gtsam::PreintegratedCombinedMeasurements::Params::MakeSharedD(env_config_.gravity);


  pim_params_->accelerometerCovariance = fgo_config_.accel_noise * fgo_config_.accel_noise * gtsam::I_3x3;
  pim_params_->gyroscopeCovariance = fgo_config_.gyro_noise * fgo_config_.gyro_noise * gtsam::I_3x3;
  pim_params_->integrationCovariance = 1e-8 * gtsam::I_3x3;
  pim_params_->biasAccCovariance = fgo_config_.accel_rw * fgo_config_.accel_rw * gtsam::I_3x3;
  pim_params_->biasOmegaCovariance = fgo_config_.gyro_rw * fgo_config_.gyro_rw * gtsam::I_3x3;
  pim_params_->biasAccOmegaInt = 1e-3 * gtsam::I_6x6;

  bias_ = gtsam::imuBias::ConstantBias();
  pim_ = std::make_unique<gtsam::PreintegratedCombinedMeasurements>(pim_params_, bias_);

  // ------------------------------------------------------------------
  // ROS I/O — BOAT
  // ------------------------------------------------------------------

  imu_sub_ = this->create_subscription<Imu>(
      topics_config_.imu, rclcpp::SensorDataQoS(),
      std::bind(&FactorGraphTrackingNode::imuCallback, this,
                std::placeholders::_1));

  gnss_sub_ = this->create_subscription<GNSSNavPvt>(
      topics_config_.gnss, 10,
      std::bind(&FactorGraphTrackingNode::gnssCallback, this,
                std::placeholders::_1));

  state_pub_ = this->create_publisher<BoatState>(topics_config_.boat_state_pub, 10);

  // ------------------------------------------------------------------
  // ROS I/O — ROV
  // ------------------------------------------------------------------
  acoustic_comm_sub_ = this->create_subscription<AcousticCommReceive>(
      topics_config_.acoustic_rx, 10,
      std::bind(&FactorGraphTrackingNode::acousticCommCallback, this,
                std::placeholders::_1));
  
  usbl_sub_ = this->create_subscription<USBLMessage>(
      topics_config_.usbl, 10,
      std::bind(&FactorGraphTrackingNode::usblCallback, this,
                std::placeholders::_1));


  rov_state_pub_ = this->create_publisher<ROVState>(topics_config_.rov_state_pub, 10);

  RCLCPP_INFO(get_logger(), "Expanded factor graph node initialized.");
}

void FactorGraphTrackingNode::loadConfigurations() {
  declareAndLoadTopics(*this, topics_config_);
  declareAndLoadEnv(*this, env_config_);
  declareAndLoadFgo(*this, fgo_config_);
}

FactorGraphTrackingNode::~FactorGraphTrackingNode() = default;

// ============================================================
// ASV CALLBACKS
// ============================================================

void FactorGraphTrackingNode::imuCallback(const Imu::SharedPtr msg) {
  // 1. Basic timestamp and raw gyro z handling (for yaw-rate output)
  rclcpp::Time stamp = msg->header.stamp;
  double current_time = stamp.seconds();
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

  // 2. Prepare GTSAM vectors
  gtsam::Vector3 acc(msg->linear_acceleration.x, msg->linear_acceleration.y,
                     msg->linear_acceleration.z);
  gtsam::Vector3 gyro(msg->angular_velocity.x, msg->angular_velocity.y,
                      msg->angular_velocity.z);

  // 3. Thread-safe operations
  std::lock_guard<std::mutex> lock(graph_mutex_);

  // A. Live integration for the NEXT real-time node (GNSS or next USBL)
  // Note: The actual bridging to the correct timestamp is handled in getAsvKeyAtTime() when we process a GNSS measurement, 
  // which calls getPimFromBuffer() to integrate the relevant IMU data from the buffer.
  // pim_->integrateMeasurement(acc, gyro, dt);

  // B. Store in historical buffer for delayed USBL/ROV measurements
  imu_buffer_.push_back({current_time, acc, gyro});

  // C. Prune buffer (remove data older than 10s)
  while (!imu_buffer_.empty() && (current_time - imu_buffer_.front().timestamp) > max_imu_buffer_duration_) {
    imu_buffer_.pop_front();
  }
}

void FactorGraphTrackingNode::initializeDatumFromGNSS(const GNSSNavPvt &msg) {
  ned_.setDatum(msg.lat, msg.lon, msg.height);
  datum_initialised_ = true;
  RCLCPP_INFO(get_logger(), "Datum initialised: lat=%.8f lon=%.8f h=%.2f",
              msg.lat, msg.lon, msg.height);
}

void FactorGraphTrackingNode::gnssCallback(const GNSSNavPvt::SharedPtr msg) {
  RCLCPP_INFO(get_logger(), "Received GNSS measurement at time %.2f with fix type %u and h_acc %f mm", 
              rclcpp::Time(msg->header.stamp).seconds(), msg->fix_type, msg->h_acc);
  constexpr uint8_t FIX_3D = 3;
  if (msg->fix_type < FIX_3D || !msg->gnss_fix_ok) {
    return;
  }

  if (!datum_initialised_) {
    initializeDatumFromGNSS(*msg);
  }

  std::lock_guard<std::mutex> lock(graph_mutex_);

  if (!graph_initialised_) {
    initializeGraphWithGNSS(msg);
    graph_initialised_ = true;
    return;
  }

  double current_gnss_time = rclcpp::Time(msg->header.stamp).seconds();
  gtsam::Key current_asv_key = getAsvKeyAtTime(current_gnss_time);
  uint64_t current_idx = gtsam::Symbol(current_asv_key).index();

  double n, e, d;
  ned_.gnssToNED(msg->lat, msg->lon, msg->height, n, e, d);
  double pos_sigma = std::clamp(static_cast<double>(msg->h_acc) / 1000.0,
                                fgo_config_.gps_sigma_floor, fgo_config_.gps_sigma_max);

  auto gps_noise = gtsam::noiseModel::Isotropic::Sigma(3, pos_sigma);
  graph_.add(gtsam::GPSFactor(X(current_idx), gtsam::Point3(n, e, d), gps_noise));

  isam2_.update(graph_, values_);
  // isam2_.update();
  graph_.resize(0);
  values_.clear();

  bias_ = isam2_.calculateEstimate<gtsam::imuBias::ConstantBias>(B(current_idx));
  pim_->resetIntegrationAndSetBias(bias_);

  last_asv_timestamp_ = current_gnss_time;
  asv_timeline_[last_asv_timestamp_] = X(current_idx);

  gtsam::Values final_est = isam2_.calculateEstimate();
  publishBoatState(final_est);
  // if (!rov_initialised_.empty()) {
  //   publishROVState(final_est);
  // }
  RCLCPP_INFO(get_logger(), "Processed GNSS measurement at time %.2f, position (%.2f, %.2f, %.2f)", 
              current_gnss_time, n, e, d);
}

// ============================================================
// HELPERS
// ============================================================
gtsam::PreintegratedCombinedMeasurements FactorGraphTrackingNode::getPimFromBuffer(
    double t_start, double t_end,
    const gtsam::imuBias::ConstantBias& bias) const {
  gtsam::PreintegratedCombinedMeasurements sub_pim(pim_params_, bias);
  
  double last_t = t_start;
  for (const auto& data : imu_buffer_) {
      if (data.timestamp > t_start && data.timestamp <= t_end) {
          double dt = data.timestamp - last_t;
          sub_pim.integrateMeasurement(data.acc, data.gyro, dt);
          last_t = data.timestamp;
      }
  }
  return sub_pim;
}

std::pair<double, double> FactorGraphTrackingNode::resolveAcousticTimestamps(
    const rclcpp::Time& header_stamp,
    uint64_t t_sent_us,
    uint64_t t_received_us) const {
  constexpr double MIN_EPOCH_SEC = 1.0e8;  // ~1973
  constexpr double MAX_EPOCH_SEC = 4.0e9;  // ~2096
  constexpr double MAX_HEADER_MISMATCH_SEC = 5.0;

  const double header_time = header_stamp.seconds();
  const double raw_t_s = static_cast<double>(t_sent_us) / 1e6;
  const double raw_t_r = static_cast<double>(t_received_us) / 1e6;
  const double tof = raw_t_r - raw_t_s;

  const bool sent_epoch_like = raw_t_s > MIN_EPOCH_SEC && raw_t_s < MAX_EPOCH_SEC;
  const bool recv_epoch_like = raw_t_r > MIN_EPOCH_SEC && raw_t_r < MAX_EPOCH_SEC;
  const bool recv_matches_header = std::abs(raw_t_r - header_time) < MAX_HEADER_MISMATCH_SEC;

  // If modem timestamps are not ROS-epoch aligned, anchor receive time to the ROS header.
  if (!(sent_epoch_like && recv_epoch_like && recv_matches_header)) {
    const double t_r = header_time;
    const double t_s = t_r - tof;
    return {t_s, t_r};
  }

  return {raw_t_s, raw_t_r};
}

// This function checks if an ASV node already exists at the target_time (within 1ms). If it does, it returns that key.
// If not, it creates a new ASV node at target_time by bridging from the last known ASV node using IMU data from the buffer.
gtsam::Key FactorGraphTrackingNode::getAsvKeyAtTime(double target_time) {
  RCLCPP_INFO(get_logger(), "Getting ASV key for target time %.2f, last index %lu", target_time, asv_index_);
    // 1. Check if a node at this exact timestamp (within 1ms) already exists
    auto it = asv_timeline_.lower_bound(target_time - 0.001);
    if (it != asv_timeline_.end() && std::abs(it->first - target_time) < 0.001) {
        return it->second; // Already exists (likely from a previous ROV in same TDMA slot)
    }

    if (target_time < last_asv_timestamp_ - 0.001) {
        auto upper = asv_timeline_.lower_bound(target_time);
        if (upper == asv_timeline_.begin()) {
            RCLCPP_WARN(get_logger(),
                        "Requested past ASV time %.3f before timeline start. Reusing earliest node.",
                        target_time);
            return upper->second;
        }
        if (upper == asv_timeline_.end()) {
            auto latest = std::prev(asv_timeline_.end());
            return latest->second;
        }

        auto lower = std::prev(upper);
        const double dt_upper = std::abs(upper->first - target_time);
        const double dt_lower = std::abs(lower->first - target_time);
        auto chosen = (dt_upper < dt_lower) ? upper : lower;
        RCLCPP_WARN(get_logger(),
                    "Requested past ASV time %.3f while latest ASV node is %.3f. Reusing nearest existing node at %.3f.",
                    target_time, last_asv_timestamp_, chosen->first);
        return chosen->second;
    }

    // 2. Find the most recent node in the graph
    double t_prev = last_asv_timestamp_;
    uint64_t idx_prev = asv_index_;
    uint64_t idx_curr = asv_index_ + 1;

    auto prev_pose = isam2_.calculateEstimate<gtsam::Pose3>(X(idx_prev));
    auto prev_vel  = isam2_.calculateEstimate<gtsam::Vector3>(V(idx_prev));
    auto prev_bias = isam2_.calculateEstimate<gtsam::imuBias::ConstantBias>(B(idx_prev));

    // 3. Bridge the gap using the IMU buffer with the parent node bias.
    auto sub_pim = getPimFromBuffer(t_prev, target_time, prev_bias);
    
    // 4. Predict the state at target_time
    gtsam::NavState prev_state(prev_pose, prev_vel);
    auto predicted = sub_pim.predict(prev_state, prev_bias);

    // 5. Add to Graph and Initial Values
    values_.insert(X(idx_curr), predicted.pose());
    values_.insert(V(idx_curr), predicted.v());
    values_.insert(B(idx_curr), prev_bias); // Start with previous bias

    graph_.add(gtsam::CombinedImuFactor(X(idx_prev), V(idx_prev), 
                                        X(idx_curr), V(idx_curr),
                                        B(idx_prev), B(idx_curr), sub_pim));

    // 6. Housekeeping
    asv_timeline_[target_time] = X(idx_curr);
    last_asv_timestamp_ = target_time;
    asv_index_ = idx_curr;

    // IMPORTANT: Reset your main PIM so it starts from this new 'tip'
    pim_->resetIntegrationAndSetBias(prev_bias); 

    RCLCPP_INFO(get_logger(), "Created new ASV node at time %.2f with index X(%lu)", target_time, idx_curr);
    return X(idx_curr);
}

// Helper function to generate unique keys per ROV
// Encodes prefix (R=position, W=velocity), ROV ID, and step count into a single Key
// Example: ROV position at step 5 for ROV ID 2 -> getRovKey('R', 2, 5)
gtsam::Key FactorGraphTrackingNode::getRovKey(unsigned char prefix, uint32_t rov_id, uint32_t step_count) {
    uint64_t packed_index = (uint64_t(rov_id) << 32) | step_count;
    return gtsam::Symbol(prefix, packed_index);
}

// ============================================================
// ROV CALLBACKS
// ============================================================

void FactorGraphTrackingNode::usblCallback(
    const USBLMessage::SharedPtr msg) {
  double header_time = rclcpp::Time(msg->header.stamp).seconds();
  RCLCPP_INFO(get_logger(), "Received USBL message for ROV %u at time %.2f", msg->rov_id, header_time);
  if (!graph_initialised_) {
    return;
  }

  uint8_t rov_id = msg->rov_id;
  double sound_speed = env_config_.sound_speed;
  auto [t_s, t_r] = resolveAcousticTimestamps(rclcpp::Time(msg->header.stamp), msg->t_sent, msg->t_received);
  double tof = t_r - t_s;
  if (tof <= 0.0 || tof > 5.0) {
    RCLCPP_WARN(get_logger(),
                "Invalid USBL TOF %.6f s for ROV %u (t_s=%.6f, t_r=%.6f), skipping update",
                tof, rov_id, t_s, t_r);
    return;
  }

  std::lock_guard<std::mutex> lock(graph_mutex_);

  if (!rov_initialised_[rov_id]) {
    // First time seeing this ROV, initialize its state in the graph
    Key rKey = getRovKey('R', rov_id, 0); // Position key for time step 0
    Key wKey = getRovKey('W', rov_id, 0); // Velocity key for time step 0
    initializeNewRov(rov_id, rKey, wKey, msg, sound_speed);
    return;
  }
  
  // 1. ASV SIDE: Use the ASV state at acoustic receive time.
  gtsam::Key asv_received = getAsvKeyAtTime(t_r);

  // 2. ROV SIDE: Handle the sequence
  uint32_t current_rov_step = rov_step_counters_[rov_id];
  gtsam::Key r_curr = getRovKey('R', rov_id, current_rov_step);
  gtsam::Key w_curr = getRovKey('W', rov_id, current_rov_step);
  gtsam::Key r_prev = getRovKey('R', rov_id, current_rov_step - 1);
  gtsam::Key w_prev = getRovKey('W', rov_id, current_rov_step - 1);

  double dt_rov = t_s - last_rov_timestamp_[rov_id];
  if (dt_rov <= 0.0) {
    RCLCPP_WARN(get_logger(),
                "Non-positive ROV dt=%.6f for ROV %u (t_s=%.6f, last=%.6f), skipping USBL update",
                dt_rov, rov_id, t_s, last_rov_timestamp_[rov_id]);
    return;
  }

  // 3. PREDICTION: Provide Initial Values for the Optimizer
  // We predict where the ROV is now based on its last known position and velocity
  gtsam::Point3 p_prev = isam2_.calculateEstimate<gtsam::Point3>(r_prev);
  gtsam::Vector3 v_prev = isam2_.calculateEstimate<gtsam::Vector3>(w_prev);

  values_.insert(r_curr, (p_prev + (v_prev * dt_rov)).eval()); // Linear prediction
  values_.insert(w_curr, v_prev.eval());                    // Assume constant velocity

  // // NB: no constraint on velocity. Adding a weak prior to prevent unbounded growth in early stages before we have good measurements. FGO crashes witout any velocity constraint.
  // auto weak_vel_noise = gtsam::noiseModel::Isotropic::Sigma(3, 1.0); // tune
  // graph_.add(gtsam::PriorFactor<gtsam::Vector3>(w_curr, v_prev, weak_vel_noise));
  
  // 4. MOTION MODEL FACTORS: Connect to the previous ROV state
  // Add Constant Velocity Factor: (P_prev, V_prev, P_curr)
  auto rov_noise = gtsam::noiseModel::Isotropic::Sigma(3, fgo_config_.rov_process_vel_sigma);
  graph_.add(boost::make_shared<ConstantVelocityFactor>(
      r_prev, w_prev, r_curr, dt_rov, rov_noise));
  // Also add a "Velocity Smoothness" factor (V_prev == V_curr)
  graph_.add(boost::make_shared<gtsam::BetweenFactor<gtsam::Vector3>>(
      w_prev, w_curr, gtsam::Vector3::Zero(), rov_noise));

  // 5. MEASUREMENT FACTORS
  // Add USBL factor
  gtsam::Point3 usbl_offset(env_config_.usbl_offset_x, env_config_.usbl_offset_y, env_config_.usbl_offset_z);
  double roll_rad = env_config_.usbl_roll_deg * M_PI / 180.0;
  double pitch_rad = env_config_.usbl_pitch_deg * M_PI / 180.0;
  double yaw_rad = env_config_.usbl_yaw_deg * M_PI / 180.0;
  gtsam::Rot3 usbl_rotation = gtsam::Rot3::RzRyRx(roll_rad, pitch_rad, yaw_rad);
  gtsam::Pose3 body_P_sensor(usbl_rotation, usbl_offset);
  auto usbl_noise = gtsam::noiseModel::Diagonal::Sigmas((gtsam::Vector(2) << fgo_config_.usbl_azimuth_sigma, fgo_config_.usbl_elevation_sigma).finished());
  graph_.add(boost::make_shared<UsblFactor>(asv_received, r_curr, msg->azimuth, msg->elevation, body_P_sensor, usbl_noise));

  // Add Depth factor
  auto depth_noise = gtsam::noiseModel::Isotropic::Sigma(1, fgo_config_.rov_depth_sigma);
  graph_.add(boost::make_shared<DepthFactor>(r_curr, msg->position.z, depth_noise));

  // Add Acoustic Range factor (Psuedo-Range)
  auto acoustic_noise = gtsam::noiseModel::Isotropic::Sigma(1, fgo_config_.acoustic_range_sigma / sound_speed); // Convert range sigma to time sigma using speed of sound
  graph_.add(boost::make_shared<PsudoRangeFactor>(asv_received, r_curr, tof, sound_speed, body_P_sensor, acoustic_noise)); // TODO: Change from hardcoded value

  // 6. UPDATE & CLEANUP
  isam2_.update(graph_, values_);
  graph_.resize(0);
  values_.clear();
  last_rov_timestamp_[rov_id] = t_s;
  rov_step_counters_[rov_id]++;

  gtsam::Values final_est = isam2_.calculateEstimate();
  publishBoatState(final_est);
  publishROVState(final_est);
  
  RCLCPP_DEBUG(get_logger(), "Added USBL, Depth, and Psuedo-Range factors for ROV %u at step %u", rov_id, rov_step_counters_[rov_id]);
}

// Will only receive this if there is something wrong with the USBL measurement.
// TODO: fix to match USBL pattern
void FactorGraphTrackingNode::acousticCommCallback(
    const AcousticCommReceive::SharedPtr msg) {
  if (!graph_initialised_) {
    return;
  }

  uint8_t rov_id = msg->node_id;
  double sound_speed = env_config_.sound_speed;
  auto [t_s, t_r] = resolveAcousticTimestamps(rclcpp::Time(msg->header.stamp), msg->t_sent, msg->t_received);
  double tof = t_r - t_s;
  if (tof <= 0.0 || tof > 5.0) {
    RCLCPP_WARN(get_logger(),
                "Invalid acoustic TOF %.6f s for ROV %u (t_s=%.6f, t_r=%.6f), skipping update",
                tof, rov_id, t_s, t_r);
    return;
  }

  std::lock_guard<std::mutex> lock(graph_mutex_);

  if (!rov_initialised_[rov_id]) {
    // For now only initialize upon receiving full USBL message, since we need the initial position to make sense of the acoustic comms.
    return;
  }
  
  // 1. ASV SIDE: Create/Retrieve the ASV Pose at Time of Arrival (t_received)
  auto asv_received = getAsvKeyAtTime(t_r);

  // 2. ROV SIDE: Handle the sequence
  uint32_t current_rov_step = rov_step_counters_[rov_id];
  gtsam::Key r_curr = getRovKey('R', rov_id, current_rov_step);
  gtsam::Key w_curr = getRovKey('W', rov_id, current_rov_step);
  gtsam::Key r_prev = getRovKey('R', rov_id, current_rov_step - 1);
  gtsam::Key w_prev = getRovKey('W', rov_id, current_rov_step - 1);

  // Use double for precision: (usec - usec) / 1e6
  double dt_rov = t_s - last_rov_timestamp_[rov_id];
  if (dt_rov <= 0.0) {
    RCLCPP_WARN(get_logger(),
                "Non-positive ROV dt=%.6f for ROV %u (t_s=%.6f, last=%.6f), skipping acoustic update",
                dt_rov, rov_id, t_s, last_rov_timestamp_[rov_id]);
    return;
  }

  // 3. PREDICTION: Provide Initial Values for the Optimizer
  // We predict where the ROV is now based on its last known position and velocity
  gtsam::Point3 p_prev = isam2_.calculateEstimate<gtsam::Point3>(r_prev);
  gtsam::Vector3 v_prev = isam2_.calculateEstimate<gtsam::Vector3>(w_prev);

  values_.insert(r_curr, (p_prev + (v_prev * dt_rov)).eval()); // Linear prediction
  values_.insert(w_curr, v_prev.eval());                    // Assume constant velocity
  
  // 4. MOTION MODEL FACTORS: Connect to the previous ROV state
  // Add Constant Velocity Factor: (P_prev, V_prev, P_curr)
  auto rov_noise = gtsam::noiseModel::Isotropic::Sigma(3, fgo_config_.rov_process_vel_sigma);
  graph_.add(boost::make_shared<ConstantVelocityFactor>(
      r_prev, w_prev, r_curr, dt_rov, rov_noise));
  // Also add a "Velocity Smoothness" factor (V_prev == V_curr)
  graph_.add(boost::make_shared<gtsam::BetweenFactor<gtsam::Vector3>>(
      w_prev, w_curr, gtsam::Vector3::Zero(), rov_noise));

  // 5. MEASUREMENT FACTORS
  // Add Depth factor
  auto depth_noise = gtsam::noiseModel::Isotropic::Sigma(1, fgo_config_.rov_depth_sigma);
  graph_.add(boost::make_shared<DepthFactor>(r_curr, msg->position.z, depth_noise));

  // Add Acoustic Range factor (Psuedo-Range)
  gtsam::Point3 usbl_offset(env_config_.usbl_offset_x, env_config_.usbl_offset_y, env_config_.usbl_offset_z);
  double roll_rad = env_config_.usbl_roll_deg * M_PI / 180.0;
  double pitch_rad = env_config_.usbl_pitch_deg * M_PI / 180.0;
  double yaw_rad = env_config_.usbl_yaw_deg * M_PI / 180.0;
  gtsam::Rot3 usbl_rotation = gtsam::Rot3::RzRyRx(roll_rad, pitch_rad, yaw_rad);
  gtsam::Pose3 body_P_sensor(usbl_rotation, usbl_offset);
  auto acoustic_noise = gtsam::noiseModel::Isotropic::Sigma(1, fgo_config_.acoustic_range_sigma / sound_speed); // Convert range sigma to time sigma using speed of sound
  graph_.add(boost::make_shared<PsudoRangeFactor>(asv_received, r_curr, tof, sound_speed, body_P_sensor, acoustic_noise)); // TODO: Change from hardcoded value

  // 6. UPDATE & CLEANUP
  isam2_.update(graph_, values_);
  graph_.resize(0);
  values_.clear();
  last_rov_timestamp_[rov_id] = t_s;
  rov_step_counters_[rov_id]++;

  gtsam::Values final_est = isam2_.calculateEstimate();
  publishBoatState(final_est);
  publishROVState(final_est);

  RCLCPP_DEBUG(get_logger(), "Added Acoustic Comm factors for ROV %u at step %u", rov_id, rov_step_counters_[rov_id]);
}

// ============================================================
// ROV INITIALIZATION
// ============================================================

void FactorGraphTrackingNode::initializeNewRov(uint8_t rov_id, Key rKey, Key wKey, 
                                               const USBLMessage::SharedPtr& usbl,
                                               double speed_of_sound) {
    double header_time = rclcpp::Time(usbl->header.stamp).seconds();
    auto [t_s, t_r] = resolveAcousticTimestamps(rclcpp::Time(usbl->header.stamp), usbl->t_sent, usbl->t_received);
    double tof = t_r - t_s;
    if (tof <= 0.0 || tof > 5.0) {
        RCLCPP_WARN(get_logger(),
                    "Invalid initialization TOF %.6f s for ROV %u (t_s=%.6f, t_r=%.6f), skipping init",
                    tof, rov_id, t_s, t_r);
        return;
    }
    RCLCPP_INFO(get_logger(), "Initializing new ROV %u with first USBL measurement at time %.2f", rov_id, header_time);
    
    // Check if we have any ASV timeline entries (i.e., at least one GNSS fix processed)
    if (asv_timeline_.empty()) {
        RCLCPP_ERROR(get_logger(), "Cannot initialize ROV before ASV has at least one GNSS measurement. Ignoring USBL for ROV %u.", rov_id);
        return;
    }
    
    gtsam::Key asv_key = getAsvKeyAtTime(t_r);
    
    // DON'T update iSAM2 yet - we need to add the ROV initialization factors first
    // to avoid an underconstrained system. The ASV node created by getAsvKeyAtTime()
    // only has an IMU factor, which isn't enough to constrain the system alone.
    
    // For now, predict the ASV pose using the values we just added
    // (they're in values_ but not yet in isam2_)
    gtsam::Pose3 world_T_asv;
    if (values_.exists(asv_key)) {
        // Use the predicted value from getAsvKeyAtTime()
        world_T_asv = values_.at<gtsam::Pose3>(asv_key);
    } else {
        // Fallback: get from isam2 (shouldn't happen in normal flow)
        world_T_asv = isam2_.calculateEstimate<gtsam::Pose3>(asv_key);
    }

    // Calculate local position from Az, El, Range
    double az = (usbl->azimuth) * M_PI / 180.0;
    double el = (usbl->elevation) * M_PI / 180.0;
    double r  = tof * speed_of_sound; // Speed of sound in water ~1500 m/s

    gtsam::Point3 p_local(r * cos(el) * cos(az), 
                          r * cos(el) * sin(az), 
                          r * sin(el));

    // Transform to world and force depth from the depth sensor (it's more reliable)
    gtsam::Point3 usbl_offset(env_config_.usbl_offset_x, env_config_.usbl_offset_y, env_config_.usbl_offset_z);
    double roll_rad = env_config_.usbl_roll_deg * M_PI / 180.0;
    double pitch_rad = env_config_.usbl_pitch_deg * M_PI / 180.0;
    double yaw_rad = env_config_.usbl_yaw_deg * M_PI / 180.0;
    gtsam::Rot3 usbl_rotation = gtsam::Rot3::RzRyRx(roll_rad, pitch_rad, yaw_rad);
    gtsam::Pose3 body_P_sensor(usbl_rotation, usbl_offset);

    gtsam::Point3 p_body = p_local;
    gtsam::Point3 p_world = world_T_asv.transformFrom(body_P_sensor.transformFrom(p_body));
    p_world = gtsam::Point3(p_world.x(), p_world.y(), usbl->position.z);

    values_.insert(rKey, p_world);
    values_.insert(wKey, gtsam::Vector3::Zero().eval());

    // Initial Priors
    auto prior_pos_noise = gtsam::noiseModel::Isotropic::Sigma(3, fgo_config_.rov_prior_pos_sigma);
    auto prior_vel_noise = gtsam::noiseModel::Isotropic::Sigma(3, fgo_config_.rov_prior_vel_sigma);
    graph_.add(gtsam::PriorFactor<gtsam::Point3>(rKey, p_world, prior_pos_noise));
    graph_.add(gtsam::PriorFactor<gtsam::Vector3>(wKey, gtsam::Vector3::Zero(), prior_vel_noise));

    auto usbl_noise = gtsam::noiseModel::Diagonal::Sigmas(
        (gtsam::Vector(2) << fgo_config_.usbl_azimuth_sigma,
         fgo_config_.usbl_elevation_sigma).finished());
    auto depth_noise = gtsam::noiseModel::Isotropic::Sigma(1, fgo_config_.rov_depth_sigma);
    auto acoustic_noise = gtsam::noiseModel::Isotropic::Sigma(
        1, fgo_config_.acoustic_range_sigma / speed_of_sound);

    graph_.add(boost::make_shared<UsblFactor>(
        asv_key, rKey, usbl->azimuth, usbl->elevation, body_P_sensor, usbl_noise));
    graph_.add(boost::make_shared<DepthFactor>(rKey, usbl->position.z, depth_noise));
    graph_.add(boost::make_shared<PsudoRangeFactor>(
        asv_key, rKey, tof, speed_of_sound, body_P_sensor, acoustic_noise));

    last_rov_timestamp_[rov_id] = t_s;
    rov_step_counters_[rov_id] = 0;

    // Now update iSAM2 with BOTH the ASV node (from getAsvKeyAtTime) AND the ROV initialization
    // This ensures the system is properly constrained
    isam2_.update(graph_, values_);
    graph_.resize(0);
    values_.clear();

    rov_initialised_[rov_id] = true;
    rov_step_counters_[rov_id] ++;

    gtsam::Values final_est = isam2_.calculateEstimate();
    publishBoatState(final_est);
    publishROVState(final_est);

    RCLCPP_INFO(get_logger(), "Initialized new ROV %u at time %.2f with position (%.2f, %.2f, %.2f)", 
                rov_id, header_time, p_world.x(), p_world.y(), p_world.z());
}

// ============================================================
// GRAPH INITIALIZATION
// ============================================================

void FactorGraphTrackingNode::initializeGraphWithGNSS(const GNSSNavPvt::SharedPtr msg) {
  RCLCPP_INFO(get_logger(), "Initializing factor graph with first GNSS at time %.2f", 
              rclcpp::Time(msg->header.stamp).seconds());

  double n, e, d;
  ned_.gnssToNED(msg->lat, msg->lon, msg->height, n, e, d);

  gtsam::Pose3 priorPose(gtsam::Rot3::RzRyRx(0, 0, 0), gtsam::Point3(n, e, d));
  gtsam::Vector3 priorVel(0.0, 0.0, 0.0);
  gtsam::imuBias::ConstantBias priorBias;

  auto pose_noise = gtsam::noiseModel::Diagonal::Sigmas(
      (gtsam::Vector(6) << fgo_config_.prior_pose_sigma, fgo_config_.prior_pose_sigma,
       fgo_config_.prior_pose_sigma, fgo_config_.prior_pose_sigma, fgo_config_.prior_pose_sigma,
       fgo_config_.prior_pose_sigma).finished());

  // Priors on pose, velocity, bias
  graph_.add(gtsam::PriorFactor<gtsam::Pose3>(X(0), priorPose, pose_noise));
  graph_.add(gtsam::PriorFactor<gtsam::Vector3>(V(0), priorVel,
      gtsam::noiseModel::Isotropic::Sigma(3, fgo_config_.prior_vel_sigma)));
  graph_.add(gtsam::PriorFactor<gtsam::imuBias::ConstantBias>(B(0), priorBias,
      gtsam::noiseModel::Isotropic::Sigma(6, fgo_config_.prior_bias_sigma)));

  // First GNSS constraint
  double pos_sigma = std::clamp(static_cast<double>(msg->h_acc) / 1000.0,
                                fgo_config_.gps_sigma_floor, fgo_config_.gps_sigma_max);
  auto gps_noise = gtsam::noiseModel::Isotropic::Sigma(3, pos_sigma);
  graph_.add(gtsam::GPSFactor(X(0), gtsam::Point3(n, e, d), gps_noise));

  values_.insert(X(0), priorPose);
  values_.insert(V(0), priorVel);
  values_.insert(B(0), priorBias);

  isam2_.update(graph_, values_);
  graph_.resize(0);
  values_.clear();

  // Record initial state
  last_asv_timestamp_ = rclcpp::Time(msg->header.stamp).seconds();
  asv_index_ = 0;
  asv_timeline_[last_asv_timestamp_] = X(0);

  RCLCPP_INFO(get_logger(), "Graph initialized with first GNSS at (%.2f, %.2f, %.2f)", n, e, d);

}

gtsam::Values FactorGraphTrackingNode::updateAndGetEstimate() {
  return isam2_.calculateEstimate();
}

// ============================================================
// PUBLISH STATES
// ============================================================

void FactorGraphTrackingNode::publishBoatState(const gtsam::Values &est) {
  if (!est.exists(X(asv_index_)) || !est.exists(V(asv_index_)) || !est.exists(B(asv_index_))) {
    return;
  }

  auto pose = est.at<gtsam::Pose3>(X(asv_index_));
  auto vel = est.at<gtsam::Vector3>(V(asv_index_));
  auto bias = est.at<gtsam::imuBias::ConstantBias>(B(asv_index_));

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
  for (const auto& [rov_id, initialized] : rov_initialised_) {
    if (initialized) {
      uint32_t current_rov_step = rov_step_counters_[rov_id] - 1; // Last updated step
      gtsam::Key rKey = getRovKey('R', rov_id, current_rov_step);
      gtsam::Key wKey = getRovKey('W', rov_id, current_rov_step);

      if (!est.exists(rKey) || !est.exists(wKey)) {
        return;
      }

      auto rov_pos = est.at<gtsam::Point3>(rKey);
      auto rov_vel = est.at<gtsam::Vector3>(wKey);

      ROVState rs;
      rs.header.stamp = now();

      rs.rov_id = rov_id;

      rs.position.x = rov_pos.x();
      rs.position.y = rov_pos.y();
      rs.position.z = rov_pos.z();

      rs.velocity.x = rov_vel.x();
      rs.velocity.y = rov_vel.y();
      rs.velocity.z = rov_vel.z();


      rov_state_pub_->publish(rs);
    }
  }
}


// ============================================================
// MAIN
// ============================================================
int main(int argc, char **argv) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<FactorGraphTrackingNode>());
  rclcpp::shutdown();
  return 0;
}
