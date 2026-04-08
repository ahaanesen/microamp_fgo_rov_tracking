#include "microamp_fgo_rov_tracking/fgo_tracking_node.hpp"
#include "microamp_fgo_rov_tracking/rov_factors.hpp"
#include "microamp_fgo_rov_tracking/config.hpp"

#include <gtsam/geometry/Point3.h>
#include <gtsam/nonlinear/NonlinearFactor.h>
#include <gtsam/base/numericalDerivative.h>


// ============================================================
// CONSTRUCTOR
// ============================================================

FactorGraphTrackingNode::FactorGraphTrackingNode()
    : Node("microamp_factor_graph_tracking"), isam2_(gtsam::ISAM2Params()),
      gravity_(9.82145996), datum_initialised_(false),
      graph_initialised_(false), asv_index_(0) {
  
  loadConfigurations();

  // ------------------------------------------------------------------
  // IMU Preintegration (NED)
  // ------------------------------------------------------------------
  auto params =
      gtsam::PreintegratedCombinedMeasurements::Params::MakeSharedD(env_config_.gravity);

  params->accelerometerCovariance = fgo_config_.accel_noise * fgo_config_.accel_noise * gtsam::I_3x3;
  params->gyroscopeCovariance = fgo_config_.gyro_noise * fgo_config_.gyro_noise * gtsam::I_3x3;
  params->integrationCovariance = 1e-8 * gtsam::I_3x3;
  params->biasAccCovariance = fgo_config_.accel_rw * fgo_config_.accel_rw * gtsam::I_3x3;
  params->biasOmegaCovariance = fgo_config_.gyro_rw * fgo_config_.gyro_rw * gtsam::I_3x3;
  params->biasAccOmegaInt = 1e-3 * gtsam::I_6x6;

  bias_ = gtsam::imuBias::ConstantBias();
  pim_ =
      std::make_unique<gtsam::PreintegratedCombinedMeasurements>(params, bias_);

  // ------------------------------------------------------------------
  // ROS I/O — BOAT
  // ------------------------------------------------------------------
  imu_sub_ = create_subscription<Imu>(
      topics_cfg_.imu, rclcpp::SensorDataQoS(),
      std::bind(&FactorGraphTrackingNode::imuCallback, this,
                std::placeholders::_1));

  gnss_sub_ = create_subscription<GNSSNavPvt>(
      topics_cfg_.gnss, 10,
      std::bind(&FactorGraphTrackingNode::gnssCallback, this,
                std::placeholders::_1));

  state_pub_ = create_publisher<BoatState>(topics_cfg_.boat_state_pub, 10);

  // ------------------------------------------------------------------
  // ROS I/O — ROV
  // ------------------------------------------------------------------
  acoustic_comm_sub_ = create_subscription<AcousticCommReceive>(
      topics_cfg_.acoustic_rx, 10,
      std::bind(&FactorGraphTrackingNode::acousticCommCallback, this,
                std::placeholders::_1));
  
  usbl_sub_ = create_subscription<USBLMessage>(
      topics_cfg_.usbl, 10,
      std::bind(&FactorGraphTrackingNode::usblCallback, this,
                std::placeholders::_1));


  rov_state_pub_ = create_publisher<ROVState>(topics_cfg_.rov_state_pub, 10);

  RCLCPP_INFO(get_logger(), "Expanded factor graph node initialized.");
}

void FactorGraphTrackingNode::loadConfigs() {
  declareAndLoadTopics(*this, topics_cfg_);
  declareAndLoadEnv(*this, env_cfg_);
  declareAndLoadFgo(*this, fgo_cfg_);
}

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

  double current_gnss_time = msg->header.stamp.seconds();
  // This handles the IMU integration and CombinedImuFactor automatically!
  gtsam::Key current_asv_key = getAsvKeyAtTime(current_gnss_time); // Key for the ASV node corresponding to GNSS measurement (either existing or newly created). IMU bridging is handled inside this function.
  uint64_t current_idx = gtsam::Symbol(current_asv_key).index();
  uint64_t asv_idx_next = asv_index_ + 1;

  // Add the GPS factor to the node we just ensured exists
  auto gps_noise = gtsam::noiseModel::Isotropic::Sigma(3, pos_sigma);
  graph_.add(gtsam::GPSFactor(X(current_idx), gtsam::Point3(n, e, d), gps_noise));

  // Predict boat state
  gtsam::Values est = updateAndGetEstimate();

  if (!est.exists(X(current_idx)) || !est.exists(V(current_idx)) || !est.exists(B(current_idx))) {
    RCLCPP_ERROR(get_logger(), "Boat keys missing at key=%lu", current_idx);
    graph_.resize(0);
    values_.clear();
    pim_->resetIntegrationAndSetBias(bias_);
    return;
  }

  auto prev_pose = est.at<gtsam::Pose3>(X(current_idx));
  auto prev_vel = est.at<gtsam::Vector3>(V(current_idx));
  auto prev_bias = est.at<gtsam::imuBias::ConstantBias>(B(current_idx));

  gtsam::NavState prev_state(prev_pose, prev_vel);
  auto predicted = pim_->predict(prev_state, prev_bias);

  values_.insert(X(asv_idx_next), predicted.pose());
  values_.insert(V(asv_idx_next), predicted.v());
  values_.insert(B(asv_idx_next), prev_bias);

  // Update iSAM2
  isam2_.update(graph_, values_);
  isam2_.update();

  graph_.resize(0);
  values_.clear();

  bias_ = isam2_.calculateEstimate<gtsam::imuBias::ConstantBias>(B(asv_idx_next));
  pim_->resetIntegrationAndSetBias(bias_);

  asv_index_ = asv_idx_next;
  asv_timeline_[msg->header.stamp.seconds()] = X(asv_index_);

  gtsam::Values final_est = isam2_.calculateEstimate();
  publishBoatState(final_est);
  if (!rov_initialised_.empty()) {
    publishROVState(final_est);
  }
}

// ============================================================
// HELPERS
// ============================================================
gtsam::PreintegratedCombinedMeasurements FactorGraphTrackingNode::getPimFromBuffer(double t_start, double t_end) {
    // Clone your existing PIM parameters (bias, noise models)
    gtsam::PreintegratedCombinedMeasurements sub_pim(pim_->params(), bias_);
    
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

// This function checks if an ASV node already exists at the target_time (within 1ms). If it does, it returns that key.
// If not, it creates a new ASV node at target_time by bridging from the last known ASV node using IMU data from the buffer.
gtsam::Key FactorGraphTrackingNode::getAsvKeyAtTime(double target_time) {
    // 1. Check if a node at this exact timestamp (within 1ms) already exists
    auto it = asv_timeline_.lower_bound(target_time - 0.001);
    if (it != asv_timeline_.end() && std::abs(it->first - target_time) < 0.001) {
        return it->second; // Already exists (likely from a previous ROV in same TDMA slot)
    }

    // 2. Find the most recent node in the graph
    double t_prev = last_asv_timestamp_;
    uint64_t idx_prev = asv_index_;
    uint64_t idx_curr = ++asv_index_;

    // 3. Bridge the gap using the IMU Buffer
    // Create a temporary PIM for this specific sub-interval
    auto sub_pim = getPimFromBuffer(t_prev, target_time);

    // 4. Predict the state at target_time
    auto prev_pose = isam2_.calculateEstimate<gtsam::Pose3>(X(idx_prev));
    auto prev_vel  = isam2_.calculateEstimate<gtsam::Vector3>(V(idx_prev));
    auto prev_bias = isam2_.calculateEstimate<gtsam::imuBias::ConstantBias>(B(idx_prev));
    
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
  if (!graph_initialised_) {
    return;
  }

  uint8_t rov_id = msg->rov_id;
  double sound_speed = env_cfg_.sound_speed;
  double t_sent = msg->t_sent;
  double t_received = msg->t_received;

  std::lock_guard<std::mutex> lock(graph_mutex_);

  if (!rov_initialised_[rov_id]) {
    // First time seeing this ROV, initialize its state in the graph
    Key rKey = getRovKey('R', rov_id, 0); // Position key for time step 0
    Key wKey = getRovKey('W', rov_id, 0); // Velocity key for time step 0
    initializeNewRov(rov_id, rKey, wKey, msg, sound_speed);
    return;
  }
  
  // 1. ASV SIDE: Create/Retrieve the ASV Pose at Time of Arrival (t_received)
  double time = t_received / 1e6; // Convert microseconds to seconds
  auto asv_received = getAsvKeyAtTime(time);
  // uint64_t current_asv_idx = gtsam::Symbol(asv_received).index();
  // uint64_t asv_idx_next = asv_index_ + 1;

  // 2. ROV SIDE: Handle the sequence
  uint32_t current_rov_step = rov_step_counters_[rov_id];
  gtsam::Key r_curr = getRovKey('R', rov_id, current_rov_step);
  gtsam::Key w_curr = getRovKey('W', rov_id, current_rov_step);
  gtsam::Key r_prev = getRovKey('R', rov_id, current_rov_step - 1);
  gtsam::Key w_prev = getRovKey('W', rov_id, current_rov_step - 1);

  // Use double for precision: (usec - usec) / 1e6
  double t_s = static_cast<double>(msg->t_sent) / 1e6;
  double t_r = static_cast<double>(msg->t_received) / 1e6;
  double dt_rov = t_s - last_rov_timestamp_[rov_id];

  // 3. PREDICTION: Provide Initial Values for the Optimizer
  // We predict where the ROV is now based on its last known position and velocity
  gtsam::Point3 p_prev = isam2_.calculateEstimate<gtsam::Point3>(r_prev);
  gtsam::Vector3 v_prev = isam2_.calculateEstimate<gtsam::Vector3>(w_prev);

  values_.insert(r_curr, p_prev + (v_prev * dt_rov)); // Linear prediction
  values_.insert(w_curr, v_prev);                    // Assume constant velocity
  
  // 4. MOTION MODEL FACTORS: Connect to the previous ROV state
  // Add Constant Velocity Factor: (P_prev, V_prev, P_curr)
  auto rov_noise = gtsam::noiseModel::Isotropic::Sigma(3, rov_process_vel_sigma_);
  graph_.add(boost::make_shared<ConstantVelocityFactor>(
      r_prev, w_prev, r_curr, dt_rov, rov_noise));
  // Also add a "Velocity Smoothness" factor (V_prev == V_curr)
  graph_.add(boost::make_shared<gtsam::BetweenFactor<gtsam::Vector3>>(
      w_prev, w_curr, gtsam::Vector3::Zero(), rov_noise));

  // 5. MEASUREMENT FACTORS
  // Add USBL factor
  auto body_P_sensor = gtsam::Pose3(usbl_rotation_, usbl_offset_);
  auto usbl_noise = gtsam::noiseModel::Diagonal::Sigmas((gtsam::Vector(2) << usbl_azimuth_sigma_, usbl_elevation_sigma_).finished());
  graph_.add(boost::make_shared<UsblFactor>(asv_received, r_curr, msg->azimuth, msg->elevation, body_P_sensor, usbl_noise));

  // Add Depth factor
  auto depth_noise = gtsam::noiseModel::Isotropic::Sigma(1, rov_depth_sigma_);
  graph_.add(boost::make_shared<DepthFactor>(r_curr, msg->position.z, depth_noise));

  // Add Acoustic Range factor (Psuedo-Range)
  double tof = t_r - t_s; 
  auto acoustic_noise = gtsam::noiseModel::Isotropic::Sigma(1, acoustic_range_sigma_ / sound_speed); // Convert range sigma to time sigma using speed of sound
  graph_.add(boost::make_shared<PsudoRangeFactor>(asv_received, r_curr, tof, sound_speed, body_P_sensor, acoustic_noise)); // TODO: Change from hardcoded value

  // 6. UPDATE & CLEANUP
  isam2_.update(graph_, values_);
  graph_.resize(0);
  values_.clear();
  last_rov_timestamp_[rov_id] = t_sent;
  rov_step_counters_[rov_id]++;
  
  RCLCPP_DEBUG(get_logger(), "Added USBL, Depth, and Psuedo-Range factors for ROV %u at step %u", rov_id, rov_step_counters_[rov_id]);
}

// Will only receive this if there is something wrong with the USBL measurement.
void FactorGraphTrackingNode::acousticCommCallback(
    const AcousticCommReceive::SharedPtr msg) {
  if (!graph_initialised_) {
    return;
  }

  uint8_t rov_id = msg->rov_id;
  double sound_speed = env_cfg_.sound_speed;
  double t_sent = msg->t_sent;
  double t_received = msg->t_received;

  std::lock_guard<std::mutex> lock(graph_mutex_);

  if (!rov_initialised_[rov_id]) {
    // For now only initialize upon receiving full USBL message, since we need the initial position to make sense of the acoustic comms.
    return;
  }
  
  // 1. ASV SIDE: Create/Retrieve the ASV Pose at Time of Arrival (t_received)
  double time = rclcpp::Time(t_received).seconds();
  auto asv_received = getAsvKeyAtTime(time);
  uint64_t current_asv_idx = gtsam::Symbol(asv_received).index();
  uint64_t asv_idx_next = asv_index_ + 1;

  // 2. ROV SIDE: Handle the sequence
  uint32_t current_rov_step = rov_step_counters_[rov_id];
  gtsam::Key r_curr = getRovKey('R', rov_id, current_rov_step);
  gtsam::Key w_curr = getRovKey('W', rov_id, current_rov_step);
  gtsam::Key r_prev = getRovKey('R', rov_id, current_rov_step - 1);
  gtsam::Key w_prev = getRovKey('W', rov_id, current_rov_step - 1);

  // Use double for precision: (usec - usec) / 1e6
  double t_s = static_cast<double>(msg->t_sent) / 1e6;
  double t_r = static_cast<double>(msg->t_received) / 1e6;
  double dt_rov = t_s - last_rov_timestamp_[rov_id];

  // 3. PREDICTION: Provide Initial Values for the Optimizer
  // We predict where the ROV is now based on its last known position and velocity
  gtsam::Point3 p_prev = isam2_.calculateEstimate<gtsam::Point3>(r_prev);
  gtsam::Vector3 v_prev = isam2_.calculateEstimate<gtsam::Vector3>(w_prev);

  values_.insert(r_curr, p_prev + (v_prev * dt_rov)); // Linear prediction
  values_.insert(w_curr, v_prev);                    // Assume constant velocity
  
  // 4. MOTION MODEL FACTORS: Connect to the previous ROV state
  // Add Constant Velocity Factor: (P_prev, V_prev, P_curr)
  auto rov_noise = gtsam::noiseModel::Isotropic::Sigma(3, rov_process_vel_sigma_);
  graph_.add(boost::make_shared<ConstantVelocityFactor>(
      r_prev, w_prev, r_curr, dt_rov, rov_noise));
  // Also add a "Velocity Smoothness" factor (V_prev == V_curr)
  graph_.add(boost::make_shared<gtsam::BetweenFactor<gtsam::Vector3>>(
      w_prev, w_curr, gtsam::Vector3::Zero(), rov_noise));

  // 5. MEASUREMENT FACTORS
  // Add Depth factor
  auto depth_noise = gtsam::noiseModel::Isotropic::Sigma(1, rov_depth_sigma_);
  graph_.add(boost::make_shared<DepthFactor>(r_curr, msg->position.z, depth_noise));

  // Add Acoustic Range factor (Psuedo-Range)
  double tof = t_r - t_s; 
  auto body_P_sensor = gtsam::Pose3(usbl_rotation_, usbl_offset_);
  auto acoustic_noise = gtsam::noiseModel::Isotropic::Sigma(1, acoustic_range_sigma_ / sound_speed); // Convert range sigma to time sigma using speed of sound
  graph_.add(boost::make_shared<PsudoRangeFactor>(asv_received, r_curr, tof, sound_speed, body_P_sensor, acoustic_noise)); // TODO: Change from hardcoded value

  // 6. UPDATE & CLEANUP
  isam2_.update(graph_, values_);
  graph_.resize(0);
  values_.clear();
  last_rov_timestamp_[rov_id] = t_sent;
  rov_step_counters_[rov_id]++;

  RCLCPP_DEBUG(get_logger(), "Added Acoustic Comm factors for ROV %u at step %u", rov_id, rov_step_counters_[rov_id]);
}

// ============================================================
// ROV INITIALIZATION
// ============================================================

void FactorGraphTrackingNode::initializeNewRov(uint8_t rov_id, Key rKey, Key wKey, 
                                               const USBLMessage::SharedPtr& usbl,
                                               double speed_of_sound) {
    gtsam::Key asv_key = getAsvKeyAtTime(usbl->t_received);
    gtsam::Pose3 world_T_asv = isam2_.calculateEstimate<gtsam::Pose3>(asv_key);

    // Calculate local position from Az, El, Range
    double az = usbl->azimuth * M_PI / 180.0;
    double el = usbl->elevation * M_PI / 180.0;
    double tof = (usbl->t_received - usbl->t_sent) / 1e6; // TODO: ensure this is in seconds with appropriate precision. might need to change either the seatrac driver or this conversion.
    double r  = tof * speed_of_sound; // Speed of sound in water ~1500 m/s

    gtsam::Point3 p_local(r * cos(el) * cos(az), 
                          r * cos(el) * sin(az), 
                          r * sin(el));

    // Transform to world and force depth from the depth sensor (it's more reliable)
    gtsam::Pose3 body_P_sensor(usbl_rotation_, usbl_offset_);
    gtsam::Point3 p_world = world_T_asv.transformFrom(body_P_sensor.transformFrom(p_local));
    p_world = gtsam::Point3(p_world.x(), p_world.y(), usbl->position.z); // Assuming Z is Depth

    values_.insert(rKey, p_world);
    values_.insert(wKey, gtsam::Vector3::Zero());

    // Initial Priors
    auto prior_pos_noise = gtsam::noiseModel::Isotropic::Sigma(3, rov_prior_pos_sigma_);
    auto prior_vel_noise = gtsam::noiseModel::Isotropic::Sigma(3, rov_prior_vel_sigma_);
    graph_.add(gtsam::PriorFactor<gtsam::Point3>(rKey, p_world, prior_pos_noise));
    graph_.add(gtsam::PriorFactor<gtsam::Vector3>(wKey, gtsam::Vector3::Zero(), prior_vel_noise));

    last_rov_timestamp_[rov_id] = usbl->t_sent;
    rov_step_counters_[rov_id] = 0;

    isam2_.update(graph_, values_);
    graph_.resize(0);
    values_.clear();

    rov_initialised_[rov_id] = true;
    rov_step_counters_[rov_id] ++;

    RCLCPP_INFO(get_logger(), "Initialized new ROV %u at time %.2f with position (%.2f, %.2f, %.2f)", 
                rov_id, usbl->t_received, p_world.x(), p_world.y(), p_world.z());
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

  graph_.add(gtsam::PriorFactor<gtsam::Pose3>(X(asv_index_), priorPose, pose_noise));
  graph_.add(gtsam::PriorFactor<gtsam::Vector3>(
      V(asv_index_), priorVel,
      gtsam::noiseModel::Isotropic::Sigma(3, prior_vel_sigma_)));
  graph_.add(gtsam::PriorFactor<gtsam::imuBias::ConstantBias>(
      B(asv_index_), priorBias,
      gtsam::noiseModel::Isotropic::Sigma(6, prior_bias_sigma_)));
  
  last_asv_timestamp_ = 0.0;
  asv_index_ = 0;
  asv_timeline_[last_asv_timestamp_] = X(asv_index_);

  values_.insert(X(asv_index_), priorPose);
  values_.insert(V(asv_index_), priorVel);
  values_.insert(B(asv_index_), priorBias);

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
