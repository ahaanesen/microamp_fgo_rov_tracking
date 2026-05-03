#include "microamp_fgo_rov_tracking/fgo_tracking_node.hpp"
#include "microamp_fgo_rov_tracking/rov_factors.hpp"
#include "microamp_fgo_rov_tracking/config.hpp"

#include <gtsam/geometry/Point3.h>
#include <gtsam/nonlinear/NonlinearFactor.h>
#include <gtsam/base/numericalDerivative.h>


gtsam::Pose3 getBodyToUsblPose(const EnvConfig& env_config) {
  const gtsam::Point3 usbl_offset(
      env_config.usbl_offset_x,
      env_config.usbl_offset_y,
      env_config.usbl_offset_z);

  const double roll_rad = env_config.usbl_roll_deg * M_PI / 180.0;
  const double pitch_rad = env_config.usbl_pitch_deg * M_PI / 180.0;
  const double yaw_rad = env_config.usbl_yaw_deg * M_PI / 180.0;
  const gtsam::Rot3 usbl_rotation = gtsam::Rot3::RzRyRx(yaw_rad, pitch_rad, roll_rad);

  return gtsam::Pose3(usbl_rotation, usbl_offset);
}



// ============================================================
// CONSTRUCTOR, DESTRUCTOR, AND INITIALIZATION
// ============================================================

FactorGraphTrackingNode::FactorGraphTrackingNode()
    : Node("microamp_factor_graph_tracking"), isam2_(gtsam::ISAM2Params()),
      graph_initialised_(false), asv_index_(0), datum_initialised_(false) {
  
  loadConfigurations();
  scenario_id_ = resolveScenarioId();
  RCLCPP_INFO(get_logger(), "Using scenario_id=%u", static_cast<unsigned>(scenario_id_));

  // Initialize NED converter with default datum (can be overridden by first GNSS)
  auto default_datum_lat = 60.3913;
  auto default_datum_lon = 5.3221;
  auto default_datum_h = 0.0;
  ned_.setDatum(default_datum_lat, default_datum_lon, default_datum_h);
  datum_initialised_ = true;

  // ------------------------------------------------------------------
  // IMU Preintegration (NED)
  // ------------------------------------------------------------------
  pim_params_ = gtsam::PreintegratedCombinedMeasurements::Params::MakeSharedD(env_config_.gravity);


  // IMU preintegration covariances (from ESKF model) 
  pim_params_->accelerometerCovariance = fgo_config_.accel_noise * fgo_config_.accel_noise * gtsam::I_3x3;
  pim_params_->gyroscopeCovariance = fgo_config_.gyro_noise * fgo_config_.gyro_noise * gtsam::I_3x3;
  pim_params_->integrationCovariance = 1e-8 * gtsam::I_3x3;
  // Bias random walk covariances
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

  state_pub_ = this->create_publisher<Odometry>(topics_config_.boat_state_pub, 10);

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

uint8_t FactorGraphTrackingNode::resolveScenarioId() {
  this->declare_parameter<int>("scenario_id", bearing_range_depth);
  const int configured_scenario = this->get_parameter("scenario_id").as_int();

  if (configured_scenario < bearing_only || configured_scenario > bearing_range_depth) {
    RCLCPP_WARN(
        get_logger(),
        "Invalid scenario_id=%d. Falling back to scenario %d.",
        configured_scenario,
        bearing_range_depth);
    return bearing_range_depth;
  }

  return static_cast<uint8_t>(configured_scenario);
}

FactorGraphTrackingNode::~FactorGraphTrackingNode() = default;

// ============================================================
// ASV CALLBACKS
// ============================================================

// IMU callback: preintegrate measurements only 
// Adding the factor and reset the preintegrator when we get the next GNSS measurement (to ensure correct bias handling)
void FactorGraphTrackingNode::imuCallback(const Imu::SharedPtr msg) {
  if (!graph_initialised_) {
    // RCLCPP_INFO(get_logger(), "Graph not initialized yet, skipping IMU data");
    return;
  }
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

  // if (dt <= 0.0) {
  //   RCLCPP_WARN(get_logger(), "Non-positive IMU dt=%.6f, skipping", dt);
  //   return;
  // }

  gtsam::Vector3 acc(msg->linear_acceleration.x, msg->linear_acceleration.y,
                     msg->linear_acceleration.z);

  gtsam::Vector3 gyro(msg->angular_velocity.x, msg->angular_velocity.y,
                      msg->angular_velocity.z);
  pim_->integrateMeasurement(acc, gyro, dt);
}

void FactorGraphTrackingNode::initializeDatumFromGNSS(const GNSSNavPvt &msg) {
  ned_.setDatum(msg.lat, msg.lon, msg.height);
  datum_initialised_ = true;
  RCLCPP_INFO(get_logger(), "Datum initialised: lat=%.8f lon=%.8f h=%.2f",
              msg.lat, msg.lon, msg.height);
}

//GNSS callback: if graph not initialized, initialize with GNSS. Else, add new node and factors.
void FactorGraphTrackingNode::gnssCallback(const GNSSNavPvt::SharedPtr gnss_msg) {
  // 0. Validity of gnss fix
  //    fix_type: 0=no fix, 1=dead reck, 2=2D, 3=3D, 4=GNSS+DR, 5=time
  //    flags bit 0: gnssFixOK (carrier solution valid)
  constexpr uint8_t FIX_3D = 3;
  if (gnss_msg->fix_type < FIX_3D) {RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 5000, "Waiting for 3D fix (current fix_type=%d)",
                              gnss_msg->fix_type);
    return;
  }
  if (!gnss_msg->gnss_fix_ok) {RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 5000, "gnss_fix_ok not set, skipping");
    return;
  }

  // 1. Initialize datum and graph if not already initialized

  // if (!datum_initialised_) { // NB. Commented out for simulation with fixed datum.
  //   initializeDatumFromGNSS(*gnss_msg);
  // }

 if (!graph_initialised_) {
    if (ne_init_ == std::tuple<double, double>{0.0, 0.0}) {
      NedCoordinates ned_coords = ned_.gnssToNED(gnss_msg->lat, gnss_msg->lon, gnss_msg->height);
      ne_init_ = std::make_tuple(ned_coords.n, ned_coords.e);
      return; // Wait for the next GNSS message to initialize the graph with a full pose (including heading)
    }
    initializeGraphWithGNSS(gnss_msg);
    graph_initialised_ = true;
    pim_->resetIntegrationAndSetBias(bias_); // Ensure preintegrator is reset after graph initialization
    return;
  }

  // 2. Value initialization for new ASV node (based on prediction from previous node)
  double current_asv_time = rclcpp::Time(gnss_msg->header.stamp).seconds();
  uint64_t prev_asv_index = asv_index_;
  uint64_t curr_asv_index = asv_index_ + 1;

  auto prev_asv_pose = isam2_.calculateEstimate<gtsam::Pose3>(X(prev_asv_index));
  auto prev_asv_vel = isam2_.calculateEstimate<gtsam::Vector3>(V(prev_asv_index));
  auto prev_asv_bias = isam2_.calculateEstimate<gtsam::imuBias::ConstantBias>(B(prev_asv_index));

  gtsam::NavState prev_asv_state(prev_asv_pose, prev_asv_vel);
  gtsam::NavState predicted_asv_state = pim_->predict(prev_asv_state, prev_asv_bias);

  values_.insert(X(curr_asv_index), predicted_asv_state.pose());
  values_.insert(V(curr_asv_index), predicted_asv_state.v());
  values_.insert(B(curr_asv_index), prev_asv_bias); // Initial guess for bias is that it hasn't changed since the last node
  
  // 3. Add IMU factor between previous and new ASV node
  graph_.add(gtsam::CombinedImuFactor(X(prev_asv_index), V(prev_asv_index),
                                      X(curr_asv_index), V(curr_asv_index),
                                      B(prev_asv_index), B(curr_asv_index),
                                      *pim_));


  // 4. Add GPS factor to new node
  NedCoordinates ned_coords = ned_.gnssToNED(gnss_msg->lat, gnss_msg->lon, gnss_msg->height);
  double sigma_ne = std::clamp(fgo_config_.gps_sigma_ne,
                               fgo_config_.gps_sigma_floor, fgo_config_.gps_sigma_max);
  double sigma_d = std::clamp(fgo_config_.gps_sigma_d,
                              fgo_config_.gps_sigma_floor, fgo_config_.gps_sigma_max);
  auto gps_noise = gtsam::noiseModel::Diagonal::Sigmas(
    (gtsam::Vector(3) << sigma_ne, sigma_ne, sigma_d).finished());
  auto gps_leverarm = gtsam::Point3(env_config_.gps_offset[0], env_config_.gps_offset[1], env_config_.gps_offset[2]);
  graph_.add(gtsam::GPSFactorArm(X(curr_asv_index), gtsam::Point3(ned_coords.n, ned_coords.e, ned_coords.d), gps_leverarm, gps_noise));


  // 5. Process pending USBL measurements and add corresponding factors for associated ROV nodes
  std::vector<std::tuple<uint8_t, double>> updated_rov_ids;

  while (!usbl_queue_.empty()) {
    const auto& usbl_msg = usbl_queue_.front();
    uint8_t rov_id = usbl_msg->rov_id;
    double rov_time = static_cast<double>(usbl_msg->t_sent) / 1e6;
    gtsam::Key associated_asv_key = getAsvKeyForRovAssociation(current_asv_time, curr_asv_index, rov_time);

    if (!rov_initialised_[rov_id]) {
      // If this is the first measurement for this ROV, initialize its position based on the associated ASV state and the USBL measurement
      initializeNewRov(associated_asv_key, usbl_msg);
      rov_initialised_[rov_id] = true;
    }
    else {
      // In function value and graph updates for subsequent measurements of this ROV 
      // (new node + CV factor + USBL factor, optionally acoustic range and depth factors depending on scenario)
      rovUpdateWithUsbl(rov_id, rov_time, associated_asv_key, usbl_msg);
      // Migth want to return new values and factors?
    }

    updated_rov_ids.push_back(std::make_tuple(rov_id, rov_time));
    usbl_queue_.pop_front();

  }

  // 6. Graph optimization
  isam2_.update(graph_, values_);
  graph_.resize(0);
  values_.clear();

  // 7. Housekeeping: reset preintegrator, update last ASV node timestamp/index, etc.
  bias_ = isam2_.calculateEstimate<gtsam::imuBias::ConstantBias>(B(curr_asv_index));
  pim_->resetIntegrationAndSetBias(bias_);

  last_asv_timestamp_ = rclcpp::Time(gnss_msg->header.stamp).seconds();
  asv_index_ = curr_asv_index;
  asv_timeline_[last_asv_timestamp_] = X(curr_asv_index);

  for (const auto& rov_info : updated_rov_ids) {
    auto rov_id = std::get<0>(rov_info);
    double rov_time = std::get<1>(rov_info);
    rov_step_counters_[rov_id] += 1;
    last_rov_timestamp_[rov_id] = rov_time;
  }

  // 8. Publish updated ASV and ROV states
  auto current_estimate = isam2_.calculateEstimate();
  publishBoatOdometry(current_estimate);
  publishROVState(current_estimate);

  RCLCPP_INFO(get_logger(), "Processed GNSS measurement at time %.2f, position (%.2f, %.2f, %.2f).", 
            current_asv_time, ned_coords.n, ned_coords.e, ned_coords.d);
}

// ============================================================
// HELPERS
// ============================================================

// Helper function to generate unique keys per ROV
// Encodes prefix (R=position, W=velocity), ROV ID, and step count into a single Key
// Example: ROV position at step 5 for ROV ID 2 -> getRovKey('R', 2, 5)
gtsam::Key FactorGraphTrackingNode::getRovKey(unsigned char prefix, uint32_t rov_id, uint32_t step_count) {
  uint64_t packed_index = (uint64_t(rov_id) << 32) | step_count;
  return gtsam::Symbol(prefix, packed_index);
}

// Helper function to find the closest ASV node key for a given ROV measurement time.
// To enable correct association of USBL measurements with the ASV state at the time of measurement
gtsam::Key FactorGraphTrackingNode::getAsvKeyForRovAssociation(double current_asv_time, uint64_t current_asv_index, double rov_time) {
  // Find the ASV node with the closest timestamp to the measurement time 
  gtsam::Key closest_asv_key = X(current_asv_index); // Default to the latest ASV node
  double smallest_time_diff = std::abs(current_asv_time - rov_time);

  // Iterate backwards through the ASV timeline to find the closest node
  for (auto it = asv_timeline_.rbegin(); it != asv_timeline_.rend(); ++it) { // ++it or it++?
    double asv_time = it->first;
    double time_diff = std::abs(asv_time - rov_time);

    if (time_diff < smallest_time_diff) {
      gtsam::Key asv_key = it->second;
      smallest_time_diff = time_diff;
      closest_asv_key = asv_key;
    }
    else {
      break; // Since the timeline is ordered, we can stop once the time difference starts increasing
    }
  }

  return closest_asv_key;
}

// Helper function to add a new ROV node and corresponding factors based on a USBL measurement and the associated ASV state
// Adds values for a new ROV node and updates the graph with the CV factor, a USBL factor, and optionally acoustic range and depth factors depending on the scenario
void FactorGraphTrackingNode::rovUpdateWithUsbl(uint8_t rov_id, 
                                                double rov_time, 
                                                gtsam::Key associated_asv_key, 
                                                const USBLMessage::SharedPtr& usbl_msg) {
  // Check validity of measurement timestamp
  double dt_rov = rov_time - last_rov_timestamp_[rov_id];
  if (dt_rov <= 0.0) {
    RCLCPP_WARN(get_logger(), "Non-positive dt for ROV %d: %.6f, skipping USBL measurement", rov_id, dt_rov);
    return;
  }
  // Make a new ROV node for this measurement
  uint32_t current_rov_step = rov_step_counters_[rov_id];
  gtsam::Key r_curr = getRovKey('R', rov_id, current_rov_step);
  gtsam::Key w_curr = getRovKey('W', rov_id, current_rov_step);
  gtsam::Key r_prev = getRovKey('R', rov_id, current_rov_step - 1);
  gtsam::Key w_prev = getRovKey('W', rov_id, current_rov_step - 1);

  // Value initialization of new ROV node (based on prediction from previous node)
  gtsam::Point3 r_pred = isam2_.calculateEstimate<gtsam::Point3>(r_prev);
  gtsam::Vector3 w_pred = isam2_.calculateEstimate<gtsam::Vector3>(w_prev);

  values_.insert(r_curr, (r_pred + (w_pred * dt_rov)).eval());  // CV linear prediction
  values_.insert(w_curr, w_pred.eval()); // Assume CV

  // Add motion model factor between previous and new ROV node (CV model)
  // const double q = fgo_config_.rov_cv_continous_sigma * fgo_config_.rov_cv_continous_sigma;
  // gtsam::Matrix66 cov = gtsam::Matrix66::Zero();
  // // Piecewise Constant Acceleration (PCA) model
  // // cov.block<3,3>(0,0) = gtsam::Matrix33::Identity() * (q * std::pow(dt_rov, 4) / 4.0);
  // // cov.block<3,3>(0,3) = gtsam::Matrix33::Identity() * (q * std::pow(dt_rov, 3) / 2.0);
  // // cov.block<3,3>(3,0) = gtsam::Matrix33::Identity() * (q * std::pow(dt_rov, 3) / 2.0);
  // // cov.block<3,3>(3,3) = gtsam::Matrix33::Identity() * (q * std::pow(dt_rov, 2));
  
  // // Standard CV model
  // cov.block<3,3>(0,0) = gtsam::Matrix33::Identity() * (q * std::pow(dt_rov, 3) / 3.0);
  // cov.block<3,3>(0,3) = gtsam::Matrix33::Identity() * (q * std::pow(dt_rov, 2) / 2.0);
  // cov.block<3,3>(3,0) = gtsam::Matrix33::Identity() * (q * std::pow(dt_rov, 2) / 2.0);
  // cov.block<3,3>(3,3) = gtsam::Matrix33::Identity() * (q * dt_rov);

  const double q_h = fgo_config_.rov_cv_continous_sigma * fgo_config_.rov_cv_continous_sigma;
  const double q_v = (q_h * 0.1); // 10x tighter in Down
  gtsam::Matrix66 cov = gtsam::Matrix66::Zero();
  // Position block
  cov(0,0) = q_h * std::pow(dt_rov,3)/3; cov(1,1) = q_h * std::pow(dt_rov,3)/3; cov(2,2) = q_v * std::pow(dt_rov,3)/3;
  // Cross terms
  cov(0,3) = q_h * std::pow(dt_rov,2)/2; cov(3,0) = q_h * std::pow(dt_rov,2)/2;
  cov(1,4) = q_h * std::pow(dt_rov,2)/2; cov(4,1) = q_h * std::pow(dt_rov,2)/2;
  cov(2,5) = q_v * std::pow(dt_rov,2)/2; cov(5,2) = q_v * std::pow(dt_rov,2)/2;
  // Velocity block
  cov(3,3) = q_h * dt_rov; cov(4,4) = q_h * dt_rov; cov(5,5) = q_v * dt_rov;


  auto cv_noise = gtsam::noiseModel::Gaussian::Covariance(cov);
  graph_.add(std::make_shared<ConstantVelocityFactor>(r_prev, w_prev, r_curr, w_curr, dt_rov, cv_noise));


  // Add USBL factor between ASV node and new ROV node
  auto body_P_sensor = getBodyToUsblPose(env_config_);
  auto usbl_noise = gtsam::noiseModel::Diagonal::Sigmas(
    (gtsam::Vector(2) << fgo_config_.usbl_azimuth_sigma, fgo_config_.usbl_elevation_sigma).finished());
  graph_.add(std::make_shared<UsblFactor>(
    associated_asv_key, r_curr, usbl_msg->azimuth, usbl_msg->elevation, body_P_sensor, usbl_noise));

    // In rovUpdateWithUsbl, after the USBL factor:
  if (fgo_config_.use_rov_depth_prior) {
      auto depth_prior_noise = gtsam::noiseModel::Isotropic::Sigma(1, fgo_config_.rov_depth_prior_sigma);
      graph_.add(std::make_shared<DepthFactor>(
          r_curr, fgo_config_.rov_depth_prior_mean, depth_prior_noise));
  }
  
  if (scenario_id_ >= bearing_range) {
    // Add acoustic range factor between ASV and ROV
    auto tof = (usbl_msg->t_received - usbl_msg->t_sent) / 1e6; // Convert microseconds to seconds
    auto acoustic_noise = gtsam::noiseModel::Isotropic::Sigma(1, fgo_config_.acoustic_range_sigma / env_config_.sound_speed); // Convert range sigma to time-of-flight sigma
    graph_.add(std::make_shared<PsudoRangeFactor>(
      associated_asv_key, r_curr, tof, env_config_.sound_speed, body_P_sensor, acoustic_noise));
  }
  if (scenario_id_ >= bearing_range_depth) {
    // Add depth factor for ROV
    auto depth_noise = gtsam::noiseModel::Isotropic::Sigma(1, fgo_config_.rov_depth_sigma);
    graph_.add(std::make_shared<DepthFactor>(r_curr, usbl_msg->position.z, depth_noise));
  }

  RCLCPP_INFO(get_logger(), "Added new ROV node for ROV ID %d at time %.2f based on USBL measurement sent at %.2f (received at %.2f)", 
              rov_id, rov_time, static_cast<double>(usbl_msg->t_sent) / 1e6, static_cast<double>(usbl_msg->t_received) / 1e6);
}

// ============================================================
// ROV CALLBACKS
// ============================================================

void FactorGraphTrackingNode::usblCallback(const USBLMessage::SharedPtr msg) {
  if (!graph_initialised_) {
    RCLCPP_WARN(get_logger(), "Graph not initialized yet, skipping USBL measurement");
    return;
  }
  usbl_queue_.push_back(msg);
}

// Will only receive this if there is something wrong with the USBL measurement.
// TODO: fix to match USBL pattern
void FactorGraphTrackingNode::acousticCommCallback(
    const AcousticCommReceive::SharedPtr msg) {
  (void)msg;
}

// ============================================================
// ROV INITIALIZATION
// ============================================================

// Initialize a new ROV in the graph based on the associated ASV state and the initial USBL measurement
void FactorGraphTrackingNode::initializeNewRov(gtsam::Key associated_asv_key, const USBLMessage::SharedPtr& usbl_msg) {
  uint8_t rov_id = usbl_msg->rov_id;
  double t_sent = static_cast<double>(usbl_msg->t_sent) / 1e6;
  double t_receive = static_cast<double>(usbl_msg->t_received) / 1e6;

  // Get the ASV pose at the time of the USBL measurement (using the associated ASV key)
  gtsam::Pose3 world_T_asv;
  if (values_.exists(associated_asv_key)) {
      // Use the predicted value
      world_T_asv = values_.at<gtsam::Pose3>(associated_asv_key);
  } else {
      // Fallback: get from isam2 (shouldn't happen in normal flow)
      world_T_asv = isam2_.calculateEstimate<gtsam::Pose3>(associated_asv_key);
  }

  // Calculate local position from Az, El, Range 
  double az_rad = (usbl_msg->azimuth) * M_PI / 180.0;   // (NB. seatrac modems return azimuth and elevation in degrees)
  double el_rad = (usbl_msg->elevation) * M_PI / 180.0; // (NB. seatrac modems return azimuth and elevation in degrees)
  double tof = t_receive - t_sent;
  double r = tof * env_config_.sound_speed;

  gtsam::Pose3 body_P_sensor = getBodyToUsblPose(env_config_);
  gtsam::Pose3 world_T_sensor = world_T_asv.compose(body_P_sensor);
  gtsam::Point3 sensor_pos = world_T_sensor.translation();

  // Direct NED coordinates of ROV:
  gtsam::Point3 p_world(
      sensor_pos.x() + r * std::cos(el_rad) * std::cos(az_rad),       // North
      sensor_pos.y() + r * std::cos(el_rad) * std::sin(az_rad),       // East
      usbl_msg->position.z // sensor_pos.z() + r * std::sin(el_rad)   // Down (depth)
  );

  // Value initialization for new ROV node
  gtsam::Key rKey = getRovKey('R', rov_id, 0);
  gtsam::Key wKey = getRovKey('W', rov_id, 0);
  values_.insert(rKey, p_world);
  values_.insert(wKey, gtsam::Vector3::Zero().eval());

  // Add prior factors
  auto prior_pos_noise = gtsam::noiseModel::Isotropic::Sigma(3, fgo_config_.rov_prior_pos_sigma);
  auto prior_vel_noise = gtsam::noiseModel::Isotropic::Sigma(3, fgo_config_.rov_prior_vel_sigma);
  graph_.add(gtsam::PriorFactor<gtsam::Point3>(rKey, p_world, prior_pos_noise));
  graph_.add(gtsam::PriorFactor<gtsam::Vector3>(wKey, gtsam::Vector3::Zero(), prior_vel_noise));

  // Add USBL factor for this initial measurement
  auto usbl_noise = gtsam::noiseModel::Diagonal::Sigmas(
    (gtsam::Vector(2) << fgo_config_.usbl_azimuth_sigma, fgo_config_.usbl_elevation_sigma).finished());
  graph_.add(std::make_shared<UsblFactor>(
    associated_asv_key, rKey, usbl_msg->azimuth, usbl_msg->elevation, body_P_sensor, usbl_noise));
  
  if (scenario_id_ >= bearing_range) {
    // Add acoustic range factor between ASV and ROV
    auto tof = (usbl_msg->t_received - usbl_msg->t_sent) / 1e6; // Convert microseconds to seconds
    auto acoustic_noise = gtsam::noiseModel::Isotropic::Sigma(1, fgo_config_.acoustic_range_sigma / env_config_.sound_speed); // Convert range sigma to time-of-flight sigma
    graph_.add(std::make_shared<PsudoRangeFactor>(
      associated_asv_key, rKey, tof, env_config_.sound_speed, body_P_sensor, acoustic_noise));
  }
  if (scenario_id_ >= bearing_range_depth) {
    // Add depth factor for ROV
    auto depth_noise = gtsam::noiseModel::Isotropic::Sigma(1, fgo_config_.rov_depth_sigma);
    graph_.add(std::make_shared<DepthFactor>(rKey, usbl_msg->position.z, depth_noise));
  }

  // Initialize step counter and last timestamp for this ROV
  rov_step_counters_[rov_id] = 0; 
  last_rov_timestamp_[rov_id] = t_sent;

  RCLCPP_INFO(get_logger(), "Initialized new ROV with ID %d based on associated ASV node at time %.2f and USBL measurement sent at %.2f (received at %.2f)", 
              rov_id, rclcpp::Time(isam2_.calculateEstimate<gtsam::Pose3>(associated_asv_key).translation().x()).seconds(), t_sent, t_receive);
}

// ============================================================
// GRAPH INITIALIZATION
// ============================================================

void FactorGraphTrackingNode::initializeGraphWithGNSS(const GNSSNavPvt::SharedPtr gnss_msg) {
  RCLCPP_INFO(get_logger(), "Initializing factor graph with first GNSS at time %.2f", 
              rclcpp::Time(gnss_msg->header.stamp).seconds());

  NedCoordinates ned_coords = ned_.gnssToNED(gnss_msg->lat, gnss_msg->lon, gnss_msg->height);

  // TODO: take velocity into account for better initial guess. Currently assuming forward motion along init heading.
  double delta_n = ned_coords.n - std::get<0>(ne_init_);
  double delta_e = ned_coords.e - std::get<1>(ne_init_);
  double initial_yaw = std::atan2(delta_e, delta_n); // Heading from the first two gnss points
  gtsam::Rot3 initial_rot = gtsam::Rot3::Ypr(initial_yaw, 0.0, 0.0); //Ypr or Rz?
  gtsam::Pose3 priorPose(initial_rot, gtsam::Point3(ned_coords.n, ned_coords.e, ned_coords.d));
  gtsam::Vector3 priorVel(0.0, 0.0, 0.0);
  gtsam::imuBias::ConstantBias priorBias;

  auto pose_noise = gtsam::noiseModel::Diagonal::Sigmas(
      (gtsam::Vector(6) << M_PI, M_PI, M_PI, 
       fgo_config_.prior_translation_sigma, fgo_config_.prior_translation_sigma, fgo_config_.prior_translation_sigma).finished());

  // Priors on pose, velocity, bias
  graph_.add(gtsam::PriorFactor<gtsam::Pose3>(X(0), priorPose, pose_noise));
  graph_.add(gtsam::PriorFactor<gtsam::Vector3>(V(0), priorVel,
      gtsam::noiseModel::Isotropic::Sigma(3, fgo_config_.prior_vel_sigma)));
  // Bias prior: first 3 DOF accel bias, last 3 DOF gyro bias
  auto bias_noise = gtsam::noiseModel::Diagonal::Sigmas(
    (gtsam::Vector(6) << fgo_config_.prior_accel_bias_sigma, fgo_config_.prior_accel_bias_sigma, fgo_config_.prior_accel_bias_sigma,
                         fgo_config_.prior_bias_sigma, fgo_config_.prior_bias_sigma, fgo_config_.prior_bias_sigma).finished());
  graph_.add(gtsam::PriorFactor<gtsam::imuBias::ConstantBias>(B(0), priorBias, bias_noise));

  // First GNSS constraint
  double sigma_ne = std::clamp(fgo_config_.gps_sigma_ne,
                               fgo_config_.gps_sigma_floor, fgo_config_.gps_sigma_max);
  double sigma_d = std::clamp(fgo_config_.gps_sigma_d,
                              fgo_config_.gps_sigma_floor, fgo_config_.gps_sigma_max);
  auto gps_noise_init = gtsam::noiseModel::Diagonal::Sigmas(
    (gtsam::Vector(3) << sigma_ne, sigma_ne, sigma_d).finished());
  auto gps_leverarm = gtsam::Point3(env_config_.gps_offset[0], env_config_.gps_offset[1], env_config_.gps_offset[2]);
  graph_.add(gtsam::GPSFactorArm(X(0), gtsam::Point3(ned_coords.n, ned_coords.e, ned_coords.d), gps_leverarm, gps_noise_init));

  values_.insert(X(0), priorPose);
  values_.insert(V(0), priorVel);
  values_.insert(B(0), priorBias);

  isam2_.update(graph_, values_);
  graph_.resize(0);
  values_.clear();

  // Record initial state
  last_asv_timestamp_ = rclcpp::Time(gnss_msg->header.stamp).seconds();
  asv_index_ = 0;
  asv_timeline_[last_asv_timestamp_] = X(0);

  RCLCPP_INFO(get_logger(), "Graph initialized with first GNSS at (%.2f, %.2f, %.2f)", ned_coords.n, ned_coords.e, ned_coords.d);

}

gtsam::Values FactorGraphTrackingNode::updateAndGetEstimate() {
  return isam2_.calculateEstimate();
}

// ============================================================
// PUBLISH STATES
// ============================================================

void FactorGraphTrackingNode::publishBoatOdometry(const gtsam::Values &est) {
  if (!est.exists(X(asv_index_)) || !est.exists(V(asv_index_)) || !est.exists(B(asv_index_))) {
    return;
  }

  auto pose = est.at<gtsam::Pose3>(X(asv_index_));
  auto vel  = est.at<gtsam::Vector3>(V(asv_index_));
  auto bias = est.at<gtsam::imuBias::ConstantBias>(B(asv_index_));

  // Marginal covariances from ISAM2 Bayes tree
  gtsam::Matrix pose_cov_gtsam = isam2_.marginalCovariance(X(asv_index_));
  gtsam::Matrix vel_cov        = isam2_.marginalCovariance(V(asv_index_));

  // GTSAM Pose3 tangent ordering: [rx, ry, rz, tx, ty, tz]
  // ROS Odometry covariance ordering: [x, y, z, rx, ry, rz]
  // Permutation: indices {3,4,5,0,1,2}
  const int perm[6] = {3, 4, 5, 0, 1, 2};

  gtsam::Quaternion q = pose.rotation().toQuaternion();

  Odometry odom;
  odom.header.stamp    = rclcpp::Time(last_asv_timestamp_);
  odom.header.frame_id = "ned";
  odom.child_frame_id  = "base_link";

  odom.pose.pose.position.x    = pose.translation().x();
  odom.pose.pose.position.y    = pose.translation().y();
  odom.pose.pose.position.z    = pose.translation().z();
  odom.pose.pose.orientation.x = q.x();
  odom.pose.pose.orientation.y = q.y();
  odom.pose.pose.orientation.z = q.z();
  odom.pose.pose.orientation.w = q.w();

  for (int i = 0; i < 6; ++i)
    for (int j = 0; j < 6; ++j)
      odom.pose.covariance[i * 6 + j] = pose_cov_gtsam(perm[i], perm[j]);

  odom.twist.twist.linear.x  = vel.x();
  odom.twist.twist.linear.y  = vel.y();
  odom.twist.twist.linear.z  = vel.z();
  odom.twist.twist.angular.z = last_gyro_z_ - bias.gyroscope()(2);

  // Velocity covariance in top-left 3x3 of 6x6 twist covariance
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j)
      odom.twist.covariance[i * 6 + j] = vel_cov(i, j);

  // Angular velocity diagonal from gyro noise
  const double gyro_var = fgo_config_.gyro_noise * fgo_config_.gyro_noise;
  odom.twist.covariance[21] = gyro_var;
  odom.twist.covariance[28] = gyro_var;
  odom.twist.covariance[35] = gyro_var;

  state_pub_->publish(odom);
}

void FactorGraphTrackingNode::publishROVState(const gtsam::Values &est) {
  for (const auto& [rov_id, initialized] : rov_initialised_) {
    if (initialized && rov_step_counters_[rov_id] >= 1) { // Ensure we have at least one measurement for this ROV to publish a state
      uint32_t current_rov_step = rov_step_counters_[rov_id] - 1; // Last updated step
      gtsam::Key rKey = getRovKey('R', rov_id, current_rov_step);
      gtsam::Key wKey = getRovKey('W', rov_id, current_rov_step);

      if (!est.exists(rKey) || !est.exists(wKey)) {
        return;
      }

      auto rov_pos = est.at<gtsam::Point3>(rKey);
      auto rov_vel = est.at<gtsam::Vector3>(wKey);

      ROVState rs;
      // rs.header.stamp = now();
      rs.header.stamp = rclcpp::Time(last_rov_timestamp_[rov_id]); // For comparison with GT

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
