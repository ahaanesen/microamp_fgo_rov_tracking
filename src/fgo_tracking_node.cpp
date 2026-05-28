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

static rclcpp::Time timeFromSeconds(double seconds) {
  return rclcpp::Time(static_cast<int64_t>(seconds * 1e9));
}

static gtsam::ISAM2Params make_isam2_params() {
  gtsam::ISAM2Params p;
  gtsam::ISAM2GaussNewtonParams gn;
  gn.wildfireThreshold = 1e-3;
  p.optimizationParams = gn;  // std::variant assignment in 4.3a

  p.factorization          = gtsam::ISAM2Params::CHOLESKY; 
  p.relinearizeThreshold   = 0.01;
  p.relinearizeSkip        = 5;
  p.enableRelinearization  = true;
  p.evaluateNonlinearError = false;
  p.cacheLinearizedFactors = true;

  return p;
}


// ============================================================
// CONSTRUCTOR, DESTRUCTOR, AND INITIALIZATION
// ============================================================

FactorGraphTrackingNode::FactorGraphTrackingNode()
    : Node("microamp_factor_graph_tracking"), isam2_(make_isam2_params()),
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
  // datum_initialised_ = false;

  // ------------------------------------------------------------------
  // IMU Preintegration (NED)
  // ------------------------------------------------------------------
  pim_params_ = gtsam::PreintegratedCombinedMeasurements::Params::MakeSharedD(env_config_.gravity);

  pim_params_->accelerometerCovariance = fgo_config_.accel_noise * fgo_config_.accel_noise * gtsam::I_3x3;
  pim_params_->gyroscopeCovariance = fgo_config_.gyro_noise * fgo_config_.gyro_noise * gtsam::I_3x3;
  pim_params_->integrationCovariance = 1e-8 * gtsam::I_3x3;
  pim_params_->biasAccCovariance = fgo_config_.accel_bias * fgo_config_.accel_bias * gtsam::I_3x3;
  pim_params_->biasOmegaCovariance = fgo_config_.gyro_bias * fgo_config_.gyro_bias * gtsam::I_3x3;
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

void FactorGraphTrackingNode::imuCallback(const Imu::SharedPtr msg) {
  if (!graph_initialised_) {
    return;
  }
  rclcpp::Time stamp = msg->header.stamp;

  last_gyro_z_ = msg->angular_velocity.z;

  if (!imu_initialised_) {
    last_imu_time_ = stamp;
    imu_initialised_ = true;
    return;
  }

  double dt = (stamp - last_imu_time_).seconds();
  last_imu_time_ = stamp;

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

void FactorGraphTrackingNode::gnssCallback(const GNSSNavPvt::SharedPtr gnss_msg) {
  constexpr uint8_t FIX_3D = 3;
  if (gnss_msg->fix_type < FIX_3D) {
    RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 5000,
                         "Waiting for 3D fix (current fix_type=%d)", gnss_msg->fix_type);
    return;
  }
  if (!gnss_msg->gnss_fix_ok) {
    RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 5000, "gnss_fix_ok not set, skipping");
    return;
  }

  // if (!datum_initialised_) {
  //   initializeDatumFromGNSS(*gnss_msg);
  // }

  if (!graph_initialised_) {
    if (ne_init_ == std::tuple<double, double>{0.0, 0.0}) {
      NedCoordinates ned_coords = ned_.gnssToNED(gnss_msg->lat, gnss_msg->lon, gnss_msg->height);
      ne_init_ = std::make_tuple(ned_coords.n, ned_coords.e);
      return;
    }
    initializeGraphWithGNSS(gnss_msg);
    graph_initialised_ = true;
    pim_->resetIntegrationAndSetBias(bias_);
    return;
  }

  // 2. Value initialization for new ASV node
  rclcpp::Time current_asv_time = gnss_msg->header.stamp;
  uint64_t prev_asv_index = asv_index_;
  uint64_t curr_asv_index = asv_index_ + 1;

  auto prev_asv_pose = isam2_.calculateEstimate<gtsam::Pose3>(X(prev_asv_index));
  auto prev_asv_vel  = isam2_.calculateEstimate<gtsam::Vector3>(V(prev_asv_index));
  auto prev_asv_bias = isam2_.calculateEstimate<gtsam::imuBias::ConstantBias>(B(prev_asv_index));

  gtsam::NavState prev_asv_state(prev_asv_pose, prev_asv_vel);
  gtsam::NavState predicted_asv_state = pim_->predict(prev_asv_state, prev_asv_bias);

  values_.insert(X(curr_asv_index), predicted_asv_state.pose());
  values_.insert(V(curr_asv_index), predicted_asv_state.v());
  values_.insert(B(curr_asv_index), prev_asv_bias);

  // 3. Add IMU factor
  graph_.add(gtsam::CombinedImuFactor(X(prev_asv_index), V(prev_asv_index),
                                      X(curr_asv_index), V(curr_asv_index),
                                      B(prev_asv_index), B(curr_asv_index),
                                      *pim_));

  // 4. Add GPS factor
  NedCoordinates ned_coords = ned_.gnssToNED(gnss_msg->lat, gnss_msg->lon, gnss_msg->height);
  double sigma_ne = std::clamp(fgo_config_.gps_sigma_ne,
                               fgo_config_.gps_sigma_floor, fgo_config_.gps_sigma_max);
  double sigma_d  = std::clamp(fgo_config_.gps_sigma_d,
                               fgo_config_.gps_sigma_floor, fgo_config_.gps_sigma_max);
  auto gps_noise = gtsam::noiseModel::Diagonal::Sigmas(
    (gtsam::Vector(3) << sigma_ne, sigma_ne, sigma_d).finished());
  auto gps_leverarm = gtsam::Point3(
      env_config_.gps_offset[0], env_config_.gps_offset[1], env_config_.gps_offset[2]);
  graph_.add(gtsam::GPSFactorArm(
      X(curr_asv_index),
      gtsam::Point3(ned_coords.n, ned_coords.e, ned_coords.d),
      gps_leverarm, gps_noise));


  // auto rp_noise = gtsam::noiseModel::Diagonal::Sigmas(
  //   (gtsam::Vector(2) << 0.05, 0.05).finished());  // 50 mrad ≈ 3° roll/pitch sigma
  // graph_.add(std::make_shared<RollPitchPriorFactor>(
  //     X(curr_asv_index), rp_noise));

  // 5. Process queued USBL measurements.
  //
  //    The step counter is incremented immediately inside the loop so that
  //    processing multiple USBL messages in one GNSS callback (possible when
  //    USBL rate > GNSS rate or measurements are buffered) uses distinct GTSAM
  //    keys for each new ROV node.
  while (!usbl_queue_.empty()) {
    const auto& usbl_msg = usbl_queue_.front();
    uint8_t rov_id   = usbl_msg->rov_id;
    rclcpp::Time rov_time = timeFromSeconds(static_cast<double>(usbl_msg->t_sent) / 1e6);
    gtsam::Key associated_asv_key =
        getAsvKeyForRovAssociation(current_asv_time, curr_asv_index, rov_time);

    if (!rov_initialised_[rov_id]) {
      initializeNewRov(associated_asv_key, usbl_msg);
      rov_initialised_[rov_id] = true;
      // Advance to step 1 so the first rovUpdateWithUsbl call uses step 0 as r_prev.
      // Without this, current_rov_step would be 0 and r_prev would underflow to UINT32_MAX.
      rov_step_counters_[rov_id] += 1;
    } else {
      
      if (rovUpdateWithUsbl(rov_id, rov_time, associated_asv_key, usbl_msg)) {
        rov_step_counters_[rov_id] += 1;
        last_rov_timestamp_[rov_id] = rov_time;
      }
      // rovUpdateWithUsbl(rov_id, rov_time, associated_asv_key, usbl_msg);
      // // Increment immediately so the next iteration (if any) uses the correct key
      // rov_step_counters_[rov_id] += 1;
      // last_rov_timestamp_[rov_id] = rov_time;
    }

    usbl_queue_.pop_front();
  }

  // 6. Graph optimization (first pass)
  RCLCPP_INFO(get_logger(), "Graph before update: %zu factors, %zu values",
              graph_.size(), values_.size());
  for (const auto& kv : values_) {
    RCLCPP_INFO(get_logger(), "  key %lu  dim %zu",
                kv.key, kv.value.dim());
  }
  if (graph_.size() > 0) {
    graph_.print("Factors:\n");
  }

  isam2_.update(graph_, values_);
  graph_.resize(0);
  values_.clear();

  // Extra optimization pass with no new factors: improves convergence for the
  // nonlinear USBL bearing factors, especially early in tracking when position
  // uncertainty is still large and the linearisation point changes significantly.
  // isam2_.update();

  // 7. Housekeeping
  bias_ = isam2_.calculateEstimate<gtsam::imuBias::ConstantBias>(B(curr_asv_index));
  pim_->resetIntegrationAndSetBias(bias_);

  last_asv_timestamp_ = gnss_msg->header.stamp;
  asv_index_ = curr_asv_index;
  asv_timeline_[last_asv_timestamp_] = X(curr_asv_index);

  // 8. Publish
  auto current_estimate = isam2_.calculateEstimate();
  publishBoatOdometry(current_estimate);
  publishROVState(current_estimate);

  RCLCPP_INFO(get_logger(),
              "Processed GNSS at %.2f, NED (%.2f, %.2f, %.2f).",
              current_asv_time, ned_coords.n, ned_coords.e, ned_coords.d);
}

// ============================================================
// HELPERS
// ============================================================

// gtsam::Key FactorGraphTrackingNode::getRovKey(
//     unsigned char prefix, uint32_t rov_id, uint32_t step_count) {
//   uint64_t packed_index = (uint64_t(rov_id) << 32) | step_count;
//   return gtsam::Symbol(prefix, packed_index);
// }
gtsam::Key FactorGraphTrackingNode::getRovKey(
    unsigned char prefix, uint32_t rov_id, uint32_t step_count) {
  // Encode as a flat index: rov_id * MAX_STEPS + step_count.
  // MAX_STEPS must exceed the maximum number of acoustic updates
  // expected in any mission; 1000000 is safe for all practical cases.
  constexpr uint64_t MAX_STEPS = 1'000'000ULL;
  uint64_t flat_index = static_cast<uint64_t>(rov_id) * MAX_STEPS
                      + static_cast<uint64_t>(step_count);
  return gtsam::Symbol(prefix, flat_index);
}

gtsam::Key FactorGraphTrackingNode::getAsvKeyForRovAssociation(
    rclcpp::Time current_asv_time, uint64_t current_asv_index, rclcpp::Time rov_time) {
  gtsam::Key closest_asv_key = X(current_asv_index);
  double smallest_time_diff  = std::abs((current_asv_time.seconds() - rov_time.seconds()));

  for (auto it = asv_timeline_.rbegin(); it != asv_timeline_.rend(); ++it) {
    rclcpp::Time asv_time  = it->first;
    double time_diff = std::abs((asv_time.seconds() - rov_time.seconds()));
    if (time_diff < smallest_time_diff) {
      smallest_time_diff = time_diff;
      closest_asv_key    = it->second;
    } else {
      break;
    }
  }
  return closest_asv_key;
}

bool FactorGraphTrackingNode::rovUpdateWithUsbl(
    uint8_t rov_id,
    rclcpp::Time rov_time,
    gtsam::Key associated_asv_key,
    const USBLMessage::SharedPtr& usbl_msg)
{
  double dt_rov = (rov_time - last_rov_timestamp_[rov_id]).seconds();
  if (dt_rov <= 0.0) {
    RCLCPP_WARN(get_logger(),
                "Non-positive dt for ROV %d: %.6f, skipping USBL measurement",
                rov_id, dt_rov);
    return false;
  }

  uint32_t current_rov_step = rov_step_counters_[rov_id];
  gtsam::Key r_curr = getRovKey('R', rov_id, current_rov_step);
  gtsam::Key w_curr = getRovKey('W', rov_id, current_rov_step);
  gtsam::Key r_prev = getRovKey('R', rov_id, current_rov_step - 1);
  gtsam::Key w_prev = getRovKey('W', rov_id, current_rov_step - 1);

  // Initialise new node from previous estimate.
  // Fall back to values_ when r_prev was just inserted in this GNSS callback
  // (not yet in the iSAM2 Bayes tree) to avoid calculateEstimate failures
  // when multiple USBL messages are processed in one gnssCallback invocation.
  gtsam::Point3  r_pred;
  gtsam::Vector3 w_pred;
  if (values_.exists(r_prev)) {
    r_pred = values_.at<gtsam::Point3>(r_prev);
    w_pred = values_.at<gtsam::Vector3>(w_prev);
  } else {
    r_pred = isam2_.calculateEstimate<gtsam::Point3>(r_prev);
    w_pred = isam2_.calculateEstimate<gtsam::Vector3>(w_prev);
  }

  values_.insert(r_curr, (r_pred + w_pred * dt_rov).eval());
  values_.insert(w_curr, w_pred.eval());

      // Constant velocity factor between previous and current ROV node
  const double sigma_a = fgo_config_.rov_cv_continous_sigma;  // m/s²
  const double sigma_p_h = sigma_a * dt_rov * dt_rov / 2.0;   // horizontal pos
  const double sigma_p_v = sigma_a * dt_rov * dt_rov / 2.0 * std::sqrt(0.1);
  const double sigma_v_h = sigma_a * dt_rov;                  // horizontal vel
  const double sigma_v_v = sigma_a * dt_rov * std::sqrt(0.1);

  auto cv_noise = gtsam::noiseModel::Diagonal::Sigmas(
      (gtsam::Vector(6) << sigma_p_h, sigma_p_h, sigma_p_v,
                          sigma_v_h, sigma_v_h, sigma_v_v).finished());
  graph_.add(std::make_shared<ConstantVelocityFactor>(
      r_prev, w_prev, r_curr, w_curr, dt_rov, cv_noise));

      // Alternative split CV factorisation (position integration + velocity random walk)
  // const double sigma_a = fgo_config_.rov_cv_continous_sigma;       // m/s²
  // // Position-integration noise (double-integrated accel noise)
  // const double sigma_p_h = sigma_a * dt_rov * dt_rov / 2.0;
  // const double sigma_p_v = sigma_p_h * std::sqrt(0.1);             // tighter down-axis
  // // Velocity random-walk noise (single-integrated accel noise)
  // const double sigma_v_h = sigma_a * dt_rov;
  // const double sigma_v_v = sigma_v_h * std::sqrt(0.1);
  
  // auto pos_noise = gtsam::noiseModel::Diagonal::Sigmas(
  //     (gtsam::Vector(3) << sigma_p_h, sigma_p_h, sigma_p_v).finished());
  // auto vel_noise = gtsam::noiseModel::Diagonal::Sigmas(
  //     (gtsam::Vector(3) << sigma_v_h, sigma_v_h, sigma_v_v).finished());
  
  // // Split CV: position integration + velocity random walk (decoupled, SPD-safe)
  // graph_.add(std::make_shared<ConstantVelocityIntegrationFactor>(
  //     r_prev, w_prev, r_curr, dt_rov, pos_noise));
  // graph_.add(std::make_shared<VelocityRandomWalkFactor>(
  //     w_prev, w_curr, vel_noise));

// USBL factor
  auto body_P_sensor = getBodyToUsblPose(env_config_);
  auto usbl_noise = gtsam::noiseModel::Diagonal::Sigmas(
    (gtsam::Vector(2) << fgo_config_.usbl_azimuth_sigma,
                         fgo_config_.usbl_elevation_sigma).finished());
  graph_.add(std::make_shared<UsblFactor>(
      associated_asv_key, r_curr,
      usbl_msg->azimuth, usbl_msg->elevation,
      body_P_sensor, usbl_noise));

  // // Optional soft depth prior (bearing-only stabilisation)
  // if (fgo_config_.use_rov_depth_prior) {
  //   auto depth_prior_noise = gtsam::noiseModel::Isotropic::Sigma(
  //       1, fgo_config_.rov_depth_prior_sigma);
  //   graph_.add(std::make_shared<DepthFactor>(
  //       r_curr, fgo_config_.rov_depth_prior_mean, depth_prior_noise));
  // }

  if (scenario_id_ >= bearing_range) {
    auto tof = (usbl_msg->t_received - usbl_msg->t_sent) / 1e6;
    auto acoustic_noise = gtsam::noiseModel::Isotropic::Sigma(
        1, fgo_config_.acoustic_range_sigma / env_config_.sound_speed);
    graph_.add(std::make_shared<PsudoRangeFactor>(
        associated_asv_key, r_curr, tof, env_config_.sound_speed,
        body_P_sensor, acoustic_noise));
  }
  if (scenario_id_ >= bearing_range_depth) {
    auto depth_noise = gtsam::noiseModel::Isotropic::Sigma(
        1, fgo_config_.rov_depth_sigma);
    graph_.add(std::make_shared<DepthFactor>(
        r_curr, usbl_msg->position.z, depth_noise));
  }

  RCLCPP_INFO(get_logger(),
              "Added ROV %d node (step %u) at t=%.2f, dt=%.3f s",
              rov_id, current_rov_step, rov_time, dt_rov);
  return true;
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

void FactorGraphTrackingNode::acousticCommCallback(
    const AcousticCommReceive::SharedPtr msg) {
  (void)msg;
}

// ============================================================
// ROV INITIALIZATION
// ============================================================

void FactorGraphTrackingNode::initializeNewRov(
    gtsam::Key associated_asv_key,
    const USBLMessage::SharedPtr& usbl_msg)
{
  uint8_t rov_id    = usbl_msg->rov_id;
  if (fgo_config_.rov_use_gt) {
    gtsam::Point3 gt_pos(usbl_msg->position.x, usbl_msg->position.y, usbl_msg->position.z);
    gtsam::Point3 gt_vel(0.0, 0.225, 0.072);
    values_.insert(getRovKey('R', rov_id, 0), gt_pos);
    values_.insert(getRovKey('W', rov_id, 0), gt_vel);
    RCLCPP_INFO(get_logger(),
                "Initialized ROV %d with GT at NED (%.2f, %.2f, %.2f)",
                rov_id, gt_pos.x(), gt_pos.y(), gt_pos.z());
    return;
  }

  double  t_sent    = static_cast<double>(usbl_msg->t_sent)    / 1e6;
  double  t_receive = static_cast<double>(usbl_msg->t_received) / 1e6;

  gtsam::Pose3 world_T_asv;
  if (values_.exists(associated_asv_key)) {
    world_T_asv = values_.at<gtsam::Pose3>(associated_asv_key);
  } else {
    world_T_asv = isam2_.calculateEstimate<gtsam::Pose3>(associated_asv_key);
  }

  double az_rad = (usbl_msg->azimuth)   * M_PI / 180.0;
  double el_rad = (usbl_msg->elevation) * M_PI / 180.0;
  // double tof    = t_receive - t_sent;
  // double r      = tof * env_config_.sound_speed;

  gtsam::Pose3  body_P_sensor  = getBodyToUsblPose(env_config_);
  gtsam::Pose3  world_T_sensor = world_T_asv.compose(body_P_sensor);
  gtsam::Point3 sensor_pos     = world_T_sensor.translation();

  double r = fgo_config_.rov_initial_range_guess;  // Use fixed initial range guess instead of time-of-flight
  gtsam::Point3 p_world = gtsam::Point3(
      sensor_pos.x() + r * std::cos(el_rad) * std::cos(az_rad),
      sensor_pos.y() + r * std::cos(el_rad) * std::sin(az_rad),
      sensor_pos.z() + r * std::sin(el_rad));

  if (scenario_id_ >= bearing_range) {
    r = (t_receive - t_sent) * env_config_.sound_speed;
    // Use actual range from time-of-flight, but still ignore depth measurement (if any) for initial guess
    p_world = gtsam::Point3(
        sensor_pos.x() + r * std::cos(el_rad) * std::cos(az_rad),
        sensor_pos.y() + r * std::cos(el_rad) * std::sin(az_rad),
        sensor_pos.z() + r * std::sin(el_rad));
    if (scenario_id_ >= bearing_range_depth) {
      // If depth measurement is available, use it to refine the initial guess
      p_world.z() = usbl_msg->position.z;
    }
  } 

  gtsam::Key rKey = getRovKey('R', rov_id, 0);
  gtsam::Key wKey = getRovKey('W', rov_id, 0);
  values_.insert(rKey, p_world);
  values_.insert(wKey, gtsam::Vector3::Zero().eval());

  auto prior_pos_noise = gtsam::noiseModel::Isotropic::Sigma(3, fgo_config_.rov_prior_pos_sigma);
  auto prior_vel_noise = gtsam::noiseModel::Isotropic::Sigma(3, fgo_config_.rov_prior_vel_sigma);
  graph_.add(gtsam::PriorFactor<gtsam::Point3>(rKey, p_world, prior_pos_noise));
  graph_.add(gtsam::PriorFactor<gtsam::Vector3>(wKey, gtsam::Vector3::Zero(), prior_vel_noise));

  auto usbl_noise = gtsam::noiseModel::Diagonal::Sigmas(
    (gtsam::Vector(2) << fgo_config_.usbl_azimuth_sigma,
                         fgo_config_.usbl_elevation_sigma).finished());
  graph_.add(std::make_shared<UsblFactor>(
      associated_asv_key, rKey,
      usbl_msg->azimuth, usbl_msg->elevation,
      body_P_sensor, usbl_noise));

  if (scenario_id_ >= bearing_range) {
    double tof    = t_receive - t_sent;
    auto acoustic_noise = gtsam::noiseModel::Isotropic::Sigma(
        1, fgo_config_.acoustic_range_sigma / env_config_.sound_speed);
    graph_.add(std::make_shared<PsudoRangeFactor>(
        associated_asv_key, rKey, tof, env_config_.sound_speed,
        body_P_sensor, acoustic_noise));
  }
  if (scenario_id_ >= bearing_range_depth) {
    auto depth_noise = gtsam::noiseModel::Isotropic::Sigma(
        1, fgo_config_.rov_depth_sigma);
    graph_.add(std::make_shared<DepthFactor>(rKey, usbl_msg->position.z, depth_noise));
  }

  rov_step_counters_[rov_id]   = 0;
  last_rov_timestamp_[rov_id]  = timeFromSeconds(t_sent);

  RCLCPP_INFO(get_logger(),
              "Initialized ROV %d at NED (%.2f, %.2f, %.2f), t_sent=%.2f",
              rov_id, p_world.x(), p_world.y(), p_world.z(), t_sent);
}

// ============================================================
// GRAPH INITIALIZATION
// ============================================================

void FactorGraphTrackingNode::initializeGraphWithGNSS(
    const GNSSNavPvt::SharedPtr gnss_msg)
{
  RCLCPP_INFO(get_logger(), "Initializing factor graph with first GNSS at time %.2f",
              rclcpp::Time(gnss_msg->header.stamp).seconds());

  NedCoordinates ned_coords =
      ned_.gnssToNED(gnss_msg->lat, gnss_msg->lon, gnss_msg->height);

  double delta_n    = ned_coords.n - std::get<0>(ne_init_);
  double delta_e    = ned_coords.e - std::get<1>(ne_init_);
  double initial_yaw = std::atan2(delta_e, delta_n);
  gtsam::Rot3  initial_rot(gtsam::Rot3::Ypr(initial_yaw, 0.0, 0.0));
  gtsam::Pose3 priorPose(initial_rot,
                         gtsam::Point3(ned_coords.n, ned_coords.e, ned_coords.d));
  gtsam::Vector3 priorVel(0.0, 0.0, 0.0);
  gtsam::imuBias::ConstantBias priorBias;

  // auto pose_noise = gtsam::noiseModel::Diagonal::Sigmas(
  //     (gtsam::Vector(6) << M_PI, M_PI, M_PI,
  //      fgo_config_.prior_translation_sigma,
  //      fgo_config_.prior_translation_sigma,
  //      fgo_config_.prior_translation_sigma).finished());
  auto pose_noise = gtsam::noiseModel::Diagonal::Sigmas(
    (gtsam::Vector(6) << 
       0.1,    // roll: 6 deg
       0.1,    // pitch: 6 deg
       0.5,    // yaw: 30 deg (consistent with 2-GNSS init uncertainty)
       fgo_config_.prior_translation_sigma,
       fgo_config_.prior_translation_sigma,
       fgo_config_.prior_translation_sigma).finished());

  graph_.add(gtsam::PriorFactor<gtsam::Pose3>(X(0), priorPose, pose_noise));
  
  // auto rp_noise_init = gtsam::noiseModel::Diagonal::Sigmas(
  //   (gtsam::Vector(2) << 0.05, 0.05).finished());
  // graph_.add(std::make_shared<RollPitchPriorFactor>(X(0), rp_noise_init));

  graph_.add(gtsam::PriorFactor<gtsam::Vector3>(
      V(0), priorVel,
      gtsam::noiseModel::Isotropic::Sigma(3, fgo_config_.prior_vel_sigma)));
  auto bias_noise = gtsam::noiseModel::Diagonal::Sigmas(
    (gtsam::Vector(6) << fgo_config_.prior_accel_bias_sigma,
                         fgo_config_.prior_accel_bias_sigma,
                         fgo_config_.prior_accel_bias_sigma,
                         fgo_config_.prior_gyro_bias_sigma,
                         fgo_config_.prior_gyro_bias_sigma,
                         fgo_config_.prior_gyro_bias_sigma).finished());
  graph_.add(gtsam::PriorFactor<gtsam::imuBias::ConstantBias>(
      B(0), priorBias, bias_noise));

  double sigma_ne = std::clamp(fgo_config_.gps_sigma_ne,
                               fgo_config_.gps_sigma_floor, fgo_config_.gps_sigma_max);
  double sigma_d  = std::clamp(fgo_config_.gps_sigma_d,
                               fgo_config_.gps_sigma_floor, fgo_config_.gps_sigma_max);
  auto gps_noise_init = gtsam::noiseModel::Diagonal::Sigmas(
    (gtsam::Vector(3) << sigma_ne, sigma_ne, sigma_d).finished());
  auto gps_leverarm = gtsam::Point3(
      env_config_.gps_offset[0], env_config_.gps_offset[1], env_config_.gps_offset[2]);
  graph_.add(gtsam::GPSFactorArm(
      X(0),
      gtsam::Point3(ned_coords.n, ned_coords.e, ned_coords.d),
      gps_leverarm, gps_noise_init));

  values_.insert(X(0), priorPose);
  values_.insert(V(0), priorVel);
  values_.insert(B(0), priorBias);

  isam2_.update(graph_, values_);
  graph_.resize(0);
  values_.clear();

  last_asv_timestamp_ = gnss_msg->header.stamp;
  asv_index_          = 0;
  asv_timeline_[last_asv_timestamp_] = X(0);

  RCLCPP_INFO(get_logger(),
              "Graph initialized at NED (%.2f, %.2f, %.2f)",
              ned_coords.n, ned_coords.e, ned_coords.d);
}

gtsam::Values FactorGraphTrackingNode::updateAndGetEstimate() {
  return isam2_.calculateEstimate();
}

// ============================================================
// PUBLISH STATES
// ============================================================

void FactorGraphTrackingNode::publishBoatOdometry(const gtsam::Values &est) {
  if (!est.exists(X(asv_index_)) || !est.exists(V(asv_index_)) ||
      !est.exists(B(asv_index_))) {
    return;
  }

  auto pose = est.at<gtsam::Pose3>(X(asv_index_));
  auto vel  = est.at<gtsam::Vector3>(V(asv_index_));
  auto bias = est.at<gtsam::imuBias::ConstantBias>(B(asv_index_));

  gtsam::Matrix pose_cov_gtsam = isam2_.marginalCovariance(X(asv_index_));
  gtsam::Matrix vel_cov        = isam2_.marginalCovariance(V(asv_index_));

  const int perm[6] = {3, 4, 5, 0, 1, 2};

  gtsam::Quaternion q = pose.rotation().toQuaternion();

  Odometry odom;
  odom.header.stamp    = last_asv_timestamp_;
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

  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j)
      odom.twist.covariance[i * 6 + j] = vel_cov(i, j);

  const double gyro_var = fgo_config_.gyro_noise * fgo_config_.gyro_noise;
  odom.twist.covariance[21] = gyro_var;
  odom.twist.covariance[28] = gyro_var;
  odom.twist.covariance[35] = gyro_var;

  state_pub_->publish(odom);
}

void FactorGraphTrackingNode::publishROVState(const gtsam::Values &est) {
  for (const auto& [rov_id, initialized] : rov_initialised_) {
    if (initialized && rov_step_counters_[rov_id] >= 1) {
      uint32_t current_rov_step = rov_step_counters_[rov_id] - 1;
      gtsam::Key rKey = getRovKey('R', rov_id, current_rov_step);
      gtsam::Key wKey = getRovKey('W', rov_id, current_rov_step);

      if (!est.exists(rKey) || !est.exists(wKey)) {
        continue;
      }

      auto rov_pos = est.at<gtsam::Point3>(rKey);
      auto rov_vel = est.at<gtsam::Vector3>(wKey);
      gtsam::Matrix rov_pos_cov = isam2_.marginalCovariance(rKey);

      ROVState rs;
      rs.header.stamp = last_rov_timestamp_[rov_id];
      rs.rov_id       = rov_id;

      rs.position.x = rov_pos.x();
      rs.position.y = rov_pos.y();
      rs.position.z = rov_pos.z();

      rs.velocity.x = rov_vel.x();
      rs.velocity.y = rov_vel.y();
      rs.velocity.z = rov_vel.z();

      for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
          rs.position_covariance[i * 3 + j] = rov_pos_cov(i, j);

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
