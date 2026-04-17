#include "microamp_fgo_rov_tracking/config.hpp"


void declareAndLoadTopics(rclcpp::Node& node, TopicsConfig& cfg) {
  node.declare_parameter("topics.imu", cfg.imu);
  node.declare_parameter("topics.gnss", cfg.gnss);
  node.declare_parameter("topics.usbl", cfg.usbl);
  node.declare_parameter("topics.acoustic_rx", cfg.acoustic_rx);
  node.declare_parameter("topics.boat_state_pub", cfg.boat_state_pub);
  node.declare_parameter("topics.rov_state_pub", cfg.rov_state_pub);

  cfg.imu = node.get_parameter("topics.imu").as_string();
  cfg.gnss = node.get_parameter("topics.gnss").as_string();
  cfg.usbl = node.get_parameter("topics.usbl").as_string();
  cfg.acoustic_rx = node.get_parameter("topics.acoustic_rx").as_string();
  cfg.boat_state_pub = node.get_parameter("topics.boat_state_pub").as_string();
  cfg.rov_state_pub = node.get_parameter("topics.rov_state_pub").as_string();
}

void declareAndLoadEnv(rclcpp::Node& node, EnvConfig& cfg) {
  node.declare_parameter("env.imu_rate_hz", cfg.imu_rate_hz);
  node.declare_parameter("env.gnss_rate_hz", cfg.gnss_rate_hz);
  node.declare_parameter("env.usbl_rate_hz", cfg.usbl_rate_hz);
  node.declare_parameter("env.gravity", cfg.gravity);
  node.declare_parameter("env.sound_speed", cfg.sound_speed);
  node.declare_parameter("env.usbl_offset.x", cfg.usbl_offset_x);
  node.declare_parameter("env.usbl_offset.y", cfg.usbl_offset_y);
  node.declare_parameter("env.usbl_offset.z", cfg.usbl_offset_z);
  node.declare_parameter("env.usbl_rpy_deg.roll", cfg.usbl_roll_deg);
  node.declare_parameter("env.usbl_rpy_deg.pitch", cfg.usbl_pitch_deg);
  node.declare_parameter("env.usbl_rpy_deg.yaw", cfg.usbl_yaw_deg);
  node.declare_parameter("env.imu_buffer_duration_sec", cfg.imu_buffer_duration_sec);

  cfg.imu_rate_hz = node.get_parameter("env.imu_rate_hz").as_double();
  cfg.gnss_rate_hz = node.get_parameter("env.gnss_rate_hz").as_double();
  cfg.usbl_rate_hz = node.get_parameter("env.usbl_rate_hz").as_double();
  cfg.gravity = node.get_parameter("env.gravity").as_double();
  cfg.sound_speed = node.get_parameter("env.sound_speed").as_double();
  cfg.usbl_offset_x = node.get_parameter("env.usbl_offset.x").as_double();
  cfg.usbl_offset_y = node.get_parameter("env.usbl_offset.y").as_double();
  cfg.usbl_offset_z = node.get_parameter("env.usbl_offset.z").as_double();
  cfg.usbl_roll_deg = node.get_parameter("env.usbl_rpy_deg.roll").as_double();
  cfg.usbl_pitch_deg = node.get_parameter("env.usbl_rpy_deg.pitch").as_double();
  cfg.usbl_yaw_deg = node.get_parameter("env.usbl_rpy_deg.yaw").as_double();
  cfg.imu_buffer_duration_sec = node.get_parameter("env.imu_buffer_duration_sec").as_double();
}

void declareAndLoadFgo(rclcpp::Node& node, FgoConfig& cfg) {
#define LOAD_D(name) do { node.declare_parameter("fgo." #name, cfg.name); cfg.name = node.get_parameter("fgo." #name).as_double(); } while (0)
  LOAD_D(accel_noise); LOAD_D(gyro_noise); LOAD_D(accel_rw); LOAD_D(gyro_rw);
  LOAD_D(prior_pose_sigma); LOAD_D(prior_vel_sigma); LOAD_D(prior_bias_sigma);
  LOAD_D(gps_sigma_floor); LOAD_D(gps_sigma_max);
  LOAD_D(rov_cv_continous_sigma);
  LOAD_D(rov_prior_pos_sigma); LOAD_D(rov_prior_vel_sigma); LOAD_D(rov_process_vel_sigma);
  LOAD_D(usbl_azimuth_sigma); LOAD_D(usbl_elevation_sigma);
  LOAD_D(acoustic_range_sigma); LOAD_D(rov_depth_sigma);
#undef LOAD_D
}
