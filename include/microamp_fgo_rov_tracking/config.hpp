#pragma once
#include <string>
#include <rclcpp/rclcpp.hpp>


struct TopicsConfig {
  std::string imu{"microampere/imu/data"};
  std::string gnss{"microampere/gnss/nav_pvt"};
  std::string usbl{"microampere/sensors/usbl"};
  std::string acoustic_rx{"microampere/acoustic/receive"};
  std::string boat_state_pub{"/state/boat"};
  std::string rov_state_pub{"/state/rov"};
};

struct EnvConfig {
    double imu_rate_hz{100.0}; 
    double gnss_rate_hz{1.0}; 
    double usbl_rate_hz{0.2}; 

    double gravity{9.82145996};
    double sound_speed{1500.0};
    std::vector<double> gps_offset{0.3, 0.3, 0.1}; // x, y, z offsets for GPS antenna from ASV center (in meters)
    double usbl_offset_x{0.0};
    double usbl_offset_y{0.0};
    double usbl_offset_z{1.2};

    double usbl_roll_deg{0.0};
    double usbl_pitch_deg{0.0};
    double usbl_yaw_deg{0.0};
    double imu_buffer_duration_sec{10.0};
};

struct FgoConfig {
  // ASV
  double accel_noise{1e-2};
  double gyro_noise{1e-3};
  double accel_rw{1e-4};
  double gyro_rw{1e-5};
  double prior_pose_sigma{1e-2};
  double prior_vel_sigma{1e-2};
  double prior_bias_sigma{1e-3};
  double gps_sigma_floor{0.5};
  double gps_sigma_max{50.0};

  // ROV
  double rov_cv_continous_sigma{0.5};  // m/s (how much we expect ROV velocity to change between timesteps)
  double rov_prior_pos_sigma{10.0};
  double rov_prior_vel_sigma{1.0};
  double rov_process_vel_sigma{0.5};
  double usbl_azimuth_sigma{0.05};
  double usbl_elevation_sigma{0.05};
  double acoustic_range_sigma{1.0};
  double rov_depth_sigma{0.2};
};

void declareAndLoadTopics(rclcpp::Node& node, TopicsConfig& cfg);
void declareAndLoadEnv(rclcpp::Node& node, EnvConfig& cfg);
void declareAndLoadFgo(rclcpp::Node& node, FgoConfig& cfg);
