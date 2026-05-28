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

    double gravity{9.82};
    double sound_speed{1500.0};
    std::vector<double> gps_offset{0.3, 0.3, 0.1};
    double usbl_offset_x{0.0};
    double usbl_offset_y{0.0};
    double usbl_offset_z{1.2};

    double usbl_roll_deg{0.0};
    double usbl_pitch_deg{0.0};
    double usbl_yaw_deg{0.0};
    double imu_buffer_duration_sec{10.0};
};

struct FgoConfig {
  // ASV IMU noise
  double accel_noise{4.363e-3};
  double gyro_noise{1.1667e-3};
  double accel_bias{4.9e-4};
  double gyro_bias{2.4e-6};

  // ASV prior uncertainties
  double prior_translation_sigma{1.0};
  double prior_rotation_sigma{5};
  double prior_vel_sigma{0.1};
  double prior_gyro_bias_sigma{0.01};
  double prior_accel_bias_sigma{0.001};

  // GPS measurement noise
  double gps_sigma_ne{1.5};
  double gps_sigma_d{2.0};
  double gps_sigma_floor{0.3};
  double gps_sigma_max{50.0};

  // ROV CV process noise.
  bool rov_use_gt{false};  // whether to use ROV ground truth as a measurement
  std::vector<double> rov_pos_gt{5, -5, 10}; 
  double rov_cv_continous_sigma{0.020};   // m/s²

  double rov_initial_range_guess{10.0};   // initial guess on range from ASV to ROV [m]

  // ROV priors
  double rov_prior_pos_sigma{2.0};
  double rov_prior_vel_sigma{0.1};     

  // USBL / range / depth sensor noise
  double usbl_azimuth_sigma{0.01745};
  double usbl_elevation_sigma{0.01745};
  double acoustic_range_sigma{0.3};
  double rov_depth_sigma{1.5};

  // // Bearing-only: weak depth stabilising prior
  // bool   use_rov_depth_prior{false};
  // double rov_depth_prior_mean{10.0};
  // double rov_depth_prior_sigma{10.0};
};


void declareAndLoadTopics(rclcpp::Node& node, TopicsConfig& cfg);
void declareAndLoadEnv(rclcpp::Node& node, EnvConfig& cfg);
void declareAndLoadFgo(rclcpp::Node& node, FgoConfig& cfg);
