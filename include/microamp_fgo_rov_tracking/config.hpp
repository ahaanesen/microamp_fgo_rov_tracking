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
    double usbl_rate_hz{1.0};

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
  double accel_noise{0.001167};
  double gyro_noise{0.0000436};
  double accel_rw{0.004};
  double gyro_rw{0.00005};

  // ASV prior uncertainties
  double prior_translation_sigma{1.0};
  double prior_rotation_sigma{0.087};
  double prior_vel_sigma{0.1};
  double prior_gyro_bias_sigma{1e-16};
  double prior_accel_bias_sigma{1e-16};

  // GPS measurement noise
  double gps_sigma_ne{0.3};
  double gps_sigma_d{0.5};
  double gps_sigma_floor{0.3};
  double gps_sigma_max{50.0};

  // ROV CV process noise.
  // MUST match ESKF ModelCV sigma_a.  Previously 0.02 (10x too small).
  // Q matrix uses DWPA formulation: Q_pp = sigma^2 * dt^4/4,
  //                                 Q_pv = sigma^2 * dt^3/2,
  //                                 Q_vv = sigma^2 * dt^2.
  double rov_cv_continous_sigma{0.020};   // m/s²
  double rov_process_vel_sigma{0.20};    // not used todo: remove

  double rov_initial_range_guess{10.0};   // initial guess on range from ASV to ROV [m]

  // ROV priors
  double rov_prior_pos_sigma{2.0};
  double rov_prior_vel_sigma{0.1};     

  // USBL / range / depth sensor noise
  double usbl_azimuth_sigma{0.01745};
  double usbl_elevation_sigma{0.01745};
  double acoustic_range_sigma{0.5};
  double rov_depth_sigma{0.3};

  // // Bearing-only: weak depth stabilising prior
  // bool   use_rov_depth_prior{false};
  // double rov_depth_prior_mean{10.0};
  // double rov_depth_prior_sigma{10.0};
};


void declareAndLoadTopics(rclcpp::Node& node, TopicsConfig& cfg);
void declareAndLoadEnv(rclcpp::Node& node, EnvConfig& cfg);
void declareAndLoadFgo(rclcpp::Node& node, FgoConfig& cfg);
