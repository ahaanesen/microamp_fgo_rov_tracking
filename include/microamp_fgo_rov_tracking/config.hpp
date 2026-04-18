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
  // ASV IMU noise (from ESKF ModelIMU)
  double accel_noise{0.001167};  // m/s² white noise std
  double gyro_noise{0.0000436};  // rad/s white noise std
  double accel_rw{0.004};        // m/s² random walk std (accm_bias_std)
  double gyro_rw{0.00005};       // rad/s random walk std (gyro_bias_std)

  // ASV prior uncertainties (from ESKF initial state)
  double prior_pose_sigma{1.0};           // metres and radians
  double prior_vel_sigma{0.1};            // m/s
  double prior_bias_sigma{0.001};         // IMU gyro bias prior
  double prior_accel_bias_sigma{0.01};    // IMU accel bias prior

  // GPS measurement noise (from ESKF SensorGNSS_ASV)
  double gps_sigma_ne{0.3};       // horizontal GPS sigma (metres)
  double gps_sigma_d{0.5};        // depth GPS sigma (metres)
  double gps_sigma_floor{0.3};    // minimum GPS sigma (metres)
  double gps_sigma_max{50.0};     // maximum GPS sigma (metres)

  // ROV priors (from ESKF initial state)
  double rov_cv_continous_sigma{0.2};    // m/s (ModelCV process noise)
  double rov_prior_pos_sigma{2.0};       // metres
  double rov_prior_vel_sigma{0.1};       // m/s
  double rov_process_vel_sigma{0.2};     // m/s (matches ModelCV sigma_a)

  // USBL/Range sensor noise (from ESKF sensors)
  double usbl_azimuth_sigma{0.01745};    // radians (~1 degree)
  double usbl_elevation_sigma{0.01745};  // radians (~1 degree)
  double acoustic_range_sigma{0.5};      // metres (SensorRange_Joint)
  double rov_depth_sigma{0.3};           // metres (SensorDepth_ROV)
};

void declareAndLoadTopics(rclcpp::Node& node, TopicsConfig& cfg);
void declareAndLoadEnv(rclcpp::Node& node, EnvConfig& cfg);
void declareAndLoadFgo(rclcpp::Node& node, FgoConfig& cfg);
