#pragma once

#include <gtsam/navigation/ImuBias.h>
#include <memory>
#include <blueboat_interfaces/msg/boat_state.hpp>
#include <blueboat_interfaces/msg/gnss_nav_pvt.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/imu.hpp>

#include <gtsam/inference/Symbol.h>
#include <gtsam/navigation/CombinedImuFactor.h>
#include <gtsam/navigation/GPSFactor.h>
#include <gtsam/navigation/NavState.h>
#include <gtsam/navigation/PreintegrationCombinedParams.h>
#include <gtsam/nonlinear/ISAM2.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/Values.h>

#include "microamp_fgo_rov_tracking/ned_converter.hpp"

using blueboat_interfaces::msg::BoatState;
using blueboat_interfaces::msg::GNSSNavPvt;
using sensor_msgs::msg::Imu;

using gtsam::symbol_shorthand::B;
using gtsam::symbol_shorthand::V;
using gtsam::symbol_shorthand::X;

class FactorGraphNode : public rclcpp::Node {
public:
  FactorGraphNode();

private:
  // --- Callbacks ---
  void imuCallback(const Imu::SharedPtr msg);
  void gnssCallback(const GNSSNavPvt::SharedPtr msg);

  // --- Initialization ---
  void initializeGraph(); // Priors
  void initializeDatumFromGNSS(const GNSSNavPvt &msg);

  // --- Helpers ---
  gtsam::Values updateAndGetEstimate();
  void publishState(const gtsam::Values &est);

  // --- ROS ---
  rclcpp::Subscription<Imu>::SharedPtr imu_sub_;
  rclcpp::Subscription<GNSSNavPvt>::SharedPtr gnss_sub_;
  rclcpp::Publisher<BoatState>::SharedPtr state_pub_;

  // Explicit init flag
  bool imu_initialised_;
  rclcpp::Time last_imu_time_;

  // Latest raw IMU measurement for yaw-rate output
  double last_gyro_z_;

  // --- Graph objects ---
  gtsam::ISAM2 isam2_;
  gtsam::NonlinearFactorGraph graph_;
  gtsam::Values values_;

  // --- IMU ---
  std::unique_ptr<gtsam::PreintegratedCombinedMeasurements> pim_;
  gtsam::imuBias::ConstantBias bias_;
  double gravity_;

  // --- Noise parameters (set from ROS params) ---
  double accel_noise_;
  double gyro_noise_;
  double accel_rw_;
  double gyro_rw_;
  double prior_pose_sigma_;
  double prior_vel_sigma_;
  double prior_bias_sigma_;
  double gps_sigma_floor_; // metres
  double gps_sigma_max_;   // metres — cap bad fixes

  // --- Datum ---
  NedConverter ned_;
  bool datum_initialised_;

  // --- State ---
  bool graph_initialised_;
  uint64_t key_;

  // --- Thread safety ---
  std::mutex graph_mutex_;
};