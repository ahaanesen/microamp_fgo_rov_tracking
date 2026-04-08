#ifndef MICROAMP_FACTOR_GRAPH_TRACKING_NODE_HPP
#define MICROAMP_FACTOR_GRAPH_TRACKING_NODE_HPP

#include <gtsam/geometry/Pose3.h>
#include <gtsam/inference/Symbol.h>
#include <gtsam/navigation/CombinedImuFactor.h>
#include <gtsam/navigation/PreintegratedCombinedMeasurements.h>
#include <gtsam/navigation/GPSFactor.h>
#include <gtsam/navigation/ImuBias.h>
#include <gtsam/nonlinear/ISAM2.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/Values.h>
#include <gtsam/slam/BetweenFactor.h>
#include <gtsam/slam/PriorFactor.h>

#include <memory>
#include <mutex>
#include <deque>

#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/imu.hpp"

// Custom message types (replace with your actual message packages)
#include "blueboat_interfaces/msg/gnss_nav_pvt.hpp"
#include "blueboat_interfaces/msg/boat_state.hpp"
#include "blueboat_interfaces/msg/usbl.hpp"
#include "blueboat_interfaces/msg/acoustic_comm_receive.hpp"
#include "blueboat_interfaces/msg/rov_state.hpp"

// NED conversion utility (your existing utility)
#include "ned_converter.hpp"

using gtsam::symbol_shorthand::B; // Bias  (b)
using gtsam::symbol_shorthand::V; // Velocity (v)
using gtsam::symbol_shorthand::X; // ASV Pose (x)

using gtsam::symbol_shorthand::R; // ROV Position (r)
using gtsam::symbol_shorthand::W; // ROV Velocity (w)

class FactorGraphTrackingNode : public rclcpp::Node {
public:
  using Imu = sensor_msgs::msg::Imu;
  using GNSSNavPvt = blueboat_interfaces::msg::GNSSNavPvt;
  using BoatState = blueboat_interfaces::msg::BoatState;
  using USBLMessage = blueboat_interfaces::msg::USBL;
  using AcousticCommReceive = blueboat_interfaces::msg::AcousticCommReceive;
  using ROVState = blueboat_interfaces::msg::ROVState;

  FactorGraphTrackingNode();

private:
  // ==================== ASV STATE ====================
  void imuCallback(const Imu::SharedPtr msg);
  void gnssCallback(const GNSSNavPvt::SharedPtr msg);
  void initializeDatumFromGNSS(const GNSSNavPvt &msg);
  void initializeGraph();
  void publishBoatState(const gtsam::Values &est);

  // ==================== ROV STATE ====================
  void usblCallback(const USBLMessage::SharedPtr msg);
  void acousticCommCallback(const AcousticCommReceive::SharedPtr msg);
  void initializeNewRov(uint8_t rov_id, Key rKey, Key wKey, const USBLMessage::SharedPtr& usbl, double speed_of_sound);
  void publishROVState(const gtsam::Values &est);

  // ==================== HELPERS ====================
  gtsam::PreintegratedCombinedMeasurements getPimFromBuffer(double t_start, double t_end);
  gtsam::Key getAsvKeyAtTime(double target_time);
  gtsam::Key getRovKey(unsigned char prefix, uint32_t rov_id, uint32_t time_step);


  // ==================== SHARED ====================
  gtsam::Values updateAndGetEstimate();

  // ==================== GTSAM CORE ====================
  gtsam::ISAM2 isam2_;
  gtsam::NonlinearFactorGraph graph_;
  gtsam::Values values_;
  std::mutex graph_mutex_;
  
  std::map<double, gtsam::Key> asv_timeline_; // Maps timestamp (seconds) to the GTSAM Key for the ASV
  uint64_t asv_index_ = 0; // Counter for the ASV symbol index
  double last_asv_timestamp_ = -1.0; // The timestamp of the very last ASV node added to the graph


  // ==================== ASV NAVIGATION ====================
  std::unique_ptr<gtsam::PreintegratedCombinedMeasurements> pim_;
  gtsam::imuBias::ConstantBias bias_;
  double gravity_;

  NEDConverter ned_;
  bool datum_initialised_;
  bool graph_initialised_;
  bool imu_initialised_ = false;

  rclcpp::Time last_imu_time_; 
  double last_gyro_z_ = 0.0; // Latest raw gyro z for yaw-rate output (bias correction applied later)

  // ==================== IMU buffer ====================
  struct ImuMeasurement {
    double timestamp;
    gtsam::Vector3 acc;
    gtsam::Vector3 gyro;
  };

   // IMU buffer for out of order handling for ROV measurements
  std::deque<ImuMeasurement> imu_buffer_; // Use a deque for efficient pushing to back and popping from front
  double max_imu_buffer_duration_ = 10.0; // seconds, adjust as needed

  // ==================== ROV STATES ====================
  std::map<uint8_t, bool> rov_initialised_; // Maps ROV ID to its initialization status
  std::map<uint8_t, uint32_t> rov_step_counters_; // Maps ROV ID to its current step counter
  std::map<uint8_t, double> last_rov_timestamp_; // Maps ROV ID to the timestamp of its last update

  // ==================== PARAMETERS ====================
  // ASV IMU/GNSS
  double accel_noise_, gyro_noise_;
  double accel_rw_, gyro_rw_;
  double prior_pose_sigma_, prior_vel_sigma_, prior_bias_sigma_;
  double gps_sigma_floor_, gps_sigma_max_;

  // ROV
  double rov_prior_pos_sigma_;
  double rov_prior_vel_sigma_;
  double rov_process_vel_sigma_; // Constant velocity model noise
  double usbl_azimuth_sigma_;
  double usbl_elevation_sigma_;
  double acoustic_range_sigma_;
  double rov_depth_sigma_;

  // USBL mounting offset (relative to ASV IMU frame)
  gtsam::Point3 usbl_offset_; // (x, y, z) in ASV body frame
  gtsam::Rot3 usbl_rotation_; // Rotation from ASV body frame to USBL frame (if needed)

  // ==================== ROS SUBSCRIPTIONS ====================
  rclcpp::Subscription<Imu>::SharedPtr imu_sub_;
  rclcpp::Subscription<GNSSNavPvt>::SharedPtr gnss_sub_;
  rclcpp::Subscription<USBLMessage>::SharedPtr usbl_sub_;
  rclcpp::Subscription<AcousticCommReceive>::SharedPtr acoustic_comm_sub_;

  // ==================== ROS PUBLISHERS ====================
  rclcpp::Publisher<BoatState>::SharedPtr state_pub_;
  rclcpp::Publisher<ROVState>::SharedPtr rov_state_pub_;
};

#endif // MICROAMP_FACTOR_GRAPH_TRACKING_NODE_HPP
