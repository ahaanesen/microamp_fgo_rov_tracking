#ifndef MICROAMP_FACTOR_GRAPH_TRACKING_NODE_HPP
#define MICROAMP_FACTOR_GRAPH_TRACKING_NODE_HPP

#include <gtsam/geometry/Pose3.h>
#include <gtsam/inference/Symbol.h>
#include <gtsam/navigation/CombinedImuFactor.h>
#include <gtsam/navigation/GPSFactor.h>
#include <gtsam/navigation/ImuBias.h>
#include <gtsam/nonlinear/ISAM2.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/Values.h>
#include <gtsam/slam/BetweenFactor.h>
#include <gtsam/slam/PriorFactor.h>

#include <memory>
#include <deque>

#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/imu.hpp"

// Custom message types (replace with your actual message packages)
#include "blueboat_interfaces/msg/gnss_nav_pvt.hpp"
#include "blueboat_interfaces/msg/usbl.hpp"
#include "blueboat_interfaces/msg/acoustic_comm_receive.hpp"
#include "blueboat_interfaces/msg/rov_state.hpp"
#include "nav_msgs/msg/odometry.hpp"

#include "ned_converter.hpp"
#include "config.hpp"

// For simulation purposes, we can define different scenarios with different measurement types. 
// This can be used to test the factor graph in different conditions and with different sensor availability.
enum scenarios {
  bearing_only = 1,
  bearing_range = 2,
  bearing_range_depth = 3
};

using gtsam::symbol_shorthand::B; // Bias  (b)
using gtsam::symbol_shorthand::V; // Velocity (v)
using gtsam::symbol_shorthand::X; // ASV Pose (x)

using gtsam::symbol_shorthand::R; // ROV Position (r)
using gtsam::symbol_shorthand::W; // ROV Velocity (w)

using namespace gtsam;

class FactorGraphTrackingNode : public rclcpp::Node {
public:
  using Imu = sensor_msgs::msg::Imu;
  using GNSSNavPvt = blueboat_interfaces::msg::GNSSNavPvt;
  using Odometry = nav_msgs::msg::Odometry;
  using USBLMessage = blueboat_interfaces::msg::USBL;
  using AcousticCommReceive = blueboat_interfaces::msg::AcousticCommReceive;
  using ROVState = blueboat_interfaces::msg::ROVState;

  FactorGraphTrackingNode();
  ~FactorGraphTrackingNode() override;

private:
  // ==================== INITIALIZATION & CONFIG ====================
  void loadConfigurations();
  uint8_t resolveScenarioId();
  void initializeDatumFromGNSS(const GNSSNavPvt &msg);
  void initializeGraphWithGNSS(const GNSSNavPvt::SharedPtr msg);

  // ==================== ASV STATE ====================
  void imuCallback(const Imu::SharedPtr msg);
  void gnssCallback(const GNSSNavPvt::SharedPtr msg);
  void publishBoatOdometry(const gtsam::Values &est);

  // ==================== ROV STATE ====================
  void usblCallback(const USBLMessage::SharedPtr msg);
  void acousticCommCallback(const AcousticCommReceive::SharedPtr msg);
  void initializeNewRov(gtsam::Key associated_asv_key, const USBLMessage::SharedPtr& usbl_msg);
  void publishROVState(const gtsam::Values &est);

  gtsam::Key getRovKey(unsigned char prefix, uint32_t rov_id, uint32_t step_count);
  gtsam::Key getAsvKeyForRovAssociation(rclcpp::Time current_asv_time, uint64_t current_asv_index, rclcpp::Time rov_time);
  bool rovUpdateWithUsbl(uint8_t rov_id, rclcpp::Time rov_time, gtsam::Key associated_asv_key, const USBLMessage::SharedPtr& usbl_msg);

  // ==================== GTSAM CORE ====================
  gtsam::Values updateAndGetEstimate();

  gtsam::ISAM2 isam2_;
  gtsam::NonlinearFactorGraph graph_;
  gtsam::Values values_;
  bool graph_initialised_;
  std::tuple<double, double> ne_init_; // Initial N and E for heading initialization TODO: add velocity as well
  
  std::map<rclcpp::Time, gtsam::Key> asv_timeline_; // Maps timestamp (seconds) to the GTSAM Key for the ASV
  uint64_t asv_index_; // Counter for the ASV symbol index
  rclcpp::Time last_asv_timestamp_; // The timestamp of the very last ASV node added to the graph

  std::map<uint8_t, bool> rov_initialised_; // Maps ROV ID to its initialization status
  std::map<uint8_t, uint32_t> rov_step_counters_; // Maps ROV ID to its current step counter
  std::map<uint8_t, rclcpp::Time> last_rov_timestamp_; // Maps ROV ID to the timestamp of its last update

  // ==================== ASV NAVIGATION ====================
  std::unique_ptr<gtsam::PreintegratedCombinedMeasurements> pim_;
  std::shared_ptr<gtsam::PreintegrationCombinedParams> pim_params_;
  gtsam::imuBias::ConstantBias bias_;
  double gravity_;

  NedConverter ned_;
  bool datum_initialised_;

  bool imu_initialised_ = false;
  rclcpp::Time last_imu_time_; 
  double last_gyro_z_ = 0.0; // Latest raw gyro z for yaw-rate output (bias correction applied later)
  
   // ==================== ROV TRACKING ====================
   std::deque<USBLMessage::SharedPtr> usbl_queue_; // Queue for incoming USBL messages (timestamp, message)


  // ==================== PARAMETERS ====================
  EnvConfig env_config_;
  FgoConfig fgo_config_;
  TopicsConfig topics_config_;
  uint8_t scenario_id_ = bearing_range_depth;

  // ==================== ROS SUBSCRIPTIONS ====================
  rclcpp::Subscription<Imu>::SharedPtr imu_sub_;
  rclcpp::Subscription<GNSSNavPvt>::SharedPtr gnss_sub_;
  rclcpp::Subscription<USBLMessage>::SharedPtr usbl_sub_;
  rclcpp::Subscription<AcousticCommReceive>::SharedPtr acoustic_comm_sub_;

  // ==================== ROS PUBLISHERS ====================
  rclcpp::Publisher<Odometry>::SharedPtr state_pub_;
  rclcpp::Publisher<ROVState>::SharedPtr rov_state_pub_;
};

#endif // MICROAMP_FACTOR_GRAPH_TRACKING_NODE_HPP
