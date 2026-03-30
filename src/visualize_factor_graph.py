#!/usr/bin/env python3
"""
Visualize the expanded factor graph structure for boat + ROV tracking
Requires: matplotlib, networkx
"""

import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib.patches as mpatches

def create_factor_graph_visualization():
    fig, ax = plt.subplots(1, 1, figsize=(14, 10))
    
    # Define positions for nodes
    timesteps = [0, 1, 2]
    
    # Boat state positions
    boat_pose_y = 5
    boat_vel_y = 3.5
    boat_bias_y = 2
    
    # ROV state positions  
    rov_pos_y = 0
    rov_vel_y = -1.5
    
    # Factor positions (midpoints)
    imu_factor_y = 4.0
    gps_factor_y = 5.5
    usbl_factor_y = 2.5
    range_factor_y = 1.8
    depth_factor_y = 0.5
    vel_factor_y = -1.5
    
    x_spacing = 4
    
    # Color scheme
    boat_color = '#3498db'  # Blue
    rov_color = '#e74c3c'   # Red
    factor_color = '#95a5a6' # Gray
    imu_color = '#2ecc71'   # Green
    gps_color = '#f39c12'   # Orange
    acoustic_color = '#9b59b6' # Purple
    
    # Draw variable nodes
    for k in timesteps:
        x = k * x_spacing
        
        # Boat states
        ax.add_patch(FancyBboxPatch((x-0.3, boat_pose_y-0.3), 0.6, 0.6,
                                    boxstyle="round,pad=0.1", 
                                    edgecolor=boat_color, facecolor=boat_color,
                                    linewidth=2, alpha=0.7))
        ax.text(x, boat_pose_y, f'X({k})', ha='center', va='center', 
                fontsize=11, fontweight='bold', color='white')
        
        ax.add_patch(FancyBboxPatch((x-0.3, boat_vel_y-0.3), 0.6, 0.6,
                                    boxstyle="round,pad=0.1",
                                    edgecolor=boat_color, facecolor=boat_color,
                                    linewidth=2, alpha=0.7))
        ax.text(x, boat_vel_y, f'V({k})', ha='center', va='center',
                fontsize=11, fontweight='bold', color='white')
        
        ax.add_patch(FancyBboxPatch((x-0.3, boat_bias_y-0.3), 0.6, 0.6,
                                    boxstyle="round,pad=0.1",
                                    edgecolor=boat_color, facecolor=boat_color,
                                    linewidth=2, alpha=0.7))
        ax.text(x, boat_bias_y, f'B({k})', ha='center', va='center',
                fontsize=11, fontweight='bold', color='white')
        
        # ROV states
        ax.add_patch(FancyBboxPatch((x-0.3, rov_pos_y-0.3), 0.6, 0.6,
                                    boxstyle="round,pad=0.1",
                                    edgecolor=rov_color, facecolor=rov_color,
                                    linewidth=2, alpha=0.7))
        ax.text(x, rov_pos_y, f'R({k})', ha='center', va='center',
                fontsize=11, fontweight='bold', color='white')
        
        ax.add_patch(FancyBboxPatch((x-0.3, rov_vel_y-0.3), 0.6, 0.6,
                                    boxstyle="round,pad=0.1",
                                    edgecolor=rov_color, facecolor=rov_color,
                                    linewidth=2, alpha=0.7))
        ax.text(x, rov_vel_y, f'W({k})', ha='center', va='center',
                fontsize=11, fontweight='bold', color='white')
    
    # Draw factors as diamonds
    def draw_factor(x, y, label, color, size=0.4):
        # Diamond shape
        diamond = mpatches.FancyBboxPatch((x-size/2, y-size/2), size, size,
                                         boxstyle="round,pad=0.05",
                                         transform=ax.transData,
                                         edgecolor=color, facecolor='white',
                                         linewidth=2)
        ax.add_patch(diamond)
        ax.text(x, y, label, ha='center', va='center',
                fontsize=8, fontweight='bold')
    
    # IMU factors (connect boat states between timesteps)
    for k in range(len(timesteps) - 1):
        x_mid = (k + 0.5) * x_spacing
        draw_factor(x_mid, imu_factor_y, 'IMU', imu_color)
        
        # Connect X(k), V(k), B(k) -> X(k+1), V(k+1), B(k+1)
        x_k = k * x_spacing
        x_k1 = (k + 1) * x_spacing
        
        for (y_from, y_to) in [(boat_pose_y, boat_pose_y),
                                (boat_vel_y, boat_vel_y),
                                (boat_bias_y, boat_bias_y)]:
            ax.plot([x_k, x_mid, x_k1], [y_from, imu_factor_y, y_to],
                   'k-', linewidth=1.5, alpha=0.4)
    
    # GPS factors
    for k in timesteps[1:]:  # Start from k=1
        x = k * x_spacing
        draw_factor(x, gps_factor_y, 'GPS', gps_color)
        ax.plot([x, x], [boat_pose_y + 0.3, gps_factor_y - 0.3],
               'k-', linewidth=1.5, alpha=0.4)
    
    # USBL + Range factors (connect boat pose to ROV position)
    for k in timesteps:
        x = k * x_spacing
        
        # USBL bearing
        draw_factor(x + 0.6, usbl_factor_y, 'USBL', acoustic_color, size=0.35)
        ax.plot([x, x + 0.6], [boat_pose_y - 0.3, usbl_factor_y + 0.2],
               'k-', linewidth=1.5, alpha=0.4)
        ax.plot([x + 0.6, x], [usbl_factor_y - 0.2, rov_pos_y + 0.3],
               'k-', linewidth=1.5, alpha=0.4)
        
        # Range
        draw_factor(x - 0.6, range_factor_y, 'RNG', acoustic_color, size=0.35)
        ax.plot([x, x - 0.6], [boat_pose_y - 0.3, range_factor_y + 0.2],
               'k-', linewidth=1.5, alpha=0.4)
        ax.plot([x - 0.6, x], [range_factor_y - 0.2, rov_pos_y + 0.3],
               'k-', linewidth=1.5, alpha=0.4)
        
        # Depth
        draw_factor(x, depth_factor_y, 'D', acoustic_color, size=0.3)
        ax.plot([x, x], [rov_pos_y - 0.3, depth_factor_y + 0.2],
               'k-', linewidth=1.5, alpha=0.4)
    
    # ROV velocity factors (constant velocity model)
    for k in range(len(timesteps) - 1):
        x_mid = (k + 0.5) * x_spacing
        draw_factor(x_mid, vel_factor_y, 'CV', rov_color, size=0.35)
        
        x_k = k * x_spacing
        x_k1 = (k + 1) * x_spacing
        ax.plot([x_k, x_mid, x_k1], [rov_vel_y, vel_factor_y, rov_vel_y],
               'k-', linewidth=1.5, alpha=0.4)
    
    # Add legend
    legend_elements = [
        mpatches.Patch(facecolor=boat_color, edgecolor=boat_color, 
                      label='Boat State Variables', alpha=0.7),
        mpatches.Patch(facecolor=rov_color, edgecolor=rov_color,
                      label='ROV State Variables', alpha=0.7),
        mpatches.Patch(facecolor='white', edgecolor=imu_color, linewidth=2,
                      label='IMU Factor'),
        mpatches.Patch(facecolor='white', edgecolor=gps_color, linewidth=2,
                      label='GPS Factor'),
        mpatches.Patch(facecolor='white', edgecolor=acoustic_color, linewidth=2,
                      label='Acoustic Factors (USBL/Range/Depth)'),
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=10)
    
    # Add annotations
    ax.text(-1.5, boat_pose_y, 'Boat\nPose', ha='right', va='center',
           fontsize=10, style='italic')
    ax.text(-1.5, boat_vel_y, 'Boat\nVelocity', ha='right', va='center',
           fontsize=10, style='italic')
    ax.text(-1.5, boat_bias_y, 'IMU\nBias', ha='right', va='center',
           fontsize=10, style='italic')
    ax.text(-1.5, rov_pos_y, 'ROV\nPosition', ha='right', va='center',
           fontsize=10, style='italic', color=rov_color)
    ax.text(-1.5, rov_vel_y, 'ROV\nVelocity', ha='right', va='center',
           fontsize=10, style='italic', color=rov_color)
    
    # Add timestep labels
    for k in timesteps:
        x = k * x_spacing
        ax.text(x, -3, f'Timestep {k}', ha='center', va='center',
               fontsize=12, fontweight='bold')
    
    # Formatting
    ax.set_xlim(-2.5, (len(timesteps)-1) * x_spacing + 1.5)
    ax.set_ylim(-3.5, 6.5)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title('Expanded Factor Graph: Boat + ROV State Estimation',
                fontsize=16, fontweight='bold', pad=20)
    
    plt.tight_layout()
    return fig

def create_sensor_geometry_diagram():
    """Visualize USBL geometry and measurement model"""
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    
    # Boat position (origin)
    boat_x, boat_y = 0, 0
    ax.plot(boat_x, boat_y, 'bs', markersize=20, label='Boat')
    
    # USBL offset (below boat)
    usbl_x, usbl_y = 0, -1.5
    ax.plot([boat_x, usbl_x], [boat_y, usbl_y], 'b--', linewidth=2)
    ax.plot(usbl_x, usbl_y, 'bo', markersize=15, label='USBL Transducer')
    
    # ROV position (example)
    rov_x, rov_y = 40, -25
    ax.plot(rov_x, rov_y, 'r^', markersize=20, label='ROV')
    
    # Range line
    ax.plot([usbl_x, rov_x], [usbl_y, rov_y], 'purple', linewidth=2,
           label='Acoustic Range')
    
    # Azimuth (horizontal angle)
    # Draw horizontal reference (North)
    ax.arrow(usbl_x, usbl_y, 0, 15, head_width=1, head_length=1,
            fc='green', ec='green', linewidth=2, alpha=0.5)
    ax.text(usbl_x + 2, usbl_y + 15, 'North (0°)', fontsize=11, color='green')
    
    # Azimuth arc
    from matplotlib.patches import Arc
    azimuth_angle = np.degrees(np.arctan2(rov_x - usbl_x, 0))  # NED convention
    arc = Arc((usbl_x, usbl_y), 20, 20, angle=0, theta1=90-azimuth_angle, theta2=90,
             color='orange', linewidth=2, linestyle='--')
    ax.add_patch(arc)
    ax.text(10, usbl_y + 3, f'Azimuth\n({azimuth_angle:.1f}°)',
           fontsize=10, color='orange', fontweight='bold')
    
    # Elevation (vertical angle)
    # Horizontal reference for elevation
    horiz_len = np.sqrt((rov_x - usbl_x)**2)
    ax.plot([usbl_x, usbl_x + horiz_len], [usbl_y, usbl_y], 'g--',
           linewidth=1.5, alpha=0.5)
    
    # Elevation arc
    elevation_angle = np.degrees(np.arctan2(-(rov_y - usbl_y), horiz_len))
    mid_x = usbl_x + horiz_len / 3
    ax.annotate('', xy=(rov_x, rov_y), xytext=(mid_x, usbl_y),
               arrowprops=dict(arrowstyle='->', color='purple', lw=2,
                             linestyle='--'))
    ax.text(mid_x - 5, usbl_y - 5, f'Elevation\n({-elevation_angle:.1f}°)',
           fontsize=10, color='purple', fontweight='bold')
    
    # Depth measurement (vertical line)
    ax.plot([rov_x, rov_x], [0, rov_y], 'r--', linewidth=2, alpha=0.7)
    ax.text(rov_x + 2, rov_y / 2, f'Depth\n({-rov_y:.1f}m)',
           fontsize=10, color='red', fontweight='bold')
    
    # Water surface
    ax.axhline(y=0, color='cyan', linewidth=3, alpha=0.3, label='Water Surface')
    
    # NED frame axes
    ax.arrow(-10, -5, 8, 0, head_width=1, head_length=1, fc='black', ec='black')
    ax.text(-2, -6.5, 'East', fontsize=9)
    ax.arrow(-10, -5, 0, 8, head_width=1, head_length=1, fc='black', ec='black')
    ax.text(-12, 3, 'North', fontsize=9)
    
    ax.set_xlim(-15, 50)
    ax.set_ylim(-30, 20)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right', fontsize=11)
    ax.set_xlabel('East (m)', fontsize=12)
    ax.set_ylabel('North (m) / Down (m)', fontsize=12)
    ax.set_title('USBL Measurement Geometry (NED Frame, Side View)',
                fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    return fig

if __name__ == '__main__':
    import numpy as np
    
    # Create factor graph structure visualization
    fig1 = create_factor_graph_visualization()
    fig1.savefig('factor_graph_structure.png', dpi=300, bbox_inches='tight')
    print("Saved: factor_graph_structure.png")
    
    # Create sensor geometry diagram
    fig2 = create_sensor_geometry_diagram()
    fig2.savefig('usbl_geometry.png', dpi=300, bbox_inches='tight')
    print("Saved: usbl_geometry.png")
    
    plt.show()
