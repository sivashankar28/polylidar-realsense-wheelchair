import sys
import open3d as o3d
import matplotlib.pyplot as plt
import numpy as np
from surfacedetector.utility.line_mesh import LineMesh
from scripts.o3d_util import create_grid, create_box, tranform_vectors, rotate_vectors, create_transform, create_point
from surfacedetector.utility.helper_linefitting import compute_turning_manuever

np.set_printoptions(precision=3, suppress=True)

# Flip Y to Z, Z to -Y
MOUNT_TO_SENSOR_ROT = np.linalg.inv(np.array([
    [1, 0, 0],
    [0, 0, -1],
    [0, 1, 0]
]))

# def get_turning_manuever(platform_normal_sensor_frame, platform_center_sensor_frame, sensor_to_wheel_chair_transform, **kwargs):
#     platform_center_pos_wheel_chair_frame = tranform_vectors(platform_center_sensor_frame, sensor_to_wheel_chair_transform)
#     platform_normal_wheel_chair_frame = rotate_vectors(platform_normal_sensor_frame, sensor_to_wheel_chair_transform)

#     result = compute_turning_manuever(platform_center_pos_wheel_chair_frame, platform_normal_wheel_chair_frame,  **kwargs)

#     return result

# def compute_2D_angle_difference(vector1, vector2):
#     """Computes the 2D angle difference between vector 1 and vector 2
#     The sign of the angle is the CC rotation to move vector1 TO vector2
#     Args:
#         vector1 (np.ndarray): vector 1
#         vector2 (np.ndarray): vector 2

#     Returns:
#         float: Angle difference in degrees
#     """
#     angle = np.degrees(np.arctan2(vector2[1], vector2[0]) - np.arctan2(vector1[1], vector1[0]))

#     if angle > 180:
#         angle = angle - 360
#     elif angle < -180:
#         angle = angle + 360

#     return angle

# def compute_turning_manuever(platform_center_pos_wheel_chair_frame, platform_normal_wheel_chair_frame, poi_offset=0.5, debug=False, **kwargs):
#     """Will compute the turning manuever for the wheel chair. CC = Counter Clockwise
#     Assumes that the Wheel Chair reference frame origin is the center of rotation for wheel chair turn commands
#     Maneuvers Steps:
#       0. Calculate point of interest (POI) as `platform_normal_wheel_chair_frame * poi_offset + platform_center_pos_wheel_chair_frame`
#       1. Execute rotation of "first_turn" degrees CC. This aligns the wheel chair y-axis to point to the POI
#       2. Execute forward move (y-axis) "dist" meters
#       3. Execute rotation of "second_turn" degrees CC. This aligns the wheel chair y-axis to platform

#     This assumes perfect control and execution of these commands. In reality you will need feeback control probably for 
#     steps 2 and 3. Step 1 is 'probably' not as important to need feedback control.

#     Args:
#         platform_center_pos_wheel_chair_frame (np.ndarray): The platform center point in the WHEEL CHAIR FRAME
#         platform_normal_wheel_chair_frame (np.ndarray): The platform normal in the WHEEL CHAIR FRAME. Normal is pointing TOWARDS the wheel chair.
#         debug (bool, optional): Whether to print out data. Defaults to False.

#     Returns:
#         dict: A dictionary of the maneuver
#             alpha = angle between wheel chair y-axis (forward) and vector from wheel chair to point of interest
#             beta = angle between wheel chair y-axis (forward) and reversed platform normal
#             dist_poi = 2D distance between wheel chair center and point of interest
#             ortho_dist_platform = 2D orthogonal distance between wheel chair center and center of platform
#             first_turn = angle to turn wheel chair to align with vector directing wheel chair to point of interst (alpha)
#             second_turn = angle to turn wheel chair to align with platform normal (-alpha + beta)
#             vec_wheel_chair_to_poi_2D_unit = 2D unit vector from wheel chair to poi (wheel chair frame)
#             platform_normal_inverted_unit = 3D unit vector of the inverse of the platform normal 
#     """
#     platform_poi_pos_wheel_chair_frame = platform_normal_wheel_chair_frame * poi_offset + platform_center_pos_wheel_chair_frame
#     wheel_chair_pos_in_wheel_chair_frame = np.array([0.0,0.0,0.0]) # The position will be 0 in wheel chair frame (origin)
#     wheel_chair_dir_vec_unit = np.array([0.0, 1.0, 0.0]) # forward y-axis is wheel chair direction

#     # Orthogonal distance to the platform
#     ortho_dist_platform = np.dot(platform_center_pos_wheel_chair_frame, platform_normal_wheel_chair_frame)
#     ortho_dist_platform = np.abs(ortho_dist_platform)

#     vec_wheelchair_to_poi = platform_poi_pos_wheel_chair_frame - wheel_chair_pos_in_wheel_chair_frame # called Vec3 (blue) in diagrams
#     vec_wheel_chair_to_poi_2D = vec_wheelchair_to_poi[:2] # the z-axis is height in this reference frame
#     dist_poi = np.linalg.norm(vec_wheel_chair_to_poi_2D)
#     vec_wheel_chair_to_poi_2D_unit = vec_wheel_chair_to_poi_2D / dist_poi

#     platform_normal_inverted_unit = -platform_normal_wheel_chair_frame # called Vec2 (red) in diagrams

#     alpha = compute_2D_angle_difference(wheel_chair_dir_vec_unit, vec_wheel_chair_to_poi_2D_unit)
#     beta = compute_2D_angle_difference(wheel_chair_dir_vec_unit,platform_normal_inverted_unit )


#     first_turn = alpha
#     second_turn = -alpha + beta

#     result = dict(alpha=alpha, beta=beta, dist_poi=dist_poi, ortho_dist_platform=ortho_dist_platform,first_turn=first_turn, second_turn=second_turn, 
#                 vec_wheel_chair_to_poi_2D_unit=vec_wheel_chair_to_poi_2D_unit, platform_normal_inverted_unit=platform_normal_inverted_unit,
#                 platform_center_pos_wheel_chair_frame=platform_center_pos_wheel_chair_frame, platform_normal_wheel_chair_frame=platform_normal_wheel_chair_frame,
#                 platform_poi_pos_wheel_chair_frame=platform_poi_pos_wheel_chair_frame)
#     if debug:
#         print(f"Alpha Angle {alpha:.1f}; Beta Angle: {beta:.1f}")
#         print(f"First Turn CC: {first_turn:.1f} degres; Move Distance: {dist_poi:.2f};  Second Turn: {second_turn:.1f}")
#         plot_maneuver(result)

#     return result

# def plot_maneuver(result):
#     platform_poi_pos_wheel_chair = result['platform_poi_pos_wheel_chair_frame']
#     platform_center = result['platform_center_pos_wheel_chair_frame']

#     fig, ax = plt.subplots(1, 1)
#     ax.scatter(platform_center[0], platform_center[1], c=[[0, 1, 0]])
#     ax.text(platform_center[0] + 0.03, platform_center[1], 'platform')
#     ax.scatter(platform_poi_pos_wheel_chair[0], platform_poi_pos_wheel_chair[1], c=[[0, 1, 0]])
#     ax.text(platform_poi_pos_wheel_chair[0] + 0.03, platform_poi_pos_wheel_chair[1], 'poi')
#     ax.scatter(0, 0, c='k')
#     ax.text(0.01, 0, 'Wheel Chair')
#     arrow_(ax, 0.0, 0.0, 0, 1, ec='g', fc='g', width=.01)
#     arrow_(ax, 0.0, 0.0, result['vec_wheel_chair_to_poi_2D_unit'][0], result['vec_wheel_chair_to_poi_2D_unit'][1], ec='b', fc='b', width=.01)
#     arrow_(ax,0.0, 0.0, result['platform_normal_inverted_unit'][0], result['platform_normal_inverted_unit'][1], ec='r', fc='r', width=.01)
#     ax.text(result['vec_wheel_chair_to_poi_2D_unit'][0] / 2.0, (result['vec_wheel_chair_to_poi_2D_unit'][1] + 1) / 2.0, rf'$\alpha={result["alpha"]:.0f}^\circ$')
#     ax.text((0.0 + result['platform_normal_inverted_unit'][0]) / 2.0, 
#         (1.0 + result['platform_normal_inverted_unit'][1]) / 2.0, rf'$\beta={result["beta"]:.0f}^\circ$')
#     ax.axis('equal')


def main():
    # Contants you can change
    POI_OFFSET = 0.5
    PLATFORM_WIDTH = 1.16
    PLATFORM_HEIGHT = 0.25
    WHEEL_CHAIR_POS = np.array([1.0, -0.0, 0])
    WHEEL_CHAIR_ROT = dict(roll=0, pitch=0, yaw=45)
    SENSOR_MOUNT_POS = np.array([0.34, 0.34, 0.7])
    SENSOR_MOUNT_ROT = dict(roll=0, pitch=-25, yaw=0)
    SENSOR_POS = np.array([0.0, 0.0, 0.0]) #np.array([0, 0.025, 0.025])

    # Creates origin coordinate frame and grid
    global_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2)
    grid = create_grid(size=5, n=20) # each square is 25X25 CM, size is 5X5

    # Defnitions of the Platform Box, Platform Center Point, and Platform Point of Interest (POI) 
    platform = dict(parent=None, center_point=np.array([0, 2, 0]), depth=PLATFORM_HEIGHT, width=PLATFORM_WIDTH, height=1.0, rotation=dict(roll=0, pitch=0, yaw=0))
    platform_cp = dict(parent=platform, center_point=np.array([0, -0.5, 0.125]), size=0.02, rotation=dict(roll=0, pitch=0, yaw=0), color=[1,0,0])
    platform_poi = dict(parent=platform, center_point=np.array([0, -0.5 - POI_OFFSET, 0.125]), size=0.02, rotation=dict(roll=0, pitch=0, yaw=0), color=[0,1,0])

    # Wheel Chair Box, Sensor Mount, and Sensor
    wheel_chair = dict(parent=None, center_point=WHEEL_CHAIR_POS, depth=0.7, width=0.68, height=0.68, rotation=WHEEL_CHAIR_ROT)
                                            # offset from wheel chair center          rotation from wheel chair frame
    sensor_mount = dict(parent=wheel_chair, center_point=SENSOR_MOUNT_POS, rotation=SENSOR_MOUNT_ROT, width=0.1, depth=0.05, height=0.03)
    sensor = dict(parent=sensor_mount, center_point=SENSOR_POS, rotation=dict(roll=0, pitch=0, yaw=0), post_rot=MOUNT_TO_SENSOR_ROT, width=0.01, depth=0.025, height=0.01)

    # Create Open 3D Geometries, these geometries are defined in the WORLD FRAME, these geometries can then be visualized below
    # You don't need to understand these functions, they just returns a geometry to render
    platform_geom = create_box(platform)
    platform_cp_geom = create_point(platform_cp)
    platform_poi_geom = create_point(platform_poi)
    wheel_chair_geom = create_box(wheel_chair)
    sensor_mount_geom = create_box(sensor_mount)
    sensor_geom = create_box(sensor)

    # Get sensor, poi, and wheelchair 3D position and 2D projections in WORLD frame
    sensor_pos_world = sensor['geom'].get_center()
    platform_cp_pos_world = platform_cp['geom'].get_center()
    platform_poi_pos_world = platform_poi['geom'].get_center()
    wheel_chair_pos_world = wheel_chair['geom'].get_center()
    sensor_pos_world_proj = np.copy(sensor_pos_world)
    platform_poi_pos_world_proj = np.copy(platform_poi_pos_world)
    wheel_chair_pos_world_proj = np.copy(wheel_chair_pos_world)
    sensor_pos_world_proj[2] = 0
    platform_poi_pos_world_proj[2] = 0
    wheel_chair_pos_world_proj[2] = 0

    # Create lines between wheel chair center and poi (Blue) and INCORRECT LINE between sensor frame and poi (RED)
    vec3_geom_1 = LineMesh([sensor_pos_world, platform_poi_pos_world, sensor_pos_world_proj, platform_poi_pos_world_proj ], lines=[[0, 1], [2,3]], radius=0.005, colors=[1, 0, 0])
    vec3_geom_2 = LineMesh([wheel_chair_pos_world_proj, platform_poi_pos_world_proj], lines=[[0, 1]], radius=0.005, colors=[0, 0, 1])

    # Simulate POI and wall_normal in SENSOR frame, this is what is generated by RealSense Frame
    transform = np.linalg.inv(sensor['transform'])
    platform_cp_pos_sensor = tranform_vectors(platform_cp_pos_world, transform )
    platform_normal_sensor = rotate_vectors([0, -1, 0], transform )
    platform_poi_pos_sensor = tranform_vectors(platform_poi_pos_world, transform )


    # Transform POI3D and Wall Normal into WHEELCHAIR frame, all you need are the constants SENSOR_MOUNT_POS, SENSOR_MOUNT_ROT, SENSOR_POS, MOUNT_TO_SENSOR_ROT
    SENSOR_TO_WHEEL_CHAIR = create_transform(SENSOR_MOUNT_POS, SENSOR_MOUNT_ROT) @ create_transform(SENSOR_POS, MOUNT_TO_SENSOR_ROT)
    platform_cp_pos_wheel_chair = tranform_vectors(platform_cp_pos_sensor, SENSOR_TO_WHEEL_CHAIR)
    platform_normal_wheel_chair = rotate_vectors(platform_normal_sensor, SENSOR_TO_WHEEL_CHAIR)
    # platform_poi_pos_wheel_chair = tranform_vectors(platform_poi_pos_sensor, SENSOR_TO_WHEEL_CHAIR)
    platform_poi_pos_wheel_chair = platform_normal_wheel_chair * POI_OFFSET + platform_cp_pos_wheel_chair # this line will result in the same calculation as above, VERIFIED

    print(f"Platform POI in World Frame: {platform_poi_pos_world}")
    print(f"Platform POI in Sensor Frame: {platform_poi_pos_sensor}")
    print(f"Platform POI in Wheel Chair Frame: {platform_poi_pos_wheel_chair}")
    print(f"Platform Normal in Wheel Chair Frame: {platform_normal_wheel_chair}")

    # print(f"Platform Center in Sensor Frame: {platform_cp_pos_sensor}")
    # print(f"Platform Normal in Sensor Frame: {platform_normal_sensor}")


    print("\nCalculated Angles....Creating Turn Procedure")
    result = compute_turning_manuever(platform_cp_pos_wheel_chair, platform_normal_wheel_chair, poi_offset=POI_OFFSET, debug=True)

    plt.draw()
    plt.pause(0.01)

    o3d.visualization.draw_geometries([global_frame, grid, *wheel_chair_geom, *platform_geom, *platform_cp_geom, 
                                        *platform_poi_geom,*sensor_mount_geom, sensor_geom[1],
                                        *vec3_geom_1.cylinder_segments, *vec3_geom_2.cylinder_segments])



def arrow_(ax, x, y, dx, dy, **kwargs):
    ax.arrow(x, y, dx, dy, **kwargs)
    ax.scatter(x + dx + 0.1, y + dy + 0.1, alpha=0.0)

if __name__ == "__main__":
    main()



