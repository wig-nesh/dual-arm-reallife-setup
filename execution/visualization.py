import os

import numpy as np
import open3d as o3d

from .config import GRIPPER_MESH_PATH
from .transforms import ensure_right_handed_pose, get_T_model_grasp_robot_grasp


def load_gripper_mesh(color):
    if not os.path.exists(GRIPPER_MESH_PATH):
        print(f"No gripper mesh found at {GRIPPER_MESH_PATH}")
        return None

    mesh = o3d.io.read_triangle_mesh(GRIPPER_MESH_PATH)
    if mesh.is_empty():
        print(f"Gripper mesh at {GRIPPER_MESH_PATH} is empty")
        return None

    mesh.compute_vertex_normals()
    # Mesh/model assets are in meters. Execution visualization is in mm.
    mesh.scale(1000.0, center=(0, 0, 0))
    mesh.paint_uniform_color(color)
    return mesh


def make_gripper_mesh(T_board_robot_grasp, color):
    mesh = load_gripper_mesh(color)
    if mesh is None:
        return []

    # gripper.obj uses the model convention: local Y is approach and local X is
    # the gripper baseline. The execution grasp frame uses local Z as approach
    # and local Y as baseline.
    T_robot_grasp_model_mesh = np.linalg.inv(get_T_model_grasp_robot_grasp())
    mesh.transform(T_board_robot_grasp @ T_robot_grasp_model_mesh)
    return [mesh]


def visualize_scene(
    run_dir,
    T_robot_board_model_board,
    T_board_base7,
    T_board_base6,
    T_board_tip7,
    T_board_flange7,
    T_board_pre7,
    T_board_tip6,
    T_board_flange6,
    T_board_pre6,
):
    print("Visualizing scene in Open3D...")
    pcd_path = os.path.join(run_dir, "pcd_scaled.ply")
    if not os.path.exists(pcd_path):
        print(f"No pointcloud found at {pcd_path}")
        return

    pcd = o3d.io.read_point_cloud(pcd_path)
    # The saved pcd is scaled by 8 (1 unit = 0.125m). We want it in mm.
    # So 1 unit = 125mm.
    pcd.scale(125.0, center=(0, 0, 0))
    pcd.transform(T_robot_board_model_board)

    geometries = [pcd]

    board_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=50.0)
    geometries.append(board_frame)

    def make_frame(T, size, color=None):
        mesh = o3d.geometry.TriangleMesh.create_coordinate_frame(size=size)
        mesh.transform(T)
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=size / 4.0)
        sphere.transform(T)
        if color:
            sphere.paint_uniform_color(color)
        return [mesh, sphere]

    def make_line(T1, T2, color):
        points = [T1[:3, 3], T2[:3, 3]]
        lines = [[0, 1]]
        colors = [color]
        line_set = o3d.geometry.LineSet(
            points=o3d.utility.Vector3dVector(points),
            lines=o3d.utility.Vector2iVector(lines),
        )
        line_set.colors = o3d.utility.Vector3dVector(colors)
        return line_set

    # Base 7 (Blue)
    geometries.extend(make_frame(T_board_base7, size=100.0, color=[0, 0, 1]))
    # Base 6 (Red)
    geometries.extend(make_frame(T_board_base6, size=100.0, color=[1, 0, 0]))

    # Grasp 7 & Pre 7 (Blue)
    geometries.extend(make_gripper_mesh(T_board_tip7, [0.1, 0.25, 0.9]))
    geometries.extend(make_frame(T_board_flange7, size=30.0, color=[0, 0, 0.8]))
    geometries.extend(make_frame(T_board_pre7, size=30.0, color=[0.5, 0.5, 1]))
    geometries.append(make_line(T_board_pre7, T_board_tip7, [0, 0, 1]))

    # Grasp 6 & Pre 6 (Red)
    geometries.extend(make_gripper_mesh(T_board_tip6, [0.9, 0.15, 0.1]))
    geometries.extend(make_frame(T_board_flange6, size=30.0, color=[0.8, 0, 0]))
    geometries.extend(make_frame(T_board_pre6, size=30.0, color=[1, 0.5, 0.5]))
    geometries.append(make_line(T_board_pre6, T_board_tip6, [1, 0, 0]))

    # Camera Frame (if available)
    marker_path = os.path.join(run_dir, "marker_transform.npy")
    if os.path.exists(marker_path):
        # marker_transform maps camera-frame points into the saved board frame:
        # P_board = T_board_camera @ P_camera. That matrix is already the
        # camera pose in board coordinates, so do not invert it for drawing.
        T_board_camera = np.load(marker_path)
        T_board_camera_mm = T_board_camera.copy()
        T_board_camera_mm[:3, 3] *= 1000.0
        T_board_camera_mm = T_robot_board_model_board @ T_board_camera_mm
        T_board_camera_mm = ensure_right_handed_pose(T_board_camera_mm)

        geometries.extend(
            make_frame(T_board_camera_mm, size=80.0, color=[0, 1, 0])
        )  # Green for Camera
        print("Included Camera Frame (Green) in visualization.")
    else:
        print(
            "No camera transform (marker_transform.npy) found in this run. Run main_pipeline again to save it."
        )

    o3d.visualization.draw_geometries(
        geometries,
        window_name="Scene Visualization (Blue=xArm7, Red=xArm6, Green=Camera)",
    )
