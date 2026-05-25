import os

import numpy as np
import open3d as o3d


def visualize_grasps(
    pcd_xyz, pcd_rgb, grasp_pairs_np, pair_scores_np, gripper_mesh_path
):
    if grasp_pairs_np is None or grasp_pairs_np.shape[0] == 0:
        print("No grasp pairs to visualize.")
        return

    # Sort pairs by score (highest first)
    if pair_scores_np is not None:
        sort_idx = np.argsort(pair_scores_np)[::-1]
        grasp_pairs_np = grasp_pairs_np[sort_idx]
        pair_scores_np = pair_scores_np[sort_idx]

    # Load base meshes
    base_gripper = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=0.1
    )  # Fallback
    if os.path.exists(gripper_mesh_path):
        base_gripper = o3d.io.read_triangle_mesh(gripper_mesh_path)
        base_gripper.compute_vertex_normals()
        base_gripper.scale(8.0, center=(0, 0, 0))
    else:
        print(
            f"Warning: Gripper mesh not found at {gripper_mesh_path}. Using coordinate frame fallback."
        )

    base_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)

    g1 = o3d.geometry.TriangleMesh(base_gripper)
    g1.paint_uniform_color([0.8, 0.2, 0.2])
    g2 = o3d.geometry.TriangleMesh(base_gripper)
    g2.paint_uniform_color([0.2, 0.2, 0.8])

    f1 = o3d.geometry.TriangleMesh(base_frame)
    f2 = o3d.geometry.TriangleMesh(base_frame)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pcd_xyz)
    if pcd_rgb is not None:
        pcd.colors = o3d.utility.Vector3dVector(pcd_rgb)
    else:
        pcd.paint_uniform_color([0.5, 0.5, 0.5])

    world_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2)

    vis_state = {"idx": 0}

    def update_meshes(vis):
        idx = vis_state["idx"]
        pair = grasp_pairs_np[idx]
        score_str = (
            f" | Score: {pair_scores_np[idx]:.4f}" if pair_scores_np is not None else ""
        )
        print(f"Showing Grasp Pair {idx + 1}/{len(grasp_pairs_np)}{score_str}")

        g1.vertices = base_gripper.vertices
        g1.transform(pair[0])
        g1.compute_vertex_normals()

        g2.vertices = base_gripper.vertices
        g2.transform(pair[1])
        g2.compute_vertex_normals()

        f1.vertices = base_frame.vertices
        f1.transform(pair[0])
        f1.compute_vertex_normals()

        f2.vertices = base_frame.vertices
        f2.transform(pair[1])
        f2.compute_vertex_normals()

        vis.update_geometry(g1)
        vis.update_geometry(g2)
        vis.update_geometry(f1)
        vis.update_geometry(f2)

    def next_grasp(vis):
        vis_state["idx"] = (vis_state["idx"] + 1) % len(grasp_pairs_np)
        update_meshes(vis)
        return False

    def prev_grasp(vis):
        vis_state["idx"] = (vis_state["idx"] - 1) % len(grasp_pairs_np)
        update_meshes(vis)
        return False

    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(window_name="Dual Grasp Viewer (Right/Left Arrow to cycle)")

    vis.add_geometry(pcd)
    vis.add_geometry(world_frame)
    vis.add_geometry(g1)
    vis.add_geometry(g2)
    vis.add_geometry(f1)
    vis.add_geometry(f2)

    # GLFW Key codes: 262 is Right Arrow, 263 is Left Arrow
    vis.register_key_callback(262, next_grasp)
    vis.register_key_callback(263, prev_grasp)

    print("\nOpening Open3D Visualization...")
    print("Red Gripper: Arm 1 | Blue Gripper: Arm 2")
    print("Use RIGHT and LEFT arrow keys to cycle through grasp pairs.")

    update_meshes(vis)
    vis.run()
    vis.destroy_window()


def visualize_single_grasps(
    pcd_xyz, pcd_rgb, single_grasps_np, grasp_scores_np, gripper_mesh_path
):
    """Visualize single grasps when no valid pairs are available."""
    if single_grasps_np is None or len(single_grasps_np) == 0:
        print("No single grasps to visualize.")
        return

    # Sort by score
    if grasp_scores_np is not None:
        sort_idx = np.argsort(grasp_scores_np)[::-1]
        single_grasps_np = single_grasps_np[sort_idx]
        grasp_scores_np = grasp_scores_np[sort_idx]

    # Load gripper mesh
    base_gripper = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
    if os.path.exists(gripper_mesh_path):
        base_gripper = o3d.io.read_triangle_mesh(gripper_mesh_path)
        base_gripper.compute_vertex_normals()
        base_gripper.scale(8.0, center=(0, 0, 0))
    else:
        print(f"Warning: Gripper mesh not found, using coordinate frame fallback.")

    base_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)

    gripper_mesh = o3d.geometry.TriangleMesh(base_gripper)
    gripper_mesh.paint_uniform_color([0.8, 0.2, 0.2])
    frame_mesh = o3d.geometry.TriangleMesh(base_frame)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pcd_xyz)
    if pcd_rgb is not None:
        pcd.colors = o3d.utility.Vector3dVector(pcd_rgb)
    else:
        pcd.paint_uniform_color([0.5, 0.5, 0.5])

    world_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2)

    vis_state = {"idx": 0}

    def update_mesh(vis):
        idx = vis_state["idx"]
        grasp = single_grasps_np[idx]
        score_str = (
            f" | Score: {grasp_scores_np[idx]:.4f}"
            if grasp_scores_np is not None
            else ""
        )
        print(f"Showing Single Grasp {idx + 1}/{len(single_grasps_np)}{score_str}")

        gripper_mesh.vertices = base_gripper.vertices
        gripper_mesh.transform(grasp)
        gripper_mesh.compute_vertex_normals()

        frame_mesh.vertices = base_frame.vertices
        frame_mesh.transform(grasp)
        frame_mesh.compute_vertex_normals()

        vis.update_geometry(gripper_mesh)
        vis.update_geometry(frame_mesh)

    def next_grasp(vis):
        vis_state["idx"] = (vis_state["idx"] + 1) % len(single_grasps_np)
        update_mesh(vis)
        return False

    def prev_grasp(vis):
        vis_state["idx"] = (vis_state["idx"] - 1) % len(single_grasps_np)
        update_mesh(vis)
        return False

    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(window_name="Single Grasp Viewer (Right/Left Arrow to cycle)")

    vis.add_geometry(pcd)
    vis.add_geometry(world_frame)
    vis.add_geometry(gripper_mesh)
    vis.add_geometry(frame_mesh)

    vis.register_key_callback(262, next_grasp)
    vis.register_key_callback(263, prev_grasp)

    print("\nOpening Open3D Visualization...")
    print("Red: Single Grasp")
    print("Use RIGHT and LEFT arrow keys to cycle through grasps.")

    update_mesh(vis)
    vis.run()
    vis.destroy_window()

