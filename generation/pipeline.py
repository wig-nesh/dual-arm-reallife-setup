import argparse
import datetime
import json
import os

import cv2
import numpy as np
import open3d as o3d

from client.src.model import ModelClient
from client.src.sam import SAMClient
from realsense.src.realsense_camera import RealSenseCamera

from .aruco import MARKER_LENGTH, MARKER_WORLD
from .pointcloud import depth_to_pcd
from .selection import mouse_callback, state
from .visualization import visualize_grasps, visualize_single_grasps


def main():
    parser = argparse.ArgumentParser(
        description="End-to-End Pipeline (No Robot Execution)"
    )
    parser.add_argument(
        "--sam-url",
        type=str,
        default="http://dualarm@orion.rrcx.tk:8000",
        help="SAM Server URL",
    )
    parser.add_argument(
        "--model-url",
        type=str,
        default="http://localhost:8000",
        help="Grasp Model Server URL",
    )
    parser.add_argument(
        "--gripper",
        type=str,
        default="client/gripper.obj",
        help="Path to gripper mesh for vis",
    )
    parser.add_argument(
        "--ffs-url",
        type=str,
        default=None,
        help="FFS Server URL for depth. If omitted, uses native RealSense depth.",
    )
    args = parser.parse_args()

    # 1. RealSense Capture
    use_ffs = args.ffs_url is not None
    ffs_status = f"FFS at {args.ffs_url}" if use_ffs else "Native Depth"
    print(f"Initializing RealSense (using {ffs_status} at 640x480)...")

    run_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join("run_data", run_id)
    os.makedirs(run_dir, exist_ok=True)
    print(f"Saving run data to: {run_dir}")

    camera = RealSenseCamera(
        width=640, height=480, use_ffs=use_ffs, ffs_url=args.ffs_url
    )
    try:
        camera.start()
    except Exception as e:
        print(f"Failed to start RealSense: {e}")
        return

    print("Streaming RealSense... Press [SPACE] to capture a frame.")
    color_frame = None
    depth_frame = None
    intrinsics = None
    marker_transform = None  # Camera to world transform from ArUco

    while True:
        color_frame, depth_frame = camera.get_frames()
        if color_frame is None:
            continue

        # Detect markers for visualization overlay
        gray = cv2.cvtColor(color_frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = camera.aruco_detector.detectMarkers(gray)

        vis_frame = color_frame.copy()

        # Draw detected markers
        if ids is not None:
            cv2.aruco.drawDetectedMarkers(vis_frame, corners, ids)

            # Try to draw frame axes for pose visualization
            K, dist = camera.get_camera_matrix()
            try:
                # Use marker_world for PnP to get pose
                half_marker = MARKER_LENGTH / 2
                marker_corners_local = np.array(
                    [
                        [-half_marker, -half_marker, 0],
                        [half_marker, -half_marker, 0],
                        [half_marker, half_marker, 0],
                        [-half_marker, half_marker, 0],
                    ],
                    dtype=np.float32,
                )

                obj_points = []
                img_points = []
                for i, marker_id in enumerate(ids.flatten()):
                    if marker_id in MARKER_WORLD:
                        center = MARKER_WORLD[marker_id]
                        world_corners = marker_corners_local + center
                        obj_points.append(world_corners)
                        img_points.append(corners[i][0])

                if len(obj_points) >= 2:
                    obj_pts = np.vstack(obj_points).astype(np.float32)
                    img_pts = np.vstack(img_points).astype(np.float32)

                    # Refine corners
                    cv2.cornerSubPix(
                        gray,
                        img_pts.reshape(-1, 1, 2),
                        (5, 5),
                        (-1, -1),
                        (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001),
                    )

                    success, rvec, tvec = cv2.solvePnP(
                        obj_pts, img_pts, K, dist, cv2.SOLVEPNP_ITERATIVE
                    )
                    if success:
                        cv2.drawFrameAxes(vis_frame, K, dist, rvec, tvec, 0.1)
            except Exception as e:
                pass  # Skip axis drawing if it fails

        cv2.imshow("RealSense", vis_frame)
        key = cv2.waitKey(1)

        if key == ord(" "):
            # First capture: try to detect ArUco markers
            print("Detecting ArUco markers...")
            intrinsics = camera.get_intrinsics()

            success, marker_transform, num_markers = camera.detect_aruco_markers(
                color_frame, MARKER_WORLD, MARKER_LENGTH
            )

            if success:
                print(f"Marker pose detected using {num_markers} markers!")
                if num_markers < 4:
                    print(
                        "WARNING: Only a few markers detected. Pose may be less accurate."
                    )
                print(f"Translation: {marker_transform[:3, 3]}")
                R = marker_transform[:3, :3]
                print("Rotation matrix detected.")
            else:
                if num_markers > 0:
                    print(
                        f"Only {num_markers} markers detected (need at least 2). Using camera frame (no transform)."
                    )
                else:
                    print("No markers detected. Using camera frame (no transform).")

            break
        elif key == ord("q"):
            camera.stop()
            cv2.destroyAllWindows()
            return

    # Now show RGB + Depth view
    print("Streaming RGB+Depth... Press [SPACE] to capture.")
    while True:
        color_frame, depth_frame = camera.get_frames()
        if color_frame is None:
            continue

        # Normalize depth for display
        depth_vis = depth_frame.astype(np.float32)
        depth_vis = (depth_vis / depth_vis.max() * 255).astype(np.uint8)
        depth_vis = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)

        # Stack RGB and Depth side by side
        combined = np.hstack([color_frame, depth_vis])
        cv2.putText(
            combined, "RGB", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2
        )
        cv2.putText(
            combined,
            "Depth",
            (color_frame.shape[1] + 10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
        )

        cv2.imshow("RealSense", combined)
        key = cv2.waitKey(1)
        if key == ord(" "):
            break

    print("Refreshing ArUco pose on final RGBD capture...")
    success, refreshed_marker_transform, num_markers = camera.detect_aruco_markers(
        color_frame, MARKER_WORLD, MARKER_LENGTH
    )
    if success:
        marker_transform = refreshed_marker_transform
        print(f"Updated marker pose using {num_markers} markers.")
        print(f"Translation: {marker_transform[:3, 3]}")
    else:
        if marker_transform is not None:
            print(
                f"Could not refresh marker pose on final capture ({num_markers} markers). "
                "Using earlier marker pose."
            )
        elif num_markers > 0:
            print(
                f"Only {num_markers} markers detected on final capture. "
                "Using camera frame (no transform)."
            )
        else:
            print("No markers detected on final capture. Using camera frame (no transform).")

    camera.stop()
    cv2.destroyAllWindows()

    # Save raw captured data
    print("Saving raw frames and intrinsics...")
    cv2.imwrite(os.path.join(run_dir, "rgb.png"), color_frame)
    np.save(os.path.join(run_dir, "depth.npy"), depth_frame)
    with open(os.path.join(run_dir, "intrinsics.json"), "w") as f:
        json.dump(intrinsics, f, indent=4)

    # 2. SAM Interaction
    print("\nCapture successful!")
    print("Select target using SAM:")
    print("  [B] to switch to Bounding Box mode (click and drag)")
    print("  [P] to switch to Point mode (click once)")
    print("  [SPACE] to confirm selection and query SAM")

    cv2.namedWindow("SAM Selection")
    cv2.setMouseCallback("SAM Selection", mouse_callback)

    while True:
        vis = color_frame.copy()

        if state["mode"] == "point" and len(state["points"]) > 0:
            cv2.circle(vis, state["points"][0], 5, (0, 255, 0), -1)
        elif state["mode"] == "box" and state["box"] is not None:
            x1, y1, x2, y2 = state["box"]
            cv2.rectangle(vis, (x1, y1), (x2, y2), (255, 0, 0), 2)

        cv2.imshow("SAM Selection", vis)
        key = cv2.waitKey(10)

        if key == ord("p"):
            state["mode"] = "point"
            state["points"] = []
            state["box"] = None
            print("Mode: POINT")
        elif key == ord("b"):
            state["mode"] = "box"
            state["points"] = []
            state["box"] = None
            print("Mode: BOX")
        elif key == ord(" "):
            if (state["mode"] == "point" and len(state["points"]) > 0) or (
                state["mode"] == "box" and state["box"] is not None
            ):
                break
            else:
                print("Make a selection first!")

    cv2.destroyAllWindows()

    # Query SAM
    print(f"\nSending {state['mode']} prompt to SAM server at {args.sam_url}...")
    sam_client = SAMClient(args.sam_url)
    try:
        if state["mode"] == "point":
            mask = sam_client.predict_from_point(color_frame, state["points"][0])
        else:
            x1, y1, x2, y2 = state["box"]
            bbox = (min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2))
            mask = sam_client.predict_from_box(color_frame, bbox)

        # Preview Mask
        vis_mask = color_frame.copy()
        vis_mask[mask > 0] = [0, 255, 0]
        cv2.imshow("SAM Mask", vis_mask)
        cv2.imwrite(os.path.join(run_dir, "sam_mask.png"), mask)
        cv2.imwrite(os.path.join(run_dir, "sam_mask_vis.png"), vis_mask)
        print("Displaying mask. Press any key to continue...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    except Exception as e:
        print(f"SAM prediction failed: {e}")
        return

    # 3. Project to PointCloud
    print("\nDeprojecting depth to 3D point cloud...")
    if marker_transform is not None:
        print(
            "Transforming points from camera frame to world frame using marker pose..."
        )
    xyz, rgb = depth_to_pcd(
        color_frame, depth_frame, intrinsics, mask, marker_transform, camera
    )
    print(f"Generated masked point cloud with {len(xyz)} points.")

    if len(xyz) == 0:
        print("Error: Point cloud is empty (check mask and depth alignment).")
        return

    # --- Debugging Stats ---
    unique_pts = np.unique(xyz, axis=0)
    print(f"\n--- Point Cloud Stats ---")
    print(f"Unique points: {len(unique_pts)} / {len(xyz)}")
    if len(xyz) > 0:
        print(f"XYZ Min : {xyz.min(axis=0)}")
        print(f"XYZ Max : {xyz.max(axis=0)}")
        print(f"XYZ Mean: {xyz.mean(axis=0)}")

    if mask is not None:
        masked_depths = depth_frame[mask > 0]
        if len(masked_depths) > 0:
            print(
                f"Depth values in mask - Min: {masked_depths.min()}, Max: {masked_depths.max()}, Mean: {masked_depths.mean():.2f}"
            )
        else:
            print("No valid depth values found inside the mask!")
    print(f"-------------------------\n")

    # Filter points within 0.5m radius of board center (in world frame)
    if marker_transform is not None:
        dist = np.linalg.norm(xyz, axis=1)
        radius_mask = dist < 0.8
        xyz = xyz[radius_mask]
        rgb = rgb[radius_mask]
        print(f"After 0.5m radius filter: {len(xyz)} points")

    if len(xyz) == 0:
        print("Error: No points within 0.5m of board center.")
        return

    # Downsample to 2048 points for model input
    print("Downsampling to 2048 points for model...")
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    pcd.colors = o3d.utility.Vector3dVector(rgb)

    # Remove statistical outliers (noise)
    print("Removing statistical outliers...")
    pcd, ind = pcd.remove_statistical_outlier(nb_neighbors=100, std_ratio=1.0)
    print(f"After outlier removal: {len(pcd.points)} points")

    # Voxel downsample first to even out density (optional, but good before FPS)
    # pcd = pcd.voxel_down_sample(voxel_size=0.005)

    current_n = len(pcd.points)
    if current_n >= 2048:
        # Farthest Point Sampling (FPS)
        print("Applying Farthest Point Sampling (FPS)...")
        pcd = pcd.farthest_point_down_sample(2048)
    else:
        # Random sample with replacement to get exactly 2048
        print("Not enough points for FPS, padding with random replacement...")
        idx = np.random.choice(current_n, 2048, replace=True)
        pcd = pcd.select_by_index(idx)

    xyz = np.asarray(pcd.points)
    rgb = np.asarray(pcd.colors)
    print(f"Downsampled to {len(xyz)} points.")

    print("Previewing 3D Pointcloud before sending...")

    # Scale points by 8x to match model expectations (MUST match what's sent to server)
    xyz_scaled = xyz * 8.0

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz_scaled)
    pcd.colors = o3d.utility.Vector3dVector(rgb)

    # Save pointcloud
    print("Saving processed pointcloud...")
    o3d.io.write_point_cloud(os.path.join(run_dir, "pcd_scaled.ply"), pcd)

    # Save marker transform if available
    if marker_transform is not None:
        np.save(os.path.join(run_dir, "marker_transform.npy"), marker_transform)

    # Add coordinate frame at origin (board center after points transformed to world frame)
    world_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.15)
    world_frame.scale(8.0, center=(0, 0, 0))

    if marker_transform is not None:
        print("Showing object frame (from marker transform) - scaled 8x")
    else:
        print("Showing camera frame - scaled 8x")

    o3d.visualization.draw_geometries([pcd, world_frame], window_name="Filtered PCD")

    # 4. Send to Grasp Model (already scaled)
    platform_height_meters = 0.08
    platform_height_scaled = (
        platform_height_meters * 8.0
    )  # Scale to match the 8x pointcloud
    print(
        f"\nSending pointcloud to Grasp Model Server at {args.model_url} with platform_height={platform_height_scaled} (scaled 8x)..."
    )
    model_client = ModelClient(args.model_url, timeout=120.0)
    try:
        results = model_client.predict(
            xyz_scaled, platform_height=platform_height_scaled
        )

        print("Saving grasp results...")
        np.savez(os.path.join(run_dir, "grasps.npz"), **results)

        if "refined_grasp_pairs" in results:
            grasp_pairs = results["refined_grasp_pairs"]
            pair_scores = results.get("pair_scores", None)
            print(f"Received {grasp_pairs.shape[0]} refined grasp pairs.")
            visualize_grasps(xyz_scaled, rgb, grasp_pairs, pair_scores, args.gripper)
        elif "single_grasps" in results:
            # Fallback: show single grasps when no pairs available
            single_grasps = results["single_grasps"]
            grasp_scores = results.get("grasp_scores", None)
            print(f"Received {single_grasps.shape[0]} single grasps (no valid pairs).")
            visualize_single_grasps(
                xyz_scaled, rgb, single_grasps, grasp_scores, args.gripper
            )
        else:
            print("Server returned success but no grasps in response.")
            print(f"Keys found: {list(results.keys())}")

    except Exception as e:
        print(f"Model prediction failed: {e}")





if __name__ == "__main__":
    main()
