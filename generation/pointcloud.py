import cv2
import numpy as np
import open3d as o3d


def depth_to_pcd(
    color_image, depth_image, intrinsics, mask=None, transform=None, camera=None
):
    if camera is not None and getattr(camera, "use_ffs", False):
        # --- FFS MANUAL ALIGNMENT PATH ---
        # depth_image is float32 (meters) in Left IR frame
        # color_image is BGR in RGB frame
        # mask is in RGB frame

        K_ir = camera.K_ir
        K_rgb = camera.K_rgb
        R_ir2rgb = camera.R_ir_to_rgb
        t_ir2rgb = camera.t_ir_to_rgb

        H, W = depth_image.shape
        v, u = np.indices((H, W))

        # Valid depth mask
        valid = depth_image > 0
        u = u[valid]
        v = v[valid]
        z = depth_image[valid]

        # 1. Deproject to Left IR 3D frame
        fx, fy = K_ir[0, 0], K_ir[1, 1]
        cx, cy = K_ir[0, 2], K_ir[1, 2]

        x = (u - cx) * z / fx
        y = (v - cy) * z / fy
        pts_ir = np.stack((x, y, z), axis=-1)

        # 2. Transform 3D points from Left IR frame to RGB frame
        pts_rgb_frame = pts_ir @ R_ir2rgb.T + t_ir2rgb

        # 3. Project 3D points into RGB image to get UV coordinates for color sampling
        fx_rgb, fy_rgb = K_rgb[0, 0], K_rgb[1, 1]
        cx_rgb, cy_rgb = K_rgb[0, 2], K_rgb[1, 2]

        u_rgb = np.round(
            (pts_rgb_frame[:, 0] * fx_rgb / pts_rgb_frame[:, 2]) + cx_rgb
        ).astype(int)
        v_rgb = np.round(
            (pts_rgb_frame[:, 1] * fy_rgb / pts_rgb_frame[:, 2]) + cy_rgb
        ).astype(int)

        # Filter bounds
        H_c, W_c = color_image.shape[:2]
        in_bounds = (u_rgb >= 0) & (u_rgb < W_c) & (v_rgb >= 0) & (v_rgb < H_c)

        pts_rgb_frame = pts_rgb_frame[in_bounds]
        u_rgb = u_rgb[in_bounds]
        v_rgb = v_rgb[in_bounds]

        # 4. If SAM mask is provided, filter points
        # The mask is in the RGB frame, so we use u_rgb, v_rgb
        if mask is not None:
            in_mask = mask[v_rgb, u_rgb] > 0
            pts_rgb_frame = pts_rgb_frame[in_mask]
            u_rgb = u_rgb[in_mask]
            v_rgb = v_rgb[in_mask]

        if len(pts_rgb_frame) == 0:
            return np.array([]), np.array([])

        # 5. Sample colors
        colors_bgr = color_image[v_rgb, u_rgb]
        colors_rgb = colors_bgr[:, ::-1] / 255.0  # Normalize to 0-1 float

        # 6. Apply ArUco transform to World Frame
        # The ArUco detection ran on the RGB image, so `transform` maps from RGB frame to World frame.
        if transform is not None:
            pts_h = np.hstack((pts_rgb_frame, np.ones((pts_rgb_frame.shape[0], 1))))
            pts_w = (transform @ pts_h.T).T[:, :3]
            points = pts_w
        else:
            points = pts_rgb_frame

        return points, colors_rgb

    # --- NATIVE DEPTH (OPEN3D) PATH ---
    o3d_intr = o3d.camera.PinholeCameraIntrinsic(
        width=intrinsics["width"],
        height=intrinsics["height"],
        fx=intrinsics["fx"],
        fy=intrinsics["fy"],
        cx=intrinsics["ppx"],
        cy=intrinsics["ppy"],
    )

    if mask is not None:
        depth_image = depth_image.copy()
        depth_image[mask == 0] = 0

    # If depth image is float32, it's likely from FFS and already in meters.
    # Open3D's RGBDImage uses depth_scale=1000.0 by default, which divides the values by 1000.
    # To fix this, if it's float32, we convert it to uint16 millimeters before passing.
    if depth_image.dtype == np.float32:
        depth_image = (depth_image * 1000.0).astype(np.uint16)

    rgb_o3d = o3d.geometry.Image(cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB))
    depth_o3d = o3d.geometry.Image(depth_image)

    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
        rgb_o3d,
        depth_o3d,
        depth_scale=1000.0,
        depth_trunc=3.0,
        convert_rgb_to_intensity=False,
    )

    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, o3d_intr)

    points = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors)

    # Transform points from camera frame to world frame if transform provided
    if transform is not None:
        points_h = np.hstack([points, np.ones((points.shape[0], 1))])
        points_w = (transform @ points_h.T).T[:, :3]
        points = points_w

    return points, colors

