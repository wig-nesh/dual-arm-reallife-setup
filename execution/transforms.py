import math

import numpy as np

from .config import (
    COLLISION_CLEARANCE_MM,
    FLANGE_TO_MODEL_BASE_MM,
    GRIPPER_LENGTH_MM,
    OBJECT_ROTATION_Z_DEG,
    PREGRASP_OFFSET_MM,
    TABLE_Z_MM,
)

def rotation_matrix(roll_deg, pitch_deg, yaw_deg):
    r = np.radians(roll_deg)
    p = np.radians(pitch_deg)
    y = np.radians(yaw_deg)

    Rx = np.array([[1, 0, 0], [0, np.cos(r), -np.sin(r)], [0, np.sin(r), np.cos(r)]])

    Ry = np.array([[np.cos(p), 0, np.sin(p)], [0, 1, 0], [-np.sin(p), 0, np.cos(p)]])

    Rz = np.array([[np.cos(y), -np.sin(y), 0], [np.sin(y), np.cos(y), 0], [0, 0, 1]])

    return Rz @ Ry @ Rx


def get_T_robot_board_model_board():
    """
    Map saved/model board coordinates into the robot board coordinates used for
    base calibration, visualization, and execution.
    """
    R_z = rotation_matrix(0, 0, OBJECT_ROTATION_Z_DEG)
    T_rot_z = np.eye(4)
    T_rot_z[:3, :3] = R_z

    T_flip_y = np.eye(4)
    T_flip_y[1, 1] = -1.0

    return T_flip_y @ T_rot_z


def get_T_model_grasp_robot_grasp():
    """
    Model grasps use local Y as the approach axis and local X as the gripper
    baseline. xArm execution uses local Z as approach and local Y as baseline.
    This is a local-frame correction, so it is post-multiplied onto grasp poses.
    """
    T = np.eye(4)
    # Columns are the xArm grasp axes expressed in the model grasp frame:
    #   xArm X = model Z
    #   xArm Y = model X  (baseline)
    #   xArm Z = model Y  (approach)
    T[:3, :3] = np.array(
        [
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0],
        ]
    )
    return T


def ensure_right_handed_pose(T):
    """
    A single-axis board flip mirrors coordinates and can make grasp rotations
    left-handed. The robot can only execute proper rotations, so preserve the
    local baseline axis (Y) and approach axis (Z), and repair local X if needed.
    """
    T_fixed = T.copy()
    if np.linalg.det(T_fixed[:3, :3]) < 0:
        T_fixed[:3, 0] *= -1.0
    return T_fixed


def model_grasp_to_robot_board_mm(T_model_board_grasp_scaled):
    """
    Convert a saved model grasp into the exact board frame used by robot
    execution. Input translation is scaled by 8; output translation is mm.
    """
    T_model_board_grasp_mm = T_model_board_grasp_scaled.copy()
    T_model_board_grasp_mm[:3, 3] /= 8.0
    T_model_board_grasp_mm[:3, 3] *= 1000.0

    T_robot_board_grasp = (
        get_T_robot_board_model_board()
        @ T_model_board_grasp_mm
        @ get_T_model_grasp_robot_grasp()
    )
    return ensure_right_handed_pose(T_robot_board_grasp)


def matrix_to_pose(T):
    """Convert 4x4 matrix to [x, y, z, roll, pitch, yaw] (mm, degrees)."""
    x, y, z = T[:3, 3]
    sy = math.sqrt(T[0, 0] * T[0, 0] + T[1, 0] * T[1, 0])
    singular = sy < 1e-6
    if not singular:
        roll = math.atan2(T[2, 1], T[2, 2])
        pitch = math.atan2(-T[2, 0], sy)
        yaw = math.atan2(T[1, 0], T[0, 0])
    else:
        roll = math.atan2(-T[1, 2], T[1, 1])
        pitch = math.atan2(-T[2, 0], sy)
        yaw = 0
    return [x, y, z, math.degrees(roll), math.degrees(pitch), math.degrees(yaw)]


def pose_to_matrix(pose):
    """Convert [x, y, z, roll, pitch, yaw] back to a 4x4 matrix."""
    T = np.eye(4)
    T[:3, :3] = rotation_matrix(pose[3], pose[4], pose[5])
    T[:3, 3] = pose[:3]
    return T


def assert_pose_round_trip(label, T):
    pose = matrix_to_pose(T)
    T_round_trip = pose_to_matrix(pose)
    pos_err = np.linalg.norm(T[:3, 3] - T_round_trip[:3, 3])
    rot_err = np.linalg.norm(T[:3, :3] - T_round_trip[:3, :3])
    if pos_err > 1e-6 or rot_err > 1e-5:
        raise RuntimeError(
            f"{label} cannot be represented as an xArm pose "
            f"(pos_err={pos_err:.6f}mm, rot_err={rot_err:.6f})"
        )
    return pose


def eef_z_axis(pose):
    R = rotation_matrix(pose[3], pose[4], pose[5])
    return R[:, 2]


def compute_flange_pose(model_pose, offset=FLANGE_TO_MODEL_BASE_MM):
    """
    The model outputs the grasp pose for the BASE of the fingers.
    We need to offset this backwards along the local Z axis by the distance
    from the flange to the base of the fingers.
    """
    z_ax = eef_z_axis(model_pose)
    flange = list(model_pose)
    flange[0] -= z_ax[0] * offset
    flange[1] -= z_ax[1] * offset
    flange[2] -= z_ax[2] * offset
    return flange


def compute_pregrasp(flange_pose, offset_mm=PREGRASP_OFFSET_MM):
    z_ax = eef_z_axis(flange_pose)
    pre = list(flange_pose)
    pre[0] -= z_ax[0] * offset_mm
    pre[1] -= z_ax[1] * offset_mm
    pre[2] -= z_ax[2] * offset_mm
    return pre
