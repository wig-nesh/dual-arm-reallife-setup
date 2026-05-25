import argparse
import datetime
import os
import sys
import time

import numpy as np

# Keep compatibility with the original xarm/grasp_test.py import layout.
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "xarm"))

from dynamixel_sdk import PacketHandler, PortHandler
from gripper.src.gripper_controller import DynamixelGripper
from src.xarm_controller import XArmController

from .config import (
    FLANGE_TO_MODEL_BASE_MM,
    GRIPPER_BAUDRATE,
    GRIPPER_DEVICENAME,
    IP1,
    IP2,
    PREGRASP_OFFSET_MM,
    PROTOCOL_VERSION,
    TRAVEL_DEG,
    XARM6_BASE_TO_BOARD_X,
    XARM6_BASE_TO_BOARD_Y,
    XARM7_BASE_TO_BOARD_X,
    XARM7_BASE_TO_BOARD_Y,
    get_T_base_board,
)
from .motion import (
    CollisionError,
    SingleArmStreamer,
    check_table_collision,
    execute_grasp_primitive,
    getch,
    set_manual_mode,
)
from .transforms import (
    assert_pose_round_trip,
    matrix_to_pose,
    get_T_robot_board_model_board,
    model_grasp_to_robot_board_mm,
    pose_to_matrix,
)
from .visualization import visualize_scene
from .recording import RealSenseVideoRecorder


def apply_local_translation_offset(pose, x_mm, y_mm, z_mm):
    T = pose_to_matrix(pose)
    T = T @ np.array([[1, 0, 0, x_mm], [0, 1, 0, y_mm], [0, 0, 1, z_mm], [0, 0, 0, 1]])
    return matrix_to_pose(T)


def prompt_xyzrpy_pose(label):
    print(f"Enter {label} grasp pose as 6 values: x y z roll pitch yaw (mm, deg)")
    print("Example: 320 -150 180 180 0 90")
    while True:
        raw = input(f"{label} xyzrpy> ").strip().replace(",", " ")
        parts = [p for p in raw.split() if p]
        if len(parts) != 6:
            print("Invalid input: expected exactly 6 numeric values.")
            continue
        try:
            return [float(p) for p in parts]
        except ValueError:
            print("Invalid input: all 6 values must be numeric.")


def build_manual_execution_plan(grasp1_pose, grasp2_pose):
    T_robot_board_model_board = get_T_robot_board_model_board()

    T_base7_board = get_T_base_board(XARM7_BASE_TO_BOARD_X, XARM7_BASE_TO_BOARD_Y)
    T_base6_board = get_T_base_board(XARM6_BASE_TO_BOARD_X, XARM6_BASE_TO_BOARD_Y)
    T_board_base7 = np.linalg.inv(T_base7_board)
    T_board_base6 = np.linalg.inv(T_base6_board)

    T_base7_flange = pose_to_matrix(grasp1_pose)
    T_base6_flange = pose_to_matrix(grasp2_pose)

    T_flange_to_pre = np.eye(4)
    T_flange_to_pre[2, 3] = -PREGRASP_OFFSET_MM

    T_base7_pre = T_base7_flange @ T_flange_to_pre
    T_base6_pre = T_base6_flange @ T_flange_to_pre

    T_flange_to_tip = np.eye(4)
    T_flange_to_tip[2, 3] = FLANGE_TO_MODEL_BASE_MM
    T_base7_tip = T_base7_flange @ T_flange_to_tip
    T_base6_tip = T_base6_flange @ T_flange_to_tip

    T_vis_tip7 = T_board_base7 @ T_base7_tip
    T_vis_flange7 = T_board_base7 @ T_base7_flange
    T_vis_pre7 = T_board_base7 @ T_base7_pre

    T_vis_tip6 = T_board_base6 @ T_base6_tip
    T_vis_flange6 = T_board_base6 @ T_base6_flange
    T_vis_pre6 = T_board_base6 @ T_base6_pre

    grasp1 = assert_pose_round_trip("Arm1 grasp", T_base7_flange)
    pre1 = assert_pose_round_trip("Arm1 pregrasp", T_base7_pre)
    grasp2 = assert_pose_round_trip("Arm2 grasp", T_base6_flange)
    pre2 = assert_pose_round_trip("Arm2 pregrasp", T_base6_pre)

    return {
        "T_robot_board_model_board": T_robot_board_model_board,
        "T_board_base7": T_board_base7,
        "T_board_base6": T_board_base6,
        "T_board_model7": T_vis_tip7,
        "T_vis_flange7": T_vis_flange7,
        "T_vis_pre7": T_vis_pre7,
        "T_board_model6": T_vis_tip6,
        "T_vis_flange6": T_vis_flange6,
        "T_vis_pre6": T_vis_pre6,
        "grasp1": grasp1,
        "pre1": pre1,
        "grasp2": grasp2,
        "pre2": pre2,
        "idx_7": "manual",
        "idx_6": "manual",
        "dist_7": np.linalg.norm(T_base7_flange[:3, 3]),
        "dist_6": np.linalg.norm(T_base6_flange[:3, 3]),
    }


def build_execution_plan(grasps_scaled):
    T_robot_board_model_board = get_T_robot_board_model_board()
    grasps_board_mm = [model_grasp_to_robot_board_mm(g) for g in grasps_scaled]

    T_base7_board = get_T_base_board(XARM7_BASE_TO_BOARD_X, XARM7_BASE_TO_BOARD_Y)
    T_base6_board = get_T_base_board(XARM6_BASE_TO_BOARD_X, XARM6_BASE_TO_BOARD_Y)

    g0_mm = grasps_board_mm[0]
    g1_mm = grasps_board_mm[1]

    dist_g0_to_7 = np.linalg.norm((T_base7_board @ g0_mm)[:3, 3])
    dist_g0_to_6 = np.linalg.norm((T_base6_board @ g0_mm)[:3, 3])
    dist_g1_to_7 = np.linalg.norm((T_base7_board @ g1_mm)[:3, 3])
    dist_g1_to_6 = np.linalg.norm((T_base6_board @ g1_mm)[:3, 3])

    if (dist_g0_to_7 + dist_g1_to_6) < (dist_g1_to_7 + dist_g0_to_6):
        idx_7, idx_6 = 0, 1
        dist_7, dist_6 = dist_g0_to_7, dist_g1_to_6
    else:
        idx_7, idx_6 = 1, 0
        dist_7, dist_6 = dist_g1_to_7, dist_g0_to_6

    T_model_to_flange = np.eye(4)
    T_model_to_flange[2, 3] = -FLANGE_TO_MODEL_BASE_MM
    T_flange_to_pre = np.eye(4)
    T_flange_to_pre[2, 3] = -PREGRASP_OFFSET_MM

    T_board_model7 = grasps_board_mm[idx_7]
    T_board_model6 = grasps_board_mm[idx_6]

    T_board_flange7 = T_board_model7 @ T_model_to_flange
    T_board_pre7 = T_board_flange7 @ T_flange_to_pre
    T_board_flange6 = T_board_model6 @ T_model_to_flange
    T_board_pre6 = T_board_flange6 @ T_flange_to_pre

    T_base7_flange = T_base7_board @ T_board_flange7
    T_base7_pre = T_base7_board @ T_board_pre7
    T_base6_flange = T_base6_board @ T_board_flange6
    T_base6_pre = T_base6_board @ T_board_pre6

    T_board_base7 = np.linalg.inv(T_base7_board)
    T_board_base6 = np.linalg.inv(T_base6_board)

    T_vis_flange7 = T_board_base7 @ T_base7_flange
    T_vis_pre7 = T_board_base7 @ T_base7_pre
    T_vis_flange6 = T_board_base6 @ T_base6_flange
    T_vis_pre6 = T_board_base6 @ T_base6_pre

    for label, expected, actual in [
        ("Arm1 flange", T_board_flange7, T_vis_flange7),
        ("Arm1 pregrasp", T_board_pre7, T_vis_pre7),
        ("Arm2 flange", T_board_flange6, T_vis_flange6),
        ("Arm2 pregrasp", T_board_pre6, T_vis_pre6),
    ]:
        if not np.allclose(expected, actual, atol=1e-6):
            delta = np.linalg.norm(expected[:3, 3] - actual[:3, 3])
            raise RuntimeError(
                f"{label} visualization/execution mismatch: {delta:.6f}mm"
            )

    grasp1 = assert_pose_round_trip("Arm1 grasp", T_base7_flange)
    pre1 = assert_pose_round_trip("Arm1 pregrasp", T_base7_pre)
    grasp2 = assert_pose_round_trip("Arm2 grasp", T_base6_flange)
    pre2 = assert_pose_round_trip("Arm2 pregrasp", T_base6_pre)

    return {
        "T_robot_board_model_board": T_robot_board_model_board,
        "T_board_base7": T_board_base7,
        "T_board_base6": T_board_base6,
        "T_board_model7": T_board_model7,
        "T_vis_flange7": T_vis_flange7,
        "T_vis_pre7": T_vis_pre7,
        "T_board_model6": T_board_model6,
        "T_vis_flange6": T_vis_flange6,
        "T_vis_pre6": T_vis_pre6,
        "grasp1": grasp1,
        "pre1": pre1,
        "grasp2": grasp2,
        "pre2": pre2,
        "idx_7": idx_7,
        "idx_6": idx_6,
        "dist_7": dist_7,
        "dist_6": dist_6,
    }


def table_clearance_failures(plan):
    failures = []
    for label, pose in [
        ("Arm1 grasp", plan["grasp1"]),
        ("Arm1 pregrasp", plan["pre1"]),
        ("Arm2 grasp", plan["grasp2"]),
        ("Arm2 pregrasp", plan["pre2"]),
    ]:
        is_col, f_z, t_z, min_z = check_table_collision(pose)
        if is_col:
            failures.append((label, f_z, t_z, min_z))
    return failures


def print_table_clearance_error(grasp_idx, failures):
    print(
        f"Error: Requested grasp index {grasp_idx} was removed because it is too close to the table."
    )
    for label, f_z, t_z, min_z in failures:
        print(
            f"  {label:20s}: flange_z={f_z:.1f}  tip_z={t_z:.1f}  min_z={min_z:.1f}"
        )


def main():
    parser = argparse.ArgumentParser(description="Execute grasps from a run directory")
    parser.add_argument(
        "--run-dir", type=str, required=True, help="Path to run_data directory"
    )
    parser.add_argument(
        "--direct", action="store_true", help="Skip manual mode and execute directly"
    )
    parser.add_argument(
        "--grasp-idx",
        type=int,
        default=0,
        help="Index of the grasp pair to execute (default: 0)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Build and visualize poses, then exit before connecting to hardware",
    )
    parser.add_argument(
        "--record-realsense",
        action="store_true",
        help="Record RealSense RGB video during the grasp primitive",
    )
    parser.add_argument(
        "--manual-xyzrpy",
        action="store_true",
        help="Prompt for xArm7/xArm6 grasp xyzrpy poses and skip grasp-index selection",
    )
    args = parser.parse_args()

    if not os.path.isdir(args.run_dir):
        print(f"Error: run directory not found: {args.run_dir}")
        return

    if args.manual_xyzrpy:
        print("Manual pose mode enabled: grasp index and grasps.npz are ignored.")
        grasp1_pose = prompt_xyzrpy_pose("xArm7")
        grasp2_pose = prompt_xyzrpy_pose("xArm6")
        plan = build_manual_execution_plan(grasp1_pose, grasp2_pose)
    else:
        npz_path = os.path.join(args.run_dir, "grasps.npz")
        if not os.path.exists(npz_path):
            print(f"Error: {npz_path} not found.")
            return

        data = np.load(npz_path, allow_pickle=True)
        # Extract the requested grasp pair
        if "refined_grasp_pairs" in data and len(data["refined_grasp_pairs"]) > 0:
            pairs = data["refined_grasp_pairs"]
            if args.grasp_idx >= len(pairs) or args.grasp_idx < 0:
                print(
                    f"Error: Requested grasp index {args.grasp_idx} is out of bounds (max {len(pairs) - 1})."
                )
                return
            removed = {}
            for idx, pair in enumerate(pairs):
                candidate_plan = build_execution_plan([pair[0], pair[1]])
                failures = table_clearance_failures(candidate_plan)
                if failures:
                    removed[idx] = failures

            if removed:
                print(
                    f"Removed {len(removed)} / {len(pairs)} grasp pairs that are too close to the table."
                )

            if args.grasp_idx in removed:
                print_table_clearance_error(args.grasp_idx, removed[args.grasp_idx])
                valid_indices = [idx for idx in range(len(pairs)) if idx not in removed]
                if valid_indices:
                    print(f"First valid grasp indices: {valid_indices[:10]}")
                else:
                    print("No valid grasp pairs remain after table-clearance filtering.")
                return

            selected_pair = pairs[args.grasp_idx]
            grasps_scaled = [selected_pair[0], selected_pair[1]]
            print(f"Using Refined Grasp Pair Index: {args.grasp_idx} out of {len(pairs)}")
        elif "single_grasps" in data and len(data["single_grasps"]) >= 2:
            # Fallback to top 2 single grasps if no pairs exist
            grasps_scaled = [data["single_grasps"][0], data["single_grasps"][1]]
            fallback_plan = build_execution_plan(grasps_scaled)
            failures = table_clearance_failures(fallback_plan)
            if failures:
                print_table_clearance_error(args.grasp_idx, failures)
                return
        else:
            print("Not enough grasps found in the npz file (need at least 2).")
            return

        grasps_scaled = np.array(grasps_scaled)
        plan = build_execution_plan(grasps_scaled)

    visualize_scene(
        args.run_dir,
        plan["T_robot_board_model_board"],
        plan["T_board_base7"],
        plan["T_board_base6"],
        plan["T_board_model7"],
        plan["T_vis_flange7"],
        plan["T_vis_pre7"],
        plan["T_board_model6"],
        plan["T_vis_flange6"],
        plan["T_vis_pre6"],
    )

    idx_7 = plan["idx_7"]
    idx_6 = plan["idx_6"]
    dist_7 = plan["dist_7"]
    dist_6 = plan["dist_6"]
    grasp1 = plan["grasp1"]
    pre1 = plan["pre1"]
    grasp2 = plan["grasp2"]
    pre2 = plan["pre2"]

    print(f"Assigned Grasp {idx_7} to xArm7 (Blue, {IP1}) - Dist: {dist_7:.1f}mm")
    print(f"Assigned Grasp {idx_6} to xArm6 (Red, {IP2}) - Dist: {dist_6:.1f}mm")

    for label, pose in [
        ("Arm1 grasp", grasp1),
        ("Arm1 pregrasp", pre1),
        ("Arm2 grasp", grasp2),
        ("Arm2 pregrasp", pre2),
    ]:
        is_col, f_z, t_z, min_z = check_table_collision(pose)
        status = "⚠ COLLIDES" if is_col else "OK"
        print(
            f"  {label:20s}: flange_z={f_z:.1f}  tip_z={t_z:.1f}  min_z={min_z:.1f}  [{status}]"
        )
        if is_col:
            print(f"  ERROR: {label} collides with table. Aborting.")
            return

    print("=" * 70)
    print(f"  Grasp Arm1  (own):   {[f'{v:.1f}' for v in grasp1]}")
    print(f"  Pre-grasp Arm1 (own):{[f'{v:.1f}' for v in pre1]}")
    print(f"  Grasp Arm2  (own):   {[f'{v:.1f}' for v in grasp2]}")
    print(f"  Pre-grasp Arm2 (own):{[f'{v:.1f}' for v in pre2]}")
    print("=" * 70)

    if args.dry_run:
        print("Dry run complete. Skipping arm and gripper connections.")
        return

    # ── Connect arms ──
    print(f"\nConnecting to arms {IP1} and {IP2}...")
    arm1 = XArmController(IP1)
    arm2 = XArmController(IP2)

    for arm in [arm1, arm2]:
        curr = arm.get_position()
        if curr:
            is_col, f_z, t_z, min_z = check_table_collision(curr)
            if is_col:
                print(
                    f"  ERROR: [{arm.ip}] current pose already collides with table. Aborting."
                )
                arm1.disconnect()
                arm2.disconnect()
                return

    # ── Connect grippers ──
    port_handler = PortHandler(GRIPPER_DEVICENAME)
    packet_handler = PacketHandler(PROTOCOL_VERSION)
    if not port_handler.openPort() or not port_handler.setBaudRate(GRIPPER_BAUDRATE):
        print("Failed to open gripper port or set baudrate")
        return

    left_gripper = DynamixelGripper(
        port_handler, packet_handler, dxl_id=0, travel_deg=TRAVEL_DEG
    )
    right_gripper = DynamixelGripper(
        port_handler, packet_handler, dxl_id=1, travel_deg=TRAVEL_DEG
    )

    print("Calibrating grippers...")
    left_gripper.calibrate(getch)
    right_gripper.calibrate(getch)

    recorder = None
    try:
        # --- Arm 1 Manual Mode ---
        set_manual_mode(arm1, enable=True)
        print("\n── Manual Mode: Arm 1 (Blue / xArm7) ─────────────────────────")
        print("  Guide Arm 1 to its pre-grasp pose.")
        print("  Press [N] to lock Arm 1 and move to Arm 2.   Press [ESC] to abort.")
        print("──────────────────────────────────────────────────────────────────\n")

        streamer1 = SingleArmStreamer(arm1, pre1, "Arm 1")
        streamer1.start()

        while True:
            key = getch()
            if key.lower() == "n":
                streamer1.stop()
                set_manual_mode(arm1, enable=False)
                print("\n[Arm 1 locked.]")
                break
            elif key == chr(0x1B):
                streamer1.stop()
                print("\nESC — aborting.")
                return

        time.sleep(1)

        # --- Arm 2 Manual Mode ---
        set_manual_mode(arm2, enable=True)
        print("\n── Manual Mode: Arm 2 (Red / xArm6) ──────────────────────────")
        print("  Guide Arm 2 to its pre-grasp pose.")
        print("  Press [P] to execute grasp primitive.   Press [ESC] to abort.")
        print("──────────────────────────────────────────────────────────────────\n")

        streamer2 = SingleArmStreamer(arm2, pre2, "Arm 2")
        streamer2.start()

        while True:
            key = getch()
            if key.lower() == "p":
                streamer2.stop()
                set_manual_mode(arm2, enable=False)
                print("\n[P] pressed — switching to position control...")
                break
            elif key == chr(0x1B):
                streamer2.stop()
                print("\nESC — aborting.")
                return

        time.sleep(0.5)

        if args.record_realsense:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            video_path = os.path.join(
                args.run_dir, f"execution_realsense_{timestamp}.mp4"
            )
            try:
                recorder = RealSenseVideoRecorder(video_path)
                recorder.start()
            except Exception as e:
                recorder = None
                print(f"WARNING: Failed to start RealSense recording: {e}")

        execute_grasp_primitive(
            arm1, arm2, grasp1, grasp2, pre1, pre2, left_gripper, right_gripper
        )

    except CollisionError as e:
        print(f"\n\n  *** COLLISION ABORT *** {e}")
    except KeyboardInterrupt:
        print("\nInterrupted.")
    finally:
        if recorder is not None:
            recorder.stop()
        set_manual_mode(arm1, enable=False)
        set_manual_mode(arm2, enable=False)
        left_gripper.close()
        right_gripper.close()
        left_gripper.disable_torque()
        right_gripper.disable_torque()
        port_handler.closePort()
        arm1.disconnect()
        arm2.disconnect()
        print("Done.")





if __name__ == "__main__":
    main()
