import math
import os
import sys
import threading
import time

import numpy as np

from .config import (
    COLLISION_CLEARANCE_MM,
    GRIPPER_LENGTH_MM,
    LIFT_MM,
    SPEED,
    TABLE_Z_MM,
)
from .transforms import eef_z_axis

if os.name == "nt":
    import msvcrt

    def getch():
        return msvcrt.getch().decode()
else:
    import tty, termios

    def getch():
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        try:
            tty.setraw(fd)
            ch = sys.stdin.read(1)
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        return ch


def check_table_collision(
    pose,
    gripper_length=GRIPPER_LENGTH_MM,
    table_z=TABLE_Z_MM,
    clearance=COLLISION_CLEARANCE_MM,
):
    if pose is None or len(pose) < 6:
        return False, 0.0, 0.0, 0.0

    x, y, z, roll, pitch, yaw = pose[:6]
    z_dir = eef_z_axis(pose)

    p_flange = np.array([x, y, z])
    p_tip = p_flange + z_dir * gripper_length

    flange_z = p_flange[2]
    tip_z = p_tip[2]
    min_z = min(flange_z, tip_z)

    safe_z = table_z + clearance
    is_colliding = min_z <= safe_z

    return is_colliding, flange_z, tip_z, min_z


def assert_safe(arm_label, pose):
    is_col, f_z, t_z, min_z = check_table_collision(pose)
    if is_col:
        raise CollisionError(
            f"[{arm_label}] Table collision detected! "
            f"Flange Z={f_z:.1f}mm  Tip Z={t_z:.1f}mm  "
            f"Min Z={min_z:.1f}mm  ≤  safe limit={TABLE_Z_MM + COLLISION_CLEARANCE_MM:.1f}mm"
        )


class CollisionError(Exception):
    pass


def move_both(arm1, arm2, pose1, pose2, speed=SPEED, wait=True):
    assert_safe(arm1.ip, pose1)
    assert_safe(arm2.ip, pose2)

    def _move(arm, pose):
        arm.set_position(*pose[:6], speed=speed, wait=wait)

    t1 = threading.Thread(target=_move, args=(arm1, pose1))
    t2 = threading.Thread(target=_move, args=(arm2, pose2))
    t1.start()
    t2.start()
    t1.join()
    t2.join()


def set_manual_mode(arm, enable=True):
    if enable:
        arm.arm.set_mode(2)
        arm.arm.set_state(0)
        print(f"[{arm.ip}] Manual mode ON")
    else:
        arm.arm.set_mode(0)
        arm.arm.set_state(0)
        print(f"[{arm.ip}] Manual mode OFF — position control")


def pose_distance(current_pose, target_pose):
    return float(np.linalg.norm(np.array(current_pose[:3]) - np.array(target_pose[:3])))


# ── Pose streamer ─────────────────────────────────────────────────────────────


class SingleArmStreamer:
    def __init__(self, arm, target_pose, arm_name):
        self.arm = arm
        self.target = target_pose
        self.arm_name = arm_name
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self):
        self._thread.start()

    def stop(self):
        self._stop.set()
        self._thread.join()

    def _run(self):
        tx, ty, tz = self.target[:3]
        while not self._stop.is_set():
            p = self.arm.get_position()
            if p:
                cx, cy, cz = p[:3]
                ex = tx - cx
                ey = ty - cy
                ez = tz - cz
                dist = math.sqrt(ex**2 + ey**2 + ez**2)

                print(
                    f"\r[{self.arm_name}] "
                    f"CUR:({cx:6.1f}, {cy:6.1f}, {cz:6.1f}) | "
                    f"TGT:({tx:6.1f}, {ty:6.1f}, {tz:6.1f}) | "
                    f"ERR:(X:{ex:6.1f}, Y:{ey:6.1f}, Z:{ez:6.1f}) | "
                    f"Δ:{dist:5.1f}mm    ",
                    end="",
                    flush=True,
                )
            time.sleep(0.1)


def execute_grasp_primitive(
    arm1, arm2, grasp1, grasp2, pre1, pre2, left_gripper, right_gripper
):
    print("\n[Primitive] Moving to exact PRE-GRASP...")
    move_both(arm1, arm2, pre1, pre2)

    print("\n[Primitive] Waiting 3 seconds...")
    time.sleep(3.0)

    print("[Primitive] Opening grippers...")
    left_gripper.toggle()
    right_gripper.toggle()
    time.sleep(1.0)

    print("[Primitive] Moving FORWARD to grasp pose...")
    move_both(arm1, arm2, grasp1, grasp2)

    print("[Primitive] Closing grippers...")
    left_gripper.toggle()
    right_gripper.toggle()
    time.sleep(1.0)

    print(f"[Primitive] Lifting +{LIFT_MM:.0f}mm...")
    p1 = arm1.get_position()
    p2 = arm2.get_position()
    lift1 = list(p1)
    lift1[2] += LIFT_MM
    lift2 = list(p2)
    lift2[2] += LIFT_MM
    move_both(arm1, arm2, lift1, lift2)

    print("[Primitive] Holding for 2 seconds...")
    time.sleep(2.0)

    print("[Primitive] Lowering back down...")
    move_both(arm1, arm2, p1, p2)

    print("[Primitive] Opening grippers...")
    left_gripper.toggle()
    right_gripper.toggle()

    print("[Primitive] Done.")


