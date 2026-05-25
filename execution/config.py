import numpy as np

# xArm endpoints
IP1 = "192.168.1.242"  # xArm7 (Blue)
IP2 = "192.168.1.175"  # xArm6 (Red)

# Transform from each arm base frame to the robot board frame in mm.
XARM7_BASE_TO_BOARD_X = 324.0
XARM7_BASE_TO_BOARD_Y = -301.5

XARM6_BASE_TO_BOARD_X = 324.0
XARM6_BASE_TO_BOARD_Y = 293.0


def get_T_base_board(x_offset_mm, y_offset_mm):
    T = np.eye(4)
    T[0, 3] = x_offset_mm
    T[1, 3] = y_offset_mm
    return T


GRIPPER_DEVICENAME = "/dev/ttyUSB0"
GRIPPER_BAUDRATE = 57600
PROTOCOL_VERSION = 2.0
TRAVEL_DEG = 40
GRIPPER_MESH_PATH = "gripper/gripper.obj"

PREGRASP_OFFSET_MM = 70.0 
LIFT_MM = 100.0
SPEED = 40
APPROACH_STEPS = 20
POSITION_THRESHOLD_MM = 5.0

OBJECT_ROTATION_Z_DEG = -90.0

GRIPPER_LENGTH_MM = 220.0
FLANGE_TO_MODEL_BASE_MM = 150.0
TABLE_Z_MM = 0.0
COLLISION_CLEARANCE_MM = 5.0
