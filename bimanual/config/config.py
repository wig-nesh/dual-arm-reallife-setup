# ============================================================
# INPUT SETTINGS
# ============================================================

DATASET_ROOT = "inputs"

OUTPUT_ROOT = "outputs"

INPUT_MODE = "mask"

# ============================================================
# PHYSICAL GRIPPER SETTINGS
# ============================================================

GRIPPER_WIDTH_M = 0.08

# ============================================================
# GRID SETTINGS
# ============================================================

MIN_GRID_ROWS = 2
MIN_GRID_COLS = 2

MAX_GRID_ROWS = 8
MAX_GRID_COLS = 8

MIN_CELL_OBJECT_RATIO = 0.15

# ============================================================
# GRASP SETTINGS
# ============================================================

NUM_CANDIDATE_GRASPS = 2000

NUM_GRASP_HYPOTHESES = 5

BOX_SIZE = 40

# ============================================================
# VLM SETTINGS
# ============================================================

USE_REAL_VLM = False

VLM_MODEL = "gemini-3-pro-preview"
MAX_RETRIES = 5

TEMPERATURE = 0.0

MAX_TOKENS = 120

# ============================================================
# VISUALIZATION
# ============================================================

GRID_LINE_THICKNESS = 1

BBOX_LINE_THICKNESS = 2

LEFT_GRASP_COLOR = (0, 255, 0)

RIGHT_GRASP_COLOR = (255, 0, 0)