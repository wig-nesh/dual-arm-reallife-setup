#bimanual paper direct implementation

import os
import json
import time
import cv2
import numpy as np

from PIL import Image
from google import genai

# ============================================================
# CONFIG
# ============================================================

# ------------------------------------------------------------
# INPUT MODES
#
# "mask"
# -> segmentation image is binary mask
#
# "extracted"
# -> segmentation image already contains extracted object
# ------------------------------------------------------------

INPUT_MODE = "extracted"

SCENE_IMAGE = "inputs/color.png"
SEGMENTATION_IMAGE = "inputs/image.png"

OUTPUT_IMAGE = "outputs/final_result.png"
GRID_IMAGE = "outputs/grid.png"

# ------------------------------------------------------------
# PAPER-LIKE SETTINGS
# ------------------------------------------------------------

GRIPPER_PIXEL_SIZE = 80

MIN_GRID_ROWS = 10
MIN_GRID_COLS = 10

MAX_GRID_ROWS = 20
MAX_GRID_COLS = 20

# ------------------------------------------------------------
# GEMINI
# ------------------------------------------------------------

GEMINI_MODEL = "gemini-2.5-flash"

USE_REAL_VLM = True

MAX_RETRIES = 5

# ============================================================
# GEMINI CLIENT
# ============================================================

client = genai.Client(
    api_key=os.getenv("GEMINI_API_KEY")
)

# ============================================================
# CREATE DIRECTORIES
# ============================================================

os.makedirs("outputs", exist_ok=True)

# ============================================================
# LOAD IMAGES
# ============================================================

scene = cv2.imread(SCENE_IMAGE)

if scene is None:
    raise Exception(f"Could not load scene image: {SCENE_IMAGE}")

segmentation = cv2.imread(SEGMENTATION_IMAGE)

if segmentation is None:
    raise Exception(
        f"Could not load segmentation image: {SEGMENTATION_IMAGE}"
    )

# ============================================================
# OBJECT EXTRACTION
# ============================================================

if INPUT_MODE == "mask":

    gray = cv2.cvtColor(segmentation, cv2.COLOR_BGR2GRAY)

    _, binary = cv2.threshold(
        gray,
        127,
        255,
        cv2.THRESH_BINARY
    )

    contours, _ = cv2.findContours(
        binary,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if len(contours) == 0:
        raise Exception("No contours found in mask")

    largest_contour = max(contours, key=cv2.contourArea)

    x, y, w, h = cv2.boundingRect(largest_contour)

    cropped_scene = scene[y:y+h, x:x+w]
    cropped_mask = binary[y:y+h, x:x+w]

    object_image = cv2.bitwise_and(
        cropped_scene,
        cropped_scene,
        mask=cropped_mask
    )

elif INPUT_MODE == "extracted":

    gray = cv2.cvtColor(
        segmentation,
        cv2.COLOR_BGR2GRAY
    )

    _, binary = cv2.threshold(
        gray,
        1,
        255,
        cv2.THRESH_BINARY
    )

    contours, _ = cv2.findContours(
        binary,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if len(contours) == 0:
        raise Exception(
            "No object found in extracted image"
        )

    largest_contour = max(
        contours,
        key=cv2.contourArea
    )

    x, y, w, h = cv2.boundingRect(
        largest_contour
    )

    object_image = segmentation[
        y:y+h,
        x:x+w
    ]

else:

    raise Exception(
        "INPUT_MODE must be either 'mask' or 'extracted'"
    )

# ============================================================
# DYNAMIC GRID GENERATION
#
# THIS IS WHAT THE PAPER DOES
#
# Grid size depends on object dimensions
# and approximates gripper scale
# ============================================================

img_h, img_w = object_image.shape[:2]

grid_cols = max(
    MIN_GRID_COLS,
    min(MAX_GRID_COLS, img_w // GRIPPER_PIXEL_SIZE)
)

grid_rows = max(
    MIN_GRID_ROWS,
    min(MAX_GRID_ROWS, img_h // GRIPPER_PIXEL_SIZE)
)

print("\n===== DYNAMIC GRID =====\n")

print(f"Object Width  : {img_w}")
print(f"Object Height : {img_h}")

print(f"Grid Rows     : {grid_rows}")
print(f"Grid Cols     : {grid_cols}")

# ============================================================
# OVERLAY GRID
# ============================================================

grid_image = object_image.copy()

cell_w = img_w // grid_cols
cell_h = img_h // grid_rows

cell_mapping = {}

idx = 1

for r in range(grid_rows):

    for c in range(grid_cols):

        x1 = c * cell_w
        y1 = r * cell_h

        if c == grid_cols - 1:
            x2 = img_w
        else:
            x2 = x1 + cell_w

        if r == grid_rows - 1:
            y2 = img_h
        else:
            y2 = y1 + cell_h

        # ----------------------------------------------------
        # DRAW GRID
        # ----------------------------------------------------

        cv2.rectangle(
            grid_image,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2
        )

        # ----------------------------------------------------
        # DRAW CELL NUMBER
        # ----------------------------------------------------

        cv2.putText(
            grid_image,
            str(idx),
            (x1 + 10, y1 + 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 0, 0),
            2
        )

        cell_mapping[idx] = {
            "x1": x1,
            "y1": y1,
            "x2": x2,
            "y2": y2
        }

        idx += 1

# ============================================================
# SAVE GRID IMAGE
# ============================================================

cv2.imwrite(GRID_IMAGE, grid_image)

# ============================================================
# PAPER-LIKE VLM PROMPT
#
# THIS IS VERY CLOSE TO THE PAPER
# ============================================================

PROMPT = f"""
The image shows an object with a numbered grid.

There are TWO robot arms:

- The PHYSICAL LEFT robot arm is located on the RIGHT side of the image.
- The PHYSICAL RIGHT robot arm is located on the LEFT side of the image.

Task:
Select grasp regions for stable bimanual lifting.

Requirements:
- balanced lifting
- symmetric support
- stable grasping
- avoid weak regions
- avoid thin structures
- avoid empty background

IMPORTANT:
Image coordinates are NOT robot coordinates.

Robot mapping:
- image LEFT side  = RIGHT robot arm
- image RIGHT side = LEFT robot arm

Return:
- one cell for LEFT robot arm
- one cell for RIGHT robot arm

Return ONLY raw JSON.

Use EXACTLY this format:

{{
    "image_left_cell": ["<cell number>"],
    "image_right_cell": ["<cell number>"]
}}
"""

# ============================================================
# QUERY VLM
# ============================================================

if USE_REAL_VLM:

    pil_image = Image.open(GRID_IMAGE)

    success = False

    for attempt in range(MAX_RETRIES):

        try:

            print(f"\nAttempt {attempt + 1}/{MAX_RETRIES}")

            response = client.models.generate_content(
                model=GEMINI_MODEL,
                contents=[
                    PROMPT,
                    pil_image
                ]
            )

            response_text = response.text.strip()

            response_text = response_text.replace(
                "```json",
                ""
            )

            response_text = response_text.replace(
                "```",
                ""
            )

            start = response_text.find("{")
            end = response_text.rfind("}") + 1

            response_text = response_text[start:end]

            print("\n===== GEMINI RESPONSE =====\n")
            print(response_text)

            result = json.loads(response_text)

            success = True

            break

        except Exception as e:

            print("\nGemini Error:")
            print(e)

            wait_time = (attempt + 1) * 5

            print(f"\nRetrying in {wait_time} seconds...")

            time.sleep(wait_time)

    if not success:

        raise Exception("Gemini failed after retries.")

else:

    # --------------------------------------------------------
    # MOCK OUTPUT
    # --------------------------------------------------------

    total_cells = len(cell_mapping)

    result = {
        "cell_robot_left": ["1"],
        "cell_robot_right": [str(total_cells)]
    }

# ============================================================
# PARSE CELL IDS
# ============================================================

image_left_cell = int(
    result["image_left_cell"][0]
)

image_right_cell = int(
    result["image_right_cell"][0]
)

left_cell = image_left_cell
right_cell = image_right_cell

if left_cell not in cell_mapping:
    raise Exception(f"Invalid left cell: {left_cell}")

if right_cell not in cell_mapping:
    raise Exception(f"Invalid right cell: {right_cell}")

left_bbox = cell_mapping[left_cell]
right_bbox = cell_mapping[right_cell]

# ============================================================
# FINAL VISUALIZATION
#
# Draw ONLY selected grasp bboxes
# on ORIGINAL scene image
# ============================================================

final_image = scene.copy()

# ------------------------------------------------------------
# CONVERT LOCAL CELL COORDS
# BACK TO GLOBAL IMAGE COORDS
# ------------------------------------------------------------

left_global = {
    "x1": left_bbox["x1"] + x,
    "y1": left_bbox["y1"] + y,
    "x2": left_bbox["x2"] + x,
    "y2": left_bbox["y2"] + y
}

right_global = {
    "x1": right_bbox["x1"] + x,
    "y1": right_bbox["y1"] + y,
    "x2": right_bbox["x2"] + x,
    "y2": right_bbox["y2"] + y
}

# ------------------------------------------------------------
# LEFT ARM BBOX
# ------------------------------------------------------------

cv2.rectangle(
    final_image,
    (left_global["x1"], left_global["y1"]),
    (left_global["x2"], left_global["y2"]),
    (0, 255, 0),
    2
)

cv2.putText(
    final_image,
    "LEFT",
    (left_global["x1"], left_global["y1"] - 10),
    cv2.FONT_HERSHEY_SIMPLEX,
    0.8,
    (0, 255, 0),
    2
)

# ------------------------------------------------------------
# RIGHT ARM BBOX
# ------------------------------------------------------------

cv2.rectangle(
    final_image,
    (right_global["x1"], right_global["y1"]),
    (right_global["x2"], right_global["y2"]),
    (255, 0, 0),
    2
)

cv2.putText(
    final_image,
    "RIGHT",
    (right_global["x1"], right_global["y1"] - 10),
    cv2.FONT_HERSHEY_SIMPLEX,
    0.8,
    (255, 0, 0),
    2
)
# ============================================================
# SAVE OUTPUT
# ============================================================

cv2.imwrite(
    OUTPUT_IMAGE,
    final_image
)

# ============================================================
# PRINT RESULTS
# ============================================================

print("\n===== FINAL RESULTS =====\n")

print(f"LEFT CELL  : {left_cell}")
print(f"LEFT BBOX  : {left_bbox}")

print()

print(f"RIGHT CELL : {right_cell}")
print(f"RIGHT BBOX : {right_bbox}")

print()

print(f"Saved grid image to: {GRID_IMAGE}")
print(f"Saved final image to: {OUTPUT_IMAGE}")