import os
import json
import time
import cv2
import numpy as np
import base64
import re

from PIL import Image
from openai import OpenAI

# ============================================================
# CONFIG
# ============================================================

INPUT_MODE = "extracted"
# "mask" or "extracted"

SCENE_IMAGE = "inputs/scene.png"
SEGMENTATION_IMAGE = "inputs/segmentation.png"

OUTPUT_GRID_IMAGE = "outputs/grid.png"
OUTPUT_RESULT_IMAGE = "outputs/final_result.png"

# ============================================================
# GRID SETTINGS
# ============================================================

GRIPPER_APERTURE_PIXELS = 50

MIN_GRID_ROWS = 2
MIN_GRID_COLS = 2

MAX_GRID_ROWS = 8
MAX_GRID_COLS = 8

MIN_CELL_OBJECT_RATIO = 0.15

# ============================================================
# CANDIDATE GRASP SETTINGS
# ============================================================

NUM_CANDIDATE_GRASPS = 2000

# ============================================================
# VLM
# ============================================================

USE_REAL_VLM = True

VLM_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"

MAX_RETRIES = 5

# ============================================================
# GROQ CLIENT
# ============================================================

client = OpenAI(
    api_key=os.getenv(""),
    base_url="https://api.groq.com/openai/v1"
)

# ============================================================
# OUTPUT DIR
# ============================================================

os.makedirs("outputs", exist_ok=True)

# ============================================================
# LOAD IMAGES
# ============================================================

scene = cv2.imread(SCENE_IMAGE)

if scene is None:
    raise Exception(
        f"Could not load scene image: {SCENE_IMAGE}"
    )

segmentation = cv2.imread(SEGMENTATION_IMAGE)

if segmentation is None:
    raise Exception(
        f"Could not load segmentation image: "
        f"{SEGMENTATION_IMAGE}"
    )

# ============================================================
# OBJECT EXTRACTION
# ============================================================

if INPUT_MODE == "mask":

    gray = cv2.cvtColor(
        segmentation,
        cv2.COLOR_BGR2GRAY
    )

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
        raise Exception("No contours found")

    largest_contour = max(
        contours,
        key=cv2.contourArea
    )

    x, y, w, h = cv2.boundingRect(
        largest_contour
    )

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
            "No contours found"
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

    cropped_mask = binary[
        y:y+h,
        x:x+w
    ]

else:

    raise Exception(
        "INPUT_MODE must be "
        "'mask' or 'extracted'"
    )

# ============================================================
# GRID GENERATION
# ============================================================

img_h, img_w = object_image.shape[:2]

grid_cols = max(
    MIN_GRID_COLS,
    min(
        MAX_GRID_COLS,
        img_w // GRIPPER_APERTURE_PIXELS
    )
)

grid_rows = max(
    MIN_GRID_ROWS,
    min(
        MAX_GRID_ROWS,
        img_h // GRIPPER_APERTURE_PIXELS
    )
)

print("\n===== GRID =====\n")

print(f"Rows : {grid_rows}")
print(f"Cols : {grid_cols}")

# ============================================================
# GRID OVERLAY
# ============================================================

grid_image = object_image.copy()

cell_w = img_w // grid_cols
cell_h = img_h // grid_rows

cell_mapping = {}

valid_cells = []

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
        # OBJECT COVERAGE CHECK
        # ----------------------------------------------------

        cell_mask = cropped_mask[
            y1:y2,
            x1:x2
        ]

        object_pixels = np.count_nonzero(
            cell_mask
        )

        total_pixels = (
            cell_mask.shape[0]
            *
            cell_mask.shape[1]
        )

        ratio = object_pixels / total_pixels

        # ----------------------------------------------------
        # VALID CELL
        # ----------------------------------------------------

        if ratio >= MIN_CELL_OBJECT_RATIO:

            valid_cells.append(idx)

            color = (0, 255, 0)

        else:

            color = (80, 80, 80)

        cv2.rectangle(
            grid_image,
            (x1, y1),
            (x2, y2),
            color,
            1
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

cv2.imwrite(
    OUTPUT_GRID_IMAGE,
    grid_image
)

# ============================================================
# VLM PROMPT
# ============================================================

PROMPT = f"""
Output ONLY JSON.

Grid rows: {grid_rows}
Grid cols: {grid_cols}

Valid cells:
{valid_cells}

Choose:
- one LEFT grasp region
- one RIGHT grasp region

The object must be lifted vertically upward.

Rules:
- stable lifting
- balanced support
- avoid weak structures
- avoid background

JSON format:

{{
  "left_cell": 1,
  "right_cell": 2
}}
"""

# ============================================================
# QUERY VLM
# ============================================================

if USE_REAL_VLM:

    success = False

    with open(
        OUTPUT_GRID_IMAGE,
        "rb"
    ) as f:

        image_base64 = base64.b64encode(
            f.read()
        ).decode("utf-8")

    for attempt in range(MAX_RETRIES):

        try:

            print(
                f"\nAttempt "
                f"{attempt+1}/{MAX_RETRIES}"
            )

            response = client.chat.completions.create(

                model=VLM_MODEL,

                messages=[

                    {
                        "role": "user",

                        "content": [

                            {
                                "type": "text",
                                "text": PROMPT
                            },

                            {
                                "type": "image_url",

                                "image_url": {

                                    "url":
                                    (
                                        "data:image/png;base64,"
                                        + image_base64
                                    )
                                }
                            }
                        ]
                    }
                ],

                temperature=0.0,
                max_tokens=50
            )

            response_text = (
                response
                .choices[0]
                .message
                .content
            )

            print("\n===== RAW RESPONSE =====\n")
            print(repr(response_text))

            if response_text is None:

                raise Exception(
                    "Empty response"
                )

            response_text = response_text.strip()

            response_text = response_text.replace(
                "```json",
                ""
            )

            response_text = response_text.replace(
                "```",
                ""
            )


            json_match = re.search(
                r'\{[\s\S]*?\}',
                response_text
            )

            if json_match is None:

                raise Exception(
                    "No JSON found in response"
                )

            response_text = json_match.group(0)

            print(
                "\n===== VLM RESPONSE =====\n"
            )

            print(response_text)

            result = json.loads(
                response_text
            )

            left_cell = result["left_cell"]
            right_cell = result["right_cell"]

            if left_cell not in valid_cells:
                raise Exception(
                    f"Invalid left cell: {left_cell}"
                )

            if right_cell not in valid_cells:
                raise Exception(
                    f"Invalid right cell: {right_cell}"
                )
            
            success = True

            break

        except Exception as e:

            print("\nVLM Error:\n")
            print(e)

            wait_time = (
                attempt + 1
            ) * 5

            print(
                f"\nRetrying in "
                f"{wait_time} seconds..."
            )

            time.sleep(wait_time)

    if not success:

        raise Exception(
            "VLM failed after retries"
        )

else:

    result = {
        "left_cell": 5,
        "right_cell": 9
    }

# ============================================================
# PARSE CELLS
# ============================================================

left_cell = result["left_cell"]
right_cell = result["right_cell"]

if left_cell not in cell_mapping:
    raise Exception(
        f"Invalid left cell: {left_cell}"
    )

if right_cell not in cell_mapping:
    raise Exception(
        f"Invalid right cell: {right_cell}"
    )

# ============================================================
# GENERATE CANDIDATE GRASPS
#
# Simulates AnyGrasp proposals
# ============================================================

candidate_grasps = []

# ============================================================
# GENERATE CANDIDATE GRASPS
#
# Simulated AnyGrasp proposals
# ============================================================

candidate_grasps = []

# ------------------------------------------------------------
# OBJECT CONTOUR
# ------------------------------------------------------------

contours, _ = cv2.findContours(
    cropped_mask,
    cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_SIMPLE
)

largest_contour = max(
    contours,
    key=cv2.contourArea
)

# ------------------------------------------------------------
# SAMPLE GRASP CANDIDATES
# ------------------------------------------------------------

for i in range(NUM_CANDIDATE_GRASPS):

    # random contour point
    idx = np.random.randint(
        0,
        len(largest_contour)
    )

    point = largest_contour[idx][0]

    px = int(point[0])
    py = int(point[1])

    # inward perturbation
    offset_x = np.random.randint(
        -20,
        20
    )

    offset_y = np.random.randint(
        -20,
        20
    )

    cx = px + offset_x
    cy = py + offset_y

    # bounds
    cx = np.clip(cx, 0, img_w - 1)
    cy = np.clip(cy, 0, img_h - 1)

    # inside object only
    if cropped_mask[cy, cx] == 0:
        continue

    # random grasp angle
    angle = np.random.uniform(
        -90,
        90
    )

    # heuristic grasp score
    #
    # prefer:
    # - side regions
    # - wider support
    # - away from tiny structures
    #

    center_dist = abs(
        cx - (img_w / 2)
    ) / (img_w / 2)

    score = (
        0.6 * center_dist
        +
        0.4 * np.random.uniform(0.0, 1.0)
    )

    grasp = {

        "cx": cx,
        "cy": cy,
        "angle": angle,
        "score": score
    }

    candidate_grasps.append(grasp)

print(
    f"\nGenerated "
    f"{len(candidate_grasps)} "
    f"candidate grasps"
)

print(
    f"\nGenerated "
    f"{len(candidate_grasps)} "
    f"candidate grasps"
)

# ============================================================
# POINT -> CELL
# ============================================================

def point_to_cell(cx, cy):

    col = min(
        cx // cell_w,
        grid_cols - 1
    )

    row = min(
        cy // cell_h,
        grid_rows - 1
    )

    return (
        row * grid_cols
    ) + col + 1

# ============================================================
# FILTER CANDIDATES
# ============================================================

left_candidates = []
right_candidates = []

for grasp in candidate_grasps:

    cell_id = point_to_cell(
        grasp["cx"],
        grasp["cy"]
    )

    if cell_id == left_cell:

        left_candidates.append(grasp)

    if cell_id == right_cell:

        right_candidates.append(grasp)

print(
    f"\nLeft candidates  : "
    f"{len(left_candidates)}"
)

print(
    f"Right candidates : "
    f"{len(right_candidates)}"
)

# ============================================================
# SELECT BEST GRASPS
# ============================================================

if len(left_candidates) == 0:
    raise Exception(
        "No left grasp candidates"
    )

if len(right_candidates) == 0:
    raise Exception(
        "No right grasp candidates"
    )

best_left = max(
    left_candidates,
    key=lambda g: g["score"]
)

best_right = max(
    right_candidates,
    key=lambda g: g["score"]
)

# ============================================================
# DRAW FINAL GRASPS
# ============================================================

final_image = scene.copy()

# ============================================================
# DRAW FUNCTION
# ============================================================

def draw_grasp(
    image,
    grasp,
    color,
    label
):

    cx = grasp["cx"] + x
    cy = grasp["cy"] + y

    angle = np.deg2rad(
        grasp["angle"]
    )

    length = 40

    dx = int(
        np.cos(angle)
        * length
    )

    dy = int(
        np.sin(angle)
        * length
    )

    pt1 = (
        cx - dx,
        cy - dy
    )

    pt2 = (
        cx + dx,
        cy + dy
    )

    cv2.line(
        image,
        pt1,
        pt2,
        color,
        2
    )

    cv2.circle(
        image,
        (cx, cy),
        5,
        color,
        -1
    )

    cv2.putText(
        image,
        label,
        (cx + 10, cy - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        color,
        2
    )

# ============================================================
# DRAW LEFT
# ============================================================

draw_grasp(
    final_image,
    best_left,
    (0, 255, 0),
    "LEFT"
)

# ============================================================
# DRAW RIGHT
# ============================================================

draw_grasp(
    final_image,
    best_right,
    (255, 0, 0),
    "RIGHT"
)

# ============================================================
# SAVE OUTPUT
# ============================================================

cv2.imwrite(
    OUTPUT_RESULT_IMAGE,
    final_image
)

# ============================================================
# PRINT RESULTS
# ============================================================

print("\n===== FINAL RESULTS =====\n")

print(
    f"LEFT CELL : {left_cell}"
)

print(
    f"RIGHT CELL : {right_cell}"
)

print()

print("BEST LEFT GRASP:")
print(best_left)

print()

print("BEST RIGHT GRASP:")
print(best_right)

print()

print(
    f"Saved grid image to: "
    f"{OUTPUT_GRID_IMAGE}"
)

print(
    f"Saved final result to: "
    f"{OUTPUT_RESULT_IMAGE}"
)