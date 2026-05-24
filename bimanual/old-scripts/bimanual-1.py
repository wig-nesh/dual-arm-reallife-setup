# bimanual paper implementation with geometry affordance

import os
import json
import time

import cv2
import numpy as np

from google import genai
from PIL import Image

# ============================================================
# CONFIG
# ============================================================

SCENE_IMAGE = "inputs/color.png"
SEGMENTATION_MASK = "inputs/image.png"
INPUT_TYPE = "extracted"

OUTPUT_IMAGE = "outputs/final_result.png"
CANDIDATE_IMAGE = "outputs/candidate_regions.png"

GEMINI_MODEL = "gemini-2.5-flash"

USE_REAL_VLM = True

# ============================================================
# GEMINI CLIENT
# ============================================================

client = genai.Client(
    api_key=os.getenv("GEMINI_API_KEY")
)

# ============================================================
# PROMPT
# ============================================================

PROMPT = """
You are a robotic grasp planner.

The robot must lift the segmented object vertically upward
using two robotic grippers.

Several candidate affordance regions are shown.

Each region represents a possible grasp affordance.

Your task:
Choose the BEST PAIR of affordance regions for stable
bimanual lifting.

Goals:
- balanced lifting
- symmetric support
- stable grasping
- avoid thin structures
- avoid unstable regions
- avoid handles/protrusions if present

Reason jointly for BOTH arms together.

Prefer:
- LEFT_BODY + RIGHT_BODY
when appropriate.

Return ONLY raw JSON.

Use EXACTLY this format:

{
    "best_pair": [
        "LEFT_BODY",
        "RIGHT_BODY"
    ]
}
"""

# ============================================================
# CREATE OUTPUT DIRECTORIES
# ============================================================

os.makedirs("outputs", exist_ok=True)

# ============================================================
# LOAD INPUTS
# ============================================================

scene = cv2.imread(SCENE_IMAGE)

if scene is None:
    raise Exception(f"Could not load scene image: {SCENE_IMAGE}")

mask = cv2.imread(SEGMENTATION_MASK, 0)

if mask is None:
    raise Exception(f"Could not load segmentation mask: {SEGMENTATION_MASK}")

# ============================================================
# INPUT HANDLING
# ============================================================

if INPUT_TYPE == "mask":

    print("\nUsing segmentation MASK input.\n")

    # --------------------------------------------------------
    # LOAD MASK
    # --------------------------------------------------------

    _, binary = cv2.threshold(
        mask,
        127,
        255,
        cv2.THRESH_BINARY
    )

    # --------------------------------------------------------
    # FIND OBJECT CONTOUR
    # --------------------------------------------------------

    contours, _ = cv2.findContours(
        binary,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if len(contours) == 0:
        raise Exception("No object found in segmentation mask")

    largest_contour = max(
        contours,
        key=cv2.contourArea
    )

    # --------------------------------------------------------
    # OBJECT BBOX
    # --------------------------------------------------------

    x, y, w, h = cv2.boundingRect(
        largest_contour
    )

    cropped_scene = scene[y:y+h, x:x+w]

    cropped_mask = binary[y:y+h, x:x+w]

    # --------------------------------------------------------
    # EXTRACT OBJECT
    # --------------------------------------------------------

    object_image = cv2.bitwise_and(
        cropped_scene,
        cropped_scene,
        mask=cropped_mask
    )

elif INPUT_TYPE == "extracted":

    print("\nUsing extracted OBJECT image input.\n")

    # --------------------------------------------------------
    # LOAD EXTRACTED OBJECT IMAGE
    # --------------------------------------------------------

    object_image = cv2.imread(
        SEGMENTATION_MASK
    )

    if object_image is None:

        raise Exception(
            f"Could not load extracted object image: "
            f"{SEGMENTATION_MASK}"
        )

    # --------------------------------------------------------
    # CREATE SYNTHETIC MASK
    # --------------------------------------------------------

    gray = cv2.cvtColor(
        object_image,
        cv2.COLOR_BGR2GRAY
    )

    _, cropped_mask = cv2.threshold(
        gray,
        1,
        255,
        cv2.THRESH_BINARY
    )

    # --------------------------------------------------------
    # USE FULL IMAGE
    # --------------------------------------------------------

    h, w = object_image.shape[:2]

    x = 0
    y = 0

else:

    raise Exception(
        "INPUT_TYPE must be either "
        "'mask' or 'extracted'"
    )

# ============================================================
# OBJECT GEOMETRY
# ============================================================

mask_indices = np.where(cropped_mask > 0)

ys = mask_indices[0]
xs = mask_indices[1]

xmin = np.min(xs)
xmax = np.max(xs)

ymin = np.min(ys)
ymax = np.max(ys)

object_width = xmax - xmin
object_height = ymax - ymin

# ============================================================
# GENERATE AFFORDANCE PROPOSALS
# ============================================================

candidate_regions = {}

candidate_image = object_image.copy()

grasp_width = int(object_width * 0.20)
grasp_height = int(object_height * 0.35)

grasp_y = ymin + int(object_height * 0.60)

affordance_defs = {

    "LEFT_BODY": (
        xmin + int(object_width * 0.25),
        grasp_y
    ),

    "RIGHT_BODY": (
        xmin + int(object_width * 0.75),
        grasp_y
    ),

    "CENTER_BODY": (
        xmin + int(object_width * 0.50),
        grasp_y
    ),

    "UPPER_BODY": (
        xmin + int(object_width * 0.50),
        ymin + int(object_height * 0.30)
    ),

    "LOWER_BODY": (
        xmin + int(object_height * 0.50),
        ymin + int(object_height * 0.82)
    )
}

# ============================================================
# BUILD REGIONS
# ============================================================

img_h, img_w = object_image.shape[:2]

for label, (cx, cy) in affordance_defs.items():

    x1 = cx - grasp_width // 2
    y1 = cy - grasp_height // 2

    x2 = cx + grasp_width // 2
    y2 = cy + grasp_height // 2

    # --------------------------------------------------------
    # CLIP
    # --------------------------------------------------------

    x1 = max(0, x1)
    y1 = max(0, y1)

    x2 = min(img_w - 1, x2)
    y2 = min(img_h - 1, y2)

    bbox = [x1, y1, x2, y2]

    candidate_regions[label] = bbox

    # --------------------------------------------------------
    # DRAW REGION
    # --------------------------------------------------------

    cv2.rectangle(
        candidate_image,
        (x1, y1),
        (x2, y2),
        (0, 255, 255),
        2
    )

    cv2.putText(
        candidate_image,
        label,
        (x1, y1 - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 255),
        2
    )

# ============================================================
# SAVE CANDIDATE IMAGE
# ============================================================

cv2.imwrite(
    CANDIDATE_IMAGE,
    candidate_image
)

# ============================================================
# QUERY VLM
# ============================================================

if USE_REAL_VLM:

    MAX_RETRIES = 5

    response = None

    for attempt in range(MAX_RETRIES):

        try:

            print(f"\nAttempt {attempt+1}/{MAX_RETRIES}")

            pil_image = Image.open(CANDIDATE_IMAGE)

            response = client.models.generate_content(
                model=GEMINI_MODEL,
                contents=[
                    PROMPT,
                    pil_image
                ]
            )

            print("\nGemini request successful.\n")

            break

        except Exception as e:

            print(f"\nGemini Error:\n{e}\n")

            if attempt < MAX_RETRIES - 1:

                wait_time = 15

                print(f"Retrying in {wait_time} seconds...\n")

                time.sleep(wait_time)

            else:

                raise Exception(
                    "Gemini failed after maximum retries"
                )

    response_text = response.text.strip()

    response_text = response_text.replace("```json", "")
    response_text = response_text.replace("```", "")

    start = response_text.find("{")
    end = response_text.rfind("}") + 1

    response_text = response_text[start:end]

    print("\n===== GEMINI RESPONSE =====\n")
    print(response_text)

    result = json.loads(response_text)

else:

    result = {
        "best_pair": [
            "LEFT_BODY",
            "RIGHT_BODY"
        ]
    }

# ============================================================
# FINAL SELECTION
# ============================================================

selected_pair = result["best_pair"]

left_label = selected_pair[0]
right_label = selected_pair[1]

left_bbox = candidate_regions[left_label]
right_bbox = candidate_regions[right_label]

print("\n===== SELECTED REGIONS =====\n")

print("LEFT :", left_label)
print("RIGHT:", right_label)

# ============================================================
# DRAW FINAL OUTPUT
# ============================================================

final_image = scene.copy()

# ------------------------------------------------------------
# CONVERT TO GLOBAL COORDS
# ------------------------------------------------------------

left_global = [
    left_bbox[0] + x,
    left_bbox[1] + y,
    left_bbox[2] + x,
    left_bbox[3] + y
]

right_global = [
    right_bbox[0] + x,
    right_bbox[1] + y,
    right_bbox[2] + x,
    right_bbox[3] + y
]

# ------------------------------------------------------------
# LEFT REGION
# ------------------------------------------------------------

cv2.rectangle(
    final_image,
    (left_global[0], left_global[1]),
    (left_global[2], left_global[3]),
    (0, 255, 0),
    1
)

cv2.putText(
    final_image,
    left_label,
    (left_global[0], left_global[1] - 10),
    cv2.FONT_HERSHEY_SIMPLEX,
    1,
    (0, 255, 0),
    1
)

# ------------------------------------------------------------
# RIGHT REGION
# ------------------------------------------------------------

cv2.rectangle(
    final_image,
    (right_global[0], right_global[1]),
    (right_global[2], right_global[3]),
    (255, 0, 0),
    1
)

cv2.putText(
    final_image,
    right_label,
    (right_global[0], right_global[1] - 10),
    cv2.FONT_HERSHEY_SIMPLEX,
    1,
    (255, 0, 0),
    1
)

# ============================================================
# SAVE OUTPUT
# ============================================================

cv2.imwrite(
    OUTPUT_IMAGE,
    final_image
)

# ============================================================
# RESULTS
# ============================================================

print("\n===== FINAL RESULTS =====\n")

print("LEFT REGION :", left_label)
print("LEFT BBOX   :", left_global)

print()

print("RIGHT REGION:", right_label)
print("RIGHT BBOX  :", right_global)

print()

print(f"Saved output to: {OUTPUT_IMAGE}")