import os
import json
import cv2
import numpy as np
import time

from google import genai
from PIL import Image

#config
SCENE_IMAGE = "inputs/scene.png"
SEGMENTATION_MASK = "inputs/segmentation.png"

OUTPUT_IMAGE = "outputs/final_result.png"
TEMP_GRID_IMAGE = "temp/grid.png"

GRID_ROWS = 4
GRID_COLS = 4

GEMINI_MODEL = "gemini-2.5-flash"

USE_REAL_VLM = True

#gemini setup
client = genai.Client(
    api_key=os.getenv("GEMINI_API_KEY")
)

PROMPT = """
You are a robotic grasp planner.

The robot must lift this object vertically upward
using two robotic grippers.

Your task:
Generate TWO independent grasp bounding boxes:
1. left arm grasp region
2. right arm grasp region

Requirements:
- stable lifting
- balanced lifting
- symmetric support
- strong grasp regions
- avoid weak or thin structures
- avoid empty background

Bounding box format:
[x1, y1, x2, y2]

Rules:
- x1,y1 = top-left corner
- x2,y2 = bottom-right corner
- coordinates MUST be within image boundaries
- coordinates are in pixel units
- left and right boxes must be different
- boxes should tightly cover graspable regions

Return ONLY raw JSON.
No markdown.
No explanations.

Use EXACTLY this format:

{
    "left_arm_bbox": [x1, y1, x2, y2],
    "right_arm_bbox": [x1, y1, x2, y2]
}
"""

#output dirs
os.makedirs("outputs", exist_ok=True)
os.makedirs("temp", exist_ok=True)

#input load
scene = cv2.imread(SCENE_IMAGE)

if scene is None:
    raise Exception(f"Could not load scene image: {SCENE_IMAGE}")

mask = cv2.imread(SEGMENTATION_MASK, 0)

if mask is None:
    raise Exception(f"Could not load segmentation mask: {SEGMENTATION_MASK}")

#object extraction
_, binary = cv2.threshold(
    mask,
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
    raise Exception("No object found in segmentation mask")

largest_contour = max(contours, key=cv2.contourArea)

x, y, w, h = cv2.boundingRect(largest_contour)

cropped_scene = scene[y:y+h, x:x+w]
cropped_mask = binary[y:y+h, x:x+w]

object_image = cv2.bitwise_and(
    cropped_scene,
    cropped_scene,
    mask=cropped_mask
)

#temp object image save
cv2.imwrite(TEMP_GRID_IMAGE, object_image)

#vlm query
if USE_REAL_VLM:

    import time

    MAX_RETRIES = 5

    response = None

    for attempt in range(MAX_RETRIES):

        try:

            print(f"\nAttempt {attempt+1}/{MAX_RETRIES}")

            pil_image = Image.open(TEMP_GRID_IMAGE)

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
    
#bbox extract
left_bbox = result["left_arm_bbox"]
right_bbox = result["right_arm_bbox"]

#box validation
def validate_and_clip_bbox(bbox, img_w, img_h):

    if len(bbox) != 4:
        raise Exception(f"Invalid bbox length: {bbox}")

    x1, y1, x2, y2 = bbox

    # --------------------------------------------------------
    # CLIP TO IMAGE BOUNDARIES
    # --------------------------------------------------------

    x1 = max(0, min(x1, img_w - 1))
    y1 = max(0, min(y1, img_h - 1))

    x2 = max(0, min(x2, img_w - 1))
    y2 = max(0, min(y2, img_h - 1))

    # --------------------------------------------------------
    # ENSURE VALID ORDERING
    # --------------------------------------------------------

    if x2 <= x1:
        x2 = x1 + 1

    if y2 <= y1:
        y2 = y1 + 1

    return [x1, y1, x2, y2]

img_h, img_w = object_image.shape[:2]

left_bbox = validate_and_clip_bbox(
    left_bbox,
    img_w,
    img_h
)

right_bbox = validate_and_clip_bbox(
    right_bbox,
    img_w,
    img_h
)

#draw bbox finally
final_image = scene.copy()

\
left_global = [  #left arm bbox
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


cv2.rectangle(
    final_image,
    (left_global[0], left_global[1]),
    (left_global[2], left_global[3]),
    (0, 255, 0),
    5
)

#label
cv2.putText(
    final_image,
    "LEFT ARM",
    (left_global[0], left_global[1] - 10),
    cv2.FONT_HERSHEY_SIMPLEX,
    1,
    (0, 255, 0),
    2
)


cv2.rectangle(  #right arm bbox
    final_image,
    (right_global[0], right_global[1]),
    (right_global[2], right_global[3]),
    (255, 0, 0),
    5
)

#label
cv2.putText(
    final_image,
    "RIGHT ARM",
    (right_global[0], right_global[1] - 10),
    cv2.FONT_HERSHEY_SIMPLEX,
    1,
    (255, 0, 0),
    2
)

#output save
cv2.imwrite(
    OUTPUT_IMAGE,
    final_image
)

#results
print("\n===== FINAL RESULTS =====\n")

print(f"LEFT ARM BBOX  : {left_global}")
print(f"RIGHT ARM BBOX : {right_global}")

print()

print(f"Saved result to: {OUTPUT_IMAGE}")