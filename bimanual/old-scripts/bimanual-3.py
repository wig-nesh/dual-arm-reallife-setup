import os
import json
import time
import cv2
import numpy as np
import base64
import re

from openai import OpenAI

# ============================================================
# ROOT DATASET DIRECTORY
# ============================================================

DATASET_ROOT = "inputs"

# ============================================================
# OUTPUT DIRECTORY
# ============================================================

OUTPUT_ROOT = "outputs"

# ============================================================
# INPUT MODE
# ============================================================

INPUT_MODE = "mask"

# ============================================================
# GRID SETTINGS
# ============================================================

MIN_GRID_ROWS = 2
MIN_GRID_COLS = 2

MAX_GRID_ROWS = 8
MAX_GRID_COLS = 8

MIN_CELL_OBJECT_RATIO = 0.15

# ============================================================
# PHYSICAL GRIPPER WIDTH
# ============================================================

GRIPPER_WIDTH_M = 0.08

# ============================================================
# GRASP SETTINGS
# ============================================================

NUM_CANDIDATE_GRASPS = 2000

NUM_GRASP_HYPOTHESES = 5

BOX_SIZE = 40

# ============================================================
# VLM SETTINGS
# ============================================================

USE_REAL_VLM = True

VLM_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"

MAX_RETRIES = 5

# ============================================================
# GROQ CLIENT
# ============================================================

client = OpenAI(
    api_key=os.getenv("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1"
)

# ============================================================
# CREATE OUTPUT ROOT
# ============================================================

os.makedirs(
    OUTPUT_ROOT,
    exist_ok=True
)

# ============================================================
# HELPER FUNCTIONS
# ============================================================

def load_intrinsics(json_path):

    with open(json_path, "r") as f:

        data = json.load(f)

    # --------------------------------------------------------
    # Flexible parsing
    # --------------------------------------------------------

    if "fx" in data:

        fx = data["fx"]
        fy = data["fy"]

        cx = data["cx"]
        cy = data["cy"]

    elif "camera_matrix" in data:

        K = data["camera_matrix"]

        fx = K[0][0]
        fy = K[1][1]

        cx = K[0][2]
        cy = K[1][2]

    else:

        raise Exception(
            f"Unknown intrinsic format: {json_path}"
        )

    return fx, fy, cx, cy


def point_to_cell(
    cx,
    cy,
    cell_w,
    cell_h,
    grid_cols,
    grid_rows
):

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


def grasp_to_bbox(
    grasp,
    x_offset,
    y_offset,
    box_size
):

    cx = grasp["cx"] + x_offset
    cy = grasp["cy"] + y_offset

    half = box_size // 2

    return [

        int(cx - half),
        int(cy - half),

        int(cx + half),
        int(cy + half)
    ]


# ============================================================
# FIND OBJECT FOLDERS
# ============================================================

object_folders = sorted(

    [

        d for d in os.listdir(DATASET_ROOT)

        if os.path.isdir(
            os.path.join(DATASET_ROOT, d)
        )
    ]
)

# ============================================================
# PROCESS EACH OBJECT
# ============================================================

for object_name in object_folders:

    print("\n================================================")
    print(f"OBJECT: {object_name}")
    print("================================================")

    object_dir = os.path.join(
        DATASET_ROOT,
        object_name
    )

    output_object_dir = os.path.join(
        OUTPUT_ROOT,
        object_name
    )

    os.makedirs(
        output_object_dir,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Find RGB files
    # --------------------------------------------------------

    rgb_files = sorted(

        [

            f for f in os.listdir(object_dir)

            if f.startswith("rgb_")
        ]
    )

    # ========================================================
    # PROCESS EACH SCENE
    # ========================================================

    for rgb_file in rgb_files:

        print("\n------------------------------------------------")
        print(f"SCENE: {rgb_file}")
        print("------------------------------------------------")

        scene_id = rgb_file.replace(
            "rgb_",
            ""
        ).replace(".png", "")

        rgb_path = os.path.join(
            object_dir,
            rgb_file
        )

        mask_path = os.path.join(
            object_dir,
            f"mask_{scene_id}.png"
        )

        depth_path = os.path.join(
            object_dir,
            f"depth_{scene_id}.npy"
        )

        intrinsics_path = os.path.join(
            object_dir,
            f"intrinsics__{scene_id}.json"
        )

        required_files = [

            rgb_path,
            mask_path,
            depth_path,
            intrinsics_path
        ]

        missing = False

        for f in required_files:

            if not os.path.exists(f):

                print(f"Missing: {f}")
                missing = True

        if missing:
            continue

        # ====================================================
        # LOAD INPUTS
        # ====================================================

        scene = cv2.imread(rgb_path)

        segmentation = cv2.imread(mask_path)

        depth = np.load(depth_path)

        fx, fy, cx, cy = load_intrinsics(
            intrinsics_path
        )

        if scene is None:

            print("Failed RGB")
            continue

        if segmentation is None:

            print("Failed mask")
            continue

        # ----------------------------------------------------
        # Convert depth to meters if needed
        # ----------------------------------------------------

        if depth.dtype != np.float32:

            depth = depth.astype(np.float32)

        if np.max(depth) > 100:

            depth = depth / 1000.0

        # ====================================================
        # OBJECT EXTRACTION
        # ====================================================

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

                print("No contours")
                continue

            largest_contour = max(
                contours,
                key=cv2.contourArea
            )

            x, y, w, h = cv2.boundingRect(
                largest_contour
            )

            cropped_scene = scene[
                y:y+h,
                x:x+w
            ]

            cropped_mask = binary[
                y:y+h,
                x:x+w
            ]

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

                print("No contours")
                continue

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
                "INPUT_MODE invalid"
            )

        # ====================================================
        # PHYSICAL GRID GENERATION
        # ====================================================

        img_h, img_w = object_image.shape[:2]

        object_depths = depth[
            y:y+h,
            x:x+w
        ]

        valid_depths = object_depths[
            cropped_mask > 0
        ]

        valid_depths = valid_depths[
            valid_depths > 0
        ]

        if len(valid_depths) == 0:

            print("No valid depth")
            continue

        Z_object = np.median(
            valid_depths
        )

        obj_width_m = (
            w * Z_object
        ) / fx

        obj_height_m = (
            h * Z_object
        ) / fy

        grid_cols = int(
            np.round(
                obj_width_m
                /
                GRIPPER_WIDTH_M
            )
        )

        grid_rows = int(
            np.round(
                obj_height_m
                /
                GRIPPER_WIDTH_M
            )
        )

        grid_cols = np.clip(
            grid_cols,
            MIN_GRID_COLS,
            MAX_GRID_COLS
        )

        grid_rows = np.clip(
            grid_rows,
            MIN_GRID_ROWS,
            MAX_GRID_ROWS
        )

        print("\n===== GRID =====")

        print(f"Depth      : {Z_object:.3f} m")
        print(f"Width      : {obj_width_m:.3f} m")
        print(f"Height     : {obj_height_m:.3f} m")

        print(f"Rows       : {grid_rows}")
        print(f"Cols       : {grid_cols}")

        # ====================================================
        # GRID IMAGE
        # ====================================================

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

                if ratio >= MIN_CELL_OBJECT_RATIO:

                    valid_cells.append(idx)

                    cv2.rectangle(
                        grid_image,
                        (x1, y1),
                        (x2, y2),
                        (0, 255, 0),
                        1
                    )

                cell_mapping[idx] = {

                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2
                }

                idx += 1

        # ====================================================
        # SAVE GRID IMAGE
        # ====================================================

        grid_output_path = os.path.join(
            output_object_dir,
            f"grid_{scene_id}.png"
        )

        cv2.imwrite(
            grid_output_path,
            grid_image
        )

        # ====================================================
        # VLM PROMPT
        # ====================================================

        PROMPT = f"""
Output ONLY JSON.

Grid rows: {grid_rows}
Grid cols: {grid_cols}

Valid cells:
{valid_cells}

Generate {NUM_GRASP_HYPOTHESES}
stable bimanual grasp hypotheses.

The object must be lifted vertically upward.

Requirements:
- stable lifting
- balanced support
- symmetric support
- avoid weak structures
- avoid background

JSON format:

{{
  "grasps": [
    {{
      "left_cell": 1,
      "right_cell": 2
    }}
  ]
}}
"""

        # ====================================================
        # QUERY VLM
        # ====================================================

        if USE_REAL_VLM:

            success = False

            with open(
                grid_output_path,
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
                        max_tokens=120
                    )

                    response_text = (
                        response
                        .choices[0]
                        .message
                        .content
                    )

                    print("\n===== RAW RESPONSE =====")
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
                        r'\{[\s\S]*\}',
                        response_text
                    )

                    if json_match is None:

                        raise Exception(
                            "No JSON found"
                        )

                    response_text = json_match.group(0)

                    result = json.loads(
                        response_text
                    )

                    grasp_pairs = result["grasps"]

                    success = True

                    break

                except Exception as e:

                    print("\nVLM Error:")
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

                print("VLM failed.")
                continue

        else:

            grasp_pairs = [

                {
                    "left_cell": valid_cells[0],
                    "right_cell": valid_cells[-1]
                }
            ]

        # ====================================================
        # GENERATE CANDIDATE GRASPS
        # ====================================================

        candidate_grasps = []

        contours, _ = cv2.findContours(
            cropped_mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        largest_contour = max(
            contours,
            key=cv2.contourArea
        )

        for i in range(NUM_CANDIDATE_GRASPS):

            idx_rand = np.random.randint(
                0,
                len(largest_contour)
            )

            point = largest_contour[
                idx_rand
            ][0]

            px = int(point[0])
            py = int(point[1])

            offset_x = np.random.randint(
                -20,
                20
            )

            offset_y = np.random.randint(
                -20,
                20
            )

            cx_grasp = px + offset_x
            cy_grasp = py + offset_y

            cx_grasp = np.clip(
                cx_grasp,
                0,
                img_w - 1
            )

            cy_grasp = np.clip(
                cy_grasp,
                0,
                img_h - 1
            )

            if cropped_mask[
                cy_grasp,
                cx_grasp
            ] == 0:
                continue

            angle = np.random.uniform(
                -90,
                90
            )

            center_dist = abs(
                cx_grasp - (img_w / 2)
            ) / (img_w / 2)

            score = (
                0.6 * center_dist
                +
                0.4 * np.random.uniform(0, 1)
            )

            grasp = {

                "cx": cx_grasp,
                "cy": cy_grasp,
                "angle": angle,
                "score": score
            }

            candidate_grasps.append(grasp)

        # ====================================================
        # VISUALIZATION
        # ====================================================

        final_image = scene.copy()

        bbox_json = {

            "scene": scene_id,
            "object": object_name,
            "grasps": []
        }

        # ====================================================
        # PROCESS HYPOTHESES
        # ====================================================

        for pair_id, pair in enumerate(grasp_pairs):

            left_cell = pair["left_cell"]
            right_cell = pair["right_cell"]

            if left_cell not in valid_cells:
                continue

            if right_cell not in valid_cells:
                continue

            left_candidates = []
            right_candidates = []

            for grasp in candidate_grasps:

                cell_id = point_to_cell(

                    grasp["cx"],
                    grasp["cy"],

                    cell_w,
                    cell_h,

                    grid_cols,
                    grid_rows
                )

                if cell_id == left_cell:

                    left_candidates.append(
                        grasp
                    )

                if cell_id == right_cell:

                    right_candidates.append(
                        grasp
                    )

            if len(left_candidates) == 0:
                continue

            if len(right_candidates) == 0:
                continue

            best_left = max(
                left_candidates,
                key=lambda g: g["score"]
            )

            best_right = max(
                right_candidates,
                key=lambda g: g["score"]
            )

            left_bbox = grasp_to_bbox(

                best_left,
                x,
                y,
                BOX_SIZE
            )

            right_bbox = grasp_to_bbox(

                best_right,
                x,
                y,
                BOX_SIZE
            )

            # ------------------------------------------------
            # DRAW
            # ------------------------------------------------

            cv2.rectangle(

                final_image,

                (
                    left_bbox[0],
                    left_bbox[1]
                ),

                (
                    left_bbox[2],
                    left_bbox[3]
                ),

                (0, 255, 0),
                2
            )

            cv2.rectangle(

                final_image,

                (
                    right_bbox[0],
                    right_bbox[1]
                ),

                (
                    right_bbox[2],
                    right_bbox[3]
                ),

                (255, 0, 0),
                2
            )

            bbox_json["grasps"].append({

                "pair_id": pair_id,

                "left_cell": left_cell,
                "right_cell": right_cell,

                "left_bbox": left_bbox,
                "right_bbox": right_bbox
            })

        # ====================================================
        # SAVE VISUALIZATION
        # ====================================================

        vis_output_path = os.path.join(

            output_object_dir,

            f"grasps_{scene_id}.png"
        )

        cv2.imwrite(
            vis_output_path,
            final_image
        )

        # ====================================================
        # SAVE JSON
        # ====================================================

        json_output_path = os.path.join(

            output_object_dir,

            f"bbox_coords_{scene_id}.json"
        )

        with open(
            json_output_path,
            "w"
        ) as f:

            json.dump(
                bbox_json,
                f,
                indent=4
            )

print("\n================================================")
print("DONE")
print("================================================")