import cv2
import os

# ============================================================
# DRAW SINGLE BBOX
# ============================================================

def draw_bbox(

    image,

    bbox,

    color,

    thickness=2
):

    x1, y1, x2, y2 = bbox

    cv2.rectangle(

        image,

        (x1, y1),
        (x2, y2),

        color,
        thickness
    )

# ============================================================
# DRAW LABEL
# ============================================================

def draw_label(

    image,

    text,

    bbox,

    color,

    font_scale=0.6,
    thickness=2
):

    x1, y1, _, _ = bbox

    cv2.putText(

        image,

        text,

        (
            x1,
            max(0, y1 - 10)
        ),

        cv2.FONT_HERSHEY_SIMPLEX,

        font_scale,

        color,

        thickness
    )

# ============================================================
# DRAW BBOX PAIR
# ============================================================

def draw_grasp_pair(

    image,

    left_bbox,
    right_bbox,

    left_color=(0,255,0),
    right_color=(255,0,0),

    thickness=2
):

    # --------------------------------------------------------
    # LEFT
    # --------------------------------------------------------

    draw_bbox(

        image,

        left_bbox,

        left_color,

        thickness
    )

    draw_label(

        image,

        "LEFT",

        left_bbox,

        left_color
    )

    # --------------------------------------------------------
    # RIGHT
    # --------------------------------------------------------

    draw_bbox(

        image,

        right_bbox,

        right_color,

        thickness
    )

    draw_label(

        image,

        "RIGHT",

        right_bbox,

        right_color
    )

# ============================================================
# DRAW ALL GRASP PAIRS
# ============================================================

def draw_all_grasps(

    scene,

    bbox_results,

    left_color=(0,255,0),
    right_color=(255,0,0),

    thickness=2
):

    image = scene.copy()

    for result in bbox_results:

        draw_grasp_pair(

            image,

            result["left_bbox"],
            result["right_bbox"],

            left_color,
            right_color,

            thickness
        )

    return image

# ============================================================
# DRAW INDIVIDUAL GRASP PAIRS
# ============================================================

def draw_individual_grasps(

    scene,

    bbox_results,

    output_dir,

    scene_id,

    left_color=(0,255,0),
    right_color=(255,0,0),

    thickness=2
):

    os.makedirs(
        output_dir,
        exist_ok=True
    )

    saved_paths = []

    for result in bbox_results:

        image = scene.copy()

        draw_grasp_pair(

            image,

            result["left_bbox"],
            result["right_bbox"],

            left_color,
            right_color,

            thickness
        )

        pair_id = result["pair_id"]

        output_path = os.path.join(

            output_dir,

            f"grasp_pair_{scene_id}_{pair_id}.png"
        )

        cv2.imwrite(
            output_path,
            image
        )

        saved_paths.append(
            output_path
        )

    return saved_paths

# ============================================================
# SAVE GRID IMAGE
# ============================================================

def save_grid_image(

    grid_image,
    output_path
):

    cv2.imwrite(
        output_path,
        grid_image
    )

# ============================================================
# SAVE COMBINED VISUALIZATION
# ============================================================

def save_combined_visualization(

    scene,

    bbox_results,

    output_path
):

    image = draw_all_grasps(

        scene,
        bbox_results
    )

    cv2.imwrite(
        output_path,
        image
    )

# ============================================================
# PRINT VIS INFO
# ============================================================

def print_visualization_info(

    grid_path,

    grasp_paths
):

    print("\n===== VISUALIZATION =====")

    print(
        f"Grid Image:\n"
        f"{grid_path}"
    )

    print()

    print(
        f"Saved "
        f"{len(grasp_paths)} "
        f"grasp visualizations"
    )