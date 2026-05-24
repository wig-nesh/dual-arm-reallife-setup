import cv2
import numpy as np

# ============================================================
# COMPUTE OBJECT PHYSICAL SIZE
# ============================================================

def compute_object_physical_size(

    depth,
    cropped_mask,

    bbox,

    fx,
    fy
):

    x = bbox["x"]
    y = bbox["y"]

    w = bbox["w"]
    h = bbox["h"]

    # --------------------------------------------------------
    # Crop depth to object bbox
    # --------------------------------------------------------

    object_depths = depth[
        y:y+h,
        x:x+w
    ]

    # --------------------------------------------------------
    # Use only object pixels
    # --------------------------------------------------------

    valid_depths = object_depths[
        cropped_mask > 0
    ]

    # --------------------------------------------------------
    # Remove invalid depth
    # --------------------------------------------------------

    valid_depths = valid_depths[
        valid_depths > 0
    ]

    if len(valid_depths) == 0:

        raise Exception(
            "No valid object depth values"
        )

    # --------------------------------------------------------
    # Median object depth
    # --------------------------------------------------------

    Z_object = np.median(
        valid_depths
    )

    # --------------------------------------------------------
    # Physical size
    # --------------------------------------------------------

    obj_width_m = (
        w * Z_object
    ) / fx

    obj_height_m = (
        h * Z_object
    ) / fy

    result = {

        "depth_m": Z_object,

        "width_m": obj_width_m,
        "height_m": obj_height_m
    }

    return result

# ============================================================
# COMPUTE PHYSICAL GRID SIZE
# ============================================================

def compute_grid_size(

    object_size,

    gripper_width_m,

    min_rows,
    min_cols,

    max_rows,
    max_cols
):

    obj_width_m = object_size["width_m"]
    obj_height_m = object_size["height_m"]

    # --------------------------------------------------------
    # Cell size ≈ gripper width
    # --------------------------------------------------------

    grid_cols = int(

        np.round(
            obj_width_m
            /
            gripper_width_m
        )
    )

    grid_rows = int(

        np.round(
            obj_height_m
            /
            gripper_width_m
        )
    )

    # --------------------------------------------------------
    # Clamp limits
    # --------------------------------------------------------

    grid_cols = np.clip(

        grid_cols,

        min_cols,
        max_cols
    )

    grid_rows = np.clip(

        grid_rows,

        min_rows,
        max_rows
    )

    result = {

        "rows": int(grid_rows),
        "cols": int(grid_cols)
    }

    return result

# ============================================================
# GENERATE GRID CELLS
# ============================================================

def generate_grid_cells(

    object_image,
    cropped_mask,

    grid_rows,
    grid_cols,

    min_cell_object_ratio
):

    img_h, img_w = object_image.shape[:2]

    # --------------------------------------------------------
    # Cell size
    # --------------------------------------------------------

    cell_w = img_w // grid_cols
    cell_h = img_h // grid_rows

    cell_mapping = {}

    valid_cells = []

    idx = 1

    for r in range(grid_rows):

        for c in range(grid_cols):

            x1 = c * cell_w
            y1 = r * cell_h

            # ------------------------------------------------
            # Handle boundary cells
            # ------------------------------------------------

            if c == grid_cols - 1:
                x2 = img_w
            else:
                x2 = x1 + cell_w

            if r == grid_rows - 1:
                y2 = img_h
            else:
                y2 = y1 + cell_h

            # ------------------------------------------------
            # Cell mask
            # ------------------------------------------------

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

            # ------------------------------------------------
            # Valid semantic cell
            # ------------------------------------------------

            if ratio >= min_cell_object_ratio:

                valid_cells.append(idx)

            # ------------------------------------------------
            # Store mapping
            # ------------------------------------------------

            cell_mapping[idx] = {

                "x1": x1,
                "y1": y1,

                "x2": x2,
                "y2": y2
            }

            idx += 1

    result = {

        "cell_w": cell_w,
        "cell_h": cell_h,

        "valid_cells": valid_cells,

        "cell_mapping": cell_mapping
    }

    return result


# ============================================================
# COMPLETE PHYSICAL GRID PIPELINE
# ============================================================

def generate_physical_grid(

    object_image,
    cropped_mask,

    depth,
    bbox,

    fx,
    fy,

    gripper_width_m,

    min_rows,
    min_cols,

    max_rows,
    max_cols,

    min_cell_object_ratio
):

    # --------------------------------------------------------
    # Physical object size
    # --------------------------------------------------------

    object_size = compute_object_physical_size(

        depth,
        cropped_mask,

        bbox,

        fx,
        fy
    )

    # --------------------------------------------------------
    # Grid size
    # --------------------------------------------------------

    grid_size = compute_grid_size(

        object_size,

        gripper_width_m,

        min_rows,
        min_cols,

        max_rows,
        max_cols
    )

    # --------------------------------------------------------
    # Generate cells
    # --------------------------------------------------------

    cells = generate_grid_cells(

        object_image,
        cropped_mask,

        grid_size["rows"],
        grid_size["cols"],

        min_cell_object_ratio
    )



    result = {


        "rows": grid_size["rows"],
        "cols": grid_size["cols"],

        "cell_w": cells["cell_w"],
        "cell_h": cells["cell_h"],

        "valid_cells": cells["valid_cells"],

        "cell_mapping": cells["cell_mapping"],

        "object_depth_m": object_size["depth_m"],

        "object_width_m": object_size["width_m"],
        "object_height_m": object_size["height_m"]
    }

    return result

# ============================================================
# PRINT GRID INFO
# ============================================================

def print_grid_info(grid_result):

    print("\n===== PHYSICAL GRID =====")

    print(
        f"Depth  : "
        f"{grid_result['object_depth_m']:.3f} m"
    )

    print(
        f"Width  : "
        f"{grid_result['object_width_m']:.3f} m"
    )

    print(
        f"Height : "
        f"{grid_result['object_height_m']:.3f} m"
    )

    print(
        f"Rows   : "
        f"{grid_result['rows']}"
    )

    print(
        f"Cols   : "
        f"{grid_result['cols']}"
    )

    print(
        f"Valid Cells : "
        f"{len(grid_result['valid_cells'])}"
    )