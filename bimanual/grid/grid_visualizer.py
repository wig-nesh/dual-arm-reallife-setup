import cv2

# ============================================================
# DRAW GRID OVERLAY
# ============================================================

def draw_grid(

    object_image,

    cell_mapping,
    valid_cells,

    color=(0,255,0),

    thickness=1
):

    grid_image = object_image.copy()

    for cell_id in valid_cells:

        cell = cell_mapping[cell_id]

        cv2.rectangle(

            grid_image,

            (
                cell["x1"],
                cell["y1"]
            ),

            (
                cell["x2"],
                cell["y2"]
            ),

            color,
            thickness
        )

    return grid_image

# ============================================================
# PRINT GRID VIS INFO
# ============================================================

def print_grid_visualization_info(

    rows,
    cols,

    num_valid_cells
):

    print("\n===== GRID VISUALIZATION =====")

    print(f"Rows : {rows}")
    print(f"Cols : {cols}")

    print(
        f"Valid Cells : "
        f"{num_valid_cells}"
    )