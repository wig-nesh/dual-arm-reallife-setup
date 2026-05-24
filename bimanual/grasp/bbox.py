# ============================================================
# GRASP -> BBOX
# ============================================================

def grasp_to_bbox(

    grasp,

    x_offset,
    y_offset,

    box_size
):

    cx = grasp["cx"] + x_offset
    cy = grasp["cy"] + y_offset

    half = box_size // 2

    bbox = [

        int(cx - half),
        int(cy - half),

        int(cx + half),
        int(cy + half)
    ]

    return bbox

# ============================================================
# CONVERT FILTERED GRASPS TO BBOXES
# ============================================================

def grasps_to_bboxes(

    filtered_grasps,

    bbox,

    box_size
):

    x_offset = bbox["x"]
    y_offset = bbox["y"]

    results = []

    for grasp_pair in filtered_grasps:

        left_bbox = grasp_to_bbox(

            grasp_pair["left_grasp"],

            x_offset,
            y_offset,

            box_size
        )

        right_bbox = grasp_to_bbox(

            grasp_pair["right_grasp"],

            x_offset,
            y_offset,

            box_size
        )

        result = {

            "pair_id": grasp_pair["pair_id"],

            "left_cell":
            grasp_pair["left_cell"],

            "right_cell":
            grasp_pair["right_cell"],

            "left_bbox": left_bbox,

            "right_bbox": right_bbox
        }

        results.append(result)

    return results

# ============================================================
# VALIDATE BBOX
# ============================================================

def validate_bbox(

    bbox,

    image_width,
    image_height
):

    if len(bbox) != 4:
        return False

    x1, y1, x2, y2 = bbox

    if x1 < 0:
        return False

    if y1 < 0:
        return False

    if x2 > image_width:
        return False

    if y2 > image_height:
        return False

    if x2 <= x1:
        return False

    if y2 <= y1:
        return False

    return True

# ============================================================
# VALIDATE ALL BBOXES
# ============================================================

def validate_all_bboxes(

    bbox_results,

    image_width,
    image_height
):

    filtered = []

    for result in bbox_results:

        left_ok = validate_bbox(

            result["left_bbox"],

            image_width,
            image_height
        )

        right_ok = validate_bbox(

            result["right_bbox"],

            image_width,
            image_height
        )

        if not left_ok:
            continue

        if not right_ok:
            continue

        filtered.append(result)

    return filtered

# ============================================================
# PRINT BBOX INFO
# ============================================================

def print_bbox_info(bbox_results):

    print("\n===== FINAL BBOXES =====")

    print(
        f"Generated: "
        f"{len(bbox_results)}"
    )

    for result in bbox_results:

        print()

        print(
            f"Pair ID : "
            f"{result['pair_id']}"
        )

        print(
            f"Left Cell : "
            f"{result['left_cell']}"
        )

        print(
            f"Right Cell : "
            f"{result['right_cell']}"
        )

        print(
            f"Left BBOX : "
            f"{result['left_bbox']}"
        )

        print(
            f"Right BBOX : "
            f"{result['right_bbox']}"
        )