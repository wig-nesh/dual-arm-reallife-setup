# ============================================================
# POINT -> CELL
# ============================================================

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

    cell_id = (
        row * grid_cols
    ) + col + 1

    return int(cell_id)

# ============================================================
# FILTER INVALID VLM PAIRS
# ============================================================

def filter_grasp_pairs(

    grasp_pairs,
    valid_cells
):

    filtered_pairs = []

    for pair in grasp_pairs:

        left_cell = pair["left_cell"]
        right_cell = pair["right_cell"]

        # ----------------------------------------------------
        # Both cells must exist
        # ----------------------------------------------------

        if left_cell not in valid_cells:
            continue

        if right_cell not in valid_cells:
            continue

        # ----------------------------------------------------
        # Avoid identical cells
        # ----------------------------------------------------

        if left_cell == right_cell:
            continue

        filtered_pairs.append(pair)

    return filtered_pairs

# ============================================================
# FIND CANDIDATES INSIDE CELL
# ============================================================

def candidates_in_cell(

    candidate_grasps,

    target_cell,

    cell_w,
    cell_h,

    grid_cols,
    grid_rows
):

    selected = []

    for grasp in candidate_grasps:

        cell_id = point_to_cell(

            grasp["cx"],
            grasp["cy"],

            cell_w,
            cell_h,

            grid_cols,
            grid_rows
        )

        if cell_id == target_cell:

            selected.append(grasp)

    return selected

# ============================================================
# SELECT BEST GRASP
# ============================================================

def select_best_grasp(candidate_grasps):

    if len(candidate_grasps) == 0:

        return None

    best = max(

        candidate_grasps,

        key=lambda g: g["score"]
    )

    return best

# ============================================================
# FILTER SEMANTIC GRASPS
# ============================================================

def filter_semantic_grasps(

    grasp_pairs,

    candidate_grasps,

    valid_cells,

    cell_w,
    cell_h,

    grid_cols,
    grid_rows
):

    # --------------------------------------------------------
    # Remove invalid VLM pairs
    # --------------------------------------------------------

    grasp_pairs = filter_grasp_pairs(

        grasp_pairs,
        valid_cells
    )

    final_grasps = []

    # ========================================================
    # PROCESS SEMANTIC PAIRS
    # ========================================================

    for pair_id, pair in enumerate(grasp_pairs):

        left_cell = pair["left_cell"]
        right_cell = pair["right_cell"]

        # ----------------------------------------------------
        # LEFT candidates
        # ----------------------------------------------------

        left_candidates = candidates_in_cell(

            candidate_grasps,

            left_cell,

            cell_w,
            cell_h,

            grid_cols,
            grid_rows
        )

        # ----------------------------------------------------
        # RIGHT candidates
        # ----------------------------------------------------

        right_candidates = candidates_in_cell(

            candidate_grasps,

            right_cell,

            cell_w,
            cell_h,

            grid_cols,
            grid_rows
        )

        # ----------------------------------------------------
        # Reject invalid cells
        # ----------------------------------------------------

        if len(left_candidates) == 0:
            continue

        if len(right_candidates) == 0:
            continue

        # ----------------------------------------------------
        # Best grasps
        # ----------------------------------------------------

        best_left = select_best_grasp(
            left_candidates
        )

        best_right = select_best_grasp(
            right_candidates
        )

        if best_left is None:
            continue

        if best_right is None:
            continue

        result = {

            "pair_id": pair_id,

            "left_cell": left_cell,
            "right_cell": right_cell,

            "left_grasp": best_left,
            "right_grasp": best_right
        }

        final_grasps.append(result)

    # ========================================================
    # GUARANTEE MINIMUM 1 GRASP
    # ========================================================

    if len(final_grasps) == 0:

        if len(candidate_grasps) >= 2:

            sorted_candidates = sorted(

                candidate_grasps,

                key=lambda g: g["score"],

                reverse=True
            )

            fallback = {

                "pair_id": 0,

                "left_cell": -1,
                "right_cell": -1,

                "left_grasp": sorted_candidates[0],
                "right_grasp": sorted_candidates[1]
            }

            final_grasps.append(
                fallback
            )

    # ========================================================
    # SORT BY SCORE
    # ========================================================

    final_grasps = sorted(

        final_grasps,

        key=lambda g:
        (
            g["left_grasp"]["score"]
            +
            g["right_grasp"]["score"]
        ),

        reverse=True
    )

    # ========================================================
    # KEEP TOP 5
    # ========================================================

    final_grasps = final_grasps[:5]

    return final_grasps

# ============================================================
# PRINT FILTERING INFO
# ============================================================

def print_filtering_info(

    grasp_pairs,
    final_grasps
):

    print("\n===== GRASP FILTERING =====")

    print(
        f"Semantic pairs : "
        f"{len(grasp_pairs)}"
    )

    print(
        f"Final grasps   : "
        f"{len(final_grasps)}"
    )