import numpy as np
import cv2

# ============================================================
# GENERATE DENSE CANDIDATE GRASPS
# ============================================================

def generate_candidate_grasps(

    cropped_mask,

    object_image,

    num_candidates=2000
):

    img_h, img_w = object_image.shape[:2]

    candidate_grasps = []

    # --------------------------------------------------------
    # Find object contour
    # --------------------------------------------------------

    contours, _ = cv2.findContours(

        cropped_mask,

        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if len(contours) == 0:

        raise Exception(
            "No contours found for grasp generation"
        )

    largest_contour = max(
        contours,
        key=cv2.contourArea
    )

    # --------------------------------------------------------
    # Generate random candidates
    # --------------------------------------------------------

    for i in range(num_candidates):

        idx_rand = np.random.randint(

            0,
            len(largest_contour)
        )

        point = largest_contour[
            idx_rand
        ][0]

        px = int(point[0])
        py = int(point[1])

        # ----------------------------------------------------
        # Random local offsets
        # ----------------------------------------------------

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

        # ----------------------------------------------------
        # Clamp
        # ----------------------------------------------------

        cx = np.clip(
            cx,
            0,
            img_w - 1
        )

        cy = np.clip(
            cy,
            0,
            img_h - 1
        )

        # ----------------------------------------------------
        # Must lie on object
        # ----------------------------------------------------

        if cropped_mask[cy, cx] == 0:
            continue

        # ----------------------------------------------------
        # Random orientation
        # ----------------------------------------------------

        angle = np.random.uniform(
            -90,
            90
        )

        # ----------------------------------------------------
        # Heuristic score
        #
        # Favor side grasps over center grasps
        # ----------------------------------------------------

        center_dist = abs(
            cx - (img_w / 2)
        ) / (img_w / 2)

        score = (
            0.6 * center_dist
            +
            0.4 * np.random.uniform(0, 1)
        )

        grasp = {

            "cx": int(cx),
            "cy": int(cy),

            "angle": float(angle),

            "score": float(score)
        }

        candidate_grasps.append(
            grasp
        )

    return candidate_grasps

# ============================================================
# FILTER INVALID CANDIDATES
# ============================================================

def filter_boundary_grasps(

    candidate_grasps,

    image_width,
    image_height,

    box_size
):

    filtered = []

    half = box_size // 2

    for grasp in candidate_grasps:

        cx = grasp["cx"]
        cy = grasp["cy"]

        # ----------------------------------------------------
        # Reject grasps too close to image boundaries
        # ----------------------------------------------------

        if cx < half:
            continue

        if cy < half:
            continue

        if cx >= image_width - half:
            continue

        if cy >= image_height - half:
            continue

        filtered.append(grasp)

    return filtered

# ============================================================
# PRINT GRASP INFO
# ============================================================

def print_candidate_info(candidate_grasps):

    print("\n===== CANDIDATE GRASPS =====")

    print(
        f"Generated: "
        f"{len(candidate_grasps)}"
    )