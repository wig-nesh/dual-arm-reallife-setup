import cv2
import numpy as np

# ============================================================
# EXTRACT OBJECT FROM MASK
# ============================================================

def extract_from_mask(
    scene,
    segmentation
):

    # --------------------------------------------------------
    # Convert mask to grayscale
    # --------------------------------------------------------

    gray = cv2.cvtColor(
        segmentation,
        cv2.COLOR_BGR2GRAY
    )

    # --------------------------------------------------------
    # Binary threshold
    # --------------------------------------------------------

    _, binary = cv2.threshold(

        gray,

        127,
        255,

        cv2.THRESH_BINARY
    )

    # --------------------------------------------------------
    # Find contours
    # --------------------------------------------------------

    contours, _ = cv2.findContours(

        binary,

        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if len(contours) == 0:

        raise Exception(
            "No contours found in mask"
        )

    # --------------------------------------------------------
    # Largest contour
    # --------------------------------------------------------

    largest_contour = max(
        contours,
        key=cv2.contourArea
    )

    # --------------------------------------------------------
    # Bounding rectangle
    # --------------------------------------------------------

    x, y, w, h = cv2.boundingRect(
        largest_contour
    )

    # --------------------------------------------------------
    # Crop scene
    # --------------------------------------------------------

    cropped_scene = scene[
        y:y+h,
        x:x+w
    ]

    # --------------------------------------------------------
    # Crop binary mask
    # --------------------------------------------------------

    cropped_mask = binary[
        y:y+h,
        x:x+w
    ]

    # --------------------------------------------------------
    # Extract object
    # --------------------------------------------------------

    object_image = cv2.bitwise_and(

        cropped_scene,
        cropped_scene,

        mask=cropped_mask
    )

    # --------------------------------------------------------
    # Return everything
    # --------------------------------------------------------

    result = {

        "object_image": object_image,

        "cropped_mask": cropped_mask,

        "bbox": {

            "x": x,
            "y": y,

            "w": w,
            "h": h
        }
    }

    return result

# ============================================================
# EXTRACT FROM ALREADY-CROPPED OBJECT
# ============================================================

def extract_from_extracted(
    segmentation
):

    # --------------------------------------------------------
    # Convert to grayscale
    # --------------------------------------------------------

    gray = cv2.cvtColor(
        segmentation,
        cv2.COLOR_BGR2GRAY
    )

    # --------------------------------------------------------
    # Threshold
    # --------------------------------------------------------

    _, binary = cv2.threshold(

        gray,

        1,
        255,

        cv2.THRESH_BINARY
    )

    # --------------------------------------------------------
    # Find contours
    # --------------------------------------------------------

    contours, _ = cv2.findContours(

        binary,

        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if len(contours) == 0:

        raise Exception(
            "No contours found"
        )

    # --------------------------------------------------------
    # Largest contour
    # --------------------------------------------------------

    largest_contour = max(
        contours,
        key=cv2.contourArea
    )

    # --------------------------------------------------------
    # Bounding box
    # --------------------------------------------------------

    x, y, w, h = cv2.boundingRect(
        largest_contour
    )

    # --------------------------------------------------------
    # Crop extracted object
    # --------------------------------------------------------

    object_image = segmentation[
        y:y+h,
        x:x+w
    ]

    # --------------------------------------------------------
    # Crop mask
    # --------------------------------------------------------

    cropped_mask = binary[
        y:y+h,
        x:x+w
    ]

    # --------------------------------------------------------
    # Return result
    # --------------------------------------------------------

    result = {

        "object_image": object_image,

        "cropped_mask": cropped_mask,

        "bbox": {

            "x": x,
            "y": y,

            "w": w,
            "h": h
        }
    }

    return result

# ============================================================
# MAIN EXTRACTION ROUTER
# ============================================================

def extract_object(
    input_mode,
    scene,
    segmentation
):

    if input_mode == "mask":

        return extract_from_mask(
            scene,
            segmentation
        )

    elif input_mode == "extracted":

        return extract_from_extracted(
            segmentation
        )

    else:

        raise Exception(

            "INPUT_MODE must be "
            "'mask' or 'extracted'"
        )

# ============================================================
# PRINT EXTRACTION INFO
# ============================================================

def print_extraction_info(result):

    bbox = result["bbox"]

    print("\n===== OBJECT EXTRACTION =====")

    print(f"x : {bbox['x']}")
    print(f"y : {bbox['y']}")

    print(f"w : {bbox['w']}")
    print(f"h : {bbox['h']}")