import numpy as np

# ============================================================
# EUCLIDEAN DISTANCE
# ============================================================

def euclidean_distance(

    x1, y1,
    x2, y2
):

    dist = np.sqrt(

        (x2 - x1) ** 2
        +
        (y2 - y1) ** 2
    )

    return float(dist)

# ============================================================
# BBOX CENTER
# ============================================================

def bbox_center(bbox):

    x1, y1, x2, y2 = bbox

    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2

    return cx, cy

# ============================================================
# CLAMP VALUE
# ============================================================

def clamp(

    value,

    min_value,
    max_value
):

    return max(

        min_value,

        min(
            value,
            max_value
        )
    )

# ============================================================
# NORMALIZE VALUE
# ============================================================

def normalize(

    value,

    min_value,
    max_value
):

    if max_value == min_value:
        return 0.0

    return (

        value - min_value
    ) / (

        max_value - min_value
    )

# ============================================================
# IMAGE CENTER
# ============================================================

def image_center(

    image_width,
    image_height
):

    cx = image_width / 2
    cy = image_height / 2

    return cx, cy

# ============================================================
# POINT INSIDE BBOX
# ============================================================

def point_inside_bbox(

    px,
    py,

    bbox
):

    x1, y1, x2, y2 = bbox

    inside = (

        px >= x1
        and
        px <= x2
        and
        py >= y1
        and
        py <= y2
    )

    return inside

# ============================================================
# BBOX AREA
# ============================================================

def bbox_area(bbox):

    x1, y1, x2, y2 = bbox

    width = x2 - x1
    height = y2 - y1

    area = width * height

    return area