import json

# ============================================================
# LOAD CAMERA INTRINSICS
# ============================================================

def load_intrinsics(json_path):

    with open(json_path, "r") as f:

        data = json.load(f)

    # --------------------------------------------------------
    # FORMAT 1
    #
    # {
    #   "fx": ...
    #   "fy": ...
    #   "cx": ...
    #   "cy": ...
    # }
    # --------------------------------------------------------

    if "fx" in data:

        fx = data["fx"]
        fy = data["fy"]

        cx = data["cx"]
        cy = data["cy"]

    # --------------------------------------------------------
    # FORMAT 2
    #
    # {
    #   "camera_matrix":
    #   [
    #       [fx, 0, cx],
    #       [0, fy, cy],
    #       [0, 0, 1]
    #   ]
    # }
    # --------------------------------------------------------

    elif "camera_matrix" in data:

        K = data["camera_matrix"]

        fx = K[0][0]
        fy = K[1][1]

        cx = K[0][2]
        cy = K[1][2]

    else:

        raise Exception(

            f"Unknown intrinsic format:\n"
            f"{json_path}"
        )

    intrinsics = {

        "fx": fx,
        "fy": fy,

        "cx": cx,
        "cy": cy
    }

    return intrinsics

# ============================================================
# PRINT INTRINSICS
# ============================================================

def print_intrinsics(intrinsics):

    print("\n===== CAMERA INTRINSICS =====")

    print(f"fx : {intrinsics['fx']}")
    print(f"fy : {intrinsics['fy']}")

    print(f"cx : {intrinsics['cx']}")
    print(f"cy : {intrinsics['cy']}")