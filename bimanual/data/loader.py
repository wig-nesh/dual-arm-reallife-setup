import os
import cv2 # type: ignore
import numpy as np

# ============================================================
# FIND OBJECT FOLDERS
# ============================================================

def find_object_folders(dataset_root):

    object_folders = sorted(

        [

            d for d in os.listdir(dataset_root)

            if os.path.isdir(
                os.path.join(dataset_root, d)
            )
        ]
    )

    return object_folders

# ============================================================
# FIND SCENES INSIDE OBJECT FOLDER
# ============================================================

def find_scene_ids(object_dir):

    rgb_files = sorted(

        [

            f for f in os.listdir(object_dir)

            if f.startswith("rgb_")
        ]
    )

    scene_ids = []

    for f in rgb_files:

        scene_id = (
            f.replace("rgb_", "")
             .replace(".png", "")
        )

        scene_ids.append(scene_id)

    return scene_ids

# ============================================================
# BUILD FILE PATHS
# ============================================================

def build_scene_paths(
    object_dir,
    scene_id
):

    paths = {

        "rgb": os.path.join(
            object_dir,
            f"rgb_{scene_id}.png"
        ),

        "mask": os.path.join(
            object_dir,
            f"mask_{scene_id}.png"
        ),

        "depth": os.path.join(
            object_dir,
            f"depth_{scene_id}.npy"
        ),

        "intrinsics": os.path.join(
            object_dir,
            f"intrinsics__{scene_id}.json"
        )
    }

    return paths

# ============================================================
# VERIFY SCENE FILES
# ============================================================

def verify_scene_files(paths):

    for key, path in paths.items():

        if not os.path.exists(path):

            print(f"Missing {key}: {path}")

            return False

    return True

# ============================================================
# LOAD RGB IMAGE
# ============================================================

def load_rgb(path):

    image = cv2.imread(path)

    if image is None:

        raise Exception(
            f"Failed to load RGB: {path}"
        )

    return image

# ============================================================
# LOAD MASK IMAGE
# ============================================================

def load_mask(path):

    mask = cv2.imread(path)

    if mask is None:

        raise Exception(
            f"Failed to load mask: {path}"
        )

    return mask

# ============================================================
# LOAD DEPTH MAP
# ============================================================

def load_depth(path):

    depth = np.load(path)

    if depth is None:

        raise Exception(
            f"Failed to load depth: {path}"
        )

    # --------------------------------------------------------
    # Convert to float32
    # --------------------------------------------------------

    if depth.dtype != np.float32:

        depth = depth.astype(np.float32)

    # --------------------------------------------------------
    # Convert mm -> meters
    # if needed
    # --------------------------------------------------------

    if np.max(depth) > 100:

        depth = depth / 1000.0

    return depth

# ============================================================
# LOAD COMPLETE SCENE
# ============================================================

def load_scene(paths):

    scene = {

        "rgb": load_rgb(
            paths["rgb"]
        ),

        "mask": load_mask(
            paths["mask"]
        ),

        "depth": load_depth(
            paths["depth"]
        )
    }

    return scene