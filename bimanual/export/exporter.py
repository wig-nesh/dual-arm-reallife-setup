import os
import json

# ============================================================
# CREATE OUTPUT DIRECTORY
# ============================================================

def create_output_dir(path):

    os.makedirs(
        path,
        exist_ok=True
    )

# ============================================================
# BUILD OBJECT OUTPUT DIR
# ============================================================

def build_object_output_dir(

    output_root,
    object_name
):

    output_dir = os.path.join(

        output_root,
        object_name
    )

    create_output_dir(
        output_dir
    )

    return output_dir

# ============================================================
# SAVE JSON
# ============================================================

def save_json(

    data,
    output_path
):

    with open(
        output_path,
        "w"
    ) as f:

        json.dump(

            data,
            f,

            indent=4
        )

# ============================================================
# BUILD BBOX JSON
# ============================================================

def build_bbox_json(

    object_name,
    scene_id,

    bbox_results
):

    result = {

        "object": object_name,

        "scene": scene_id,

        "num_grasps": len(
            bbox_results
        ),

        "grasps": []
    }

    for grasp in bbox_results:

        grasp_json = {

            "pair_id":
            grasp["pair_id"],

            "left_cell":
            grasp["left_cell"],

            "right_cell":
            grasp["right_cell"],

            "left_bbox":
            grasp["left_bbox"],

            "right_bbox":
            grasp["right_bbox"]
        }

        result["grasps"].append(
            grasp_json
        )

    return result

# ============================================================
# EXPORT SCENE RESULTS
# ============================================================

def export_scene_results(

    output_dir,

    object_name,
    scene_id,

    bbox_results
):

    bbox_json = build_bbox_json(

        object_name,
        scene_id,

        bbox_results
    )

    json_output_path = os.path.join(

        output_dir,

        f"bbox_coords_{scene_id}.json"
    )

    save_json(

        bbox_json,

        json_output_path
    )

    return json_output_path

# ============================================================
# PRINT EXPORT INFO
# ============================================================

def print_export_info(

    json_output_path
):

    print("\n===== EXPORT =====")

    print(
        f"Saved JSON:\n"
        f"{json_output_path}"
    )