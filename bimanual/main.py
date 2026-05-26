import os

# ============================================================
# CONFIG
# ============================================================

from config.config import *

# ============================================================
# DATA
# ============================================================

from data.loader import (
    find_object_folders,
    find_scene_ids,
    build_scene_paths,
    verify_scene_files,
    load_scene
)

from data.intrinsics import (
    load_intrinsics,
    print_intrinsics
)

# ============================================================
# SEGMENTATION
# ============================================================

from segmentation.extractor import (
    extract_object,
    print_extraction_info
)

# ============================================================
# GRID
# ============================================================

from grid.physical_grid import (
    generate_physical_grid,
    print_grid_info
)

from grid.grid_visualizer import (
    draw_grid,
    print_grid_visualization_info
)

# ============================================================
# VLM
# ============================================================

from vlm.prompt import (
    build_prompt
)

from vlm.groq_client import (
    create_groq_client,
    query_vlm
)

from vlm.parser import (
    validate_grasp_response,
    print_parsed_response
)

# ============================================================
# GRASP
# ============================================================

from grasp.candidates import (
    generate_candidate_grasps,
    filter_boundary_grasps,
    print_candidate_info
)

from grasp.filtering import (
    filter_semantic_grasps,
    print_filtering_info
)

from grasp.bbox import (
    grasps_to_bboxes,
    validate_all_bboxes,
    print_bbox_info
)

# ============================================================
# VISUALIZATION
# ============================================================

from visualization.draw import (
    save_grid_image,
    save_combined_visualization
)

# ============================================================
# EXPORT
# ============================================================

from export.exporter import (
    build_object_output_dir,
    export_scene_results,
    print_export_info
)

# ============================================================
# CREATE VLM CLIENT
# ============================================================

client = create_groq_client(
    os.getenv("GEMINI_API_KEY")
)

# ============================================================
# FIND OBJECTS
# ============================================================

object_folders = find_object_folders(
    DATASET_ROOT
)

# ============================================================
# PROCESS OBJECTS
# ============================================================

for object_name in object_folders:

    print("\n================================================")
    print(f"OBJECT: {object_name}")
    print("================================================")

    object_dir = os.path.join(
        DATASET_ROOT,
        object_name
    )

    output_dir = build_object_output_dir(

        OUTPUT_ROOT,
        object_name
    )

    scene_ids = find_scene_ids(
        object_dir
    )

    # ========================================================
    # PROCESS SCENES
    # ========================================================

    for scene_id in scene_ids:

        print("\n------------------------------------------------")
        print(f"SCENE: {scene_id}")
        print("------------------------------------------------")

        try:

            # =================================================
            # PATHS
            # =================================================

            paths = build_scene_paths(

                object_dir,
                scene_id
            )

            if not verify_scene_files(paths):

                continue

            # =================================================
            # LOAD DATA
            # =================================================

            scene_data = load_scene(
                paths
            )

            scene = scene_data["rgb"]

            segmentation = scene_data["mask"]

            depth = scene_data["depth"]

            # =================================================
            # INTRINSICS
            # =================================================

            intrinsics = load_intrinsics(
                paths["intrinsics"]
            )

            print_intrinsics(
                intrinsics
            )

            # =================================================
            # OBJECT EXTRACTION
            # =================================================

            extraction = extract_object(

                INPUT_MODE,

                scene,
                segmentation
            )

            print_extraction_info(
                extraction
            )

            object_image = extraction[
                "object_image"
            ]

            cropped_mask = extraction[
                "cropped_mask"
            ]

            bbox = extraction[
                "bbox"
            ]

            # =================================================
            # PHYSICAL GRID
            # =================================================

            grid_result = generate_physical_grid(

                object_image,
                cropped_mask,

                depth,
                bbox,

                intrinsics["fx"],
                intrinsics["fy"],

                GRIPPER_WIDTH_M,

                MIN_GRID_ROWS,
                MIN_GRID_COLS,

                MAX_GRID_ROWS,
                MAX_GRID_COLS,

                MIN_CELL_OBJECT_RATIO
            )

            print_grid_info(
                grid_result
            )

            # =================================================
            # GRID VISUALIZATION
            # =================================================

            grid_image = draw_grid(

                object_image,

                grid_result["cell_mapping"],
                grid_result["valid_cells"],

                thickness=GRID_LINE_THICKNESS
            )

            print_grid_visualization_info(

                grid_result["rows"],
                grid_result["cols"],

                len(grid_result["valid_cells"])
            )

            # =================================================
            # SAVE GRID
            # =================================================

            grid_output_path = os.path.join(

                output_dir,

                f"grid_{scene_id}.png"
            )

            save_grid_image(

                grid_image,

                grid_output_path
            )

            # =================================================
            # PROMPT
            # =================================================

            prompt = build_prompt(

                grid_result["rows"],
                grid_result["cols"],

                grid_result["valid_cells"],

                NUM_GRASP_HYPOTHESES
            )

            # =================================================
            # QUERY VLM
            # =================================================

            result = query_vlm(

                client,

                VLM_MODEL,

                grid_output_path,

                prompt,

                TEMPERATURE,

                MAX_TOKENS,

                MAX_RETRIES
            )

            validate_grasp_response(
                result
            )

            print_parsed_response(
                result
            )

            grasp_pairs = result[
                "grasps"
            ]

            # =================================================
            # CANDIDATE GRASPS
            # =================================================

            candidate_grasps = generate_candidate_grasps(

                cropped_mask,

                object_image,

                NUM_CANDIDATE_GRASPS
            )

            candidate_grasps = filter_boundary_grasps(

                candidate_grasps,

                object_image.shape[1],
                object_image.shape[0],

                BOX_SIZE
            )

            print_candidate_info(
                candidate_grasps
            )

            # =================================================
            # FILTER GRASPS
            # =================================================

            filtered_grasps = filter_semantic_grasps(

                grasp_pairs,

                candidate_grasps,

                grid_result["valid_cells"],

                grid_result["cell_w"],
                grid_result["cell_h"],

                grid_result["cols"],
                grid_result["rows"]
            )

            print_filtering_info(

                grasp_pairs,

                filtered_grasps
            )

            # =================================================
            # BBOX CONVERSION
            # =================================================

            bbox_results = grasps_to_bboxes(

                filtered_grasps,

                bbox,

                BOX_SIZE
            )

            bbox_results = validate_all_bboxes(

                bbox_results,

                scene.shape[1],
                scene.shape[0]
            )

            print_bbox_info(
                bbox_results
            )

            # =================================================
            # COMBINED VISUALIZATION
            # =================================================

            combined_output_path = os.path.join(

                output_dir,

                f"grasps_{scene_id}.png"
            )

            save_combined_visualization(

                scene,

                bbox_results,

                combined_output_path
            )

            print("\n===== VISUALIZATION =====")

            print(
                f"Grid Image:\n"
                f"{grid_output_path}"
            )

            print()

            print(
                f"Combined Grasps:\n"
                f"{combined_output_path}"
            )

            # =================================================
            # EXPORT JSON
            # =================================================

            json_output_path = export_scene_results(

                output_dir,

                object_name,
                scene_id,

                bbox_results
            )

            print_export_info(
                json_output_path
            )

        except Exception as e:

            print("\nERROR:\n")

            print(e)

            continue

print("\n================================================")
print("DONE")
print("================================================")