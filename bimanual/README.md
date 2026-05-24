# Bimanual Semantic Grasping

## Overview

Physical-grid semantic bimanual grasp planning
using RGB-D perception and VLM grounding.

Pipeline:
RGB + Depth + Mask
→ physical adaptive grid
→ VLM semantic grounding
→ grasp proposal filtering
→ final bimanual grasps

---

## Installation

conda env create -f environment.yml

conda activate bimanual

---

## Run

python main.py

---

## Dataset Structure

inputs/
    object_name/
        rgb_000.png
        mask_000.png
        depth_000.npy
        intrinsics__000.json

---

## Outputs

outputs/
    object_name/
        grid_000.png
        grasp_pair_000_0.png
        bbox_coords_000.json

# Project Structure

```text
bimanual_grasping/

├── main.py
│   Main orchestration pipeline.

├── README.md
│   Project documentation and usage instructions.

├── requirements.txt
│   Python package dependencies.

├── environment.yml
│   Conda environment configuration.

├── setup.py
│   Package installation setup.

├── run.sh
│   Simple script to run the pipeline.

├── .gitignore
│   Files/folders ignored by git.

├── config/
│
│   ├── __init__.py
│   │   Makes config a Python package.
│
│   └── config.py
│       Global project configuration and hyperparameters.

├── data/
│
│   ├── __init__.py
│   │   Makes data a Python package.
│
│   ├── loader.py
│   │   Dataset traversal, file loading, and scene management.
│
│   └── intrinsics.py
│       Camera intrinsics loading and parsing.

├── segmentation/
│
│   ├── __init__.py
│   │   Makes segmentation a Python package.
│
│   └── extractor.py
│       Object extraction using segmentation masks.

├── grid/
│
│   ├── __init__.py
│   │   Makes grid a Python package.
│
│   ├── physical_grid.py
│   │   Physical adaptive grid generation using depth and intrinsics.
│
│   └── grid_visualizer.py
│       Grid rendering and visualization utilities.

├── vlm/
│
│   ├── __init__.py
│   │   Makes vlm a Python package.
│
│   ├── prompt.py
│   │   Prompt generation for semantic grasp reasoning.
│
│   ├── groq_client.py
│   │   VLM API communication and retry handling.
│
│   └── parser.py
│       VLM response cleaning, parsing, and validation.

├── grasp/
│
│   ├── __init__.py
│   │   Makes grasp a Python package.
│
│   ├── candidates.py
│   │   Dense geometric grasp proposal generation.
│
│   ├── filtering.py
│   │   Semantic-geometric grasp filtering and ranking.
│
│   └── bbox.py
│       Conversion of grasp centers into image-space bounding boxes.

├── visualization/
│
│   ├── __init__.py
│   │   Makes visualization a Python package.
│
│   └── draw.py
│       Final grasp rendering and visualization utilities.

├── export/
│
│   ├── __init__.py
│   │   Makes export a Python package.
│
│   └── exporter.py
│       JSON export and output file generation.

├── utils/
│
│   ├── __init__.py
│   │   Makes utils a Python package.
│
│   ├── geometry.py
│   │   Geometry helper functions and spatial utilities.
│
│   └── file_utils.py
│       Generic filesystem helper utilities.

├── inputs/
│   Input RGB-D dataset organized by object.
│
│   Example:
│
│   toolbox/
│       rgb_000.png
│       mask_000.png
│       depth_000.npy
│       intrinsics.json

├── outputs/
│   Final generated grids, grasp visualizations, and JSON files.

├── assets/
│   Static resources such as diagrams, figures, prompts, and demos.

├── notebooks/
│   Jupyter notebooks for debugging, visualization, and experiments.

└── logs/
    Runtime logs, VLM responses, profiling, and debugging information.