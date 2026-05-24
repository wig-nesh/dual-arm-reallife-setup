# ============================================================
# BUILD VLM PROMPT
# ============================================================

def build_prompt(

    grid_rows,
    grid_cols,

    valid_cells,

    num_grasp_hypotheses
):

    prompt = f"""
Output ONLY JSON.

Grid rows: {grid_rows}
Grid cols: {grid_cols}

Valid cells:
{valid_cells}

Generate {num_grasp_hypotheses}
stable bimanual grasp hypotheses.

The object must be lifted vertically upward.

Requirements:
- stable lifting
- balanced support
- symmetric support
- avoid weak structures
- avoid thin structures
- avoid background
- avoid empty cells

Each grasp hypothesis must contain:
- one LEFT grasp region
- one RIGHT grasp region

The regions should:
- support the object symmetrically
- be physically graspable
- avoid unstable contact areas

Return ONLY valid JSON.

JSON format:

{{
  "grasps": [

    {{
      "left_cell": 1,
      "right_cell": 2
    }}

  ]
}}
"""

    return prompt

# ============================================================
# PRINT PROMPT
# ============================================================

def print_prompt(prompt):

    print("\n===== VLM PROMPT =====\n")

    print(prompt)