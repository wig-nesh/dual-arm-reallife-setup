import json
import re

# ============================================================
# CLEAN RAW RESPONSE
# ============================================================

def clean_response(response_text):

    if response_text is None:

        raise Exception(
            "Empty VLM response"
        )

    response_text = response_text.strip()

    # --------------------------------------------------------
    # Remove markdown wrappers
    # --------------------------------------------------------

    response_text = response_text.replace(
        "```json",
        ""
    )

    response_text = response_text.replace(
        "```",
        ""
    )

    # --------------------------------------------------------
    # Extract JSON region
    # --------------------------------------------------------

    json_match = re.search(
        r'\{[\s\S]*\}',
        response_text
    )

    if json_match is None:

        raise Exception(
            "No JSON found in response"
        )

    response_text = json_match.group(0)

    return response_text

# ============================================================
# PARSE JSON
# ============================================================

def parse_json_response(response_text):

    try:

        result = json.loads(
            response_text
        )

    except Exception:

        raise Exception(

            f"Failed to parse JSON:\n"
            f"{response_text}"
        )

    return result

# ============================================================
# VALIDATE RESPONSE STRUCTURE
# ============================================================

def validate_grasp_response(result):

    if "grasps" not in result:

        raise Exception(
            "Missing 'grasps' field"
        )

    grasps = result["grasps"]

    if not isinstance(grasps, list):

        raise Exception(
            "'grasps' must be a list"
        )

    for grasp in grasps:

        if "left_cell" not in grasp:

            raise Exception(
                "Missing left_cell"
            )

        if "right_cell" not in grasp:

            raise Exception(
                "Missing right_cell"
            )

        left_cell = grasp["left_cell"]
        right_cell = grasp["right_cell"]

        if not isinstance(left_cell, int):

            raise Exception(
                "left_cell must be int"
            )

        if not isinstance(right_cell, int):

            raise Exception(
                "right_cell must be int"
            )

    return True

# ============================================================
# PRINT PARSED RESPONSE
# ============================================================

def print_parsed_response(result):

    print("\n===== PARSED VLM RESPONSE =====\n")

    grasps = result["grasps"]

    for i, grasp in enumerate(grasps):

        print(
            f"Grasp {i}"
        )

        print(
            f"  Left Cell  : "
            f"{grasp['left_cell']}"
        )

        print(
            f"  Right Cell : "
            f"{grasp['right_cell']}"
        )

        print()