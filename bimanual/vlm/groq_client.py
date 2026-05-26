import time
import json
import re
from PIL import Image
from google import genai

# ============================================================
# CREATE CLIENT
# ============================================================

def create_groq_client(api_key):

    client = genai.Client(
        api_key=api_key
    )

    return client

# ============================================================
# CLEAN RESPONSE
# ============================================================

def clean_response(response_text):

    if response_text is None:

        raise Exception(
            "Empty VLM response"
        )

    response_text = response_text.strip()

    response_text = response_text.replace(
        "```json",
        ""
    )

    response_text = response_text.replace(
        "```",
        ""
    )

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
# QUERY VLM
# ============================================================

def query_vlm(

    client,

    model,

    image_path,

    prompt,

    temperature=0.0,

    max_tokens=120,

    max_retries=5
):

    success = False

    result = None

    for attempt in range(max_retries):

        try:

            print(
                f"\nAttempt "
                f"{attempt+1}/{max_retries}"
            )

            # ------------------------------------------------
            # Gemini request
            # ------------------------------------------------

            image = Image.open(
                image_path
            )

            response = client.models.generate_content(

                model=model,

                contents=[

                    prompt,
                    image
                ]
            )

            response_text = response.text

            print("\n===== RAW RESPONSE =====\n")

            print(repr(response_text))

            # ------------------------------------------------
            # Clean
            # ------------------------------------------------

            response_text = clean_response(
                response_text
            )

            print("\n===== CLEAN RESPONSE =====\n")

            print(response_text)

            # ------------------------------------------------
            # Parse
            # ------------------------------------------------

            result = parse_json_response(
                response_text
            )

            success = True

            break

        except Exception as e:

            print("\nVLM Error:\n")

            print(e)

            wait_time = (
                attempt + 1
            ) * 5

            print(
                f"\nRetrying in "
                f"{wait_time} seconds..."
            )

            time.sleep(wait_time)

    if not success:

        raise Exception(
            "VLM failed after retries"
        )

    return result

# ============================================================
# VALIDATE GRASP RESPONSE
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

    return True