import time
import json
import base64
import re

from openai import OpenAI

from vlm.parser import (
    clean_response,
    parse_json_response
)

# ============================================================
# CREATE CLIENT
# ============================================================

def create_groq_client(api_key):

    client = OpenAI(

        api_key=api_key,

        base_url="https://api.groq.com/openai/v1"
    )

    return client

# ============================================================
# ENCODE IMAGE
# ============================================================

def encode_image_base64(image_path):

    with open(image_path, "rb") as f:

        image_base64 = base64.b64encode(
            f.read()
        ).decode("utf-8")

    return image_base64


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

    image_base64 = encode_image_base64(
        image_path
    )

    success = False

    result = None

    for attempt in range(max_retries):

        try:

            print(
                f"\nAttempt "
                f"{attempt+1}/{max_retries}"
            )

            response = client.chat.completions.create(

                model=model,

                messages=[

                    {

                        "role": "user",

                        "content": [

                            {
                                "type": "text",

                                "text": prompt
                            },

                            {

                                "type": "image_url",

                                "image_url": {

                                    "url":
                                    (
                                        "data:image/png;base64,"
                                        + image_base64
                                    )
                                }
                            }
                        ]
                    }
                ],

                temperature=temperature,

                max_tokens=max_tokens
            )

            response_text = (

                response
                .choices[0]
                .message
                .content
            )

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
            # Parse JSON
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
