import json
import datetime
import os
import openai

from openai import OpenAI
api_key = os.environ.get('OPENAI_REASONING_KEY')

# Compare its choices to the human's actual choices
client = OpenAI()
MODEL = "gpt-4"

with open("human_participants_blinded/data_human_1.csv", "r") as file:
    input_data = file.read()

def select_gallery_shapes(input_data):
    messages = [
        {
            "role": "user",
            "content": f"""
You will see shapes from a game called “the search for the shifting shape”. Your goal is to explore the world of shifting shapes and discover those you consider as interesting and beautiful. I will give you a csv of shape data, where each shape is represented as a matrix of 1s and 0s in the shape_matrix column. Each 1 represents a square and each 0 represents a blank on a grid. All shapes are composed of exactly 10 connected squares. When you land on a shape you find particularly interesting and beautiful, save it to the ‘Shape Gallery’. You have to choose at least five shapes to save to the gallery. For each gallery save, provide its unique ID and a brief explanation of why you chose it. 

SHAPES TO CHOOSE FROM:
{input_data}
            """
        }
    ]

    response = client.chat.completions.create(
        model=MODEL,
        messages=messages
    )

    model_response = response.choices[0].message.content
    print(model_response)
    # response_content = response.choices[0].message.content.replace('```json', '').replace('```', '').strip()
    
    # try:
    #     if isinstance(response_content, dict):
    #         response_dict = response_content
    #     else:
    #         response_dict = json.loads(response_content)
    #     return response_dict
    # except json.JSONDecodeError as e:
    #     print(f"Failed to decode JSON response: {response_content}")
    #     raise e