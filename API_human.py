import json
import datetime
import os
import openai

from openai import OpenAI
api_key = os.environ.get('OPENAI_REASONING_KEY')

# Compare its choices to the human's actual choices
client = OpenAI()
MODEL = "gpt-4"

# with open("human_participants_blinded/data_human_1.csv", "r") as file:
#     input_data = file.read()

# def select_gallery_shapes(input_data):
#     messages = [
#         {
#             "role": "user",
#             "content": f"""
# You will see shapes from a game called “the search for the shifting shape”. Your goal is to explore the world of shifting shapes and discover those you consider as interesting and beautiful. I will give you a csv of shape data, where each shape is represented as a matrix of 1s and 0s in the shape_matrix column. Each 1 represents a square and each 0 represents a blank on a grid. All shapes are composed of exactly 10 connected squares. When you land on a shape you find particularly interesting and beautiful, save it to the ‘Shape Gallery’. You have to choose at least five shapes to save to the gallery. For each gallery save, provide its unique ID and a brief explanation of why you chose it. 

# SHAPES TO CHOOSE FROM:
# {input_data}
#             """
#         }
#     ]

#     response = client.chat.completions.create(
#         model=MODEL,
#         messages=messages
#     )

#     model_response = response.choices[0].message.content
#     print(model_response)

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

import pandas as pd
import csv
from io import StringIO
from openai import OpenAI

client = OpenAI()
MODEL = "gpt-4"

def select_gallery_shapes(csv_filepath):
    # Read the CSV file using pandas for better handling
    try:
        # First attempt to read with pandas
        df = pd.read_csv(csv_filepath)
        
        # Extract just the necessary columns to reduce token usage
        if 'id' in df.columns and 'shape_matrix' in df.columns:
            df = df[['shape_ID', 'shape_matrix']]
        
        # Convert shape matrices to visual representations for better understanding
        shape_descriptions = []
        
        for idx, row in df.iterrows():
            shape_ID = row.get('shape_ID', f'Shape_{idx}')
            shape_matrix = row['shape_matrix']
            
            # If shape_matrix is stored as a string, convert it to a proper matrix
            if isinstance(shape_matrix, str):
                # This handles various string formats - adjust based on your actual data
                import ast
                try:
                    # Try to parse as a Python literal (list of lists)
                    matrix = ast.literal_eval(shape_matrix)
                except:
                    # If that fails, try processing as a string representation
                    # Example: "[[1,0,0],[0,1,0],[0,0,1]]"
                    shape_matrix = shape_matrix.replace('[', '').replace(']', '')
                    rows = shape_matrix.split('],[')
                    matrix = []
                    for row in rows:
                        matrix.append([int(cell) for cell in row.split(',')])
            else:
                matrix = shape_matrix
            
            # Create a visual representation
            visual = "\nVisual representation:\n"
            for row in matrix:
                visual += "".join(["■" if cell == 1 else "□" for cell in row]) + "\n"
                
            shape_descriptions.append(f"Shape ID: {shape_ID}{visual}")
        
    except Exception as e:
        # Fallback: If pandas processing fails, use the raw CSV but with better formatting
        print(f"Pandas processing failed: {e}. Using fallback method.")
        
        # Read the raw file
        with open(csv_filepath, "r") as file:
            raw_data = file.read()
            
        # Basic CSV parsing to extract shape IDs and matrices
        shape_descriptions = []
        reader = csv.DictReader(StringIO(raw_data))
        
        for idx, row in enumerate(reader):
            shape_ID = row.get('shape_ID')
            shape_matrix = row.get('shape_matrix', '')
            shape_descriptions.append(f"Shape ID: {shape_ID}\nMatrix: {shape_matrix}\n")
    
    # Limit number of shapes if there are too many (to manage token usage)
    if len(shape_descriptions) > 20:
        import random
        selected_shapes = random.sample(shape_descriptions, 20)
        sample_note = "(Showing 20 random samples due to size limitations)"
    else:
        selected_shapes = shape_descriptions
        sample_note = f"(Showing all {len(shape_descriptions)} shapes)"
    
    prompt = f"""
You will see shapes from a game called "the search for the shifting shape". Your goal is to explore the world of shifting shapes and discover those you consider as interesting and beautiful.

Each shape is represented as a matrix where:
- 1s represent filled squares
- 0s represent blank spaces
- All shapes are composed of exactly 10 connected squares

For shapes with visual representations, ■ represents filled squares and □ represents empty spaces.

When you find shapes that are particularly interesting and beautiful, select them for the 'Shape Gallery'. 
You must choose at least five shapes to save to the gallery.

For each gallery save, provide:
1. Its unique ID
2. A brief explanation of why you chose it (aesthetic quality, symmetry, resemblance to real objects, etc.)

SHAPES TO CHOOSE FROM {sample_note}:
{"".join(selected_shapes)}
"""

    messages = [{"role": "user", "content": prompt}]
    
    response = client.beta.chat.completions.parse(
        model=MODEL,
        messages=messages,
        response_format=GameMove
    )

    model_response = response.choices[0].message.content
    print(model_response)
    return model_response

# Use the function with your file
select_gallery_shapes("human_participants_blinded/data_human_1.csv")