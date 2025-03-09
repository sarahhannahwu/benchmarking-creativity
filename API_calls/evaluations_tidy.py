import os
import csv
import json
import datetime
from numpy import shape
import pandas as pd
from io import StringIO
from pydantic import BaseModel
from openai import OpenAI
from typing import List

client = OpenAI()
api_key = os.environ.get('OPENAI_API_KEY')
MODEL = "o3-mini" # replace as needed



# Define classes to structure shape evaluation responses from the LLM
class ShapeEvaluation(BaseModel):
    shape: str
    explanation: str

class ShapeEvaluationList(BaseModel):
    items: List[ShapeEvaluation]

def select_gallery_shapes(csv_filepath):
    df = pd.read_csv(csv_filepath, sep='\t')#.reset_index()
    #print(df.head())
    #df.rename(columns={'index': 'shape_ID'}, inplace=True)
    subset = df.query("timestamp_gallery != ' '")[['shape', 'shape_matrix_str']] # Subset to rows with unique shapes
    shape_descriptions = subset.apply(lambda row: f"Shape ID: {row['shape']}\nMatrix:\n{row['shape_matrix_str']}\n", axis=1).tolist()
    return shape_descriptions # Text descriptions for shapes with shape ID and matrix representation


def get_openai_response(user_prompt, model=MODEL) -> dict: # stores the output as a dictionary
    try:
        response = client.beta.chat.completions.parse(
            model=model,  # Use the appropriate model
            messages=[
                {"role": "system", "content": user_prompt}
            ],
            # max_tokens=150,  # Adjust the number of tokens as needed
            response_format=ShapeEvaluationList,   # Use class to structure the response
        )
        # Extract the text from the response
        print("openai response success")
        return response.choices[0].message.content
    except Exception as e:
        print("openai response failed")
        return {"error": f"An error occurred: {e}"}


# Load base instructions
with open("instructions/generation_instructions_v5.txt", "r") as file:
    user_prompt = file.read()
    print(get_openai_response(user_prompt))

# Load the shapes
fp = "data/all-games.tsv"
shape_descriptions = select_gallery_shapes(fp)


# Save the instructions
fp_instructions = "instructions/evaluation_instructions_v2.txt"
with open(fp_instructions, 'r') as f:
    instructions = f.read()

# Save the instructions with shapes
fp_instructions_with_shapes = "instructions/evaluation_instructions_with_shapes_v2.txt"
with open(fp_instructions_with_shapes, 'w') as f:
    f.write(instructions)
    f.write("\n\n")
    for idx, desc in enumerate(shape_descriptions):
        f.write(desc)
        f.write("\n")
        if idx >= 30:
            break

# Load the instructions with shapes
with open(fp_instructions_with_shapes, 'r') as f:
    user_prompt = f.read()

# Call the OpenAI API for each game file in the human dataset
fp_openai_responses = "openai_evaluations_Mar9.jsonl" # JSONL separates JSON objects into individual lines
human_participants_blinded_folder = "human_participants_blinded"
csv_files = [f for f in os.listdir(human_participants_blinded_folder) if f.endswith('.csv')]

for csv_file in csv_files:
    csv_path = os.path.join(human_participants_blinded_folder, csv_file)
    print(f"Processing CSV file: {csv_path}")

df = pd.read_csv(csv_path)

def matrix_to_coordinates(matrix):
    """
    Convert a 2D array of bits to a list of (x,y) coordinate tuples on a 10x10 grid.
    
    Args:
        matrix: A 2D array/list where 1 represents a filled square and 0 represents an empty square
        
    Returns:
        A list of (x,y) tuples representing the coordinates of filled squares
        where (0,0) is the top-left corner
    """
    coordinates = []
    
    # Check if matrix is provided as a string representation
    if isinstance(matrix, str):
        import ast
        try:
            # Try to parse as a Python literal (list of lists)
            matrix = ast.literal_eval(matrix)
        except:
            # If that fails, try processing as a string representation
            matrix = matrix.replace('[', '').replace(']', '')
            rows = matrix.split('],[')
            parsed_matrix = []
            for row in rows:
                parsed_matrix.append([int(cell) for cell in row.split(',')])
            matrix = parsed_matrix
    
    # Iterate through the matrix and collect coordinates of filled cells
    for y, row in enumerate(matrix):
        for x, cell in enumerate(row):
            if cell == 1:
                coordinates.append((x, y))
    
    return coordinates


# Start with just the first csv file of the human_participants_blinded_folder
csv_file = csv_files[0]
csv_path = os.path.join(human_participants_blinded_folder, csv_file)

TODO: # Iterate through folder and process all csv files

openai_response = get_openai_response(user_prompt)
with open(fp_openai_responses, 'a') as f: # appends to the file, instead of overwriting
    output = {
        "csv_file": csv_file,
        "response": openai_response,
        "timestamp": datetime.datetime.now().isoformat(), 
        "prompt_file": fp_instructions_with_shapes,
        "model": MODEL
    }
    f.write(json.dumps(output))
    f.write("\n")
