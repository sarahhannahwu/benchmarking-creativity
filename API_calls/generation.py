import json
import datetime
import os
import openai

# from functions.tools import is_valid_move
from openai import OpenAI
from pydantic import BaseModel

# from API_calls.evaluation import MODEL

# Set your OpenAI API key
api_key = os.environ.get('OPENAI_API_KEY')
MODEL = "o3-mini"

# Define a class to represent the structured output from the OpenAI API
# unsure how these values are extracted from the API response
class GameMove(BaseModel):
    shape: str
    next_shape: str
    response: str 

client = OpenAI()

# Define a function to get game responses from the OpenAI API
def get_openai_response(user_prompt):
    try:
        # Call the OpenAI API using the new interface
        response = client.beta.chat.completions.parse(
            model=MODEL,  # Use the appropriate model
            messages=[
                {"role": "system", "content": user_prompt}
            ],
            # max_tokens=150,  # Adjust the number of tokens as needed
            response_format=GameMove,   # Use the GameMove class to structure the response
        )
        # Extract the text from the response
        return response.choices[0].message.content
    except Exception as e:
        return f"An error occurred: {e}"

generation_instructions = "instructions/generation_instructions_v5.txt"

with open(generation_instructions, "r") as f:
    user_prompt = f.read()

# Call the OpenAI API
fp_openai_generations = "openai_generations.jsonl" # JSONL separates JSON objects into individual lines
openai_generations = get_openai_response(user_prompt)
with open(fp_openai_generations, 'a') as f: # appends to the file, instead of overwriting
    output = {
        "response": openai_generations,
        "timestamp": datetime.datetime.now().isoformat(), 
        "prompt_file": generation_instructions,
        "model": MODEL
    }
    f.write(json.dumps(output))
    f.write("\n")
