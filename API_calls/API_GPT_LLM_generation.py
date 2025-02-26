import json
import datetime
import os
import openai

from functions.tools import is_valid_move
from openai import OpenAI
from pydantic import BaseModel

# Set your OpenAI API key
api_key = os.environ.get('OPENAI_KEY')

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
            model="gpt-4o-mini",  # Use the appropriate model
            messages=[
                {"role": "system", "content": user_prompt}
            ],
            max_tokens=150,  # Adjust the number of tokens as needed
            response_format=GameMove,   # Use the GameMove class to structure the response
        )
        # Extract the text from the response
        return response.choices[0].message.content
    except Exception as e:
        return f"An error occurred: {e}"

with open("instructions_v4.txt", "r") as file:
    user_prompt = file.read()
    print(get_openai_response(user_prompt))