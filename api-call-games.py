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

class GameMove(BaseModel):
    messages: list
    shape: str
    next_shape: str
    response: str
    timestamp: str
    valid_move: bool

client = OpenAI()

# Define a function to get game responses from the OpenAI API
def get_openai_response(user_prompt, prompt):
    try:
        # Call the OpenAI API using the new interface
        response = openai.chat.completions.create(
            model="gpt-4o-mini",  # Use the appropriate model
            messages=[
                {"role": "system", "content": user_prompt},
                {"role": "user", "content": prompt}
            ],
            max_tokens=150  # Adjust the number of tokens as needed
        )
        # Extract the text from the response
        return response
    except Exception as e:
        return f"An error occurred: {e}"


user_prompt = "instructions_v4.txt"
prompt={
    "current_shape": "1023", 
    "last_shape": ""
}
get_openai_response(user_prompt, prompt)