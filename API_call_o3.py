import json
import datetime
import os
import openai

from functions.tools import is_valid_move
from openai import OpenAI
from pydantic import BaseModel

# Set your OpenAI API key
api_key = os.environ.get('OPENAI_REASONING_KEY')


    
# response = openai.chat.completions.create(

#     model="o1-mini",

#     messages=[

#             {"role": "user", "content": "Solve: The derivative of f(x) = x^2 * sin(x)"}

#     ]

#     )

# print(response.choices[0].message.content) 

# Define a class to represent the structured output from the OpenAI API
# unsure how these values are extracted from the API response
class GameMove(BaseModel):
    shape: str
    next_shape: str
    response: str 

client = OpenAI()

with open("instructions_v5.txt", "r") as file:
    prompt = file.read()

response = client.chat.completions.create(
    model="gpt-4",
    messages=[
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": prompt
                },
            ],
        }
    ]
)

print(response.choices[0].message.content)