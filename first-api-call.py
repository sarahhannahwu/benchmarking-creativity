import openai
import json as js
import os

api_key = os.environ.get('OPENAI_KEY')

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
user_prompt = "You are my creative partner and we are trying to develop creative uses for everyday objects. I will provide the object and you will provide the creative use."
prompt = "The object is a paperclip."
response = get_openai_response(user_prompt, prompt)

print(response)