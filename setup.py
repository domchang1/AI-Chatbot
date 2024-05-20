from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()
api_key = os.getenv('API_KEY')
client = OpenAI(
    api_key = api_key,
)

def getResponse(text):
    response = client.chat.completions.create(
        model="gpt-3.5",
        messages=[
        {
            "role": "system",
            "content": "You are a friendly and helpful chatbot."
        },
        {
            "role": "user",
            "content": text
        }
        ],
        temperature=0.5,
        max_tokens=64,
        top_p=1
    )
    return response
