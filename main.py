from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()
api_key = os.getenv('API_KEY')
client = OpenAI(
    api_key = api_key,
)

messages = []
messages.append({"role": "system", "content": "You are a friendly and helpful chatbot."})
while (True):
    msg = input("Message Chatbot: ")
    if "exit" in msg:
        break
    messages.append({"role": "user", "content": msg})
    response = client.chat.completions.create(
        model="gpt-3.5",
        messages=messages,
        stream=True
    )
    for chunk in response:
        print(chunk.choices[0].delta)
    messages.append({"role": "assistant", "content": response["choices"][0]["message"]["content"]})
    # for chunk in response:
    #     print(chunk.choices[0].delta.content or "", end="")
