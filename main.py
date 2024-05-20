from openai import OpenAI
from dotenv import load_dotenv
import os
import datetime

load_dotenv()
api_key = os.getenv('API_KEY')
client = OpenAI(
    api_key = api_key,
)

messages = []
messages.append({"role": "system", "content": "You are a friendly and helpful chatbot."})

history = open("chat_history.txt", "w")
history.write(datetime.datetime.now().strftime("%c") + "\n \n")

while (True):
    msg = input("Message Chatbot: ")
    history.write("User Input: " + msg + "\n")
    if "exit()" in msg:
        break
    messages.append({"role": "user", "content": msg})
    response = client.chat.completions.create(
        # model="gpt-4o",
        model ="gpt-3.5-turbo",
        messages=messages
    ).choices[0].message.content
    print(response)
    history.write("Chatbot Response: " + response + "\n \n")
    messages.append({"role": "assistant", "content": response})

history.close()