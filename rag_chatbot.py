from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import MessagesPlaceholder
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from dotenv import load_dotenv
import os

load_dotenv()
os.environ['OPENAI_API_KEY'] = os.getenv('OPEN_API_KEY')
os.environ['PINECONE_API_KEY'] = os.getenv('PINECONE_API_KEY')

embedding = OpenAIEmbeddings(model="text-embedding-3-small")
index_name = "harry-potter-bot"
db = PineconeVectorStore(index_name=index_name, embedding=embedding)
retriever = db.as_retriever(search_kwargs={"k": 3})

def getContext(query):
    docs = db.similarity_search(query)
    return docs

llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.4)

prompt = ChatPromptTemplate.from_messages([
        ("system", "Answer the user's questions based on the context: {context}"),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}")
    ])

chain = create_stuff_documents_chain(llm, prompt)

retriever_prompt = ChatPromptTemplate.from_messages([
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}"),
    ("human", "Given the above conversation, generate a search query to look up in order to get information relevant to the conversation")
])

history_aware_retriever = create_history_aware_retriever(llm, retriever, retriever_prompt)

retrieval_chain = create_retrieval_chain(history_aware_retriever, chain)
chat_history = []

while(True):
    q = input("Ask anything from Harry Potter Books 1-7: ")
    if 'exit()' in q:
        break
    response = retrieval_chain.invoke({
        "input": q,
        "chat_history": chat_history
    })
    response = response["answer"]
    chat_history.append(HumanMessage(content=q))
    chat_history.append(AIMessage(content=response))
    print("Chatbot: " + response)