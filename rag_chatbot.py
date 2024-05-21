from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_core.prompts import ChatPromptTemplate
from langchain.memory import ConversationBufferMemory
from dotenv import load_dotenv
import os

load_dotenv()
os.environ['OPENAI_API_KEY'] = os.getenv('OPEN_API_KEY')
os.environ['PINECONE_API_KEY'] = os.getenv('PINECONE_API_KEY')
#pinecone_env = os.getenv('PINECONE_ENV')

embedding = OpenAIEmbeddings(model="text-embedding-3-small")
index_name = "ai-chatbot"
db = PineconeVectorStore(index_name=index_name, embedding=embedding)

def getContext(query):
    #embedding_vector = embedding.embed_query(query)
    #docs = docsearch.similarity_search_by_vector(embedding_vector)
    docs = db.similarity_search(query)
    return docs[0].page_content

llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

template = """
Answer the question based on the following context:
{context}
---
Answer the question based on the above context: {query}
"""

prompt = ChatPromptTemplate.from_template(template)

while(True):
    q = input("Ask anything about Harry Potter Sorcerer's Stone: ")
    if 'exit()' in q:
        break
    context = getContext(q)
    messages = prompt.format(context=context, query=q)
    result = llm.invoke(messages).content
    print(result)