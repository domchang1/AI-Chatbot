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
#pinecone_env = os.getenv('PINECONE_ENV')

embedding = OpenAIEmbeddings(model="text-embedding-3-small")
index_name = "ai-chatbot"
db = PineconeVectorStore(index_name=index_name, embedding=embedding)
retriever = db.as_retriever

def getContext(query):
    #embedding_vector = embedding.embed_query(query)
    #docs = docsearch.similarity_search_by_vector(embedding_vector)
    docs = db.similarity_search(query)
    return docs

llm = ChatOpenAI(model="gpt-3.5-turbo")

# prompt = ChatPromptTemplate.from_messages([
#         ("system", "Answer the user's questions based on the context: {context}"),
#         MessagesPlaceholder(variable_name="chat_history"),
#         ("human", "{input}")
#     ])

# chain = create_stuff_documents_chain(llm, prompt)

# retriever_prompt = ChatPromptTemplate.from_messages([
#     MessagesPlaceholder(variable_name="chat_history"),
#     ("human", "{input}"),
#     ("human", "Given the above conversation, generate a search query to look up in order to get information relevant to the conversation")
# ])

# history_aware_retriever = create_history_aware_retriever(
#     llm,
#     retriever,
#     retriever_prompt
# )

# retrieval_chain = create_retrieval_chain(
#     # retriever,
#     history_aware_retriever,
#     chain
# )
template = """
Given chat history: {history}
---
Answer the question based on the following context:
{context}
---
Answer the question based on the above context: {query}
"""

prompt = ChatPromptTemplate.from_template(template)

chat_history = ""

while(True):
    q = input("Ask anything about Harry Potter Sorcerer's Stone: ")
    if 'exit()' in q:
        break
    # response = chain.invoke({
    #     "input": q,
    #     "chat_history": chat_history
    # })["answer"]
    context = getContext(q)
    content = ""
    for i in context:
        content += str(i.page_content)
    messages = prompt.format(history=chat_history, context=content, query=q)
    result = llm.invoke(messages).content
    chat_history += q + "\n"
    chat_history += result + "\n"
    # chat_history.append(HumanMessage(content=q))
    # chat_history.append(AIMessage(content=response))
    print("Chatbot: " + result)