from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_pinecone import PineconeVectorStore
from dotenv import load_dotenv
import os

load_dotenv()
os.environ['OPENAI_API_KEY'] = os.getenv('OPEN_API_KEY')

embedding = OpenAIEmbeddings(model="text-embedding-3-small")

#Embed Harry Potter Books 1-7
documents = []
for i in range(1,8):
    s = "./harry_potter_texts/harry_potter" + str(i) + ".txt"
    loader = TextLoader(s)
    documents.extend(loader.load())
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
docs = text_splitter.split_documents(documents)

index_name = "harry-potter-bot"
docsearch = PineconeVectorStore.from_documents(docs, embedding, index_name=index_name)