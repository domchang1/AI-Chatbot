from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from langchain_pinecone import PineconeVectorStore
from dotenv import load_dotenv
import os

load_dotenv()
os.environ['OPENAI_API_KEY'] = os.getenv('OPEN_API_KEY')

embedding = OpenAIEmbeddings(model="text-embedding-3-small")

#Embed the whole first Harry Potter Book

loader = TextLoader("harry_potter.txt")
documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
docs = text_splitter.split_documents(documents)

index_name = "ai-chatbot"
docsearch = PineconeVectorStore.from_documents(docs, embedding, index_name=index_name)