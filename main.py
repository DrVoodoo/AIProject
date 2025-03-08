import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import InMemoryVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

llm = ChatGroq(
  model_name="llama-3.3-70b-versatile",
  temperature=0.7,
  api_key=GROQ_API_KEY
)

embeddings_model = HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-large-instruct")

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

loader = TextLoader("yoda_galactic_feasts.txt")

documents = loader.load()

print(documents)