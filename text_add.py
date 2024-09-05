import getpass
import os
from langchain.document_loaders import TextLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Pinecone
from langchain.document_loaders import TextLoader
import pinecone

loader = TextLoader("data.txt")
documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
docs = text_splitter.split_documents(documents)
print(docs)

embeddings = OpenAIEmbeddings()

# initialize pinecone
pinecone.init(
    api_key="365c78b6-e389-4adf-b48f-0383d7c8676f",
    environment="gcp-starter"
)

index_name = "fishingchatbot"

# First, check if our index already exists. If it doesn't, we create it
if index_name not in pinecone.list_indexes():
    
    pinecone.create_index(name=index_name, metric="cosine", dimension=1536)

docsearch = Pinecone.from_documents(docs, embeddings, index_name=index_name)


query = "Gift Card"
docs = docsearch.similarity_search(query)
print(docs)