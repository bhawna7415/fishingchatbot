from lxml import etree
import requests
from bs4 import BeautifulSoup
from bs4.element import Comment
from langchain.text_splitter import RecursiveCharacterTextSplitter
import getpass
import os
from langchain.document_loaders import TextLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Pinecone
from langchain.document_loaders import TextLoader
import pinecone
from langchain.document_loaders import AsyncChromiumLoader
from langchain.document_transformers import BeautifulSoupTransformer
from langchain.document_loaders.sitemap import SitemapLoader

embeddings = OpenAIEmbeddings()

sitemap_loader = SitemapLoader(web_path="data/sitemap.xml", is_local=True)

docs = sitemap_loader.load()
# print(docs)

pinecone.init(
                api_key="365c78b6-e389-4adf-b48f-0383d7c8676f",  # find at app.pinecone.io
                environment="gcp-starter",  # next to api key in console
            )

index_name = "fishingchatbot"

if index_name not in pinecone.list_indexes():
    # we create a new index
    pinecone.create_index(name=index_name, metric="cosine", dimension=1536)
docsearch = Pinecone.from_documents(docs, embeddings, index_name=index_name)


   

       