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

embeddings = OpenAIEmbeddings()
sitemap_url = "https://www.northlandtackle.com/author-sitemap.xml"

def get_sitemap_urls(url):
    try: 
        headers = {
            'User-Agent': 'Your User Agent String',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            # Other headers if needed
        }
        resp = requests.get(url, headers=headers)
       
        tree = etree.fromstring(resp.content)
        return [loc.text for loc in tree.findall('{*}url/{*}loc')]
    except (requests.exceptions.RequestException, etree.XMLSyntaxError) as e:
        print(f"An error occurred: {e}")
        return []
    
def tag_visible(element):
    if element.parent.name in ['style', 'script', 'head', 'title', 'meta', '[document]']:
        return False
    if isinstance(element, Comment):
        return False
    return True

def text_from_html(body):
    soup = BeautifulSoup(body, 'html.parser')
    texts = soup.findAll(text=True)
    visible_texts = filter(tag_visible, texts)  
    return u" ".join(t.strip() for t in visible_texts)

urls = get_sitemap_urls(sitemap_url)
i = 0 
for url in urls:
    if i==0: 
        # print(url)
        headers = {
            'User-Agent': 'Your User Agent String',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            # Other headers if needed
        }
        response = requests.get(url,headers=headers)
        # print(response.status_code)
        if response.status_code == 200:
            page_content = response.text
            # data = text_from_html(page_content)

            # Load HTML
            loader = AsyncChromiumLoader([url])
            html = loader.load()

            # Transform
            bs_transformer = BeautifulSoupTransformer()
            docs_transformed = bs_transformer.transform_documents(
                html, tags_to_extract=["p", "li", "div", "a"]
            )
            print(docs_transformed)
            # text_splitter = RecursiveCharacterTextSplitter(
            #     # Set a really small chunk size, just to show.
            #     chunk_size = 500,
            #     chunk_overlap  = 50,
            #     length_function = len,
            #     is_separator_regex = False,
            # )
            # docs = text_splitter.create_documents([docs_transformed])
            # initialize pinecone
            pinecone.init(
                api_key=os.getenv("PINECONE_API_KEY"),  # find at app.pinecone.io
                environment=os.getenv("PINECONE_ENV"),  # next to api key in console
            )

            index_name = "fishingchatbot"

            # First, check if our index already exists. If it doesn't, we create it
            if index_name not in pinecone.list_indexes():
                # we create a new index
                pinecone.create_index(name=index_name, metric="cosine", dimension=1536)
            # The OpenAI embedding model `text-embedding-ada-002 uses 1536 dimensions`
            docsearch = Pinecone.from_documents(docs_transformed, embeddings, index_name=index_name)

       