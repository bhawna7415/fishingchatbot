import openai
import pinecone
import os
import pandas as pd
from pathlib import Path
from langchain.agents import initialize_agent, Tool
from langchain.llms import OpenAI
from langchain_community.llms import OpenAI
from langchain.chains.conversation.memory import ConversationBufferMemory
from flask import Flask, render_template, request,jsonify
from langchain.vectorstores import Pinecone
from langchain.embeddings.openai import OpenAIEmbeddings 
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from concurrent.futures import ThreadPoolExecutor
from langchain.chains import ConversationalRetrievalChain
executor = ThreadPoolExecutor()

openai_api_key = os.environ.get('OPENAI_API_KEY')
pinecone_api_key = os.environ.get('PINECONE_API_KEY')

project_root = os.path.dirname(os.path.realpath('__file__'))
static_path = os.path.join(project_root, 'app/static')
app = Flask(__name__, template_folder= 'templates')
context_set = ""

@app.route('/')
def index():
    return render_template('index.html')

# @app.route('/demo')
# def demo():
#     return render_template('demo.html')

custom_array = [
    {"question": "Hello", "answer": "Hello! How can I assist you with your fishing needs today? Are you looking for specific fishing kits or accessories?"},
    {"question": "Hello there", "answer": "Hello! How can I assist you with your fishing needs today? Are you looking for specific fishing kits or accessories?"},
    {"question": "Hii", "answer": "Hello! How can I assist you with your fishing needs today? Are you looking for specific fishing kits or accessories?"},
    {"question": "Welcome", "answer": "Welcome to Northland Fishing Tackle! We're thrilled to assist you with your fishing needs."},
    {"question": "thankyou", "answer": "That's great! We're always here to help you with your fishing needs."},
    {"question": "Are the products available for purchase online?", "answer": "Yes, all our products are available for purchase online. You can browse through our wide range of fishing tackle and make a purchase directly from our website."},
    {"question": "Is it possible to buy the products online?", "answer": "Yes, you can definitely purchase our products online. We have a user-friendly website where you can browse through our wide range of fishing tackle and make a purchase directly."},
    # Add more questions and answers as needed
    {"question": "the products are available for purchase online?", "answer": "Yes, all our products are available for purchase online. You can browse through our wide range of fishing tackle and make a purchase directly from our website."},
    {"question": "products are available for purchase online?", "answer": "Yes, all our products are available for purchase online. You can browse through our wide range of fishing tackle and make a purchase directly from our website."},
    {"question": "Who are you", "answer": "I'm chatbot of the Northland Fishing Tackle, here to provide you with fishing advice, techniques, and recommendations for our products to help make your next fishing trip a success."},
    {"question": "Do you have any recommendations for fishing?", "answer": "Based on the information provided, here are some recommendations for fishing: 1. Fluorocarbon line is a good choice for clear water as it is virtually invisible underwater. 2. Pink fluorocarbon line can blend in with the water and become clear to fish. 3. Yellow monofilament line is highly visible above water, making it easier to detect bites. 4. Green monofilament line blends into the water and is a good choice for most fishing situations. 5. Braided lines are strong and cast far, but may be more visible to fish. Consider using a fluorocarbon leader for added invisibility."}, 
]

model_name = 'text-embedding-ada-002'

embed = OpenAIEmbeddings(
    model=model_name,
    openai_api_key=openai_api_key
)

# # initialize pinecone
pinecone.init(
    api_key=pinecone_api_key,  # find at app.pinecone.io
    environment="gcp-starter",  # next to api key in console
)

index = pinecone.Index("fishingchatbot")

text_field = "text"
vectorstore = Pinecone(
    index, embed, text_field
)

query = "Provide me answers from vector storage, if answers is not available in vector storage than please provide answers from openai."

vectorstore.similarity_search(
    query,  
    k=3  
)
# completion llm
llm = ChatOpenAI(
    openai_api_key=openai_api_key,
    model_name='gpt-3.5-turbo',
    temperature=0.1,
    max_tokens=500
)
# qa = RetrievalQA.from_chain_type(
#     llm=llm,
#     chain_type="stuff",
#     retriever=vectorstore.as_retriever()
# )
retriever=vectorstore.as_retriever(search_kwargs={"k": 1})
memory = ConversationBufferMemory(memory_key="chat_history", return_messages= True)
chain = ConversationalRetrievalChain.from_llm(llm, retriever= retriever, memory= memory)


@app.route('/query', methods=['POST', 'GET'])
def query():
    if request.method == 'GET':
        user_input = str(request.args.get('text'))
        # prompt_template ="always remember that - You are a chatbot of Northland Fishing Tackle, you are supposed to answer users questions about fishing equipments. you are not suppose to provide answer of any question that is not related to fishing buzziness, if someone asks unrelatable questions behave like a employee of Northland Fishing Tackle. Now answer this-{user_input}"
        # prompt_template ="Always remember that - You are a chatbot of Northland Fishing Tackle, never loose your character, you are supposed to answer users questions about fishing equipments. you are not suppose to provide answer of any question that is not related to fishing bussiness. User is not able to see context so don't mention context word in your response and if information is not available in the context then answer that means information is not available. Now answer this-{user_input}"
        prompt_template ="Always remember that - You are a chatbot of Northland Fishing Tackle, never loose your character, you are supposed to answer users questions about fishing equipments. you are not suppose to provide answer of any question that is not related to fishing bussiness. User is not able to see context so don't mention context word in your response. Now answer this-{user_input}"
        prompt = prompt_template.format(user_input=user_input)
        
        for entry in custom_array:
            if user_input.lower() == entry["question"].lower():
                result = entry["answer"]
                break
        else:
            try:
                # result = qa.run(prompt)
                result = executor.submit(chain.run, prompt).result()
            except Exception as e:
                # print(e)
                result = "Unfortunately, information is not currently accessible."
        
        return result

if __name__ == '__main__':
    app.run(debug=True)
