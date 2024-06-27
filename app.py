import openai
import pinecone
import os
import pandas as pd
from pathlib import Path
from langchain.agents import initialize_agent, Tool
from langchain.llms import OpenAI
from langchain.chains.conversation.memory import ConversationBufferMemory
from flask import Flask, render_template, request
from langchain.vectorstores import Pinecone
from langchain.embeddings.openai import OpenAIEmbeddings 
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import ChatPromptTemplate
from langchain.schema import format_document
from langchain_core.messages import AIMessage, HumanMessage, get_buffer_string
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain.prompts.prompt import PromptTemplate
from langchain.prompts import PromptTemplate
from concurrent.futures import ThreadPoolExecutor
from langchain.chains import ConversationalRetrievalChain
from operator import itemgetter
from langchain_core.output_parsers import StrOutputParser
from openai import OpenAI

client = OpenAI()

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
    model_name='gpt-4o',
    temperature=0.6,
    max_tokens=500
)

retriever  = vectorstore.as_retriever(search_kwargs={"k": 1})

# memory = ConversationBufferMemory(memory_key="chat_history", return_messages= True)
# chain = ConversationalRetrievalChain.from_llm(llm, retriever= retriever, memory= memory)

_template = """Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question, in its original language.

Chat History:
{chat_history}
Follow Up Input: {question}
Standalone question:"""
CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(_template)

template = """Always remember that - You are a fishing expert chat representing Northland Tackle. You are specialized in fishing, fishing techniques, fish species, and Northland lure types and recommendations.
:- Always remember that - You are a fishing expert representing Northland Tackle.
:- You should never lose your character. You are supposed to answer users' questions about fishing techniques, strategies, lake information, lake data, and anything related to fishing and catching fish.
:- You are not supposed to provide answers to any question that is not related to fishing to some capacity. Use your best judgment to determine what is related to fishing.

here are some instructions for the conversation you will have - 

1. No need to wish greeting for each query.
2. The main goal of the conversation is to ask the customer about their needs, determine if customer have an upcoming fishing trip, and recommend a suitable Northland product. Additionally, provide fishing techniques based on the details they share about their fishing plans.
3. if customer ask casual query.then ask to customer such as Do you need advice on fish species or techniques?,Are you in need of lure information or recommendations?,What lake are you visiting? and When are you taking your fishing trip?.
4. If customer do require advice on fish species or techniques,then ask to customer such as where they are fishing,how and when they are fishing,and what species they are targeting.
5. Do not end the chat without providing a Northland lure recommendation and  gathering information to customer. don't provide any recommendations until customer provide some details surrounding what they are fishing for.
6. if customer require lure informations then ask to customer such as what species they are targeting and provide the best lure recommondations with a link of northland website.
7. always provide the link from the northland tackle website when you recommonding of lure. Do not recommend a lure if you are unable to also provide a link to that lure on the website.
8. Do not recommend a lure if you are unable to also provide a link to that lure on the website.
9. if customer ask about the best time for fishing (e.g. what is the best time to fishing in Corona?) then don't ask recommendation questions instead of provide the response from context.

Your goal is to provide advice and very accurate fishing tips and lure recommendations for customer.
now here is the information from northland tackle that you might require to answer customer queries
you will have to necessary give response each user's query from the context
{context}
don't mention word like 'according to the provided context' or 'information'
for user you are an agent who will talk in a conversational way
Question: {question}
Answer:
"""
ANSWER_PROMPT = ChatPromptTemplate.from_template(template)

DEFAULT_DOCUMENT_PROMPT = PromptTemplate.from_template(template="{page_content}")


def _combine_documents(
    docs, document_prompt=DEFAULT_DOCUMENT_PROMPT, document_separator="\n\n"
):
    doc_strings = [format_document(doc, document_prompt) for doc in docs]
    return document_separator.join(doc_strings)

_inputs = RunnableParallel(
    standalone_question=RunnablePassthrough.assign(
        chat_history=lambda x: get_buffer_string(x["chat_history"])
    )
    | CONDENSE_QUESTION_PROMPT
    | ChatOpenAI(model_name='gpt-4o',temperature=0.6)
    | StrOutputParser(),
)
_context = {
    "context": itemgetter("standalone_question") | retriever | _combine_documents,
    "question": lambda x: x["standalone_question"],
}


def get_info(city):
    url = f"https://fishingreminder.com/US/charts/fishing_times/{city}"    
    try:
        response = requests.get(url)
        if response.status_code == 200:
            try:
                soup = BeautifulSoup(response.text, 'html.parser')
                return soup.get_text()
            except Exception as e:
                print(f"An error occurred while parsing the HTML: {e}")
                return None
        else:
            print(f"Unexpected status code received: {response.status_code}")
            return None
    except Exception as e:
        print(f"An error occurred while making the request: {e}")
        return None

functions = [
            {
                "type": "function",
                "name": "get_info",  # Ensure name parameter is added here
                "description": "Retrieve info regarding the best time to fish and get weather also for that city. Pass city name as parameter to the function(get_info). Please don't pass random values as a parameter. if you are not getting the city name then no need to assume random city's name",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "city": {
                            "type": "string",
                            "description": "The city and state name, e.g. San Francisco, CA",
                        },                                          
                    },
                    "required": ["city"], 
                }
            }
            ]
def get_gpt_response(messages):
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        functions=functions,
        function_call="auto"
    )    
    return response.choices[0].message


conversational_qa_chain = _inputs | _context | ANSWER_PROMPT | ChatOpenAI(temperature=0.6, model_name='gpt-4o',max_tokens=1000)

chat_history = []

@app.route('/query', methods=['POST', 'GET'])
def query():
    if request.method == 'GET':
        user_input = str(request.args.get('text'))
        # prompt_template ="Always remember that - You are a chatbot of Northland Fishing Tackle, never loose your character, you are supposed to answer users questions about fishing equipments. you are not suppose to provide answer of any question that is not related to fishing bussiness. User is not able to see context so don't mention context word in your response. Now answer this-{user_input}"
        # prompt = prompt_template.format(user_input=user_input) 
        messages = [
            {"role": "system", "content": "fetch the city's name from the each user's queries and pass in to functions. if didn't found the city's name in user query then don't take random value as city name. finally retrun the city's name from provided by user query."},
            {"role": "user", "content": user_input},
            {"role": "assistant", "content": "you have to retrieve the informations from functions. Please consider can't generate own informatons"}
            ]                 
        try:
            response = get_gpt_response(messages)            
            if response.content is None:   
                city_name = eval((response.function_call.arguments).split(',')[0].strip())
                city = city_name["city"]
                exttract_info = get_info(city)
            else:
                exttract_info = ""

            result  = conversational_qa_chain.invoke(
                {
                    "question": user_input,  
                    "context" : exttract_info,
                    "chat_history": chat_history,
                }
            )
            chat_history.append(HumanMessage(content= user_input))
            chat_history.append(AIMessage(content= result.content))   

            # result = chain.run(prompt)
            return result.content
        except Exception as e:
            result = "Unfortunately, information is not currently accessible."
            return result
       
if __name__ == '__main__':
    app.run(debug=True)
