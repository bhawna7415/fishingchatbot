import pandas as pd
from langchain.vectorstores import Pinecone
import pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
import os
from langchain.text_splitter import RecursiveCharacterTextSplitter

embeddings = OpenAIEmbeddings()
excel_file_path = 'Copy of Northland Tackle - TackleBot Data Sheet.xlsx'

def xlsx_to_table(xlsx_file_path):
    # Initialize an empty string to store the combined table text
    combined_table_text = ""

    # Read the Excel file into a dictionary of DataFrames (one DataFrame per sheet)
    xls = pd.ExcelFile(xlsx_file_path)
    sheet_names = xls.sheet_names

    for sheet_name in sheet_names:
        # Read each sheet into a pandas DataFrame
        df = pd.read_excel(xlsx_file_path, sheet_name=sheet_name)

        # Get headers and rows
        headers = df.columns.tolist()
        rows = df.values.tolist()

        # Generate table text for the current sheet
        table_text = "| " + " | ".join(headers) + " |\n"
        table_text += "| " + " | ".join("----" for _ in headers) + " |\n"
        
        for row in rows:
            table_text += "| " + " | ".join(map(str, row)) + " |\n"

        # Add a newline between sheets
        table_text += "\n\n"

        # Append the table text for the current sheet to the combined text
        combined_table_text += table_text

    return combined_table_text

# Example usage:
xlsx_file_path = 'Copy of Northland Tackle - TackleBot Data Sheet.xlsx'
table_text = xlsx_to_table(xlsx_file_path)
print(table_text)


# xls = pd.ExcelFile(excel_file_path)
# excel_data = {sheet_name: xls.parse(sheet_name) for sheet_name in xls.sheet_names}

# combined_df = pd.concat(excel_data.values(), ignore_index=True)

# text_data = combined_df.to_csv(index=False, sep='\t')

text_splitter = RecursiveCharacterTextSplitter(
            # Set a really small chunk size, just to show.
            chunk_size = 500,
            chunk_overlap  = 150,
            length_function = len,
            is_separator_regex = False,
        )
docs = text_splitter.create_documents([table_text])

pinecone.init(
                api_key=os.getenv("PINECONE_API_KEY"),  # find at app.pinecone.io
                environment=os.getenv("PINECONE_ENV"),  # next to api key in console
            )
index_name = 'fishingchatbot'

if index_name not in pinecone.list_indexes():

    pinecone.create_index(name=index_name, metric="cosine", dimension=1536)

vectorstore = Pinecone.from_documents(
    documents=docs,
    embedding=embeddings,
    index_name=index_name
)


