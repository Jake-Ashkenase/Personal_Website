# Import required libraries
import streamlit as st
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings.openai import OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain_community.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from dotenv import load_dotenv

import os
import openai

# Load the OpenAI API key from the environment variables
openai.api_key = os.getenv("OPENAI_API_KEY")
load_dotenv()

# Load documents from a CSV file for the chatbot's knowledge base
loader = CSVLoader(file_path="ME.csv")
documents = loader.load()

# Initialize embeddings and create a FAISS index for similarity search
embeddings = OpenAIEmbeddings()
db = FAISS.from_documents(documents, embeddings)

# Function to retrieve information based on a query
def retrieve_info(query):
    # Perform a similarity search and return the top 3 similar documents
    similar_response = db.similarity_search(query, k=3)
    page_contents_array = [doc.page_content for doc in similar_response]
    return page_contents_array

# Initialize the ChatOpenAI model with specific settings
llm = ChatOpenAI(temperature=0, model="gpt-4-1106-preview")

# Define a template for the chatbot's responses
template = """ Add your prompt here """

# Create a PromptTemplate object with the defined template
prompt = PromptTemplate(
    input_variables=["question", "relevant_data"],
    template=template
)

# Create a chain of language model and prompt for generating responses
chain = LLMChain(llm=llm, prompt=prompt)

# Function to generate response based on the user's question
def generate_response(question):
    relevant_data = retrieve_info(question)
    response = chain.run(question=question, relevant_data=relevant_data)
    return response

# Main function to run the Streamlit app
def main():
    # Set up the Streamlit page configuration and customize it
    st.set_page_config(
        page_title="Get to know me", page_icon=":male-technologist:",
        layout="wide")
    
    # Custom CSS 
    st.markdown(
        """
        <style>
        .stApp {
            background-color: #193f64;
            color: #ebc57e;
        }
        .stTextInput label {
            font-size: 18px;
            color: #333;
        }
        .stTextInput div {
            background-color: #fff;
            border-radius: 10px;
            padding: 10px;
        }
        .stTextInput textarea {
            border-radius: 10px;
            border: 1px solid #ccc;
            padding: 10px;
            font-size: 16px;
        }
        .stButton button {
            background-color: #f1ab65;
            color: #f1ab65;
            border-radius: 10px;
            padding: 10px 20px;
            font-size: 16px;
        }
        .stButton button:hover {
            background-color: #e28743;
        }
        </style>
        """, unsafe_allow_html=True
    )
    # Streamlit UI layout with columns
    col1, col2, col3 = st.columns([1, 2, 1])

    col1.markdown("<h1 style='text-align: center;'>Get to know me</h1>", unsafe_allow_html=True)

    col2.image("Jake Ashkenase Headshot.png", width=200)
    with open("Jake_Ashkenase_Resume_2024.pdf", "rb") as file:
        col3.download_button(label="Download my Resume", data=file, file_name="Jake_Ashkenase_Resume_2024.pdf", mime="application/pdf")

    # Input area for user to ask questions
    message = st.text_area("Hi, I am Jake Ashkenase. What would you like to know about me.")

    # Process and display the response
    if message:
        st.write("Typing...")
        result = generate_response(message)
        st.info(result)

# Entry point for the Streamlit application
if __name__ == '__main__':
    main()
