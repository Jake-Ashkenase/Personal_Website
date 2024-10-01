import requests
import streamlit as st
from streamlit_lottie import st_lottie
from PIL import Image
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


openai.api_key = os.getenv("OPENAI_API_KEY")

load_dotenv()

loader = CSVLoader(file_path="ME.csv")
documents = loader.load()

embeddings = OpenAIEmbeddings()
db = FAISS.from_documents(documents, embeddings)


def retrieve_info(query):
    similar_response = db.similarity_search(query, k=4)
    page_contents_array = [doc.page_content for doc in similar_response]
    return page_contents_array


llm = ChatOpenAI(temperature=0, model="gpt-4-1106-preview")


template = """You are Jake Ashkenase, below is data containing information about him.

Relevant data:

{relevant_data}

Use the previous data to answer the presented interview question from the perspective of Jake Ashkenase. 

Presented question:

{question}


Instructions:
    ~Respond as Jake Ashkenase, maintaining a polite and professional tone.
    ~Keep responses under 200 words, focusing on only providing information that is relevant to the question. 
    ~ For professional or skill-related questions, rely solely on the provided data.
    ~Avoid stating "As Jake Ashkenase, I would..." or "As Jake Ashkenase, I will..." (this is implied).
    ~For very personal questions where data is not present you may use a witty response, keeping it under 50 words.

"""

prompt = PromptTemplate(
    input_variables=["question", "relevant_data"],
    template=template
)

chain = LLMChain(llm=llm, prompt=prompt)


def generate_response(question):
    relevant_data = retrieve_info(question)
    response = chain.run(question=question, relevant_data=relevant_data)
    return response



# Find more emojis here: https://www.webfx.com/tools/emoji-cheat-sheet/
st.set_page_config(page_title="My Webpage", page_icon=":tada:", layout="wide")


def load_lottieurl(url):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()


# Use local CSS
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


local_css("style/style.css")

# ---- LOAD ASSETS ----
lottie_coding = load_lottieurl("https://assets5.lottiefiles.com/packages/lf20_fcfjwiyb.json")
innovatema = Image.open("images/InnovateMA.png")
lightcast_article = Image.open("images/lightcast.jpg")

# ---- HEADER SECTION ----
with st.container():
    left_column, right_column = st.columns(2)

    with left_column:
        st.subheader("Hi, my name is Jake:wave:")
        st.title("A Data Scientist Passionate about AI")
        st.write(
            "I am passionate about utilizing AI to make positive change in the communities around me."
        )
        st.write("[Learn More >](https://www.linkedin.com/in/jake-ashkenase/)")

    with right_column:
        with open("Jake_Ashkenase_Resume_2024.pdf", "rb") as file:
            st.download_button(label="Download my Resume", data=file, file_name="Jake_Ashkenase_Resume_2024.pdf", mime="application/pdf")

        # Use st.image() and adjust styling using the image argument
        st.image("Jake Ashkenase Headshot.png", width=200, use_column_width=False, output_format="auto")

        # Apply inline CSS using st.markdown()
        st.markdown(
            """
            <style>
            img {
                border-radius: 10px;
            }
            </style>
            """,
            unsafe_allow_html=True
        )
 

# ---- WHAT I DO ----
with st.container():
    st.write("---")
    left_column, right_column = st.columns(2)
    with left_column:
        st.header("Learn About Me")

        message = st.text_area("Hi, I am Jake Ashkenase. What would you like to know about me.")
        # Process and display the response
        if message:
            st.write("Typing...")
            result = generate_response(message)
            st.info(result)
    with right_column:
        st_lottie(lottie_coding, height=300, key="coding")

# ---- PROJECTS ----
with st.container():
    st.write("---")
    st.header("My Work")
    st.write("##")
    image_column, text_column = st.columns((1, 2))
    with image_column:
        st.image(lightcast_article)
    with text_column:
        st.subheader("Global AI: Who’s Leading the Race for Jobs and Skills")
        st.write(
            """
            Looking at trends in the global labor market surrounding AI. 
            Analysis on the countries and industries with the most demand for AI jobs, and the types of skills
            that are being requested in AI job postings
            """
        )
        st.markdown("[Read Here...](https://lightcast.io/resources/blog/global-ai-skills-jobs)")
with st.container():
    image_column, text_column = st.columns((1, 2))
    with image_column:
        st.image(innovatema)
    with text_column:
        st.subheader("InnovateMA: AI for Impact Program")
        st.write(
            """
            Collaberating with the Massachusetts Executive Office of Health and Human Services to increase the efficiency 
            of their eligibility offices and increase the ease of applying to life changing healthcare programs. 
            """
        )
        st.markdown("[Read Here...](https://news.northeastern.edu/2024/07/18/healey-ai-task-force-northeastern/?utm_source=News%40Northeastern&utm_campaign=d196f86d81-EMAIL_CAMPAIGN_2022_09_22_11_00_COPY_01&utm_medium=email&utm_term=0_508ab516a3-d196f86d81-279411200)")

    st.write("")
    st.write("")
    st.write("[Website Inspired By Dhruv Kamalesh Kumar](https://www.linkedin.com/in/dhruvkamaleshkumar/)") 

