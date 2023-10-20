import streamlit as st
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from dotenv import load_dotenv

load_dotenv()

# 1. Vectorise the sales response csv data
loader = CSVLoader(file_path="ME.csv")
documents = loader.load()

embeddings = OpenAIEmbeddings()
db = FAISS.from_documents(documents, embeddings)

# 2. Function for similarity search
def retrieve_info(query):
    similar_response = db.similarity_search(query, k=3)

    page_contents_array = [doc.page_content for doc in similar_response]

    # print(page_contents_array)

    return page_contents_array


# 3. Setup LLMChain & prompts
llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-16k-0613")

template = """
You are Dhruv Kamalesh Kumar, responding to questions as yourself. Please review the data provided to understand more about me.

I will share a prospective employer's question with you, and you will provide a response as Dhruv Kamalesh Kumar, maintaining a polite and professional tone.

Below is a question from a prospective employer:

{question}

Here is the relevant data:

{relevant_data}

Please provide a response as Dhruv Kamalesh Kumar, incorporating the data to address this prospective employer:
"""

prompt = PromptTemplate(
    input_variables=["question", "relevant_data"],
    template=template
)

chain = LLMChain(llm=llm, prompt=prompt)


# 4. Retrieval augmented generation
def generate_response(question):
    relevant_data = retrieve_info(question)
    response = chain.run(question=question, relevant_data=relevant_data)
    return response


# 5. Build an app with streamlit
def main():

    st.set_page_config(
        page_title="Get to know me", page_icon=":male-technologist:")

    announcement = "(P.S. This was built in 3 days, imagine what I can do in 30 :sunglasses:)"
    # st.write(announcement)
    st.toast(body=announcement)
    st.balloons()
    col1, col2, col3 = st.columns([1, 2, 1])
    col1.header("Get to know me")
    col2.image("memoji.png", width=200)
    with open("resume.pdf", "rb") as file:
        col3.download_button(label="Download my Resume", data=file, file_name="resume.pdf", mime="application/pdf")

    message = st.text_area("Hi, I am Dhruv Kamalesh Kumar. Ask me any questions you want to know about me.")

    if message:
        st.write("Typing...")

        result = generate_response(message)

        st.info(result)


if __name__ == '__main__':
    main()