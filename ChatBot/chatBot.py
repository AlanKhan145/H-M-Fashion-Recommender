import os
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.chat_models import ChatOpenAI
from openai.error import RateLimitError

# API key từ biến môi trường
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

st.header("My First Chatbot")

with st.sidebar:
    st.title("Your Documents")
    file = st.file_uploader("Upload a PDF file and start asking questions", type="pdf")

if file is not None:
    pdf_reader = PdfReader(file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()

    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n"],
        chunk_size=1000,
        chunk_overlap=150,
        length_function=len
    )
    chunks = text_splitter.split_text(text)

    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    vector_store = FAISS.from_texts(chunks, embeddings)

    user_question = st.text_input("Type your question here")
    if user_question and st.button("Submit Question"):
        try:
            match = vector_store.similarity_search(user_question)
            llm = ChatOpenAI(
                openai_api_key=OPENAI_API_KEY,
                temperature=0,
                max_tokens=1000,
                model_name="gpt-4"
            )
            chain = load_qa_chain(llm, chain_type="stuff")
            with st.spinner("Processing your request..."):
                response = chain.run(input_documents=match, question=user_question)
                st.write(response)
        except RateLimitError:
            st.error("Rate limit exceeded. Please wait and try again.")
