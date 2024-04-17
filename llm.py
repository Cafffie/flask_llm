from flask import Flask, request, render_template
from sentence_transformers import SentenceTransformer
import spacy
import PyPDF2

import langchain
import google.generativeai as genai
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI

from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS

import os
from dotenv import load_dotenv
load_dotenv()

genai.configure(api_key=os.environ['GOOGLE_API_KEY'])
llm = ChatGoogleGenerativeAI(model='gemini-pro',
                             temperature=0.3,
                             convert_system_message_to_human=True)
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
vectordb_filepath = "faiss_index"

app = Flask(__name__)
nlp = spacy.load("en_core_web_sm")


def create_vector_database(pdf_file):
    # Extracting text from pdf
    text = ''
    with open(pdf_file, 'rb') as f:
        reader = PyPDF2.PdfFileReader(f)
        for page_num in range(reader.pages):
            page = reader.pages[page_num]
            text += str(page.extract_text())
    text = extract_text_from_pdf(pdf_file)

    # Splitting text into sentences
    doc = nlp(text)
    sentences = [sent.text for sent in doc.sents]

    # Creating vector database
    vectordb = FAISS.from_documents(documents=sentences, embedding=embeddings)
    vectordb.save_local(vectordb_filepath)


def chain_response(question):
    vectordb = FAISS.load_local(vectordb_filepath, embeddings)
    prompt_template = """
        Given the following context and a question, generate an answer based on this context.
        Use your reasoning abilities to understand the question
        In the answer try to provide as much text as possible from each section that contains similar meaning in the source document.
        Give the correct answer to the question.

        CONTEXT: {context}
        QUESTION: {question} """

    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )

    chain = RetrievalQA.from_chain_type(llm=llm,
                                        retriever=vectordb.as_retriever(),
                                        input_key="query",
                                        return_source_documents=False,
                                        chain_type_kwargs={"prompt": prompt}
                                        )
    return chain(question)


@app.route('/')
def hello():
    return render_template("index.html")


@app.route('/get_response', methods=["POST"])
def get_response():
    if request.method == 'POST':
        question = request.form['user_question']
        response = chain_response(question)
        return render_template("index.html", res=response)
    
if __name__=="__main__":
    app.run(debug=True)