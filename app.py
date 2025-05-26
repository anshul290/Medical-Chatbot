from flask import Flask, render_template, jsonify, request
from src.helper import download_hugging_face_embeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_community.llms import CTransformers
from dotenv import load_dotenv
from src.prompt import *
from pinecone import Pinecone
import pinecone
import os

app = Flask(__name__)

load_dotenv()

# Load API keys
PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
PINECONE_API_ENV = os.environ.get('PINECONE_API_ENV')
INDEX_NAME = "medical-bot"

# Initialize Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(INDEX_NAME)

# Load Embeddings
embeddings = download_hugging_face_embeddings()

# Connect to existing Pinecone index
docsearch = PineconeVectorStore(index=index, embedding=embeddings, text_key="text")

# Create Prompt Template
PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
chain_type_kwargs = {"prompt": PROMPT}

# Load Local LLM
llm = CTransformers(
    model="D:/Medical-chatbot-using-Llama-2/Model/llama-2-7b-chat.ggmlv3.q4_0.bin",
    model_type="llama",
    config={'max_new_tokens': 512, 'temperature': 0.8}
)

# Build QA Chain
qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=docsearch.as_retriever(search_kwargs={'k': 2}),
    return_source_documents=True,
    chain_type_kwargs=chain_type_kwargs
)

@app.route("/")
def index():
    return render_template('chat.html')

@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    input = msg
    print(input)
    result=qa({"query": input})
    print("Response : ", result["result"])
    return str(result["result"])


if __name__ == '__main__':
    app.run(host="0.0.0.0", port = 8000, debug=True)


