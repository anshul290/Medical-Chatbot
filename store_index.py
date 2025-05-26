from src.helper import load_pdf, text_split, download_hugging_face_embeddings
from langchain_community.vectorstores import Pinecone
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from langchain.embeddings import HuggingFaceEmbeddings
import pinecone
from dotenv import load_dotenv
import os

load_dotenv()

PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
PINECONE_API_ENV = os.environ.get('PINECONE_API_ENV')

# print(PINECONE_API_KEY)
# print(PINECONE_API_ENV)

extracted_data = load_pdf("data/")
text_chunks = text_split(extracted_data)
embeddings = download_hugging_face_embeddings()


# Step 1: Initialize embeddings
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

index_name = "medical-bot"

# Step 2: Create or connect to index and upload data
docsearch = PineconeVectorStore.from_texts(
    [t.page_content for t in text_chunks],
    embedding=embeddings,
    index_name=index_name,
    namespace="default"  # optional
)


# Step 1: Initialize Pinecone v2
pc = Pinecone(api_key="PINECONE_API_KEY")  # replace with your real API key

# Step 2: Connect to your index
index_name = "medical-bot"
index = pc.Index(
    name=index_name,
    spec=ServerlessSpec(cloud="aws", region="us-east-1")
)

# Step 3: Initialize the embeddings model
embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Step 4: Wrap the index with LangChain's PineconeVectorStore
docsearch = PineconeVectorStore(
    index=index,
    embedding=embedding_model,
    namespace="default",      # optional
    text_key="text"           # this must match the key used during upload
)




