
import os
import sys
from pinecone import Pinecone as PineconeClient, ServerlessSpec
from langchain_community.llms import Replicate
from langchain_community.vectorstores import Pinecone
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import ConversationalRetrievalChain

# Replicate API token
os.environ['REPLICATE_API_TOKEN'] = "r8_3n9r3UXzK8SYJmNqcT8RCeJbRraeqnw0XH7qZ"

# Set Pinecone API key in environment variables
os.environ['PINECONE_API_KEY'] = '5f5ebe49-8bfd-4153-af6a-143a5504ddc8'


# Initialize Pinecone
pc = PineconeClient(api_key='5f5ebe49-8bfd-4153-af6a-143a5504ddc8')

# Load and preprocess the PDF document
loader = PyPDFLoader('data_merged.pdf')
documents = loader.load()

# Split the documents into smaller chunks for processing
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
texts = text_splitter.split_documents(documents)

# Use HuggingFace embeddings for transforming text into numerical vectors
embeddings = HuggingFaceEmbeddings()

# Set up the Pinecone vector database
index_name = "pdf-chat"
if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=768,  # adjust this according to your embeddings dimension
        metric='euclidean',
        spec=ServerlessSpec(
            cloud='gcp',
            region='us-west2'  # adjust this according to your preferred region
        )
    )
vectordb = Pinecone.from_documents(texts, embeddings, index_name='pdf-chat')

# Initialize Replicate Llama2 Model
llm = Replicate(
    model="a16z-infra/llama13b-v2-chat:df7690f1994d94e96ad9d568eac121aecf50684a0b0963b25a41cc40061269e5",
    model_kwargs={"temperature": 0.75, "max_length": 3000}
)

# Set up the Conversational Retrieval Chain
qa_chain = ConversationalRetrievalChain.from_llm(
    llm,
    vectordb.as_retriever(search_kwargs={'k': 2}),
    return_source_documents=True
)

chat_history = []
while True:
    query = input('Prompt: ')
    if query.lower() in ["exit", "quit", "q"]:
        print('Exiting')
        sys.exit()
    result = qa_chain.invoke({'question': query, 'chat_history': chat_history})
    print('Answer: ' + result['answer'] + '\n')
    chat_history.append((query, result['answer']))