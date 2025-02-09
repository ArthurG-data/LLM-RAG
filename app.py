# import langchain dependencies
from langchain.document_loaders import PyPDFLoader
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter

# import chromadd to store the vector database localy
import chromadb
from langchain_chroma import Chroma

# create a persistent DB
client = chromadb.PersistentClient(path="db/")

# create a collection 

collection = client.get_or_create_collection(name="Magic")

# Strealit for UI
import streamlit as st

#Bring an interface
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings

#Load api key
load_dotenv()

model_id = "gpt-4o-mini"
llm = ChatOpenAI(model=model_id)

# Bring the pdf loader
@st.cache_resource
def load_pdf():
  
    pdf_name = 'pdf/MagicCompRules 20250207.pdf'

    loaders = PyPDFLoader(pdf_name)
    document = loaders.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=0)
    chunked_documents = text_splitter.split_documents(document)

    vectorSore = Chroma.from_documents(
    documents=chunked_documents,
    collection_name='Magic',
    client=client,
    embedding=OpenAIEmbeddings()
)
    #return vector database
    return vectorSore

# Load er on up
index = load_pdf()

# create a Q&A chain
chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type='stuff',
    retriever= index.as_retriever(),
    input_key='question'
)

# Set the app title
st.title('Judge!')

# Session state message variable to hold all the old messages
if 'messages' not in st.session_state:
    st.session_state.messages = []

# Display al the historical messages
for message in st.session_state.messages:
    st.chat_message(message['role']).markdown(message['content'])

# Build a prompt input template to display the promps
prompt = st.chat_input('Ask your question here')

#if the user hits enter then
if prompt:
    # Display the prompt
    st.chat_message('user').markdown(prompt)
    # Store the User Prompt
    st.session_state.messages.append({'role':'user', 'content':prompt})
    #send prompt to the PDF Q&A chain
    response =chain.run(prompt)
    # Show the LLM response
    st.chat_message('judge').markdown(response)
    # Store the LLM response in state
    st.session_state.messages.append(
        {'role':'judge', 'content':response}
    )
   