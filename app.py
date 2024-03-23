import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings,HuggingFaceInstructEmbeddings # we will only go with HuggingFaceInstructEMbedding beacuse openAI embedding is chargeable
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from InstructorEmbedding import INSTRUCTOR
from langchain.memory import ConversationBufferMemory #it will store the conversation in a buffer memory
from langchain.chains import ConversationalRetrievalChain #allows chat with our vector store
from langchain_community.llms import HuggingFaceHub,HuggingFaceEndpoint 

from template import css,bot_template,user_template
import time

#this function will extract the text from each pdf and store it in a variable called text, this text will be the returned as the output of the function
def get_pdf_text(pdf_docs):
    text=""
    for pdf in pdf_docs:
        pdf_reader=PdfReader(pdf) # pdfReader will take the pdf and will take each page at a time
        for page in pdf_reader.pages:
            text+=page.extract_text()
    return text

def get_chunks(raw_text):
    text_splitter=CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks=text_splitter.split_text(raw_text)
    return chunks

def get_vectorstore(chunks_data):
    
    embeddings=HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")
    vectorstore = FAISS.from_texts(texts=chunks_data, embedding=embeddings)
    return vectorstore

def get_conversation_chain(vectorstore):
    
    llm = HuggingFaceEndpoint(repo_id="google/flan-t5-xxl")
    memory=ConversationBufferMemory(memory_key='chat_history',return_messages=True)
    conversation_chain=ConversationalRetrievalChain.from_llm(
         llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain


def handle_user_input(user_question):
    response=st.session_state.conversation({"question":user_question})
    st.write(response)




def main():
    load_dotenv()
    st.set_page_config(page_title="Chat with multiple PDFs",page_icon=":books:")
    st.write(css, unsafe_allow_html=True)
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    
    # title of the page
    

    #header of the page
    st.header("Chat with multiple PDFs :books:")

    #taking text input
    user_question=st.text_input("Ask any question about your document")
    if user_question:
        handle_user_input(user_question)

    st.write(user_template.replace("{{MSG}}","Hi AI"),unsafe_allow_html=True)
    st.write(bot_template.replace("{{MSG}}","Hi Human"),unsafe_allow_html=True)
    with st.sidebar:
        st.subheader("Your uploaded documents: ")
        pdf_docs=st.file_uploader("Upload your PDFs here and click on 'Process'",accept_multiple_files=True)
        if st.button("Process"):
            with st.spinner("Processing"):
                time.sleep(2)
                
                #get the pdf text   
                raw_text=get_pdf_text(pdf_docs)
                # st.write(raw_text)

                #get the text chunks
                chunks_data=get_chunks(raw_text)
                # st.write(chunks_data)
                
                
                #create vector store
                vectorstore=get_vectorstore(chunks_data)

                # create conversation chain
                # the variable conversation should not be reinitialized that's why we are making it as a session variable.
                # Also, we can use 'conversation' variable outside the scope of the sidebar
                st.session_state.conversation=get_conversation_chain(vectorstore)   
if __name__=='__main__':
    main()