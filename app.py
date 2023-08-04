import streamlit as st
from dotenv import load_dotenv
import pickle
from PyPDF2 import PdfReader
from streamlit_extras.add_vertical_space import add_vertical_space
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.callbacks import get_openai_callback
import os
from streamlit_chat import message as st_message

# Set page title and favicon
st.set_page_config(page_title="PDF Chatbot", page_icon=":robot_face:")

# Sidebar contents
with st.sidebar:
    st.title('LLM Chatbot')
    st.markdown('''
    ## About
    This app is an LLM-powered chatbot built using:
    - [Streamlit](https://streamlit.io/)
    - [LangChain](https://python.langchain.com/)
    - [OpenAI](https://platform.openai.com/docs/models)
    - [LLM Models](https://www.geeksforgeeks.org/large-language-model-llm/) 
    ''')

def main():
    st.header("Chat with PDF 💬")

    load_dotenv()

    # Load the API key from the environment variable
    embeddings = OpenAIEmbeddings(api_key=os.getenv("OPENAI_API_KEY"))

    # upload a PDF file
    pdf = st.file_uploader("Upload your PDF", type='pdf')

    # Initialize session state
    if 'messages' not in st.session_state:
        st.session_state.messages = []

    # st.write(pdf)
    if pdf is not None:
        pdf_reader = PdfReader(pdf)
        
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
            )
        chunks = text_splitter.split_text(text=text)

        # embeddings
        store_name = pdf.name[:-4]
        st.write(f'{store_name}')
        # st.write(chunks)

        if os.path.exists(f"{store_name}.pkl"):
            with open(f"{store_name}.pkl", "rb") as f:
                VectorStore = pickle.load(f)
            # st.write('Embeddings Loaded from the Disk')
        else:
            embeddings = OpenAIEmbeddings()
            VectorStore = FAISS.from_texts(chunks, embedding=embeddings)
            with open(f"{store_name}.pkl", "wb") as f:
                pickle.dump(VectorStore, f)

        # Accept user questions/query
        query = st.text_input("Ask questions about your PDF file:")

        if query:
            docs = VectorStore.similarity_search(query=query, k=3)

            llm = OpenAI()
            chain = load_qa_chain(llm=llm, chain_type="stuff")
            with get_openai_callback() as cb:
                response = chain.run(input_documents=docs, question=query)

            # Check if the bot's response is relevant to the PDF content
            if "I don't know." in response:
                response = "Please ask a question related to the content of the PDF."    
            
            # Update the conversation history with user's question and bot's response
            st.session_state.messages.insert(0, {"role": "user", "content": query})
            st.session_state.messages.insert(1, {"role": "assistant", "content": response})

    # Display the chat conversation using streamlit_chat
    for msg in st.session_state.messages:
        if msg['role'] == 'user':
            st_message(msg['content'], is_user=True)
        else:
            st_message(msg['content'])

if __name__ == '__main__':
    main()
