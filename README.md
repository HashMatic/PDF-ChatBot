
# LLM-Powered PDF Chatbot

![](https://miro.medium.com/v2/resize:fit:500/1*JypwrsxUzXJwRWBJUruscQ.png)

This project is a chatbot application powered by a Large Language Model (LLM) that specializes in processing and answering questions related to the content of PDF files. The chatbot employs the LangChain library for efficient text manipulation and utilizes OpenAI's GPT-3.5, a cutting-edge LLM, to provide accurate and contextually relevant responses.

Key Features-

 - PDF Content Analysis: Upload a PDF document, and the chatbot will automatically extract and process its text content.

 - Text Chunk Embeddings: The application divides the PDF content into manageable text chunks and generates embeddings using LangChain and FAISS, enabling efficient querying.

 - Interactive Questioning: Users can interact with the chatbot by asking questions about the uploaded PDF content.

 - GPT-3.5 Integration: OpenAI's GPT-3.5 model is employed to generate detailed answers to user questions.

 - User-Friendly Interface: The chat conversation is presented in a user-friendly interface using Streamlit and Streamlit-Chat.

## Table Of Contents

 - [Installation]()
 - [Usage]()
 - [Acknowledgements]()
 - [Technologies Used]()
 - [Code Description]()
 - [Screenshots]()
 - [Future Scope]()
 - [Contributing]()
 - [License]()
## Installation

To get started with the PDF Chatbot, follow these steps to set up the application environment:

1.Clone the Repository: Begin by cloning this GitHub repository to your local machine using the following command:

```bash
git clone https://github.com/abhisheksingh-17/PDF-ChatBot.git
```

2.Create a Virtual Environment: It's recommended to set up a virtual environment to isolate the project's dependencies. You can create a virtual environment using Python's built-in venv module:

```bash
python -m venv venv
```

4.Activate the Virtual Environment: Activate the virtual environment using the appropriate command based on your operating system:

```bash
source venv\Scripts\activate  # On Windows
source venv/bin/activate # On macOS and Linux
```
4.Install Dependencies: Install the required Python dependencies using pip:

```bash
pip install -r requirements.txt
```

5.Configure API Key: Obtain an API key from OpenAI and set it as an environment variable named OPENAI_API_KEY. 

```bash
OPENAI_API_KEY=your-api-key
```

6.Run the Application: Launch the PDF Chatbot application using Streamlit:

```bash
streamlit run app.py
```

7.Interact with the Chatbot: Once the application is running, open the provided URL in your web browser and start uploading PDF files, asking questions, and engaging in a conversational interaction with the chatbot.

Ensure that you have Python and pip installed on your system before proceeding with the installation. Additionally, note that the PDF Chatbot relies on external libraries such as Streamlit and LangChain, which are automatically installed when you run the pip install -r requirements.txt command.
## Usage

The PDF Chatbot is a user-friendly and interactive tool that leverages the power of natural language processing to assist users in extracting information and insights from PDF documents. To use the chatbot, follow these simple steps:

 - Upload PDF: Launch the application and upload a PDF document of your choice.

 - Ask Questions: Enter your questions related to the PDF content into the text input field.

 - Receive Responses: The chatbot will analyze the PDF content and provide relevant answers and insights to your queries.

 - Interactive Conversation: Engage in a back-and-forth conversation with the chatbot to explore different aspects of the PDF content.

The chatbot's intuitive interface and integration with LangChain's language processing capabilities make it a powerful tool for extracting valuable information from PDF documents in a conversational manner. Whether you're a researcher, student, or professional, the PDF Chatbot is designed to streamline the process of extracting insights from PDF files.
## Acknowledgements

I would like to express our sincere gratitude to the developers and contributors of the open-source libraries and frameworks that made this project possible. Special thanks to the Streamlit community for creating a user-friendly platform that enabled the creation of the interactive chatbot interface. I also extend our appreciation to the LangChain team for providing essential natural language processing components that enrich the chatbot's capabilities. Additionally, I acknowledge the pioneering work of OpenAI in the field of language models, which played a pivotal role in enhancing the chatbot's responsiveness. I am thankful for the valuable resources and documentation provided by these projects, which guided me in building a feature-rich PDF chatbot application.
## Technologies Used

This project leverages a combination of powerful technologies to create an innovative PDF chatbot experience. The key technologies used in this project include:

 - Streamlit: The user interface and interactive elements of the chatbot are built using Streamlit, a popular Python library for creating web applications with minimal code.

 - LangChain: LangChain provides essential natural language processing (NLP) capabilities, including text splitting, embeddings, and vector stores. These components enable the chatbot to understand and process text data effectively.

 - OpenAI: The project harnesses the capabilities of OpenAI's state-of-the-art language models to generate meaningful and contextually relevant responses to user queries.

 - PyPDF2: PyPDF2 is utilized to extract text content from uploaded PDF files, making the PDF data accessible for analysis and response generation.

 - FAISS: FAISS, a library for efficient similarity search and clustering, is employed to organize and search through the text data extracted from PDF documents.

 - dotenv: The dotenv library helps manage environment variables, ensuring the secure storage and access of sensitive API keys and configurations.

 - Python: The project is implemented in Python, a versatile and widely used programming language that facilitates seamless integration of various libraries and technologies.

These technologies work in harmony to create an intelligent and interactive PDF chatbot that empowers users to engage with PDF documents in a new and dynamic way.
## Code Description

```python
# Import required libraries
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
```
This section of the code imports various Python libraries and modules that are necessary for building the PDF Chatbot application. Let's break down each import:

 - import streamlit as st: Imports the Streamlit library, which is used to create interactive web applications with Python.

 - from dotenv import load_dotenv: Imports the load_dotenv function from the dotenv library, which is used to load environment variables from a .env file.

 - import pickle: Imports the pickle module, which is used for serializing and deserializing Python objects (e.g., saving and loading data structures).

 - from PyPDF2 import PdfReader: Imports the PdfReader class from the PyPDF2 library, which is used for reading PDF files.

 - from streamlit_extras.add_vertical_space import add_vertical_space: Imports the add_vertical_space function from the streamlit_extras library, which provides additional Streamlit components.

 - from langchain.text_splitter import RecursiveCharacterTextSplitter: Imports the RecursiveCharacterTextSplitter class from the langchain library, which is used for splitting text into smaller chunks.

 - from langchain.embeddings.openai import OpenAIEmbeddings: Imports the OpenAIEmbeddings class from the langchain library, which provides embeddings using OpenAI's language model.

 - from langchain.vectorstores import FAISS: Imports the FAISS class from the langchain library, which is used for creating vector stores for text data.

 - from langchain.llms import OpenAI: Imports the OpenAI class from the langchain library, which represents OpenAI's language model.

 - from langchain.chains.question_answering import load_qa_chain: Imports the load_qa_chain function from the langchain library, which loads a question-answering chain.

 - from langchain.callbacks import get_openai_callback: Imports the get_openai_callback function from the langchain library, which provides a callback for interacting with OpenAI.

 - import os: Imports the os module, which provides functions for interacting with the operating system.

 - from streamlit_chat import message as st_message: Imports the message function from the streamlit_chat library and renames it as st_message. This function is used to display messages in the chat interface.

This import section ensures that all necessary functions and classes from external libraries are available for use in the PDF Chatbot application.

```python
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
```
Within the Streamlit app, this section adds content to the sidebar. It displays the title "LLM Chatbot" and a brief description of the app along with links to the technologies used.

```python
def main():
    st.header("Chat with PDF 💬")
    
    load_dotenv()

    # Load the API key from the environment variable
    embeddings = OpenAIEmbeddings(api_key=os.getenv("OPENAI_API_KEY"))
```
The main() function is defined here. It serves as the entry point for the Streamlit app. It starts by displaying a header "Chat with PDF 💬" to indicate the app's purpose. The load_dotenv() function is called to load environment variables, and the OpenAI API key is loaded from the environment to initialize the language embeddings.

```python
    # upload a PDF file
    pdf = st.file_uploader("Upload your PDF", type='pdf')
```
This line uses Streamlit's file_uploader widget to allow the user to upload a PDF file. The user interface displays a button labeled "Upload your PDF" which, when clicked, enables the user to select a PDF file from their local device.

```python
    # Initialize session state
    if 'messages' not in st.session_state:
        st.session_state.messages = []
```
This code initializes the messages session state list. It ensures that the messages list exists in the session state and is empty when the app starts. The session state is used to maintain the chat history between interactions.

```python
if pdf is not None:
```
This line checks if the pdf variable is not None. In the context of a Streamlit application, this typically means that a PDF file has been uploaded by the user.

```python
    pdf_reader = PdfReader(pdf)
```
This line creates a PdfReader object named pdf_reader by passing the uploaded PDF file (pdf) to it. The PdfReader class from the PyPDF2 library allows you to read and manipulate PDF documents.

```python
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
```
This loop iterates over each page in the PDF document using the pdf_reader.pages generator. For each page, it extracts the text content using the extract_text() method and appends the extracted text to the text variable. This loop essentially compiles the entire text content of the PDF document.

```python
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
```
This block of code initializes a RecursiveCharacterTextSplitter object named text_splitter with specific configuration parameters. This object is used to split the compiled text content into smaller "chunks" for further processing. The parameters provided define how the text should be split: chunk_size determines the maximum size of each chunk, chunk_overlap specifies the amount of overlap between chunks, and length_function is a function that calculates the length of the text.

```python
    chunks = text_splitter.split_text(text=text)
```
This line uses the text_splitter object to split the compiled text content into smaller chunks. The resulting chunks variable holds a list of text segments, each representing a smaller portion of the original PDF content. This chunking is useful for processing large text documents more efficiently.

In summary, this section of the code takes an uploaded PDF file, reads its text content, splits the content into smaller chunks, and stores these chunks in the chunks variable for further processing.

```python
        # embeddings
        store_name = pdf.name[:-4]
        st.write(f'{store_name}')

        if os.path.exists(f"{store_name}.pkl"):
            with open(f"{store_name}.pkl", "rb") as f:
                VectorStore = pickle.load(f)
        else:
            embeddings = OpenAIEmbeddings()
            VectorStore = FAISS.from_texts(chunks, embedding=embeddings)
            with open(f"{store_name}.pkl", "wb") as f:
                pickle.dump(VectorStore, f)
```

 - store_name = pdf.name[:-4]: This line extracts the name of the uploaded PDF file using the pdf.name attribute. The [:-4] slicing is used to remove the last four characters (assumed to be the ".pdf" file extension) from the filename. The resulting store_name variable will be used as the base name for storing embeddings.

 - st.write(f'{store_name}'): This line uses Streamlit's st.write() function to display the store_name in the app's user interface. This helps visually indicate the processing of embeddings for the specific PDF file.

 - if os.path.exists(f"{store_name}.pkl"):: This line checks if a pickle file (with a .pkl extension) corresponding to the store_name already exists in the file system. The os.path.exists() function is used to verify the existence of the file.

 - with open(f"{store_name}.pkl", "rb") as f:: If the pickle file exists, this line opens the file in binary read mode ("rb") using a context manager (with statement). The file object is assigned to the variable f, which will be used to read the pickle content.

 - VectorStore = pickle.load(f): Inside the context manager, this line loads the serialized VectorStore object from the pickle file using the pickle.load() function. This precomputed vector store contains embeddings for the PDF's text chunks.

 - else:: If the pickle file does not exist (the if condition is not met), the code inside the else block is executed.

 - embeddings = OpenAIEmbeddings(): An instance of OpenAIEmbeddings is created. This prepares for generating embeddings using OpenAI's language model.

 - VectorStore = FAISS.from_texts(chunks, embedding=embeddings): A new VectorStore is created using the FAISS.from_texts() method, which takes the chunks (textual chunks of the PDF) and the embeddings instance as arguments. This step generates embeddings for the text chunks and creates an efficient vector store for similarity searches.

 - with open(f"{store_name}.pkl", "wb") as f:: After generating embeddings, this line opens a new or existing pickle file (with a .pkl extension) in binary write mode ("wb") using a context manager. The file object is assigned to the variable f, which will be used to write the pickle content.

 - pickle.dump(VectorStore, f): Inside the context manager, this line serializes and writes the VectorStore object (containing the embeddings) to the pickle file using the pickle.dump() function.

In summary, this code section manages the creation, storage, and retrieval of embeddings using pickle files. It first checks if embeddings for a specific PDF have already been generated and stored. If they exist, it loads the embeddings from the pickle file. If not, it generates new embeddings using OpenAI's language model, creates a vector store using FAISS, and then stores the embeddings in a pickle file for future use.

```python
        # Accept user questions/query
        query = st.text_input("Ask questions about your PDF file:")
```
This line presents an input box to the user where they can type their questions or queries related to the uploaded PDF.

```python
        if query:
            docs = VectorStore.similarity_search(query=query, k=3)

            llm = OpenAI()
            chain = load_qa_chain(llm=llm, chain_type="stuff")
            with get_openai_callback() as cb:
                response = chain.run(input_documents=docs, question=query)
```

 - if query: This line checks if the user has entered a non-empty query in the text input field. The subsequent code will be executed only if a query is provided.

 - docs = VectorStore.similarity_search(query=query, k=3): If a query is provided, this line uses the VectorStore (containing the embeddings of the PDF's text chunks) to perform a similarity search. It searches for the top 3 most similar chunks in the PDF text to the user's query. The search is based on the semantic similarity of embeddings.

 - llm = OpenAI(): An instance of the OpenAI language model is created. This instance will be used for generating responses to user queries.

 - chain = load_qa_chain(llm=llm, chain_type="stuff"): A question-answering (QA) chain is loaded using the load_qa_chain() function. The chain is configured to use the provided llm instance (OpenAI language model) and is of the type "stuff," indicating it can handle general questions.

 - with get_openai_callback() as cb:: This line uses the get_openai_callback() context manager to set up an OpenAI callback. The callback captures information about the OpenAI API request, which can be useful for debugging and monitoring.

 - response = chain.run(input_documents=docs, question=query): Inside the context manager, this line runs the loaded QA chain to generate a response to the user's query. It takes the docs (chunks of text) as input documents and the user's query as the question. The response variable will hold the AI-generated answer to the user's question.

In summary, this code block handles the user's query by performing a similarity search to identify relevant text chunks from the PDF. It then uses OpenAI's language model and a question-answering chain to generate a response to the user's query based on the identified relevant text chunks.

```python
            if "I don't know." in response:
                response = "Please ask a question related to the content of the PDF."
            
            st.session_state.messages.insert(0, {"role": "user", "content": query})
            st.session_state.messages.insert(1, {"role": "assistant", "content": response})
```
This block checks if the response from the question-answering chain contains the message "I don't know." If it does, the response is replaced with a message instructing the user to ask a question related to the PDF content. The user's question and the bot's response are then added to the chat history stored in the session state.

```python
    for msg in st.session_state.messages:
        if msg['role'] == 'user':
            st_message(msg['content'], is_user=True)
        else:
            st_message(msg['content'])
```
Finally, this section iterates through the chat history stored in the session state and displays each message in the chat UI. The st_message function is used to render the messages, distinguishing between user and assistant messages.

```python
if __name__ == '__main__':
    main()
```
This part of the code ensures that the main() function is executed when the script is run directly (not when imported as a module).

This detailed explanation should provide a clear understanding of each line of the code and its role in building the PDF Chatbot application.

## Screenshots

 - Home Page-

 ![](https://github.com/abhisheksingh-17/PDF-ChatBot/blob/main/Results/1.png?raw=true)

 - Questions from Indian Consitution pdf-

 ![](https://github.com/abhisheksingh-17/PDF-ChatBot/blob/main/Results/2.png?raw=true)

 ![](https://github.com/abhisheksingh-17/PDF-ChatBot/blob/main/Results/3.png?raw=true)

 - Questions from Life Processes pdf-

 ![](https://github.com/abhisheksingh-17/PDF-ChatBot/blob/main/Results/4.png?raw=true)

 - Questions from Nationalism pdf-

 ![](https://github.com/abhisheksingh-17/PDF-ChatBot/blob/main/Results/5.png?raw=true)

 - Questions from Resume pdf-

 ![](https://github.com/abhisheksingh-17/PDF-ChatBot/blob/main/Results/6.png?raw=true)

 - If Question ask different from the content-

 ![](https://github.com/abhisheksingh-17/PDF-ChatBot/blob/main/Results/7.png?raw=true)
## Future Scope

This project lays the foundation for a versatile and intelligent PDF chatbot, powered by cutting-edge language models and advanced NLP techniques. As we look to the future, several exciting possibilities emerge for further enhancing and expanding this chatbot's capabilities. Some potential avenues for development include:

 - Multi-Document Support: Extend the chatbot's functionality to handle multiple PDF documents and provide coherent responses by leveraging cross-document context.

 - Interactive Learning: Implement a feature that allows users to provide feedback on the bot's responses, enabling the chatbot to learn and improve over time.

 - Rich Media Integration: Enhance the chatbot's responses by incorporating multimedia elements such as images, charts, and graphs extracted from the PDF content.

 - Customization and Personalization: Allow users to customize the chatbot's behavior, language style, and response preferences to create a more personalized interaction.

 - Natural Conversation Flow: Develop a more dynamic and natural conversation flow that enables the bot to engage in extended dialogues, follow-up questions, and context retention.

 - Advanced NLP Techniques: Integrate more advanced NLP techniques, such as sentiment analysis, summarization, and named entity recognition, to provide deeper insights and value to users.

 - Integration with Knowledge Bases: Connect the chatbot to external knowledge bases or databases to expand its access to information and improve response accuracy.

 - User Interface Enhancements: Continuously refine and improve the user interface to enhance usability, accessibility, and overall user experience.

 - Integration with APIs: Integrate with external APIs to provide additional services, such as translation, language detection, or real-time data retrieval.

 - Collaborative Features: Implement collaborative features that allow multiple users to interact with the chatbot simultaneously, facilitating group discussions and information sharing.

The future of this project holds the promise of a sophisticated and intelligent PDF chatbot that can revolutionize the way users interact with and extract information from PDF documents.

Feel free to incorporate this future scope section into your README, and adapt it based on the specific directions you envision for your project's growth. If you have any other sections you'd like assistance with, please let me know!






## Contributing

I welcome and appreciate contributions to this project from the community. If you're interested in contributing, please follow these guidelines:

 - Fork the repository and create a new branch for your feature or bug fix.

 - Make your changes and ensure they are well-tested.

 - Commit your changes with a clear and descriptive commit message.

 - Push your changes to your forked repository.

 - Create a pull request to the main repository's main branch, explaining the changes you've made.

Please ensure that your contributions align with our code of conduct and follow the coding standards of the project. By contributing to this project, you agree to release your contributions under the terms of the project's license.
## License

This project is licensed under the MIT License. The MIT License is a permissive open-source license that grants permission to anyone to use, modify, and distribute the software for any purpose, subject to the conditions that the original copyright notice and permission notice are included in all copies or substantial portions of the software. The software is provided "as is," without any warranty, and the authors or copyright holders shall not be liable for any claims or damages arising from the use of the software.