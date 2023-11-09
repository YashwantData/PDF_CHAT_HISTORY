import streamlit as st
import pickle
from streamlit_extras.add_vertical_space import add_vertical_space
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts.prompt import PromptTemplate

import os

# Define CSS styles for the "New Chat" button
button_style = (
    "position: absolute; top: 10px; left: 10px; "
    "z-index: 1000; padding: 10px; background-color: #4CAF50; "
    "color: white; border: none; cursor: pointer;"
)

# Define a variable to store the chat history in the Streamlit session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# "New Chat" button
if st.button("New Chat", key="start_new_chat", help="Click to start a new chat"):
    st.session_state.is_chatting = True
    st.session_state.messages = []

# Storing history in session states
if "messages" not in st.session_state:
    st.session_state.messages = []
for message in st.session_state["messages"]:
    if message["role"] == "user":
        with st.chat_message("user"):
            st.markdown(message["content"])
    elif message["role"] == "assistant":
        with st.chat_message("assistant"):
            st.markdown(message["content"])

# Sidebar contents
with st.sidebar:
    st.title('PDF BASED LLM CHATBOT🤗')
    key = st.text_input("Add your API Key")
    print(key)
    pdf = st.file_uploader("Upload your PDF", type='pdf')
    os.environ["OPENAI_API_KEY"] = key
    st.subheader("Chat History")
    for message in st.session_state.messages:
        if message["role"] == "user":
            with st.expander("User"):
                st.markdown(message["content"])
        elif message["role"] == "assistant":
            with st.expander("Assistant"):
                st.markdown(message["content"])
    st.markdown('''
    ## About APP:

    The app's primary resource is utilized to create::

    - [streamlit](https://streamlit.io/)
    - [Langchain](https://docs.langchain.com/docs/)
    - [OpenAI](https://openai.com/)

    ## About me:

    - [Linkedin](https://www.linkedin.com/in/yashwant-rai-2157aa28b)

    ''')

    st.write('💡All about pdf-based chatbot, created by Yashwant Rai')

def main(pdf):
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

        # # embeddings
        store_name = pdf.name[:-4]

        if os.path.exists(f"{store_name}.pkl"):
            with open(f"{store_name}.pkl", "rb") as f:
                VectorStore = pickle.load(f)
        else:
            embeddings = OpenAIEmbeddings()
            retry_decorator = _create_retry_decorator(embeddings, exceptions= (opeanai.error.Timeout,openai.error.APIError))
            with retry_decorator:
                VectorStore = FAISS.from_texts(chunks, embedding=embeddings)

        query = st.chat_input(placeholder="Ask questions about your PDF file:")

        # Check if the "New Chat" button was clicked to reset the chat state
        if st.session_state.start_new_chat:
            st.session_state.is_chatting = True
            st.session_state.messages = []  # Reset chat history

        memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

        if query:
            chat_history = []
            with st.chat_message("user"):
                st.markdown(query)
            st.session_state.messages.append({"role": "user", "content": query})

            custom_template = """Given the following conversation and a follow-up question, rephrase the follow-up question to be a standalone question. At the end of the standalone question add this 'Answer the question in English language.' If you do not know the answer reply with 'I am sorry'.
                        Chat History:
                        {chat_history}
                        Follow Up Input: {question}
                        Standalone question:
                        Remember to greet the user with 'Hi, welcome to the PDF chatbot. How can I help you?' if the user asks 'hi' or 'hello' """

            CUSTOM_QUESTION_PROMPT = PromptTemplate.from_template(custom_template)

            llm = OpenAI(temperature=0)

            qa = ConversationalRetrievalChain.from_llm(
                llm,
                VectorStore.as_retriever(),
                condense_question_prompt=CUSTOM_QUESTION_PROMPT,
                memory=memory
            )
            response = qa({"question": query, "chat_history": chat_history})

            with st.chat_message("assistant"):
                st.markdown(response["answer"])
            st.session_state.messages.append({"role": "assistant", "content": response["answer"]})
            st.session_state.chat_history.append((query, response['answer']))


# In the sidebar, display the chat history
with st.sidebar:
    st.title('Chat History')
    for i, (user_msg, bot_response) in enumerate(st.session_state.chat_history):
        with st.expander(f"Chat {i + 1}"):
            st.markdown(f"User: {user_msg}")
            st.markdown(f"Assistant: {bot_response}")
            

if __name__ == '__main__':
    main(pdf)
