import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_experimental.agents.agent_toolkits import create_csv_agent
from langchain.agents.agent_types import AgentType
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain

api_key = st.secrets["OPENAI_API_KEY"]
st.set_page_config(page_title="Chatbot", page_icon="🤖")
st.title("Chatbot")

with st.sidebar:
    file_type = st.selectbox("Select Chatbot mode", ["DEFAULT", "CSV", "PDF"])
    if file_type == "CSV":
        user_file = st.file_uploader("Upload your CSV file", type=["csv"], accept_multiple_files=False)
        user_files = [user_file] if user_file is not None else None
    else:
        user_files = st.file_uploader("Upload your PDF files", type=["pdf"], accept_multiple_files=True)

def get_response(user_query, chat_history, user_files, file_type):
    llm = ChatOpenAI(temperature=0.5, model="gpt-3.5-turbo")

    def generate_response(response):
        yield response

    if file_type == "CSV" and user_files is not None:
        user_file = user_files[0]  # There will be only one file in the list for CSV
        agent = create_csv_agent(llm, user_file, verbose=True, agent_type=AgentType.OPENAI_FUNCTIONS)
        response = agent.invoke(user_query)
        response_stream = generate_response(response["output"])
        return response_stream
    elif file_type == "PDF" and user_files is not None:
        text = ""
        for user_file in user_files:
            pdf_reader = PdfReader(user_file)
            for page in pdf_reader.pages:
                text += page.extract_text()

        text_splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )

        chunks = text_splitter.split_text(text)
        embeddings = OpenAIEmbeddings()
        knowledge_db = FAISS.from_texts(texts=chunks, embedding=embeddings)
        docs = knowledge_db.similarity_search(user_query, k=5)
        chain = load_qa_chain(llm, chain_type="stuff")
        response = chain.invoke({"input_documents": docs, "question": user_query})
        response_stream = generate_response(response["output_text"])
        return response_stream
    else:
        template = """
        You are a helpful assistant. Answer the following questions considering the history of the conversation:

        Chat history: {chat_history}

        User question: {user_question}
        """

        prompt = ChatPromptTemplate.from_template(template)
        
        chain = prompt | llm | StrOutputParser()
        response_stream = generate_response(chain.invoke({
            "chat_history": chat_history,
            "user_question": user_query,
        }))
        return response_stream

# Session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [AIMessage(content="Hello, I am a bot. How can I help you?")]

# Display a message if file reader mode is active
if file_type == "CSV" and user_files is not None:
    st.sidebar.info("CSV reader mode activated. The bot will use the uploaded CSV file to answer your queries.")
elif file_type == "PDF" and user_files is not None:
    st.sidebar.info("PDF reader mode activated. The bot will use the uploaded PDF files to answer your queries.")

# User input
user_query = st.chat_input("Type your message here...")

# Conversation
for message in st.session_state.chat_history:
    if isinstance(message, AIMessage):
        with st.chat_message("AI"):
            st.write(message.content)
    elif isinstance(message, HumanMessage):
        with st.chat_message("Human"):
            st.write(message.content)

if user_query is not None and user_query != "":
    st.session_state.chat_history.append(HumanMessage(content=user_query))

    with st.chat_message("Human"):
        st.markdown(user_query)

    with st.chat_message("AI"):
        raw_response = get_response(user_query, st.session_state.chat_history, user_files, file_type)
        response = st.write_stream(raw_response)

    st.session_state.chat_history.append(AIMessage(content=response))
