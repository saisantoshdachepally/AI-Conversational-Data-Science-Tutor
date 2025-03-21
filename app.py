import streamlit as st
import sqlite3
import uuid
import time
from langchain_google_genai import GoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_community.chat_message_histories import SQLChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

# Load API key
API_KEY = st.secrets.get("GOOGLE_API_KEY")

# Set up the Gemini 1.5 Pro model
model = GoogleGenerativeAI(api_key=API_KEY, model="gemini-1.5-pro")

# Initialize SQLite database
db_conn = sqlite3.connect("conversation_log.db", check_same_thread=False)
db_cursor = db_conn.cursor()
db_cursor.execute("""
CREATE TABLE IF NOT EXISTS messages (
    msg_id INTEGER PRIMARY KEY AUTOINCREMENT,
    chat_session TEXT,
    sender TEXT,
    text_content TEXT
)
""")
db_conn.commit()

# Function to save messages
def log_message(chat_session, sender, text_content):
    db_cursor.execute("INSERT INTO messages (chat_session, sender, text_content) VALUES (?, ?, ?)", 
                       (chat_session, sender, text_content))
    db_conn.commit()

# Function to retrieve chat history
def get_chat_history(chat_session):
    db_cursor.execute("SELECT sender, text_content FROM messages WHERE chat_session = ?", (chat_session,))
    return db_cursor.fetchall()

# Chat history instance
def conversation_history(chat_session):
    return SQLChatMessageHistory(
        session_id=chat_session,
        connection="sqlite:///conversation_log.db"
    )

# Generate unique session ID
if "chat_session" not in st.session_state:
    st.session_state.chat_session = str(uuid.uuid4())
col1, col2 = st.columns([4, 1])
with col2:
    if st.button("🆕 Start New Chat"):
        st.session_state.chat_session = str(uuid.uuid4())  # Generate new session
        st.session_state.conversation = []  # Clear chat history
        st.rerun()  # Refresh the app
with col1:   
    # Custom CSS for UI
    st.markdown("""
        <style>
            .title-style {
                text-align: center;
                font-size: 30px;
                font-weight: bold;
                color: #FF4500;
                margin-bottom: 20px;
            }

            /* Fixed input at the bottom */
            .stTextInput {
                position: fixed;
                bottom: 10px;
                width: 80%;
                left: 10%;
                z-index: 999;
            }
        </style>
    """, unsafe_allow_html=True)
    
    # 🔹 **Animated Header Function**
    def display_animated_text(text, speed=0.05):
        placeholder = st.empty()
        shown_text = ""

        for char in text:
            shown_text += char
            placeholder.markdown(f"""
                <h1 style="text-align:center; color: #00D1FF;">{shown_text} 🚀</h1>
            """, unsafe_allow_html=True)
            time.sleep(speed)

    # 🔹 **Display Animated Title**
    display_animated_text('AI Data Science Mentor')

# Get session ID
chat_session = st.session_state.chat_session
history_instance = conversation_history(chat_session)

# Define Chat Prompt Template
chat_template = ChatPromptTemplate(
    messages=[
        ('system', """You are an AI expert specializing in Data Science.
                      Only answer Data Science-related queries.
                      For anything unrelated, politely guide users to ask a Data Science question.
                   """),
        MessagesPlaceholder(variable_name="history", optional=True),
        ('human', '{question}')
    ]
)

# Define output parser
output_parser = StrOutputParser()

# Create the chain
chat_chain = chat_template | model | output_parser

# Define Runnable with message history
chat_engine = RunnableWithMessageHistory(
    chat_chain,
    lambda session: SQLChatMessageHistory(session, "sqlite:///conversation_log.db"),
    input_messages_key="question",
    history_messages_key="history"
)

# 🔹 **Container for Chat Display**
chat_box = st.container()

# Load and show chat history
if "conversation" not in st.session_state:
    st.session_state.conversation = get_chat_history(chat_session)

with chat_box:
    for sender, text_content in st.session_state.conversation:
        with st.chat_message(sender):
            st.markdown(text_content)

# User input box at the bottom
# 🔹 **Fixed Input Field for User**
user_query = st.text_input("Ask your question here:", key="user_input")

# If user submits a message
if user_query:
    # Log user query
    log_message(chat_session, "user", user_query)
    st.session_state.conversation.append(("user", user_query))

    # Invoke AI model
    config = {'configurable': {'session_id': chat_session}}
    reply = chat_engine.invoke({'question': user_query}, config)

    # Log AI reply
    log_message(chat_session, "assistant", reply)
    st.session_state.conversation.append(("assistant", reply))

    # Display AI response
    with chat_box:
        with st.chat_message("assistant"):
            st.markdown(reply)

    # ✅ Clear the input after submission
    st.session_state.pop("user_input")
    st.session_state["user_input"] = ""
    st.rerun()
