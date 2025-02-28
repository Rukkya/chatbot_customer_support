import streamlit as st
import requests
import json
import os
from dotenv import load_dotenv
import socketio
import time
from datetime import datetime

# Load environment variables
load_dotenv()

# Initialize SocketIO client
sio = socketio.Client()

# API configuration
API_URL = "http://localhost:5000"

# Page configuration
st.set_page_config(
    page_title="Multi-Language Customer Support Chatbot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main {
        background-color: #f5f7f9;
    }
    .chat-message {
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        display: flex;
        flex-direction: column;
    }
    .chat-message.user {
        background-color: #2b313e;
        color: #ffffff;
        border-bottom-right-radius: 0;
    }
    .chat-message.bot {
        background-color: #ffffff;
        border-bottom-left-radius: 0;
    }
    .chat-message .message-content {
        display: flex;
        margin-top: 0.5rem;
    }
    .chat-message .avatar {
        width: 20%;
    }
    .chat-message .message {
        width: 80%;
    }
    .chat-message .timestamp {
        font-size: 0.8rem;
        color: #a0aec0;
        margin-top: 0.5rem;
        text-align: right;
    }
    .sidebar .sidebar-content {
        background-color: #f8fafc;
    }
    .stButton>button {
        width: 100%;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'messages' not in st.session_state:
    st.session_state.messages = []

if 'connected' not in st.session_state:
    st.session_state.connected = False

if 'user_id' not in st.session_state:
    st.session_state.user_id = f"user_{int(time.time())}"

# SocketIO event handlers
@sio.event
def connect():
    st.session_state.connected = True
    print("Connected to server")

@sio.event
def disconnect():
    st.session_state.connected = False
    print("Disconnected from server")

@sio.event
def response(data):
    if 'message' in data:
        message = data['message']
        language = data.get('language', 'en')
        model_name = data.get('model_name', os.getenv('MODEL_NAME'))
        
        # Add message to chat history
        st.session_state.messages.append({
            'role': 'bot',
            'content': message,
            'language': language,
            'model_name': model_name,
            'timestamp': datetime.now().strftime("%H:%M")
        })

# Connect to server
def connect_to_server():
    if not st.session_state.connected:
        try:
            sio.connect(API_URL)
            return True
        except Exception as e:
            st.error(f"Failed to connect to server: {str(e)}")
            return False
    return True

# Fetch supported languages
@st.cache_data
def get_supported_languages():
    try:
        response = requests.get(f"{API_URL}/api/languages")
        return response.json()
    except Exception as e:
        st.error(f"Failed to fetch supported languages: {str(e)}")
        return [{'code': 'en', 'name': 'English'}]

# Fetch available models
@st.cache_data
def get_available_models():
    try:
        response = requests.get(f"{API_URL}/api/models")
        return response.json()
    except Exception as e:
        st.error(f"Failed to fetch available models: {str(e)}")
        return [{'id': 'FacebookAI/xlm-roberta-large', 'name': 'XLM-RoBERTa Large'}]

# Fetch chat history
def get_chat_history():
    try:
        response = requests.get(f"{API_URL}/api/history/{st.session_state.user_id}")
        history = response.json()
        
        # Update session state with history
        st.session_state.messages = []
        for msg in history:
            st.session_state.messages.append({
                'role': 'user',
                'content': msg['message'],
                'language': msg['language'],
                'timestamp': datetime.fromisoformat(msg['timestamp']).strftime("%H:%M")
            })
            st.session_state.messages.append({
                'role': 'bot',
                'content': msg['response'],
                'language': msg['language'],
                'model_name': msg['model_name'],
                'timestamp': datetime.fromisoformat(msg['timestamp']).strftime("%H:%M")
            })
    except Exception as e:
        st.error(f"Failed to fetch chat history: {str(e)}")

# Send message to server
def send_message(message, language, model_name):
    if connect_to_server():
        # Add message to chat history
        st.session_state.messages.append({
            'role': 'user',
            'content': message,
            'language': language,
            'timestamp': datetime.now().strftime("%H:%M")
        })
        
        # Send message to server
        sio.emit('message', {
            'user_id': st.session_state.user_id,
            'message': message,
            'language': language,
            'model_name': model_name
        })

# Main app
def main():
    # Sidebar
    with st.sidebar:
        st.title("Settings")
        
        # Language selection
        languages = get_supported_languages()
        language_options = {lang['name']: lang['code'] for lang in languages}
        selected_language_name = st.selectbox("Select Language", list(language_options.keys()))
        selected_language = language_options[selected_language_name]
        
        # Model selection
        models = get_available_models()
        model_options = {model['name']: model['id'] for model in models}
        selected_model_name = st.selectbox("Select Model", list(model_options.keys()))
        selected_model = model_options[selected_model_name]
        
        # Load chat history
        if st.button("Load Chat History"):
            get_chat_history()
        
        # Clear chat
        if st.button("Clear Chat"):
            st.session_state.messages = []
        
        st.markdown("---")
        st.markdown("### About")
        st.markdown("""
        This is a multi-language chatbot for customer support using:
        - XLM-RoBERTa model
        - MultiWOZ dataset
        - Streamlit for frontend
        """)

    # Main content
    st.title("Multi-Language Customer Support Chatbot")
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.container():
            col1, col2 = st.columns([1, 12])
            
            with col1:
                if message['role'] == 'user':
                    st.markdown("👤")
                else:
                    st.markdown("🤖")
            
            with col2:
                if message['role'] == 'user':
                    st.markdown(f"<div class='chat-message user'><div class='message'>{message['content']}</div><div class='timestamp'>{message['timestamp']}</div></div>", unsafe_allow_html=True)
                else:
                    model_info = f"Model: {message.get('model_name', 'Unknown')}" if 'model_name' in message else ""
                    st.markdown(f"<div class='chat-message bot'><div class='message'>{message['content']}</div><div class='timestamp'>{message['timestamp']} | {model_info}</div></div>", unsafe_allow_html=True)
    
    # Chat input
    with st.container():
        user_input = st.text_input("Type your message here...", key="user_input")
        col1, col2 = st.columns([4, 1])
        
        with col2:
            send_button = st.button("Send")
        
        if send_button and user_input:
            send_message(user_input, selected_language, selected_model)
            st.experimental_rerun()

if __name__ == "__main__":
    main()