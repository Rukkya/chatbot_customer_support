import os
from flask import Flask, request, jsonify
from flask_socketio import SocketIO, emit
from dotenv import load_dotenv
import json
from database import setup_db, ChatHistory, Session
from model import ChatbotModel

# Load environment variables
load_dotenv()

# Initialize Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = 'multilingual-chatbot-secret'
socketio = SocketIO(app, cors_allowed_origins="*")

# Setup database
setup_db()

# Initialize model
model = ChatbotModel()

@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({"status": "healthy"})

@socketio.on('connect')
def handle_connect():
    print('Client connected')
    emit('response', {'data': 'Connected to server'})

@socketio.on('disconnect')
def handle_disconnect():
    print('Client disconnected')

@socketio.on('message')
def handle_message(data):
    try:
        user_id = data.get('user_id', 'anonymous')
        message = data.get('message', '')
        language = data.get('language', 'en')
        model_name = data.get('model_name', os.getenv('MODEL_NAME'))
        
        # Process message with model
        response = model.generate_response(message, language, model_name)
        
        # Save to database
        session = Session()
        chat_history = ChatHistory(
            user_id=user_id,
            message=message,
            response=response,
            language=language,
            model_name=model_name
        )
        session.add(chat_history)
        session.commit()
        session.close()
        
        # Send response back to client
        emit('response', {
            'message': response,
            'language': language,
            'model_name': model_name
        })
    except Exception as e:
        print(f"Error processing message: {str(e)}")
        emit('error', {'error': str(e)})

@app.route('/api/history/<user_id>', methods=['GET'])
def get_chat_history(user_id):
    try:
        session = Session()
        history = session.query(ChatHistory).filter_by(user_id=user_id).all()
        result = [
            {
                'id': h.id,
                'user_id': h.user_id,
                'message': h.message,
                'response': h.response,
                'language': h.language,
                'model_name': h.model_name,
                'timestamp': h.timestamp.isoformat()
            }
            for h in history
        ]
        session.close()
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/languages', methods=['GET'])
def get_supported_languages():
    languages = [
        {'code': 'en', 'name': 'English'},
        {'code': 'fr', 'name': 'French'},
        {'code': 'de', 'name': 'German'},
        {'code': 'es', 'name': 'Spanish'},
        {'code': 'it', 'name': 'Italian'},
        {'code': 'zh', 'name': 'Chinese'},
        {'code': 'ja', 'name': 'Japanese'},
        {'code': 'ko', 'name': 'Korean'},
        {'code': 'ar', 'name': 'Arabic'},
        {'code': 'ru', 'name': 'Russian'}
    ]
    return jsonify(languages)

@app.route('/api/models', methods=['GET'])
def get_available_models():
    models = [
        {'id': 'FacebookAI/xlm-roberta-large', 'name': 'XLM-RoBERTa Large'},
        {'id': 'FacebookAI/xlm-roberta-base', 'name': 'XLM-RoBERTa Base'}
    ]
    return jsonify(models)

if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=5000, debug=True)