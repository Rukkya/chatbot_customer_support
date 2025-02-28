# Multi-Language Chatbot for Customer Support

A multilingual chatbot for real-time customer support, handling complex queries and switching between languages based on user input. The chatbot allows users to choose the model to chat with and provides various functionalities.

## Features

- **Multilingual Support**: Handles queries in multiple languages using XLM-RoBERTa model
- **Fine-tuned on MultiWOZ Dataset**: Trained on a comprehensive dialogue dataset for customer support
- **Real-time Interaction**: Uses WebSockets for live chat experience
- **Chat History**: Stores conversation history in PostgreSQL database
- **User-friendly Interface**: Built with Streamlit for an intuitive user experience
- **Model Selection**: Allows users to choose between different models

## Technology Stack

- **Frontend**: Streamlit
- **Backend**: Flask with SocketIO
- **Database**: PostgreSQL
- **Models**: Hugging Face Transformers (XLM-RoBERTa)
- **Data Processing**: LangChain, Pandas, NumPy

## Project Structure

```
├── api/
│   ├── app.py           # Flask application with WebSocket support
│   ├── database.py      # Database models and connection
│   └── model.py         # Chatbot model implementation
├── models/              # Directory for storing fine-tuned models
├── .env                 # Environment variables
├── requirements.txt     # Python dependencies
├── streamlit_app.py     # Streamlit frontend application
└── README.md            # Project documentation
```

## Setup Instructions

### Prerequisites

- Python 3.8+
- PostgreSQL database
- Node.js and npm

### Installation

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/multilingual-chatbot.git
   cd multilingual-chatbot
   ```

2. Install Python dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Set up PostgreSQL database:
   - Create a database named `chatbot_db`
   - Update the `.env` file with your database credentials

4. Run the Flask API:
   ```
   python api/app.py
   ```

5. Run the Streamlit frontend:
   ```
   streamlit run streamlit_app.py
   ```

## Usage

1. Open your browser and navigate to the Streamlit app (typically http://localhost:8501)
2. Select your preferred language and model from the sidebar
3. Start chatting with the bot
4. Use the sidebar options to load chat history or clear the chat

## Fine-tuning the Model

The chatbot uses a pre-trained XLM-RoBERTa model fine-tuned on the MultiWOZ dataset. To fine-tune the model yourself:

1. Ensure you have the required dependencies installed
2. Run the fine-tuning script:
   ```python
   from api.model import ChatbotModel
   
   model = ChatbotModel()
   model.fine_tune()
   ```

This process may take several hours depending on your hardware. The fine-tuned model will be saved in the `models/` directory.

## License

This project is licensed under the MIT License - see the LICENSE file for details.