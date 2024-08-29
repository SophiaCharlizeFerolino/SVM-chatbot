import streamlit as st
from streamlit_chat import message
from dotenv import load_dotenv
from htmlTemplates import css
import json
import pandas as pd
import nltk
from sklearn.svm import SVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import joblib

nltk.download('punkt_tab')
nltk.download('wordnet')
nltk.download('stopwords')

class SessionState:
    def __init__(self, **kwargs):
        for key, val in kwargs.items():
            setattr(self, key, val)

def preprocess_text(text):
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    words = word_tokenize(text)
    words = [lemmatizer.lemmatize(word) for word in words if word.lower() not in stop_words]
    return ' '.join(words)

def get_response(user_message, data, loaded_model):
    user_processed_text = preprocess_text(user_message)
    user_label = loaded_model.predict([user_processed_text])[0]
    similarities = cosine_similarity(
        loaded_model.named_steps['tfidf'].transform(data[data['label'] == user_label]['processed_text']),
        loaded_model.named_steps['tfidf'].transform([user_processed_text])
    )
    reply_data = data.loc[data['label'] == user_label].iloc[0]

    bot_response = ""  # Set a default value

    if similarities.max() >= 0.1:
        bot_response = reply_data['responses']
    else:
        # Instead of st.warning, set a value to bot_response
        bot_response = "I currently do not possess the information needed to address your question. To assist you more effectively, might I kindly request you to consider rephrasing your question or add additional details?  This will enable me to better understand your query and offer more accurate assistance."

    return bot_response, similarities

def main():
    load_dotenv()
    st.set_page_config(page_title="ANIMO KNOWS 💬")
    st.write(css, unsafe_allow_html=True)
    st.sidebar.image("dlsud_logo.png", use_column_width=True)
    
    st.sidebar.title("ANIMO KNOWS 💬")
    st.sidebar.text("A DLSU-D CBL and Student")
    st.sidebar.text("Handbook Chatbot")
    st.sidebar.markdown(
        """
        <style>
            .css-1aumxhk {
                background-color: #ffffff;  /* Set to white (#ffffff) */
            }
        </style>
        """,
        unsafe_allow_html=True
    )

    session_state = SessionState(user_input="")

    st.header("Chat with Animo! :robot_face:")
    st.caption("Ask anything about The Conduct of Blended Learning and The Sections of the Student Handbook :books:")

    submit_button = st.chat_input("Type here", key="user_input")
    

    if submit_button:
        with st.spinner("Processing"):
            # Load intents data (Replace 'your_intent_data.json' with the actual file path)
            with open('dataset.json', 'rb') as file:
                intents_data = json.load(file)

            data = pd.DataFrame(columns=['text', 'label', 'responses'])
            for intent in intents_data['intents']:
                for pattern in intent.get('patterns', []):
                    response = intent.get('responses', [''])[0]
                    new_data = pd.DataFrame({'text': pattern, 'label': intent['tag'], 'responses': response}, index=[0])
                    data = pd.concat([data, new_data], ignore_index=True)

            data['processed_text'] = data['text'].apply(preprocess_text)

            try:
                loaded_content = joblib.load('svm_model.pkl')
                print(loaded_content)
            except Exception as e:
                print(f"Error loading the model: {e}")
                raise  # Raise the exception to stop the script

            bot_response, similarities = get_response(submit_button, data, loaded_content)

            # Initialize chat_history as an empty list if it doesn't exist in st.session_state
            st.session_state.chat_history = st.session_state.get('chat_history', [])

            if bot_response:
                # Add user message to chat history
                st.session_state.chat_history.append({'content': submit_button, 'is_user': True})

                # Add bot response to chat history
                st.session_state.chat_history.append({'content': bot_response, 'is_user': False})

            # Display chat history
            for i, msg in enumerate(st.session_state.chat_history):
                message(msg['content'], is_user=msg['is_user'], key=str(i) + ('_user' if msg['is_user'] else '_ai'))
    
    def reset_conversation():
        session_state.user_input = ""
        st.session_state.chat_history = []
    st.sidebar.button('Reset Chat', on_click=reset_conversation)

if __name__ == '__main__':
    main()
