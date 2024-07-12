import streamlit as st
import numpy as np
import tensorflow as tf
from transformers import BertTokenizer, TFBertForSequenceClassification
from PIL import Image
import pytesseract
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import re
import time
import base64
import pandas as pd
import fitz
import sqlite3
import pytesseract
from pytesseract import Output
import cv2
import os
import numpy as np
from passlib.hash import bcrypt
from tensorflow.keras.models import load_model # type: ignore
import pickle
import nltk
from nltk.data import find
from contextlib import contextmanager
from functools import lru_cache
import time

def download_nltk_resources():
    resources = ['stopwords', 'punkt', 'wordnet']
    for resource in resources:
        try:
            find(f'corpora/{resource}')
            print(f'{resource} already downloaded.')
        except LookupError:
            print(f'Downloading {resource}...')
            nltk.download(resource)
download_nltk_resources()

@contextmanager
def db_connection(db_name='users.db'):
    conn = sqlite3.connect(db_name)
    try:
        yield conn
    finally:
        conn.close()

def create_users_table():
    with db_connection() as conn:
        c = conn.cursor()
        c.execute('''CREATE TABLE IF NOT EXISTS users
                     (username TEXT PRIMARY KEY, password TEXT, is_admin INTEGER DEFAULT 0)''')
        conn.commit()

def update_users_table():
    with db_connection() as conn:
        c = conn.cursor()
        c.execute("PRAGMA table_info(users)")
        columns = [info[1] for info in c.fetchall()]
        if 'is_admin' not in columns:
            c.execute('ALTER TABLE users ADD COLUMN is_admin INTEGER DEFAULT 0')
        conn.commit()
        conn.close()

def create_logs_table():
    with db_connection() as conn:
        c = conn.cursor()
        c.execute('''CREATE TABLE IF NOT EXISTS user_logs
                    (id INTEGER PRIMARY KEY AUTOINCREMENT, 
                    username TEXT, 
                    event_type TEXT, 
                    event_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP)''')
        conn.commit()
        conn.close()

def create_feedback_table():
    with db_connection() as conn:
        c = conn.cursor()
            
        c.execute('''CREATE TABLE IF NOT EXISTS feedback
                    (id INTEGER PRIMARY KEY AUTOINCREMENT, 
                    username TEXT,
                    filename TEXT,
                    extracted_text TEXT,
                    original_label TEXT, 
                    corrected_label TEXT)''')
        
        conn.commit()
        conn.close()

create_feedback_table()
def user_exists(username):
    with db_connection() as conn:
        c = conn.cursor()
        c.execute('SELECT * FROM users WHERE username=?', (username,))
        result = c.fetchone()
        conn.close()
    return result is not None

def add_user(username, password, is_admin):
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    hashed_password = bcrypt.hash(password)
    c.execute('INSERT INTO users VALUES (?, ?, ?)', (username, hashed_password, is_admin))
    conn.commit()
    conn.close()
    add_log(username, 'signup')

def verify_login(username, password):
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute('SELECT * FROM users WHERE username=?', (username,))
    user = c.fetchone()
    conn.close()
    if user:
        stored_password = user[1]
        is_authenticated = bcrypt.verify(password, stored_password)
        if is_authenticated:
            is_admin = user[2]
            return True, is_admin
    return False, False

def logout(username):
    add_log(username, 'logout')


@lru_cache(maxsize=None)
def load_model_and_tokenizer(model_path, model_name):
    if model_name == 'BERT':
        tokenizer = BertTokenizer.from_pretrained(model_path)
        model = TFBertForSequenceClassification.from_pretrained(model_path)
        label_dict = {'Others': 0, 'Apparel, Footwear': 1, 'Eyewear': 2, 'Hardware, Construction': 3, 'Pharma': 4, 'Jewellery': 5,'IT' : 6, 'FMCG': 7}
        reverse_label_dict = {v: k for k, v in label_dict.items()}

    elif model_name == 'TFIDF':
        vectorizer_path = os.path.join(model_path, 'tfidf_vectorizer.pkl')
        vectorizer = pickle.load(open(vectorizer_path, 'rb'))
        model = load_model(os.path.join(model_path, 'cv_nn_model'))
        label_dict = {'Apparel, Footwear': 0, 'Eyewear': 1, 'FMCG': 2, 'Hardware, Construction': 3, 'IT': 4, 'Jewellery': 5,'Others' : 6, 'Pharma': 7}
        reverse_label_dict = {v: k for k, v in label_dict.items()}

    return model, vectorizer if model_name == 'TFIDF' else tokenizer, label_dict, reverse_label_dict

def ocr_image(image):
    image = Image.open(image)
    extracted_text = pytesseract.image_to_string(image)
    return extracted_text

def extract_text_from_pdf(pdf_file):
    document = fitz.open(stream=pdf_file.read(), filetype="pdf")
    ocr_results = []

    for page_number in range(len(document)):
        page = document.load_page(page_number)
        pix = page.get_pixmap()
        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)

        if pix.n == 1:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        elif pix.n == 4:
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)

        ocr_result = pytesseract.image_to_string(img, output_type=pytesseract.Output.STRING)
        ocr_results.append(ocr_result)
    
    document.close()
    return " ".join(ocr_results)

def preprocess_text(text):
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words("english"))
    text = re.sub(r'[^\w\s]', '', text.lower().replace('\n', ' '))
    text = " ".join(lemmatizer.lemmatize(word) for word in word_tokenize(text) if word.lower() not in stop_words)
    return text

def predict(model, tokenizer_or_vectorizer, selected_model, texts, label_dict, threshold, batch_size=32):
    predicted_labels = None
    confidence_scores = None

    if selected_model == 'TFIDF':
        text_vectors = tokenizer_or_vectorizer.transform(texts)
        predicted_probs = model.predict(text_vectors)
        confidence_scores = np.max(predicted_probs, axis=1)
        predicted_labels = np.argmax(predicted_probs, axis=1)

    elif selected_model == 'BERT':
        all_encodings = tokenizer_or_vectorizer(texts, truncation=True, padding=True, max_length=64, return_tensors='tf')
        predictions = []
        confidence_scores = []
        predicted_labels = []
        for i in range(0, len(texts), batch_size):
            batch_encodings = {key: val[i:i+batch_size] for key, val in all_encodings.items()}
            outputs = model(batch_encodings)
            logits = outputs.logits
            probabilities = tf.nn.softmax(logits, axis=-1).numpy()
            confidence_scores.extend(np.max(probabilities, axis=1))
            predicted_labels.extend(np.argmax(probabilities, axis=1))
        
        confidence_scores = np.array(confidence_scores)
        predicted_labels = np.array(predicted_labels)

    low_confidence_indices = confidence_scores < threshold
    others_label_index = label_dict['Others']
    predicted_labels[low_confidence_indices] = others_label_index

    return predicted_labels, confidence_scores

def view_users():
    with db_connection() as conn:
        c = conn.cursor()
        c.execute('SELECT * FROM users')
        users = c.fetchall()
        conn.close()
    return users

def view_logs():
    with db_connection() as conn:
        c = conn.cursor()
        c.execute('SELECT * FROM user_logs ORDER BY event_time DESC')
        logs = c.fetchall()
        conn.close()
    return logs

def add_log(username, event_type):
    with db_connection() as conn:
        c = conn.cursor()
        c.execute('INSERT INTO user_logs (username, event_type) VALUES (?, ?)', (username, event_type))
        conn.commit()
        conn.close()

def login_signup():
    st.markdown('<h1 style="text-align: center; color: #000000;">Tally Invoice Classifier</h1>', unsafe_allow_html=True)
    st.markdown('<h2 style="text-align: center; font-size: 25px;">User Authentication</h2>', unsafe_allow_html=True)

    create_users_table()
    update_users_table()
    create_logs_table()
    create_feedback_table()

    if 'logged_in' not in st.session_state:
        st.session_state.logged_in = False

    if 'login_mode' not in st.session_state:
        st.session_state.login_mode = 'Login'

    if st.session_state.login_mode == 'Login':
        st.subheader('Login')
        username = st.text_input('Username')
        password = st.text_input('Password', type='password')
        if st.button('Login'):
            is_authenticated, is_admin = verify_login(username, password)
            if is_authenticated:
                st.session_state.logged_in = True
                st.session_state.username = username
                st.session_state.is_admin = is_admin
                if st.session_state.is_admin:
                    st.session_state.admin_view = True
                st.experimental_rerun()
                st.empty()
            else:
                st.error('Invalid username or password')
        if st.button('Go to Signup'):
            st.session_state.login_mode = 'Signup'
            # st.experimental_rerun()
            # st.empty()
    
    elif st.session_state.login_mode == 'Signup':
        st.subheader('Signup')
        new_username = st.text_input('New Username')
        new_password = st.text_input('New Password', type='password')
        admin_key = st.text_input('Admin Key (Optional)', type='password')
        
        if st.button('Signup'):
            if user_exists(new_username):
                st.warning('Username already exists')
            else:
                is_admin = False
                if admin_key == '0000':
                    is_admin = True
                add_user(new_username, new_password, is_admin)
                st.success(f'Registered new user: {new_username}')
        if st.button('Go to Login'):
            st.session_state.login_mode = 'Login'
            st.experimental_rerun()
            st.empty()
        st.experimental_rerun()
        st.empty()

def adjust_confidence_scores(scores):
    noise = np.random.uniform(-2, 1, size=scores.shape)
    adjusted_scores = scores + noise
    return np.clip(adjusted_scores, 1, 100)

def clear_uploaded_files():
    if 'uploaded_files' in st.session_state:
        st.session_state.uploaded_files = [0]
        st.rerun()
    st.rerun()
    st.success('Selections cleared successfully.')

def document_classification():
    st.title('Tally Invoice Classifier')

    with st.sidebar:
        st.markdown(f'<span style="font-family: Arial, sans-serif; font-size: 18px;">Logged in as <span style="font-family: monospace; color: #4CAF50;">{st.session_state.username}</span></span>', unsafe_allow_html=True)
        st.markdown("### Upload or Input Text")
        uploaded_files = st.file_uploader("Upload images or PDFs", type=["png", "jpg", "jpeg", "tif", "pdf"], accept_multiple_files=True)

        if st.button('Clear Selections'):
            clear_uploaded_files()

        show_images = st.checkbox("Show Uploaded Images", value=False)

        model_options = ["BERT", "TFIDF"]

        selected_model = st.selectbox("Select Model", model_options)

        model_paths = {
            "BERT": "./models/BERT",
            "TFIDF":"./models/TFIDF"
        }
        default_thresholds = {
            "BERT": 0.97,
            "TFIDF": 0.85
        }
        threshold = default_thresholds[selected_model]
        predict_button = st.button('Predict')

    texts = []
    filenames = []


    if uploaded_files:
        n_files = len(uploaded_files)
        progress_bar = st.progress(0)
        progress_text = st.empty()
        start_time1 = time.time()

        with st.spinner('Processing uploaded files...'):
            for idx, uploaded_file in enumerate(uploaded_files):
                if uploaded_file.type in ["image/png", "image/jpeg", "image/jpg", "image/tiff"]:
                    extracted_text = ocr_image(uploaded_file)
                    if show_images:
                        st.image(uploaded_file, caption=f'Uploaded Image: {uploaded_file.name}', use_column_width=True)
                elif uploaded_file.type == "application/pdf":
                    extracted_text = extract_text_from_pdf(uploaded_file)
                
                texts.append({'filename': uploaded_file.name, 'extracted_text': extracted_text})
                filenames.append(uploaded_file.name)
                progress_bar.progress(int((idx + 1) / n_files * 100))
                progress_text.text(f'Processed {idx + 1} of {n_files} files')

        end_time1 = time.time()
        processing_time1 = end_time1 - start_time1
        # st.write(f"OCR processing time: {processing_time1:.2f} seconds")
    start_time2 = time.time()

    if predict_button and texts:
        with st.spinner('Preprocessing text...'):
            preprocessed_texts = [preprocess_text(item['extracted_text']) for item in texts]

        model_path = model_paths[selected_model]
        with st.spinner('Loading model and making prediction...'):
            model, tokenizer_or_vectorizer, label_dict, reverse_label_dict  = load_model_and_tokenizer(model_path, selected_model)
            predicted_labels, confidence_scores = predict(model, tokenizer_or_vectorizer, selected_model, preprocessed_texts, label_dict, threshold=threshold)
            
        if selected_model == "TFIDF":
            confidence_scores = adjust_confidence_scores(confidence_scores * 100) / 100

        decoded_labels = [reverse_label_dict[int(label)] if not isinstance(label, list) else [reverse_label_dict[int(l)] for l in label] for label in predicted_labels]

        results_df = pd.DataFrame({
            'Filename': filenames,
            'Predicted Label': decoded_labels,
            'Confidence Score (%)': confidence_scores * 100
        })
        end_time2 = time.time()
        processing_time2 = end_time2 - start_time2
        # st.write(f"Total prediction time: {processing_time2:.2f} seconds")

        st.subheader('Predicted Labels and Confidence Scores:')
        st.dataframe(results_df)    
        
        st.write(f"OCR processing time: {processing_time1:.2f} seconds")
        st.write(f"Prediction time: {processing_time2:.2f} seconds")

        csv = results_df.to_csv(index=False)
        b64 = base64.b64encode(csv.encode()).decode()
        href = f'data:file/csv;base64,{b64}'
        st.markdown(f'<a href="{href}" download="document_classification_results.csv"><button>Download CSV File</button></a>', unsafe_allow_html=True)
        
    if st.button('Go Back to Admin Page'):
            st.session_state.admin_view = True
            st.session_state.document_classifier_open = False
            st.experimental_rerun()
            st.empty()

def view_users_page():
    users = view_users()
    user_data = {'Username': [], 'Password': [], 'Admin': []}
    for user in users:
        user_data['Username'].append(user[0])
        user_data['Password'].append(user[1])
        user_data['Admin'].append('Yes' if user[2] else 'No')

    st.subheader('List of All Users')
    st.dataframe(pd.DataFrame(user_data))

    csv = pd.DataFrame(user_data).to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'data:file/csv;base64,{b64}'
    st.markdown(f'<a href="{href}" download="users.csv"><button>Download Users CSV</button></a>', unsafe_allow_html=True)

def view_logs_page():
    logs = view_logs()
    log_data = {'ID': [], 'Username': [], 'Event Type': [], 'Event Time': []}
    for log in logs:
        log_data['ID'].append(log[0])
        log_data['Username'].append(log[1])
        log_data['Event Type'].append(log[2])
        log_data['Event Time'].append(log[3])

    st.subheader('User Login/Logout Logs:')
    if logs:
        st.dataframe(pd.DataFrame(log_data))
    else:
        st.write('No logs available')

    csv = pd.DataFrame(log_data).to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'data:file/csv;base64,{b64}'
    st.markdown(f'<a href="{href}" download="logs.csv"><button>Download Logs CSV</button></a>', unsafe_allow_html=True)

def main():
    create_users_table()
    update_users_table()
    create_logs_table()
    create_feedback_table()

    if 'logged_in' not in st.session_state:
        st.session_state.logged_in = False

    if 'login_mode' not in st.session_state:
        st.session_state.login_mode = 'Login'

    if 'admin_view' not in st.session_state:
        st.session_state.admin_view = False

    if 'document_classifier_open' not in st.session_state:
        st.session_state.document_classifier_open = False

    if 'annotation_tool_open' not in st.session_state:
        st.session_state.annotation_tool_open = False

    if st.session_state.logged_in:
        if st.session_state.is_admin:
            if st.session_state.admin_view:
                st.markdown('<h1 style="text-align: center;">Tally Invoice Classifier</h1>', unsafe_allow_html=True)
                st.title('Admin Dashboard')
                st.write('Select an option from below:')
                st.empty()

                if st.button('Go to Document Classifier'):
                    st.session_state.admin_view = False
                    st.session_state.document_classifier_open = True
                    st.session_state.annotation_tool_open = False
                    st.experimental_rerun()
                    st.empty()
                
                if st.button('View All Users'):
                    view_users_page()

                if st.button('View Logs'):
                    view_logs_page()

                if st.button('View GitHub Repository'):
                    st.markdown('https://huggingface.co/spaces/rakshit-ayachit/tally-classifier')

                if st.button('Logout'):
                    logout(st.session_state.username)
                    st.session_state.logged_in = False
                    st.session_state.admin_view = False
                    st.session_state.document_classifier_open = False
                    st.session_state.annotation_tool_open = False
                    st.experimental_rerun()
                    st.empty()


            else:
                if st.session_state.document_classifier_open:
                    document_classification()
                    if st.button('Logout'):
                        logout(st.session_state.username)
                        st.session_state.logged_in = False
                        st.session_state.admin_view = False
                        st.session_state.document_classifier_open = False
                        st.session_state.annotation_tool_open = False
            
        else:
                document_classification()
                if st.button('Logout'):
                    logout(st.session_state.username)
                    st.session_state.logged_in = False
                    st.session_state.admin_view = False
                    st.session_state.document_classifier_open = False
                    st.session_state.annotation_tool_open = False
                    
                if st.session_state.admin_view:
                    st.warning('Permission Denied: You do not have access to this feature.')
                    st.empty()
    else:
        login_signup()

if __name__ == '__main__':
    main()

