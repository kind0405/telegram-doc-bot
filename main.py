import telebot
import io
import os
import numpy as np
import faiss
import threading
import time
import datetime
import requests
import logging

import pandas as pd
import pdfplumber
from docx import Document

from googleapiclient.discovery import build
from google.oauth2 import service_account
from googleapiclient.http import MediaIoBaseDownload

from sentence_transformers import SentenceTransformer

# --- Настройки ---

TOKEN = '7746119786:AAGm0uWy-urxACu9Q9w0lP9HQ6v610K6Vcg'
FOLDER_ID = '1SDBfV-2Zk7lriKUsgRSS6wWnyC2O7ZX0'
SERVICE_ACCOUNT_FILE = 'credentials.json'
HF_API_TOKEN = os.environ.get("HF_API_TOKEN")
HF_MODEL = "google/flan-t5-small"

SCOPES = ['https://www.googleapis.com/auth/drive.readonly']
credentials = service_account.Credentials.from_service_account_file(
    SERVICE_ACCOUNT_FILE, scopes=SCOPES)
drive_service = build('drive', 'v3', credentials=credentials)

bot = telebot.TeleBot(TOKEN)
embed_model = SentenceTransformer('all-MiniLM-L6-v2')

INDEX_FILE = 'faiss_index.bin'
TEXTS_FILE = 'texts.npy'
FILENAMES_FILE = 'filenames.npy'
LAST_UPDATE_FILE = 'last_update.txt'

index = None
texts = []
file_names = []

logging.basicConfig(
    filename='bot.log',
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s',
    encoding='utf-8'
)

# --- Загрузка и извлечение текста из файлов ---

def download_file(file_id):
    try:
        request = drive_service.files().get_media(fileId=file_id)
        fh = io.BytesIO()
        downloader = MediaIoBaseDownload(fh, request)
        done = False
        while not done:
            _, done = downloader.next_chunk()
        fh.seek(0)
        return fh.read()
    except Exception as e:
        logging.error(f"Ошибка загрузки файла {file_id}: {e}")
        return None

def extract_text_from_docx(file_content):
    try:
        with io.BytesIO(file_content) as file_stream:
            document = Document(file_stream)
            paragraphs = [p.text.strip() for p in document.paragraphs if p.text.strip()]
            return '\n'.join(paragraphs)
    except Exception as e:
        logging.error(f"Ошибка извлечения текста из DOCX: {e}")
        return ""

def extract_text_from_pdf(file_content):
    try:
        with io.BytesIO(file_content) as file_stream:
            with pdfplumber.open(file_stream) as pdf:
                texts = []
                for page in pdf.pages:
                    text = page.extract_text()
                    if text:
                        texts.append(text.strip())
                return '\n'.join(texts)
    except Exception as e:
        logging.error(f"Ошибка извлечения текста из PDF: {e}")
        return ""

def extract_text_from_excel(file_content):
    try:
        with io.BytesIO(file_content) as file_stream:
            df = pd.read_excel(file_stream, engine='openpyxl')
            return df.to_csv(index=False)
    except Exception as e:
        logging.error(f"Ошибка извлечения текста из Excel: {e}")
        return ""

def extract_text_from_google_doc(file_id):
    try:
        request = drive_service.files().export_media(
            fileId=file_id,
            mimeType='application/vnd.openxmlformats-officedocument.wordprocessingml.document'
        )
        fh = io.BytesIO()
        downloader = MediaIoBaseDownload(fh, request)
        done = False
        while not done:
            _, done = downloader.next_chunk()
        fh.seek(0)
        return extract_text_from_docx(fh.read())
    except Exception as e:
        logging.error(f"Ошибка извлечения текста из Google Doc {file_id}: {e}")
        return ""

# --- Сбор текстов из папки ---

def gather_all_texts():
    all_texts = []
    page_token = None
    while True:
        results = drive_service.files().list(
            q=f"'{FOLDER_ID}' in parents and trashed=false",
            pageSize=100,
            fields="nextPageToken, files(id, name, mimeType)",
            pageToken=page_token
        ).execute()
        files = results.get('files', [])
        page_token = results.get('nextPageToken', None)

        for file in files:
            try:
                file_id = file['id']
                name = file['name']
                mime = file.get('mimeType', '')
                text = None

                if mime == 'application/pdf' or name.lower().endswith('.pdf'):
                    content = download_file(file_id)
                    if content:
                        text = extract_text_from_pdf(content)

                elif mime == 'application/vnd.openxmlformats-officedocument.wordprocessingml.document' or name.lower().endswith('.docx'):
                    content = download_file(file_id)
                    if content:
                        text = extract_text_from_docx(content)

                elif mime == 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet' or name.lower().endswith('.xlsx'):
                    content = download_file(file_id)
                    if content:
                        text = extract_text_from_excel(content)

                elif mime == 'application/vnd.google-apps.document':
                    text = extract_text_from_google_doc(file_id)

                if text and text.strip():
                    all_texts.append((name, text))
                    logging.info(f"Файл '{name}' добавлен для индексирования")
            except Exception as e:
                logging.error(f"Ошибка обработки файла {file['name']}: {e}")

        if not page_token:
            break

    return all_texts

def split_text_to_chunks(text, max_chunk_size=1000):
    paragraphs = [p.strip() for p in text.split('\n') if p.strip()]
    chunks = []
    current_chunk = ""
    for para in paragraphs:
        if len(current_chunk) + len(para) + 1 <= max_chunk_size:
            current_chunk += para + "\n"
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = para + "\n"
    if current_chunk.strip():
        chunks.append(current_chunk.strip())
    return chunks

def save_index_and_data():
    faiss.write_index(index, INDEX_FILE)
    np.save(TEXTS_FILE, np.array(texts, dtype=object))
    np.save(FILENAMES_FILE, np.array(file_names, dtype=object))
    logging.info("Индекс и данные сохранены")

def load_index_and_data():
    global index, texts, file_names
    if not os.path.exists(INDEX_FILE):
        return False
    index = faiss.read_index(INDEX_FILE)
    texts = np.load(TEXTS_FILE, allow_pickle=True).tolist()
    file_names = np.load(FILENAMES_FILE, allow_pickle=True).tolist()
    logging.info("Индекс и данные загружены")
    return True

def build_index():
    global index, texts, file_names
    logging.info("Строим индекс...")
    docs = gather_all_texts()
    texts.clear()
    file_names.clear()
    for name, text in docs:
        chunks = split_text_to_chunks(text)
        for chunk in chunks:
            texts.append(chunk)
            file_names.append(name)
    embeddings = embed_model.encode(texts, show_progress_bar=True)
    dim = embeddings.shape[1]
    index_local = faiss.IndexFlatL2(dim)
    index_local.add(np.array(embeddings).astype('float32'))
    index = index_local
    save_index_and_data()
    with open(LAST_UPDATE_FILE, 'w') as f:
        f.write(datetime.datetime.now().isoformat())
    logging.info("Индекс построен")

def check_and_update_index():
    if not os.path.exists(LAST_UPDATE_FILE):
        build_index()
        return
    with open(LAST_UPDATE_FILE, 'r') as f:
        last = datetime.datetime.fromisoformat(f.read().strip())
    if (datetime.datetime.now() - last).days >= 7:
        build_index()
    else:
        if not load_index_and_data():
            build_index()

# --- Генерация ответа ---

def generate_answer(question, contexts):
    prompt = (
        "Ты — ассистент по внутренним документам компании. "
        "Отвечай максимально понятно, точно и по существу, используя информацию из регламентов. "
        "Если нет точного ответа — скажи об этом.\n\n"
        f"Вопрос: {question}\n\n"
        "Информация из документов:\n" +
        "\n\n".join(contexts[:3]) + "\n\nОтвет:"
    )

    headers = {"Authorization": f"Bearer {HF_API_TOKEN}"}
    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": 256,
            "temperature": 0.0
        }
    }

    try:
        response = requests.post(
            f"https://api-inference.huggingface.co/models/{HF_MODEL}",
            headers=headers,
            json=payload,
            timeout=30
        )
        result = response.json()
        if "error" in result:
            return "Ошибка Hugging Face API: " + result["error"]
        return result[0]["generated_text"].strip()
    except Exception as e:
        return f"Ошибка запроса к Hugging Face API: {e}"

# --- Поиск и отправка файла ---

def find_file_by_name(query):
    try:
        results = drive_service.files().list(
            q=f"'{FOLDER_ID}' in parents and trashed=false",
            pageSize=100,
            fields="files(id, name)"
        ).execute()
        files = results.get('files', [])
    except Exception as e:
        logging.error(f"Ошибка получения списка файлов: {e}")
        return None

    query_lower = query.lower()
    for f in files:
        name = f['name'].lower()
        if name in query_lower or query_lower in name:
            return f['id'], f['name']
    return None

def send_file_by_id(chat_id, file_id, file_name):
    content = download_file(file_id)
    if content is None:
        bot.send_message(chat_id, f"Не удалось скачать файл {file_name}.")
        return
    bio = io.BytesIO(content)
    bio.name = file_name
    bio.seek(0)
    try:
        bot.send_document(chat_id, bio)
    except Exception as e:
        logging.error(f"Ошибка отправки файла: {e}")
        bot.send_message(chat_id, "Ошибка при отправке файла.")

# --- Обработчики Telegram ---

@bot.message_handler(commands=['start'])
def start_message(message):
    bot.send_message(message.chat.id, "Привет! Я ассистент по регламентам. Задай вопрос или попроси документ.")

@bot.message_handler(content_types=['text'])
def handle_text(message):
    user_text = message.text.strip()
    chat_id = message.chat.id
    logging.info(f"Сообщение от {chat_id}: {user_text}")

    file_found = find_file_by_name(user_text)
    if file_found:
        file_id, file_name = file_found
        bot.send_message(chat_id, f"Отправляю документ: {file_name}")
        send_file_by_id(chat_id, file_id, file_name)
        return

    if index is None:
        bot.send_message(chat_id, "Индекс ещё не загружен, подождите.")
        return

    q_emb = embed_model.encode([user_text])
    D, I = index.search(np.array(q_emb).astype('float32'), 5)
    contexts = [texts[idx] for idx in I[0] if idx < len(texts)]
    answer = generate_answer(user_text, contexts)
    bot.send_message(chat_id, answer)

# --- Запуск бота и фоновое обновление ---

def background_index_updater():
    while True:
        try:
            check_and_update_index()
            time.sleep(3600 * 6)
        except Exception as e:
            logging.error(f"Ошибка обновления индекса: {e}")
            time.sleep(300)

if __name__ == '__main__':
    logging.info("Запуск бота...")
    threading.Thread(target=background_index_updater, daemon=True).start()
    bot.infinity_polling()

   
