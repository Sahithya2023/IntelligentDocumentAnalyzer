import streamlit as st
import fitz  # PyMuPDF
import pytesseract
from pdf2image import convert_from_path
from PIL import Image
import os
import google.generativeai as genai
from dotenv import load_dotenv
import tempfile
from nltk.tokenize import sent_tokenize
import nltk
nltk.download('punkt')
nltk.download('punkt_tab')


# Extra imports for ODF and DOCX
from odf import text, teletype
from odf.opendocument import load
import docx

# ------------------------------
# NLTK Punkt Safety Check
# ------------------------------
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

# ------------------------------
# Load Environment Variables
# ------------------------------
load_dotenv()

API_KEY_ENV_VAR = "GOOGLE_API_KEY"
API_KEY_SESSION_KEY = "docify_gemini_api_key"
MODEL_NAME = "models/gemini-2.5-flash"

# ------------------------------
# Gemini API Key Resolver
# ------------------------------
def _resolve_gemini_api_key():
    session_key = st.session_state.get(API_KEY_SESSION_KEY)
    if session_key:
        return session_key

    env_key = os.getenv(API_KEY_ENV_VAR)
    if env_key:
        return env_key

    try:
        secret_key = st.secrets[API_KEY_ENV_VAR]
        if secret_key:
            return secret_key
    except Exception:
        pass

    return None

def get_gemini_model():
    api_key = _resolve_gemini_api_key()
    if not api_key:
        return None
    genai.configure(api_key=api_key)
    return genai.GenerativeModel(MODEL_NAME)

# ------------------------------
# File Processing Functions
# ------------------------------
def extract_text_from_pdf(pdf_path):
    """Try extracting text directly, else OCR"""
    text = ""
    try:
        doc = fitz.open(pdf_path)
        for page in doc:
            text += page.get_text()
        if not text.strip():  # fallback to OCR
            images = convert_from_path(pdf_path)
            for image in images:
                text += pytesseract.image_to_string(image)
    except Exception as e:
        st.error(f"PDF extraction error: {e}")
    return text

def extract_text_from_odt(file_path):
    try:
        doc = load(file_path)
        paragraphs = doc.getElementsByType(text.P)
        return "\n".join(teletype.extractText(p) for p in paragraphs)
    except Exception as e:
        st.error(f"ODT extraction error: {e}")
        return ""

def extract_text_from_docx(file_path):
    try:
        doc = docx.Document(file_path)
        return "\n".join(p.text for p in doc.paragraphs)
    except Exception as e:
        st.error(f"DOCX extraction error: {e}")
        return ""

def extract_text_from_txt(file_path):
    try:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()
    except Exception as e:
        st.error(f"TXT extraction error: {e}")
        return ""

def extract_text_from_image(file_path):
    try:
        image = Image.open(file_path)
        return pytesseract.image_to_string(image)
    except Exception as e:
        st.error(f"Image OCR error: {e}")
        return ""

# ------------------------------
# Preprocessing
# ------------------------------
def preprocess_text(text):
    if not text:
        return ""
    text = text.strip()
    text = " ".join(text.split())
    return " ".join(sent_tokenize(text))

# ------------------------------
# Gemini Response Functions
# ------------------------------
def generate_response(prompt, context=""):
    model = get_gemini_model()
    if model is None:
        return "⚠️ Gemini API key missing."
    try:
        response = model.generate_content(f"Context: {context}\n\nQuery: {prompt}")
        if hasattr(response, "candidates") and response.candidates:
            return response.candidates[0].content.parts[0].text
        return "No response generated."
    except Exception as e:
        return f"Gemini error: {e}"

def extract_entities(text):
    return generate_response(f"Extract key entities (names, orgs, dates, locations) from:\n\n{text}")

def summarize_text(text):
    return generate_response(f"Summarize this:\n\n{text}")

def analyze_document(text):
    prompt = """Analyze this document and provide:
    1. Main topics/themes
    2. Key points
    3. Document type
    4. Important dates
    5. Action items
    """
    return generate_response(prompt + text)

def translate_text(text, target_language):
    return generate_response(f"Translate this to {target_language}:\n\n{text}")

# ------------------------------
# Streamlit App
# ------------------------------
def main():
    st.set_page_config(page_title="SmartDoc - Document Understanding", layout="wide")
    st.title("📄 Docify: Intelligent Document Processing")

    # API key check
    if get_gemini_model() is None:
        with st.form("gemini_api_key_form"):
            api_key_input = st.text_input("Enter Gemini API Key", type="password")
            save_key = st.form_submit_button("Save API Key")
        if save_key and api_key_input.strip():
            st.session_state[API_KEY_SESSION_KEY] = api_key_input.strip()
            st.success("API key saved. Reloading...")
            st.rerun()
        st.stop()

    uploaded_file = st.file_uploader("Upload a file", type=["pdf", "odt", "docx", "txt", "jpg", "png"])

    if uploaded_file:
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            file_path = tmp_file.name

        # Extract text depending on file type
        ext = uploaded_file.name.split(".")[-1].lower()
        if ext == "pdf":
            extracted = extract_text_from_pdf(file_path)
        elif ext == "odt":
            extracted = extract_text_from_odt(file_path)
        elif ext == "docx":
            extracted = extract_text_from_docx(file_path)
        elif ext == "txt":
            extracted = extract_text_from_txt(file_path)
        elif ext in ["jpg", "png", "jpeg"]:
            extracted = extract_text_from_image(file_path)
        else:
            extracted = ""

        combined_text = preprocess_text(extracted)

        tab1, tab2, tab3, tab4 = st.tabs(["Text Extraction", "Analysis", "Summary", "Translation"])

        with tab1:
            st.subheader("Extracted Text")
            st.text_area("Processed Text", combined_text, height=300)
            if st.button("Extract Entities"):
                st.write(extract_entities(combined_text))

        with tab2:
            st.subheader("Document Analysis")
            if st.button("Analyze Document"):
                st.write(analyze_document(combined_text))

        with tab3:
            st.subheader("Document Summary")
            if st.button("Generate Summary"):
                st.write(summarize_text(combined_text))

        with tab4:
            st.subheader("Translation")
            target_lang = st.selectbox("Target language", ["Spanish", "French", "German", "Chinese", "Japanese", "Hindi"])
            if st.button("Translate"):
                st.write(translate_text(combined_text, target_lang))

        st.markdown("---")
        st.subheader("💬 Ask Questions")
        user_q = st.text_input("Enter your question about the document:")
        if user_q:
            st.write(generate_response(user_q, context=combined_text))

if __name__ == "__main__":
    main()