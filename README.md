# 📄 Docify — Intelligent Document Understanding System

A Streamlit-based web app that extracts text from uploaded documents and uses **Google Gemini AI** to summarize, analyze, translate, extract entities, and answer questions about them.

Built as part of the BECE303L AI/ML course at **VIT Chennai**.

---

## 🚀 Features

- 📥 **Multi-format support** — PDF, DOCX, ODT, TXT, JPG, PNG
- 🔍 **OCR fallback** — scanned PDFs and images handled via Tesseract
- 🧠 **Gemini AI powered** — summarization, analysis, entity extraction, translation
- 💬 **Document Q&A** — ask any question about the uploaded document
- 🌐 **Translation** — Spanish, French, German, Chinese, Japanese, Hindi
- 🔑 **Flexible API key input** — via `.env`, Streamlit secrets, or in-app form

---

## 🛠️ Setup & Installation

### 1. Clone the repository

```bash
git clone https://github.com/your-username/docify.git
cd docify
```

### 2. Install Python dependencies

```bash
pip install streamlit google-generativeai pymupdf pdf2image pytesseract pillow python-docx odfpy nltk python-dotenv
```

### 3. Install Tesseract OCR

Tesseract is required for image and scanned PDF processing.

- **Windows**: Download from [UB-Mannheim Tesseract](https://github.com/UB-Mannheim/tesseract/wiki) and add to your system PATH
- **Linux**: `sudo apt install tesseract-ocr`
- **Mac**: `brew install tesseract`

### 4. Get a Gemini API Key

Go to [Google AI Studio](https://aistudio.google.com/app/apikey) and create a free API key.

### 5. Create a `.env` file

In the project root folder, create a file named `.env`:

```
GOOGLE_API_KEY=your_gemini_api_key_here
```

> Alternatively, you can enter the API key directly in the app's UI when prompted.

---

## ▶️ Running the App

```bash
streamlit run aiml.py
```

A browser tab will open at `http://localhost:8501`. Upload a document and use the tabs to explore its features.

---

## 📁 Project Structure

```
docify/
│
├── aiml.py          # Main Streamlit application
├── .env             # API key (do not commit this to GitHub)
├── .gitignore       # Should include .env
└── README.md        # This file
```

---

##  Important

Make sure your `.gitignore` includes `.env` so your API key is never pushed to GitHub:

```
# .gitignore
.env
__pycache__/
*.pyc
```

---

## 📝 License

This project was built for academic purposes.
