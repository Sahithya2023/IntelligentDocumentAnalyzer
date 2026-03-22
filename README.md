# 📄 Docify — Document RAG with Advanced Evaluation Metrics

## 🚀 Overview
Docify is an end-to-end **Retrieval-Augmented Generation (RAG)** system that allows users to upload documents and query them intelligently. It combines **semantic search, text generation, and advanced evaluation metrics** to provide accurate and explainable answers.

---

## ❗ Problem
Analyzing large documents manually is inefficient. Users need:
- Fast information retrieval  
- Context-aware answers  
- Reliable evaluation of generated responses  

---

## 💡 Solution
Built a **RAG-based NLP pipeline** that:
- Extracts text from multiple document formats  
- Splits content into semantic chunks  
- Retrieves relevant context using embeddings + FAISS  
- Generates answers using a transformer model  
- Evaluates output with multiple NLP metrics and visualizations  

---

## 🔑 Features
- 📂 Multi-format support (PDF, DOCX, PPTX, TXT, Images)
- 🔍 Semantic search using Sentence Transformers + FAISS
- 🤖 Answer generation using FLAN-T5
- 📊 Advanced evaluation metrics:
  - ROUGE (1,2,L)
  - BLEU
  - METEOR
  - Cosine similarity
  - BERTScore (optional)
- 🔥 Combined heatmap visualization of metrics
- 📈 Chunk-level and document-level evaluation

---

## 🧠 Tech Stack
- **Language:** Python  
- **Framework:** Streamlit  
- **NLP Models:** Sentence Transformers, HuggingFace Transformers  
- **Vector Search:** FAISS  
- **Libraries:**  
  - NLTK  
  - Scikit-learn  
  - PyMuPDF, pytesseract  
  - Seaborn, Matplotlib  
  - Pandas, NumPy  

---

## ⚙️ Workflow

1. Upload document  
2. Extract and preprocess text  
3. Chunk text into manageable segments  
4. Create embeddings and FAISS index  
5. Retrieve relevant chunks for query  
6. Generate answer using LLM  
7. Evaluate answer using multiple metrics  
8. Visualize results using heatmaps  

---

## ▶️ Usage

### Run the app
```bash
streamlit run your_file_name.py
