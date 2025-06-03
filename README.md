# Medical Chatbot with Streamlit & LangChain

An AI-powered medical chatbot built with Streamlit, LangChain, and Google Gemini to answer medical questions using PDF documents.

---

## 🚀 Features

- Chatbot interface powered by LangChain and Google Gemini
- PDF document processing for accurate medical information retrieval
- Vector search with FAISS for efficient knowledge querying
- Easy deployment on Streamlit Cloud

---

## 📋 Prerequisites

- Python 3.8 or higher
- Git installed on your system
- GitHub account
- Streamlit account (optional, for hosting)

---

## 🛠️ Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/your-repo.git
cd your-repo

python -m venv venv
source venv/bin/activate  # Linux/macOS
# OR
venv\Scripts\activate     # Windows

pip install -r requirements.txt

GOOGLE_API_KEY=your_google_api_key_here

streamlit run app.py
