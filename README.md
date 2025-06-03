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
☁️ Deploy on Streamlit Cloud
1. Push your code to GitHub
bash
Copy
Edit
git add .
git commit -m "Initial commit"
git push origin main
2. Connect GitHub repo to Streamlit Cloud
Go to Streamlit Cloud

Click New app

Select your GitHub repository and branch (main)

Select app.py as the main file

Click Deploy

3. Auto-update on GitHub pushes
Once connected, your app will auto-redeploy every time you push changes to GitHub.

📂 File Structure
bash
Copy
Edit
/app.py              # Main Streamlit app
/requirements.txt    # Python dependencies
/.env                # Environment variables (not committed)
/Data                # PDF and data files
/research            # Research scripts and notebooks
/faiss_index         # FAISS index files
🛠️ Troubleshooting
If you see ModuleNotFoundError, ensure all dependencies are listed in requirements.txt.

Check Streamlit Cloud logs for build and runtime errors.

Make sure your .env file is properly set up with all necessary API keys.

📫 Contact
For questions or help, please open an issue on GitHub or contact [your-email@example.com].

⚖️ License
MIT License

