# 🎥 YouTube RAG Chatbot

A **YouTube Retrieval-Augmented Generation (RAG) Chatbot** built with **Streamlit**, **LangChain**, and **Google Gemini**.  
Ask questions about any YouTube video, and get answers directly from its transcript.

[![Streamlit Badge](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://streamlit.io/)
[![Python](https://img.shields.io/badge/python-3.11-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 🚀 Features

- Automatically fetches transcripts from YouTube videos.
- Splits long transcripts into manageable chunks.
- Generates embeddings and stores them in FAISS for fast retrieval.
- Uses **Google Generative AI** to answer questions based on transcripts.
- Clean, interactive **Streamlit interface**.
- Gracefully handles videos without transcripts.

---

## 🖥️ Demo

Try the live demo on Streamlit:  
[🎬 Launch App](https://yourusername-streamlitapp.streamlit.app)

> **Note:** Replace `yourusername` with your Streamlit Cloud username.

---

## 📦 Installation (Local Setup)

1. Clone the repository:

```bash
git clone https://github.com/yourusername/your-repo.git
cd your-repo
Create a virtual environment (recommended):

python -m venv env
source env/bin/activate  # macOS/Linux
env\Scripts\activate     # Windows


Install dependencies:

pip install -r requirements.txt


Set environment variables (if needed) in a .env file:

# Example .env
GOOGLE_API_KEY=your_google_api_key_here


Run the app:

streamlit run ytchatbot.py
🛠️ Usage

Open the app in your browser.

Enter a YouTube Video ID (e.g., Gfr50f6ZBvo).

Enter a question about the video content.

Click Get Answer to receive responses based on the transcript.

⚙️ How It Works

Transcript Retrieval:
Fetches video transcript using youtube-transcript-api.

Text Splitting:
Uses RecursiveCharacterTextSplitter to split long transcripts into smaller chunks.

Embeddings & Vector Store:
Chunks are converted to embeddings with HuggingFaceEmbeddings and stored in FAISS.

Retrieval-Augmented Generation:
Uses ChatGoogleGenerativeAI to answer questions based on retrieved chunks.

Streamlit Frontend:
Provides a clean interface to input video IDs and questions.

📄 File Structure
ytchatbot/
│
├─ ytchatbot.py         # Main Streamlit app
├─ requirements.txt     # Python dependencies
├─ README.md            # Project documentation
└─ .gitignore           # Ignore sensitive/temporary files

💡 Tips

Some YouTube videos may not have transcripts; the app handles this gracefully.

You can switch to other embedding models for better performance.

Never push .env files with API keys to public repos.

🙌 Contributing

Contributions, suggestions, and feedback are welcome!

Fork the repository.

Create a new branch: git checkout -b feature-name.

Make your changes and commit: git commit -m "Add feature".

Push to your branch: git push origin feature-name.

Open a Pull Request.


📫 Contact

GitHub: Rohith gowa K

Email: rohithgowdak18@gmail.com