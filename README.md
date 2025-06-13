## 📄 RAG Chat Application – Chat with Your PDF using GPT-4

This is a simple **RAG (Retrieval-Augmented Generation)** application built using **Streamlit**, **LangChain**, **OpenAI**, and **Qdrant** as the vector database. The app lets you upload a PDF file and chat with it, powered by GPT-4. It stores vectorized content from the PDF and retrieves relevant chunks to answer your questions.

---

### 🚀 Features

* Upload any PDF
* PDF is split into chunks and stored as vector embeddings in Qdrant
* Ask questions about the PDF in natural language
* Get answers from GPT-4 based on the content of your uploaded PDF

---

### 🛠️ Tech Stack

* **Frontend:** Streamlit
* **LLM:** OpenAI GPT-4.1
* **Embeddings:** `text-embedding-3-large` from OpenAI
* **Vector DB:** Qdrant
* **Chunking:** LangChain's `RecursiveCharacterTextSplitter`

---

## 🧑‍💻 Getting Started

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/your-username/rag-chat-pdf.git
cd rag-chat-pdf
```

---

### 2️⃣ Set Up Your Environment

Install dependencies:

```bash
pip install -r requirements.txt
```

---

### 3️⃣ Add Your API Key

Create a `.env` file in the root directory:

```env
OPENAI_API_KEY=sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
```

> ⚠️ Never hard-code your API key into the Python files.

---

### 4️⃣ Start Qdrant Locally

You can run Qdrant using Docker:

```bash
docker run -p 6333:6333 qdrant/qdrant
```

Make sure Qdrant is running before starting the app.

---

### 5️⃣ Launch the Streamlit App

```bash
streamlit run app.py
```

This will open the Streamlit app in your browser.

---

## 📁 Project Structure

```
.
├── app.py               # Streamlit app with Upload & Chat UI
├── .env                 # Your API key is stored here (DO NOT COMMIT)
├── requirements.txt     # Python dependencies
└── README.md            # Project documentation
```

---

## ⚠️ Notes

* Only one PDF is processed at a time. If a new PDF is uploaded, the vector store is overwritten.
* GPT will answer **only based on content from the uploaded PDF**.
* You can expand this into a multi-document or session-based app using LangChain memory and persistent vector stores.

---

## ✅ To-Do (Optional Extensions)

* Support multiple documents
* Session-based chat memory
* File-type support beyond PDFs (Word, Text, etc.)
* Cloud-based deployment (Streamlit Cloud, HuggingFace Spaces)

---

## 🙏 Acknowledgements

* [LangChain](https://github.com/langchain-ai/langchain)
* [Qdrant Vector DB](https://qdrant.tech/)
* [Streamlit](https://streamlit.io/)
* [OpenAI GPT-4](https://platform.openai.com/)

