'''
Need to create a RAG chat application which processes your pdf and then you can chat with it, giving the pdf you wish and get your query resolved
using Streamlit
'''
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()


api_key=os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)

tab1, tab2 = st.tabs(["Upload PDF", "Chat with LLM"])

with tab1: 
    st.title("RAG Chat Application")
    uploaded_file = st.file_uploader("Choose a pdf file")

    # in UI we need to show that the data is being processed 
    if uploaded_file is not None:

        # the uploaded file variable holds a file like object which needs to be converted to a file path 
        with open("file-path.pdf", "wb") as file_uploaded:
            file_uploaded.write(uploaded_file.read())
                
        loader = PyPDFLoader("file-path.pdf")  
        docs = loader.load()# docs is to Read pdf 
            
        st.success("Loaded file !")

    # now we will start chunking the data 
        text_splitter= RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )

        chunks = text_splitter.split_documents(documents=docs)

        st.success(f"Successfully split into chunks.")
        
        # Now we will need to create vector embeddings of the chunks 
        # Using OpenAI embeddings model 
        embedding_model=OpenAIEmbeddings(
        model="text-embedding-3-large",
        api_key=api_key
    )
        
        vector_store = QdrantVectorStore.from_documents(
        documents=chunks,
        url="http://localhost:6333",
        collection_name="RAG-Project-UploadDoc",
        embedding=embedding_model
    )
        st.session_state.indexed=True
        st.success("Indexing of documents is done")

with tab2:
    # a way in which we take the user input in UI
    st.title("Chat with your PDF") 
    if not st.session_state.get("indexed"):
        st.warning("Please upload and process a PDF in Tab 1 first.")
    else:
        query = st.chat_input("Ask a question based on the uploaded PDF")

        if query:
            # Load vector store again (connect to existing collection)
            embeddings = OpenAIEmbeddings(
                model="text-embedding-3-large",
                api_key=api_key
            )

            vector_db = QdrantVectorStore.from_existing_collection(
                url="http://localhost:6333",
                collection_name="RAG-Project-UploadDoc",
                embedding=embeddings
            )

            # Retrieve context from vector DB
            search_results = vector_db.similarity_search(query=query)

            context = "\n\n".join([
                f"Page Content: {doc.page_content}\nPage Number: {doc.metadata.get('page_label', 'N/A')}"
                for doc in search_results
            ])

            SYSTEM_PROMPT = f"""
            You are a helpful AI assistant. Use only the following context to answer the user question.
            If you don't find relevant information, say you don't know.

            Context:
            {context}
            """

            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": query}
            ]

            response = client.chat.completions.create(
                model="gpt-4.1",
                messages=messages,
                max_tokens=500,
                temperature=0.2
            )

            answer = response.choices[0].message.content
            with st.expander(query):
                st.write("### 🤖 Answer")
                st.write(answer)    
            st.write("### User")
            st.write(query)
            st.write("### 🤖 Answer")
            st.write(answer)