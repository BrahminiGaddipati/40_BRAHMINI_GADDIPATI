import pandas as pd

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings



def load_vectorstore():
    df = pd.read_csv("data/ai_patient_safety_dataset.csv")

    documents = []
    for _, row in df.iterrows():
        content = f"""
        Category: {row['category']}
        Title: {row['title']}
        Description: {row['description']}
        Action: {row['action']}
        """
        documents.append(
            Document(
                page_content=content,
                metadata={"category": row["category"]}
            )
        )

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )

    docs = splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectordb = Chroma.from_documents(
        docs,
        embedding=embeddings,
        persist_directory="chroma_db"
    )

    return vectordb
