# 🧠 Intelligent Document Q&A with LangChain, FAISS & HuggingFace

> Ask questions from your PDF, DOCX, or TXT files using powerful open-source tools — all in Google Colab.

---

## 📌 Overview

This project allows you to upload a document and ask it natural language questions. Behind the scenes, it uses:

* 🧩 **LangChain** – to handle document loading, text splitting, and chaining logic
* 🔍 **FAISS** – to store and retrieve semantically similar chunks
* 🤗 **HuggingFace Transformers** – to run a local language model for question-answering
* ⚡ **Google Colab** – no setup required, runs in the cloud

---

## 🚀 Features

 ✅ Upload **PDF**, **DOCX**, or **TXT**  
 ✅ Extract and chunk document contents  
 ✅ Embed chunks using **sentence‑transformers**  
 ✅ Store & search in a **FAISS** vector index  
 ✅ Use a small, efficient language model (**Flan‑T5**) for answers  
 ✅ Interactive Q&A loop  

---

## 🛠️ Installation (Google Colab)

```python
!pip install langchain langchain-community
!pip install faiss-cpu
!pip install pypdf python-docx
!pip install sentence-transformers
!pip install transformers
```

---

## 📤 Upload Your Document

```python
from google.colab import files

uploaded = files.upload()
file_path = list(uploaded.keys())[0]
print("Uploaded:", file_path)
```

---

## 📄 Load & Split the Document

```python
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Choose loader based on file type
if file_path.endswith(".pdf"):
    loader = PyPDFLoader(file_path)
elif file_path.endswith(".docx"):
    loader = Docx2txtLoader(file_path)
else:
    loader = TextLoader(file_path)

docs = loader.load()

# Split into chunks
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
documents = splitter.split_documents(docs)

print(f"Total Chunks: {len(documents)}")
```

---

## 🧠 Generate Embeddings & Build Vector Store

```python
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = FAISS.from_documents(documents, embeddings)
```

---

## 🤖 Load Local Language Model

```python
from transformers import pipeline
from langchain_community.llms import HuggingFacePipeline

flan_pipeline = pipeline(
    "text2text-generation",
    model="google/flan-t5-base",  # Swap to flan-t5-small for faster inference
    max_length=512
)

llm = HuggingFacePipeline(pipeline=flan_pipeline)
```

---

## 🔗 Set Up QA Chain

```python
from langchain.chains import RetrievalQA

qa = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
    chain_type="stuff"
)
```

---

## ❓ Ask Questions

### One-Off Query

```python
query = "Give me a short summary of the document"
print(qa.run(query))
```

### Interactive Q&A Loop

```python
while True:
    q = input("Ask a question (or 'exit'): ")
    if q.lower() == "exit":
        break
    print("Answer:", qa.run(q))
```

---

## 🧪 Sample Output

```
Ask a question (or 'exit'): what is the importance of a healthy diet?
Answer: helps to protect against malnutrition in all its forms, as well as noncommunicable diseases (NCDs) such as diabetes, heart disease, stroke and cancer
```

---

## 📁 File Types Supported

* `.pdf` – parsed using `PyPDFLoader`
* `.docx` – parsed using `Docx2txtLoader`
* `.txt` – parsed using `TextLoader`

---
