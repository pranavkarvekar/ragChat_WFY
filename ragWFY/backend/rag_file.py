import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader, TextLoader, UnstructuredWordDocumentLoader
from langchain_community.vectorstores import FAISS
from langchain.chains.combine_documents.stuff import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# initialize model
llm = ChatGroq(model_name="llama-3.3-70b-versatile", temperature=0.7, api_key=GROQ_API_KEY)

prompt = ChatPromptTemplate.from_template(
    """
    You are an assistant. Use the following context to answer the question.
    If the answer is not in the context, say you don't know.

    <context>
    {context}
    </context>

    Question:
    {input}
    """
)

def load_documents_from_path(path):
    ext = os.path.splitext(path)[1].lower()
    if ext == ".pdf":
        loader = PyPDFLoader(path)
    elif ext in (".docx", ".doc"):
        loader = UnstructuredWordDocumentLoader(path)
    elif ext == ".txt":
        loader = TextLoader(path)
    else:
        raise ValueError("Unsupported file type: " + ext)
    return loader.load()

def build_faiss_from_docs(docs):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    final_docs = splitter.split_documents(docs)
    return FAISS.from_documents(final_docs, embeddings)

def query_file(file_path: str, question: str):
    """
    Loads file, builds index in-memory and answers `question`.
    Returns dict: {'answer': str, 'context': [str, ...]}
    """
    docs = load_documents_from_path(file_path)
    vectors = build_faiss_from_docs(docs)
    document_chain = create_stuff_documents_chain(llm, prompt)
    retriever = vectors.as_retriever()
    retriever_chain = create_retrieval_chain(retriever, document_chain)

    out = retriever_chain.invoke({"input": question})
    answer = out.get("answer") or out.get("result") or str(out)
    context = out.get("context", [])
    context_snips = [getattr(d, "page_content", str(d)) for d in context]
    return {"answer": answer, "context": context_snips}


