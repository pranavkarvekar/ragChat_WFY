import os
from dotenv import load_dotenv
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents.stuff import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceEmbeddings

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

llm = ChatGroq(model_name="llama-3.3-70b-versatile", api_key=GROQ_API_KEY, temperature=0.7)
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

def query_website(website_url: str, question: str) -> str:
    loader = WebBaseLoader(web_paths=(website_url,))
    documents = loader.load()
    if not documents:
        return "No readable content found on the provided website."
    clean_docs = [doc for doc in documents if doc.page_content and doc.page_content.strip()]
    if not clean_docs:
        return "No valid content found after cleaning."
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=200)
    docs = splitter.split_documents(clean_docs)
    if not docs:
        return "Could not split webpage content for embedding."
    vectors = FAISS.from_documents(docs, embeddings)
    retriever = vectors.as_retriever(search_kwargs={"k": 5})
    prompt = ChatPromptTemplate.from_template(
        """
        Answer the following question based on the provided context only.
        Provide the most accurate response based on the question.
        <context>
        {context}
        </context>
        Question:
        {input}
        """
    )
    document_chain = create_stuff_documents_chain(llm, prompt)
    retriever_chain = create_retrieval_chain(retriever, document_chain)
    result = retriever_chain.invoke({"input": question})
    return result.get("answer", str(result))


