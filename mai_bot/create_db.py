import os

from langchain.document_loaders import ConfluenceLoader, DirectoryLoader, GitbookLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain_community.document_loaders import ApifyDatasetLoader
from langchain_community.document_loaders.base import Document

def get_all_documents():
    """
    Load all documents from Gitbook, Confluence and local files.
    Adapt this function to your needs.
    """
    loader = DirectoryLoader('/Users/jospolfliet/src/vlerick/DATA/MAI-2023 dump/', silent_errors=True)
    course_docs = loader.load()


    loader = ApifyDatasetLoader(
        dataset_id="RcArHfVs80xOg9IKs",
        dataset_mapping_function=lambda dataset_item: Document(
            page_content=dataset_item["text"], metadata={"source": dataset_item["url"]}
        ),
    )
    website_docs = loader.load()

    documents = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0).split_documents(
        course_docs + website_docs
    )
    return documents


if __name__ == "__main__":
    """Recreate the vectorstore from scratch."""
    documents = get_all_documents()

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2") # keep embeddings model in sync with main.py!
    vectorstore = FAISS.from_documents(documents, embedding=embeddings)

    vectorstore.save_local("db")
