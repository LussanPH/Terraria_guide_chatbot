import os
from langchain_community.document_loaders import Docx2txtLoader, DirectoryLoader #Read the documents from a dir
from langchain_text_splitters import CharacterTextSplitter #Used for create chunks
from langchain_openai import OpenAIEmbeddings #Embedding Model
from langchain_chroma import Chroma #Vector DB that runs locally
from dotenv import load_dotenv

load_dotenv()

def load_documentation(docs_path):
    if not os.path.exists(docs_path):
        raise FileNotFoundError(f"The directory path {docs_path} does not exists.")

    loader = DirectoryLoader(
        path=docs_path,
        glob='*.docx',
        loader_cls=Docx2txtLoader
    )

    documents = loader.load()

    if len(documents) == 0:
        raise FileNotFoundError(f"No .docx file founded in the directory path {docs_path}")

    for i, doc in enumerate(documents):
        print(f"Document {i + 1}:")
        print(f"Source: {doc.metadata['source']}")
        print(f"Content lenght: {len(doc.page_content)} characters")
        print(f"Content preview: {doc.page_content[:100]}...")
        print(f"Metadata: {doc.metadata}")

    return documents

def main():
    documents = load_documentation("terraria")



if __name__ == "__main__":
    main()


