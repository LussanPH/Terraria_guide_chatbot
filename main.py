import os
from langchain_community.document_loaders import Docx2txtLoader, DirectoryLoader #Read the documents from a dir
from langchain_text_splitters import CharacterTextSplitter, RecursiveCharacterTextSplitter #Used for create chunks
from langchain_openai import OpenAIEmbeddings #Embedding Model
from langchain_chroma import Chroma #Vector DB that runs locally
from dotenv import load_dotenv

load_dotenv()

#Load the documentation localized in the documents directory
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

    """
    for i, doc in enumerate(documents):
        
        print(f"Document {i + 1}:")
        print(f"Source: {doc.metadata['source']}")
        print(f"Content lenght: {len(doc.page_content)} characters")
        print(f"Content preview: {doc.page_content[:100]}...")
        print(f"Metadata: {doc.metadata}")
    """

    return documents


#Split the documents in various chunks
def split_documents(documents, chunk_size=800, chunk_overlap=0):
    """text_splitter = CharacterTextSplitter(
        chunk_size = chunk_size,
        chunk_overlap = chunk_overlap
    )"""

    text_splitter = RecursiveCharacterTextSplitter(
            separators=['\n\n', '\n', '. ', '.', ' ', ''],
            chunk_size = chunk_size,
            chunk_overlap = chunk_overlap
        )

    chunks = text_splitter.split_documents(documents)

    """
    if chunks:
        for i, chunk in enumerate(chunks[:5], 1):
            print(f"\nChunk {i}:")
            print(f"Source: {chunk.metadata['source']}")
            print(f"Content: \n{chunk.page_content}")

        if len(chunks) > 5:
            print(f"Remain: {len(chunks) - 5} chunks")
    """
    return chunks


#Embed the documents chunks and store them in the vector DB
def embedding_and_vector_db(chunks, persist_directory="db/chroma_db"):
    embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")

    vector_db = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=persist_directory,
        collection_metadata={'hnsw:space':'cosine'}
    )


def main():
    #Load Documents
    documents = load_documentation("terraria")

    #Split Documents in Chunks
    chunks = split_documents(documents)

    #embedding and create vector DB
    embedding_and_vector_db(chunks)


if __name__ == "__main__":
    main()


