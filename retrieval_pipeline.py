import os
from langchain_community.document_loaders import Docx2txtLoader, DirectoryLoader #Read the documents from a dir
from langchain_text_splitters import CharacterTextSplitter #Used for create chunks
from langchain_openai import OpenAIEmbeddings #Embedding Model
from langchain_chroma import Chroma #Vector DB that runs locally
from dotenv import load_dotenv
import numpy as np
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

load_dotenv()

def cosine_similarity(revelant_chunks, query_embed, embedding_model=OpenAIEmbeddings(model="text-embedding-3-small")):
    text_chunks = [chunk.page_content for chunk in revelant_chunks]

    chunks_embed = embedding_model.embed_documents(text_chunks)

    cos_sim = []

    for vec in chunks_embed:
        cos_sim.append(np.dot(vec, query_embed))

    print(cos_sim)


def retrieval_procedure(query, persistent_dir = 'db/chroma_db'):

    embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")

    db = Chroma(
        persist_directory=persistent_dir,
        embedding_function=embedding_model,
        collection_metadata={'hnsw:space':'cosine'}
    )

    retriever = db.as_retriever(
        search_type='similarity_score_threshold',
        search_kwargs={
            "k":5,
            "score_threshold":0.3
        }
    )

    revelant_chunks = retriever.invoke(query)

    print(f"User query: {query}")

    for i, chunk in enumerate(revelant_chunks, 1):
        print(f"Chunk {i}\n{chunk.page_content}")

    query_embed = np.array(embedding_model.embed_query(query))

    cosine_similarity(revelant_chunks, query_embed, embedding_model)

    input = f"""Bases only in the following documents, answer this question: {query}

    Documents:
    {chr(10).join([f"- {chunk.page_content}" for chunk in revelant_chunks])}

    Provide a clear answer using only the information from the sended documents. If you can't find the answer with these documents, say that you don't have enough information to answer the question.
    """

    model = ChatOpenAI(model='gpt-4o-mini')

    messages = [
        SystemMessage(content="You are a helpful assistent about the game terraria, that guides new players and veterans that have any question about the gameplay."),
        HumanMessage(content=input)
    ]

    result = model.invoke(messages)

    print(result.content)


retrieval_procedure("How i start the game, what i have to do?")