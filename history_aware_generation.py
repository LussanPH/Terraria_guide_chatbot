from langchain_openai import OpenAIEmbeddings #Embedding Model
from langchain_chroma import Chroma #Vector DB that runs locally
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

load_dotenv()

model = ChatOpenAI(model='gpt-4o-mini')

chat_history = []


#Rewrite the user query in such a way that makes the query searchable in the vector DB
def query_reform(query):
    if chat_history:
        messages = [
            SystemMessage(content="Based on the chat history, rewrite the user query with the objective to make the query searchable in the chatbot's vector database. Retrun only the new query.")
        ] + chat_history + [
            HumanMessage(content=query)
        ]

        result = model.invoke(messages)
        new_query = result.content.strip()
        print(new_query)

    else:
        new_query = query

    return new_query


#The same procedure that is realized in the first agent, but with the chat_history applied
def retrieval_procedure(query, persistent_dir = 'db/chroma_db'):

    new_query = query_reform(query)

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

    revelant_chunks = retriever.invoke(new_query)

    input = f"""Bases only in the following documents, answer this question: {new_query}

    Documents:
    {chr(10).join([f"- {chunk.page_content}" for chunk in revelant_chunks])}

    Provide a clear answer using only the information from the sended documents. If you can't find the answer with these documents, say that you don't have enough information to answer the question.
    """

    messages =  [
            SystemMessage(content="You are a helpful assistent about the game terraria, that guides new players and veterans that have any question about the gameplay.") 
        ] + chat_history + [ 
            HumanMessage(content=input) 
        ]
    

    result = model.invoke(messages)
    answer = result.content

    chat_history.append(HumanMessage(content=new_query))
    chat_history.append(AIMessage(content=answer))

    return answer


def start():
    print("Ask me a question, type 'quit' to exit")

    while True:
        question = input("\nWrite your question: ")

        if question.lower() == 'quit':
            print("Goodbye!")
            break

        else:
            answer = retrieval_procedure(question)
            print(answer)


start()