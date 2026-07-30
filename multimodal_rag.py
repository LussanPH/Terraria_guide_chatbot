import json
from typing import List
from unstructured.partition.pdf import partition_pdf
from unstructured.chunking.title import chunk_by_title
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core import messages
from dotenv import load_dotenv

load_dotenv()

poppler_path = r"C:\Users\pedro.henrique\poppler-26.02.0\Library\bin"

def partition_pdf_documentation(file_path : str):
    elements = partition_pdf(
        filename = file_path,
        poppler_path = poppler_path,
        strategy= "hi_res",
        infer_table_structure=True,
        extract_image_block_types=["Image"],
        extract_image_block_to_payload=True
    )

    return elements


#PRECISA EDITAR VARIÁVEIS GLOBAIS
elements = partition_pdf(r"terraria\(1.1) Pre-Boss Melee.pdf")
print(f"Quantidade de elementos: {len(elements)}")