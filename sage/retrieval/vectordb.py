"""
Everything related to vectordbs
"""
import os
from typing import List, Dict
from pathlib import Path
import shutil
from langchain.vectorstores import Chroma, FAISS
from langchain.docstore.document import Document
from langchain.embeddings.base import Embeddings

from sage.base import GlobalConfig
from sage.utils.common import CONSOLE, load_embedding_model


def build_chroma_db(
    vector_dir: str, documents: List[Document], embeddings: List[Embeddings], load=True
) -> Chroma:
    """Creates or loads a chroma database"""

    if os.path.isdir(vector_dir) and load is True:
        CONSOLE.log(f"Loading vector db from {vector_dir}....")

        return Chroma(
            persist_directory=vector_dir,
            embedding_function=embeddings,
        )

    if not load:
        if os.path.isdir(vector_dir):
            shutil.rmtree(vector_dir)
            CONSOLE.log("Existing db wiped!")
        CONSOLE.log(f"Creating vector db in {vector_dir}...")

    return Chroma.from_documents(documents, embeddings, persist_directory=vector_dir)


def build_faiss_db(
    vector_dir,
    documents: List[Document],
    embeddings: List[Embeddings],
    load: bool = True,
):
    """Creates or loads a FAISS index"""

    if os.path.isdir(vector_dir) and load:
        CONSOLE.log(f"Loading vector db from {vector_dir}....")
        index = FAISS.load_local("smartie-index", embeddings)

        return index

    if not load:
        files = Path(vector_dir).glob("sage-index.*")

        if files:
            for filename in files:
                filename.unlink()

    CONSOLE.log("Creating vector db ...")
    index = FAISS.from_documents(documents=documents, embedding=embeddings)
    index.save_local(folder_path=vector_dir, index_name="sage-index")

    return index


VECTORDBS = {"chroma": build_chroma_db, "faiss": build_faiss_db}


def create_multiuser_vector_indexes(documents, db_name, embedding_model, load=False):
    from langchain.vectorstores import Chroma
    from langchain.embeddings import HuggingFaceEmbeddings

    if isinstance(documents, dict):
        # 处理用户偏好：每个用户一个小库
        indexes = {}

        for user_name, memories in documents.items():
            persist_directory = os.path.join(GlobalConfig.vectorstore_path, user_name, db_name)

            if load:
                db = Chroma(
                    persist_directory=persist_directory,
                    embedding_function=load_embedding_model(embedding_model),
                )
            else:
                db = Chroma.from_texts(
                    texts=memories,
                    embedding=load_embedding_model(embedding_model),
                    persist_directory=persist_directory,
                )
                db.persist()

            indexes[user_name] = db

        return indexes

    elif isinstance(documents, list):
        # 处理设备/环境信息：全部打到一个大库
        persist_directory = os.path.join(GlobalConfig.vectorstore_path, db_name)

        if load:
            db = Chroma(
                persist_directory=persist_directory,
                embedding_function=load_embedding_model(embedding_model),
            )
        else:
            db = Chroma.from_texts(
                texts=documents,
                embedding=load_embedding_model(embedding_model),
                persist_directory=persist_directory,
            )
            db.persist()

        return db  # 注意：直接返回 db，不是 dict

    else:
        raise ValueError("Unsupported document format: expected dict or list.")
