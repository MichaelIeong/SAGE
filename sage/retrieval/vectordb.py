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
from typing import List, Dict, Union
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


def create_multiuser_vector_indexes(
    vectorstore: str,
    documents: Union[Dict[str, List[str]], List[str]],
    embedding_model,
    load: bool = True,
):
    """Creates a vector index that offers similarity search"""

    from langchain.vectorstores import Chroma
    import os

    if isinstance(documents, dict):
        # 用户偏好，每个用户一个小库
        user_indexes = {}

        for user_name, memories in documents.items():
            user_index_dir = os.path.join(
                f"{os.getenv('SMARTHOME_ROOT')}", "user_info", user_name, vectorstore
            )

            if load and os.path.exists(user_index_dir):
                db = Chroma(
                    persist_directory=user_index_dir,
                    embedding_function=embedding_model,
                )
            else:
                db = Chroma.from_texts(
                    texts=memories,
                    embedding=embedding_model,
                    persist_directory=user_index_dir,
                )
                db.persist()

            user_indexes[user_name] = db

        return user_indexes

    elif isinstance(documents, list):
        # 设备信息/环境信息，合成一个大库
        vectorstore_dir = os.path.join(
            f"{os.getenv('SMARTHOME_ROOT')}", "user_info", vectorstore
        )

        if load and os.path.exists(vectorstore_dir):
            db = Chroma(
                persist_directory=vectorstore_dir,
                embedding_function=embedding_model,
            )
        else:
            db = Chroma.from_texts(
                texts=documents,
                embedding=embedding_model,
                persist_directory=vectorstore_dir,
            )
            db.persist()

        return db  # 注意 list 的情况下，返回是单个 db，不是字典

    else:
        raise ValueError("Unsupported document format: expected dict or list.")