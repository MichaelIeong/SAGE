"""
This is where the memory bank is constructed and updated
The memory bank consists of the following components:
    - Interaction history
    - User preferences
"""
import os
import json
import glob
from typing import List
from collections import defaultdict
from langchain.schema.document import Document
from sage.retrieval.profiler import UserProfiler
from sage.retrieval.vectordb import create_multiuser_vector_indexes
from sage.utils.common import load_embedding_model


class MemoryBank:
    """This class handles the memory bank"""

    def __init__(self):
        self.history = defaultdict(dict)
        self.user_profiler = UserProfiler()
        self.indexes = defaultdict(list)
        self.snapshot_id = 0

    def _load_user_queries(self, user_name: str, directory: str):
        """
        Loads user queries from a directory.
        This assumes that the queries are saved in JSON files
        """

        if os.path.isdir(f"{directory}/instructions") and os.listdir(
            f"{directory}/instructions"
        ):

            filenames = glob.glob(f"{directory}/instructions/*.json")

            for filename in filenames:
                instruction_info = json.load(open(filename))
                self.add_query(
                    user_name, instruction_info["instruction"], instruction_info["date"]
                )

        else:
            print(f"The instructions folder is empty. Skipping user {user_name}")

    def load(self, memory_path: str):
        """Load memory bank from directory"""

        print(f"Loading memory from {memory_path}")

        if os.path.isfile(memory_path) and memory_path.endswith(".json"):
            # Load saved json file
            self.history = json.load(open(memory_path))

        elif os.path.isdir(memory_path):
            sources = glob.glob(f"{memory_path}/*")

            for source in sources:
                if os.path.isdir(source):
                    # This assumes that the memory is a list of directories
                    # where in each directory contains the interactions of
                    # one user
                    user_name = os.path.basename(source)
                else:
                    # This assumes that the interactions are not organized by user name
                    # give a fake username
                    user_name = "all"

                self._load_user_queries(user_name, source)
        else:
            raise ValueError(f"Invalid memory path {memory_path}. Please check ")

    def add_query(self, user_name: str, query: str, date: str):
        """Add a query to the history"""

        if self.history[user_name].get("history") is None:
            self.history[user_name] = {"history": defaultdict(list)}

        if self.history[user_name]["history"].get(date) is None:
            self.history[user_name]["history"][date] = []

        self.history[user_name]["history"][date].append(query)

    def _build_user_profiles(self):
        """Build the user profiles based on the saved interactions"""

        for user_name in self.history.keys():
            user_queries = self.history[user_name]["history"]

            for date, queries in user_queries.items():
                self.user_profiler.update_daily_user_preferences(
                    user_name, queries, date
                )

            self.user_profiler.create_global_user_profile(user_name)

            self.history[user_name]["profile"] = self.user_profiler.global_profiles[
                user_name
            ]
        self.user_profiler.print_global_profiles()

    def read_from_json(self, save_path: str) -> None:
        """Read JSON file (standard JSON or line-delimited JSON)"""
        if save_path.endswith("memory_bank.json"):
            # 用户偏好：整体是一个大 JSON
            with open(save_path, "r", encoding="utf-8") as f:
                self.history = json.load(f)
        else:
            # 设备、环境信息：一行一个小 JSON
            self.history = []
            with open(save_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            self.history.append(json.loads(line))
                        except json.JSONDecodeError as e:
                            print(f"[Warning] Failed to parse line: {line}")
                            print(e)

    def save(self, save_path: str):
        """Saves the memory into a json file"""

        self._build_user_profiles()

        json.dump(self.history, open(save_path, "w"))

    def save_snapshot(self, save_path: str):
        """Save a snapshot of the memory"""

        filename = save_path

        if not save_path.endswith("json"):
            filename = os.path.join(save_path, f"snapshot_{self.snapshot_id}.json")

        json.dump(
            self.history,
            open(filename, "w"),
        )
        self.snapshot_id += 1

    def prepare_for_vector_db(self):
        """Prepare data for vector database creation"""
        if isinstance(self.history, dict):
            result = {}

            for user_name, user_data in self.history.items():
                texts = []

                # 取出每个用户下的 history
                history = user_data.get("history", {})

                for day, sentences in history.items():
                    texts.extend(sentences)  # 把每天的所有句子平铺出来

                result[user_name] = texts

            return result

        elif isinstance(self.history, list):
            # 设备信息 / 环境信息：直接是 list
            return self.history

        else:
            raise ValueError(f"Unsupported history format in MemoryBank: {type(self.history)}")

    def create_indexes(
            self, vectorstore: str, embedding_model: str, load: bool = True
    ) -> None:
        """
        Create vector indexes.
        - If `self.history` is a dict keyed by user_name → create per-user index.
        - Else → create a shared index named by `vectorstore`.
        """
        emb_function = load_embedding_model(model_name=embedding_model)

        # ✅ 情况 1：是用户偏好（按 user_name 分组）
        if isinstance(self.history, dict) and all(
                isinstance(v, dict) and "history" in v for v in self.history.values()
        ):
            for user_name, user_data in self.history.items():
                user_texts = []

                # 1. 加入历史对话内容
                for date, utterances in user_data["history"].items():
                    user_texts.extend(utterances)

                # 2. 加入结构化 profile 内容（先 JSON，再自然语言化）
                # if "profile" in user_data:
                #     try:
                #         profile_dict = json.loads(user_data["profile"])  # 可能是字符串，需要解析
                #         # profile_text = convert_profile_to_natural_language(profile_dict)
                #         user_texts.append(profile_dict)
                #     except Exception as e:
                #         print(f"[Warning] Failed to parse profile for {user_name}: {e}")

                # 3. 最终返回格式
                user_docs = {user_name.lower(): user_texts}

                index_name = f"{vectorstore}_{user_name.lower()}"
                self.indexes[user_name.lower()] = create_multiuser_vector_indexes(
                    index_name, user_docs, emb_function, load=load
                )[user_name.lower()]
                print(f"[✓] Created index for user: {user_name}")

        # ✅ 情况 2：设备信息、环境信息（整体向量集合）
        else:
            documents = self.prepare_for_vector_db()
            self.indexes[vectorstore] = create_multiuser_vector_indexes(
                vectorstore, documents, emb_function, load=load
            )
            print(f"[✓] Created global index: {vectorstore}")

    def search(self, query: str, vectorstore: str = None, user_name: str = None, top_k: int = 5) -> List[str]:
        """
        Generalized search method:
        - If user_name is provided, search in that user's memory index.
        - If vectorstore is provided, search in the named vectorstore index (for env/device info).
        """
        if user_name is not None:
            if user_name not in self.indexes:
                raise ValueError(f"No index found for user: {user_name}")
            sources = self.indexes[user_name].similarity_search(query, k=top_k)
        elif vectorstore is not None:
            if vectorstore not in self.indexes:
                raise ValueError(f"No index found for vectorstore: {vectorstore}")
            sources = self.indexes[vectorstore].similarity_search(query, k=top_k)
        else:
            raise ValueError("Must provide either user_name or vectorstore")

        return [s.page_content for s in sources]
    def contains(self, memory: str, user_name: str) -> bool:
        """Check if a specific memory exists"""

        if user_name not in self.history:
            return False

        user_memories = self.history[user_name]["history"]

        for _, value in user_memories.items():
            if value == memory:
                return True

        return False

    def __len__(self):
        total = 0

        if isinstance(self.history, dict):
            for user, data in self.history.items():
                user_total = 0
                for key, value in data.get("history", {}).items():
                    user_total += len(value)
                print(f"User {user} has {user_total} saved memories")
                total += user_total

        elif isinstance(self.history, list):
            total = len(self.history)

        return total
