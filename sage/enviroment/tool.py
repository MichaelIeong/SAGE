import json
from typing import Dict, Any, Type
from dataclasses import dataclass, field

from langchain.prompts import PromptTemplate
from langchain.llms.base import BaseLLM
from langchain import LLMChain

from sage.utils.llm_utils import LLMConfig, TGIConfig
from sage.base import SAGEBaseTool, BaseToolConfig
from sage.utils.common import parse_json
from sage.retrieval.templates import tool_template
from sage.retrieval.memory_bank import MemoryBank


@dataclass
class EnvironmentInfoToolConfig(BaseToolConfig):
    _target: Type = field(default_factory=lambda: EnvironmentInfoTool)

    name: str = "environment_info_tool"
    description: str = """
Use this to retrieve the environmental context (e.g., devices or attributes) associated with a user based on their name.
Input should be a json string with 2 keys: query and user_name.

Example:
{"user_name": "mmhu", "query": "What device should I use to watch a sci-fi movie?"}
"""
    vectordb: str = "chroma_environment"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    top_k: int = 5
    llm_config: LLMConfig = None


class EnvironmentInfoTool(SAGEBaseTool):
    llm: BaseLLM = None
    memory: MemoryBank = None
    config: EnvironmentInfoToolConfig = None

    def setup(self, config: EnvironmentInfoToolConfig, memory=None):
        self.config = config
        if isinstance(config.llm_config, TGIConfig):
            config.llm_config = TGIConfig(stop_sequences=["Question"])
        self.llm = config.llm_config.instantiate()

        # 统一传入的共享 memory 实例
        self.memory = memory
        if self.memory is None:
            raise ValueError("EnvironmentInfoTool requires a shared MemoryBank instance.")

    def _run(self, text: str) -> str:
        attr = parse_json(text)

        if not attr or "user_name" not in attr or "query" not in attr:
            return "The input should be a json string with keys 'user_name' and 'query'."

        query = attr["query"]
        user_name = attr["user_name"]

        # 检索环境信息
        search_result = self.memory.search(
            {"query": query}, 
            namespace=self.config.vectordb,
            top_k=self.config.top_k
        )

        if not search_result:
            return f"No environmental information found for user '{user_name}'."

        prompt = PromptTemplate.from_template(tool_template)
        inputs = {
            "preferences": search_result,  # 提供上下文内容
            "context": search_result,
            "username": user_name,
            "question": query,
        }

        chain = LLMChain(llm=self.llm, prompt=prompt)
        return chain.predict(**inputs)


if __name__ == "__main__":
    import langchain
    import tyro
    from sage.coordinators.coordinator import CoordinatorConfig

    langchain.verbose = True

    config = tyro.cli(CoordinatorConfig)
    coordinator = config.instantiate()

    print("\n=== Final Output ===\n")
    print(coordinator.execute("mmhu: What device should I use to watch a sci-fi movie?"))