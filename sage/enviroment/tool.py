import json
from typing import Dict, Any, Type
from dataclasses import dataclass, field

from langchain.prompts import PromptTemplate
from langchain.llms.base import BaseLLM
from langchain import LLMChain

from sage.utils.llm_utils import LLMConfig, TGIConfig,OllamaConfig
from sage.base import SAGEBaseTool, BaseToolConfig
from sage.utils.common import parse_json
from sage.enviroment.templates import environment_prompt_template
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
    embedding_model: str = "sentence-transvformers/all-MiniLM-L6-v2"
    top_k: int = 5
    llm_config: LLMConfig = None


class EnvironmentInfoTool(SAGEBaseTool):
    llm: BaseLLM = None
    memory: MemoryBank = None
    config: EnvironmentInfoToolConfig = None

    def setup(self, config: EnvironmentInfoToolConfig, memory=None):
        self.config = config
        if isinstance(config.llm_config, OllamaConfig):
            config.llm_config = OllamaConfig(stop=["Question"])
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

        # ✅ 不再检查 history 是否包含 user_name，因为是 list 类型
        try:
            search_result = self.memory.search(
                query=query,
                vectorstore=self.config.vectordb,
                top_k=self.config.top_k
            )
        except Exception as e:
            return f"[Error] Failed to retrieve environment info: {e}"

        if not search_result:
            return f"No environmental information found for user '{user_name}'."

        context = search_result

        prompt = PromptTemplate.from_template(environment_prompt_template)
        inputs = {
            "preferences": "\n".join(context),
            "context": "\n".join(context),
            "username": user_name,
            "question": query,
        }

        chain = LLMChain(llm=self.llm, prompt=prompt)
        try:
            return chain.predict(**inputs)
        except Exception as e:
            return f"[LLM Error] Failed to generate answer: {e}"


if __name__ == "__main__":
    import langchain
    import tyro
    from sage.coordinators.coordinator import CoordinatorConfig

    langchain.verbose = True

    config = tyro.cli(CoordinatorConfig)
    coordinator = config.instantiate()

    print("\n=== Final Output ===\n")
    print(coordinator.execute("mmhu: What is the ID of the space I am in?"))