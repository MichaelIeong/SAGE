import os
import json
import requests
from typing import Dict, Any, Type
from dataclasses import dataclass, field

from langchain.prompts import PromptTemplate
from langchain.llms.base import BaseLLM
from langchain import LLMChain

from sage.base import SAGEBaseTool, BaseToolConfig
from sage.retrieval.memory_bank import MemoryBank
from sage.utils.llm_utils import LLMConfig, TGIConfig
from sage.utils.common import parse_json
from sage.utils.memory_registry import init_shared_memory


@dataclass
class DeviceInfoToolConfig(BaseToolConfig):
    _target: Type = field(default_factory=lambda: DeviceInfoTool)
    name: str = "device_info_tool"
    description: str = """
Use this to retrieve information about devices in a given location.
Input should be a JSON string with two keys: 'query' and 'location'.

Example input:
{"query": "What TV is available?", "location": "Shanghai"}
"""
    vectordb: str = "chroma_deviceinfo"
    top_k: int = 5
    llm_config: LLMConfig = None


class DeviceInfoTool(SAGEBaseTool):
    llm: BaseLLM = None
    memory: MemoryBank = None

    def setup(self, config: DeviceInfoToolConfig, memory=None) -> None:
        self.config = config
        if isinstance(config.llm_config, TGIConfig):
            config.llm_config = TGIConfig(stop_sequences=["Human", "Question"])
        self.llm = config.llm_config.instantiate()

        self.memory = memory or init_shared_memory()

    def _run(self, text: str) -> str:
        attr = parse_json(text)
        if not attr or "query" not in attr or "location" not in attr:
            return "Invalid input format. Expected JSON with keys 'query' and 'location'."

        query = attr["query"]
        location = attr["location"]

        results = self.memory.search({"query": query},
                                     top_k=self.config.top_k,
                                     index_name=self.config.vectordb)

        filtered = [
            item for item in results
            if item.get("location", "").lower() == location.lower()
        ]

        if not filtered:
            return f"No matching devices found for location: {location}"

        return "\n".join(item["instruction"] for item in filtered)


if __name__ == "__main__":
    import langchain
    import tyro
    from sage.coordinators.coordinator import CoordinatorConfig

    langchain.verbose = True

    config = tyro.cli(CoordinatorConfig)
    coordinator = config.instantiate()

    print("\n=== Final Output ===\n")
    print(coordinator.execute("mmhu: Which devices can I use in Shanghai?"))
