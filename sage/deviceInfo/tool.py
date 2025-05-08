import os
import json
import requests
from typing import Dict, Any, Type
from dataclasses import dataclass, field
from typing import Optional, Any
from pydantic import Field
from langchain.prompts import PromptTemplate
from langchain.llms.base import BaseLLM
from langchain import LLMChain

from sage.base import SAGEBaseTool, BaseToolConfig
from sage.deviceInfo.templates import device_info_prompt_template
from sage.retrieval.memory_bank import MemoryBank
from sage.utils.llm_utils import LLMConfig, TGIConfig, OllamaConfig
from sage.utils.common import parse_json
from sage.chroma_registry.memory_registry import init_shared_memory


@dataclass
class DeviceInfoToolConfig(BaseToolConfig):
    _target: Type = field(default_factory=lambda: DeviceInfoTool)
    name: str = "device_info_tool"
    description: str = """
Use this to retrieve information about devices in a given spaceId.
Input should be a JSON string with two keys: 'query' and 'spaceId'.

Example input:
{"query": "What devices are available in space 1?", "spaceId": "1"}
"""
    vectordb: str = "chroma_deviceinfo"
    embedding_model: str = "sentence-transvformers/all-MiniLM-L6-v2"
    top_k: int = 10
    llm_config: LLMConfig = None


class DeviceInfoTool(SAGEBaseTool):
    config: DeviceInfoToolConfig = None


    llm: BaseLLM = None
    memory: MemoryBank = None

    def setup(self, config: DeviceInfoToolConfig, memory=None) -> None:
        self.config = config
        if isinstance(config.llm_config, OllamaConfig):
            config.llm_config = OllamaConfig(stop=["Question"])
        self.llm = config.llm_config.instantiate()

        self.memory = memory
        if self.memory is None:
            raise ValueError("DeviceInfoTool requires a shared MemoryBank instance.")

    def _run(self, text: str) -> str:
        attr = parse_json(text)
        if not attr or "query" not in attr or "spaceId" not in attr:
            return "Invalid input format. Expected JSON with keys 'query' and 'spaceId'."

        query = attr["query"]
        spaceId = str(attr["spaceId"]).strip().lower().replace("space_", "")  # 支持 space_3 或 3 格式

        try:
            search_results = self.memory.search(
                query=query,
                vectorstore=self.config.vectordb,
                top_k=self.config.top_k
            )
        except Exception as e:
            return f"[Error] Failed to retrieve device info: {e}"

        if not search_results:
            return f"No devices found for spaceId '{spaceId}'."

        # 过滤掉不是该空间的设备信息（解析 spaceId）
        filtered = []
        for item in search_results:
            if isinstance(item, str) and f"space {spaceId}" in item.lower():
                filtered.append(item)

        if not filtered:
            return f"No devices found in space {spaceId}."

        context = "\n".join(filtered)
        prompt = PromptTemplate.from_template(device_info_prompt_template)
        chain = LLMChain(llm=self.llm, prompt=prompt)

        return chain.predict(
            context=context,
            username=attr.get("user_name", "unknown"),
            question=query
        )

if __name__ == "__main__":
    import langchain
    import tyro
    from sage.coordinators.sage_coordinator import CoordinatorConfig

    langchain.verbose = True

    config = tyro.cli(CoordinatorConfig)
    coordinator = config.instantiate()

    print("\n=== Final Output ===\n")
    print(coordinator.execute("mmhu: What devices are available in my space?"))
