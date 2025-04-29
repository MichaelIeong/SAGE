import json
from typing import Dict, Any, Type
from dataclasses import dataclass, field

from langchain.prompts import PromptTemplate
from langchain.llms.base import BaseLLM
from langchain import LLMChain

from sage.utils.llm_utils import LLMConfig, TGIConfig
from sage.base import SAGEBaseTool, BaseToolConfig
from sage.utils.common import parse_json
from typing import Optional, Any
from pydantic import Field

@dataclass
class DeviceFunctionToolConfig(BaseToolConfig):
    _target: Type = field(default_factory=lambda: DeviceFunctionTool)
    name: str = "device_function_tool"
    description: str = """
Use this to retrieve the list of supported functions and APIs for a given device.
Input should be a JSON string with one key: `device_id` (str).
Example input: {"device_id": "tv_01"}
"""
    llm_config: LLMConfig = None


class DeviceFunctionTool(SAGEBaseTool):
    config: Optional[Any] = Field(default=None, exclude=True)
    llm: Optional[Any] = Field(default=None, exclude=True)
    memory: Optional[Any] = Field(default=None, exclude=True)

    def setup(self, config: DeviceFunctionToolConfig, memory=None):
        if isinstance(config.llm_config, TGIConfig):
            config.llm_config = TGIConfig(stop_sequences=["Human", "Question"])
        self.llm = config.llm_config.instantiate()
        self.memory = memory

    def _run(self, text: str) -> str:
        attr = parse_json(text)
        if attr is None or "device_id" not in attr:
            return "Invalid input format. Expected JSON with key 'device_id'."

        device_id = attr["device_id"]

        results = self.memory.search({"query": f"What can device {device_id} do?"}, top_k=1, namespace="chroma_devicefunction")
        if not results:
            return f"No capability information found for device {device_id}"

        data = results[0]["raw"]

        prompt = PromptTemplate.from_template("""
Device ID: {device_id}
Functions and APIs:
{functions}

User Query: {query}

Based on the device capabilities, generate the proper API call and reasoning.
""")
        query = attr.get("query", "What can I do with this device?")
        inputs = {
            "device_id": device_id,
            "functions": json.dumps(data, indent=2),
            "query": query
        }

        chain = LLMChain(llm=self.llm, prompt=prompt)
        response = chain.predict(**inputs)

        return response


# 🧪 Test entry point
if __name__ == "__main__":
    import langchain
    import tyro
    from sage.coordinators.coordinator import CoordinatorConfig

    langchain.verbose = True

    # 使用命令行加载配置
    config = tyro.cli(CoordinatorConfig)
    coordinator = config.instantiate()

    # 示例命令执行
    result = coordinator.execute("mmhu: Can I turn on the TV and adjust its brightness?")
    print("\n=== Final Output ===\n")
    print(result)