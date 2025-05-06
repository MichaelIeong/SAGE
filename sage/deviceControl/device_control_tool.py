import requests
from typing import Type
from dataclasses import dataclass, field

from langchain.chains.llm import LLMChain
from langchain.llms.base import BaseLLM
from langchain.prompts import PromptTemplate
from sage.utils.llm_utils import LLMConfig, TGIConfig, OllamaConfig

from sage.base import SAGEBaseTool, BaseToolConfig
from sage.deviceControl.templates import device_control_prompt_template
from sage.utils.common import parse_json


@dataclass
class DeviceControlToolConfig(BaseToolConfig):
    _target: Type = field(default_factory=lambda: DeviceControlTool)
    name: str = "device_control_tool"
    description: str = (
        "Use this to control a specific device by calling its control API.\n"
        "Input should be a JSON string with four keys: 'device_id', 'function_url', 'content_type', and 'username'.\n"
        "It will construct the final URL as: {function_url}/{device_id}/{content_type}, and make a GET request.\n"
        "Example input: {\"device_id\": \"12\", \"function_url\": \"api/tv/movie\", \"content_type\": \"sci-fi\", \"username\": \"mmhu\"}"
    )

    top_k: int = 10
    llm_config: LLMConfig = None


class DeviceControlTool(SAGEBaseTool):
    config: DeviceControlToolConfig = None
    llm: BaseLLM = None

    def setup(self, config: DeviceControlToolConfig, memory=None):
        self.config = config
        if isinstance(config.llm_config, OllamaConfig):
            config.llm_config.stop_sequences = ["Question"]
        self.llm = config.llm_config.instantiate()


    def _run(self, text: str) -> str:
        attr = parse_json(text)
        if not attr or not all(k in attr for k in ["function_url", "device_id", "content_type", "username", "question"]):
            return "Invalid input format. Expected JSON with keys: function_url, device_id, content_type, username, question."

        prompt = PromptTemplate.from_template(device_control_prompt_template)
        inputs = {
            "function_url": attr["function_url"].strip("/"),
            "device_id": attr["device_id"],
            "content_type": attr["content_type"],
            "username": attr["username"],
            "question": attr["question"]
        }
        print("dddddddddsddfgdgdhhddhghghbfgb")
        print(inputs)
        chain = LLMChain(llm=self.llm, prompt=prompt)
        llm_output = chain.predict(**inputs).strip()

        # 2. 从 LLM 输出中解析 URL
        if not llm_output.startswith("http"):
            return f"[Error] LLM did not return a valid URL: {llm_output}"

        try:
            print("-----------------------------")
            print(llm_output)
            response = requests.get(llm_output)
            if response.ok:
                return f"[Success] Executed device API: {llm_output}\nResponse: {response.text}"
            else:
                return f"[Failure] API call to {llm_output} returned status {response.status_code}: {response.text}"
        except Exception as e:
            return f"[Error] Exception while calling {llm_output}: {e}"


if __name__ == "__main__":
    # Manual test
    tool = DeviceControlTool()
    config = DeviceControlToolConfig()
    tool.setup(config)
    print(tool._run('{"device_id": "tv_01", "function_url": "http://localhost:8000/api/tv", "content_type": "movie"}'))
# mmhu : I want to watch tv