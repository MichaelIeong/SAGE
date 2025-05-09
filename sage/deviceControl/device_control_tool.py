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
    description: str = """
You must use this tool to execute **actual control of smart devices**. LLM cannot perform this action directly.

Do NOT attempt to construct or simulate device actions in your own response.

This tool is responsible for:
- Constructing the final API URL based on input parameters
- Executing the HTTP request to control the target device

To use it, you MUST provide a JSON input with the following keys:
- 'device_id' (str): The device to be controlled
- 'function_url' (str): The control endpoint (e.g. api/tv/show)
- 'content_type' (str): The genre/type the user prefers (e.g. drama, finance)
- 'username' (str): Who issued the command
- 'question' (str): The original user input

Only the tool can perform this operation. You MUST invoke it when device control is required.
"""

    top_k: int = 10
    llm_config: LLMConfig = None


class DeviceControlTool(SAGEBaseTool):
    config: DeviceControlToolConfig = None
    llm: BaseLLM = None

    def setup(self, config: DeviceControlToolConfig) -> None:
        self.config = config
        if isinstance(config.llm_config, OllamaConfig):
            config.llm_config = OllamaConfig(stop=["Question"])
        self.llm = config.llm_config.instantiate()


    def _run(self, text: str) -> str:
        print(">>> DeviceControlTool was triggered <<<")
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

        # LLM 输出
        print("Raw LLM output:\n", llm_output)

        # 从 LLM 输出中提取首个合法 URL
        lines = llm_output.strip().splitlines()
        url = next((line.strip() for line in lines if line.strip().startswith("http")), None)

        if not url:
            return f"[Error] LLM did not return a valid URL:\n{llm_output}"

        try:
            print("-----------------------------")
            print(url)
            response = requests.post(url)
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
# mmhu : I want to watch tv  dmitriy : I want to watch tv