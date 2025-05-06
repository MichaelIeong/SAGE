""" Sage Coordinator """

import os
import pickle
from dataclasses import asdict
from dataclasses import dataclass
from dataclasses import field
from datetime import date
from typing import Any, Type

from langchain.agents import initialize_agent, ZeroShotAgent, AgentExecutor
from langchain.prompts import PromptTemplate


from sage.base import BaseToolConfig
from sage.coordinators.base import AgentConfig
from sage.coordinators.base import BaseCoordinator
from sage.coordinators.base import CoordinatorConfig
from sage.human_interaction.tools import HumanInteractionToolConfig
from sage.misc_tools.weather_tool import WeatherToolConfig
from sage.retrieval.memory_bank import MemoryBank
from sage.smartthings.persistent_command_tools import ConditionCheckerToolConfig
from sage.smartthings.persistent_command_tools import NotifyOnConditionToolConfig
from sage.smartthings.smartthings_tool import SmartThingsToolConfig
from sage.retrieval.tools import UserProfileToolConfig
from sage.enviroment.tool import EnvironmentInfoToolConfig
from sage.deviceInfo.tool import DeviceInfoToolConfig
from sage.deviceControl.device_control_tool import DeviceControlToolConfig
from sage.smartthings.tv_schedules import QueryTvScheduleToolConfig
from sage.utils.llm_utils import TGIConfig
from sage.utils.llm_utils import OllamaConfig
from sage.utils.logging_utils import initialize_tool_names
from sage.chroma_registry.memory_registry import init_shared_memory


@dataclass
class SAGECoordinatorConfig(CoordinatorConfig):
    """SAGE coordinator config"""

    _target: Type = field(default_factory=lambda: SAGECoordinator)

    name: str = "SAGE"
    agent_type: str = "zero-shot-react-description"
    #agent_config: AgentConfig = AgentConfig()
    agent_config: AgentConfig = field(default_factory=lambda: AgentConfig(input_variables=["input"]))
    memory_path: str = os.path.join(
        os.getenv("SMARTHOME_ROOT"), "data/memory_data/large_memory_bank.json"
    )
    # Bool to activate the memory updating
    enable_memory_updating: bool = False

    # Bool to activate human interaction
    enable_human_interaction: bool = False

    # Bool to activate google tool
    enable_google: bool = False

    # Bool to use the same llm config for all the tools
    single_llm_config: bool = True

    # output dir for snapshots
    output_dir: str = os.path.join(
        os.getenv("SMARTHOME_ROOT"), "logs", "memory_snapshots"
    )

    # The tools config
    tool_configs: tuple[BaseToolConfig, ...] = (
        UserProfileToolConfig(),
        EnvironmentInfoToolConfig(),
        DeviceInfoToolConfig(),
        DeviceControlToolConfig()
    )

    # Save a snapshot of the memory of N interactions
    snapshot_frequency: int = 1

    def __post_init__(self):
        super().__post_init__()

        if self.enable_google:
            from sage.misc_tools.google_suite import GoogleToolConfig
            self.tool_configs = self.tool_configs + (GoogleToolConfig(),)

        if self.single_llm_config:
            self.override_llm_config(tool_configs=self.tool_configs)

    def override_llm_config(self, tool_configs: tuple[BaseToolConfig]) -> None:
        """Overrides the LLM config for the tools based on the coordinator config"""
        for config in tool_configs:
            if len(config.tool_configs) > 0:
                self.override_llm_config(config.tool_configs)

            if hasattr(config, "llm_config"):
                config.llm_config = self.llm_config


class SAGECoordinator(BaseCoordinator):
    """SAGE coordinator instantiates agents, llms and tools"""

    def __init__(self, config: SAGECoordinatorConfig):
        super().__init__(config)

        self.tooldict = {}
        self.memory = init_shared_memory()
        if isinstance(config.llm_config, OllamaConfig):
            config.llm_config = OllamaConfig(stop=["Human", "Question"])

        # 确保日志目录存在
        os.makedirs(config.global_config.logpath, exist_ok=True)

        # setup tools
        self._build_tools()

        # save tool descriptions in logs (used in visualization)
        tool_file = os.path.join(config.global_config.logpath, "tools.pickle")
        if not os.path.exists(tool_file):
            with open(tool_file, "wb") as fp:  # Pickling
                pickle.dump(initialize_tool_names(self.tooldict), fp)

        # setup agent
        toollist = [tool for _, tool in self.tooldict.items()]
        custom_prompt = PromptTemplate(
            template="{input}",
            input_variables=["input"]
        )
        self.agent = self._build_agent(toollist, self.llm, self.config.agent_config)

        self.request_idx = 0

    def _build_agent(self, toollist, llm, agent_config):
        from langchain.prompts import PromptTemplate
        from langchain.agents import ZeroShotAgent, AgentExecutor
        from langchain.chains import LLMChain

        # 手动构造一个 PromptTemplate
        prompt = PromptTemplate(
            template=(
                "Answer the following question as best you can.\n\n"
                "Available tools:\n"
                "{tools}\n\n"
                "{prefix}\n\n"
                "{suffix}\n\n"
                "Question: {input}\n"
                "Thought:{agent_scratchpad}\n"
                "You must always output a Thought, Action, and Action Input.When you have a final answer, respond with:Final Answer: [your answer here]"
            ),
            input_variables=["input", "agent_scratchpad", "tools", "prefix", "suffix"],
        )

        # 将 LLM 包成 LLMChain
        llm_chain = LLMChain(llm=llm, prompt=prompt)

        # 用 llm_chain 创建 agent
        agent = ZeroShotAgent(
            llm_chain=llm_chain,
            allowed_tools=[tool.name for tool in toollist],
        )

        # 最后生成 AgentExecutor
        return AgentExecutor.from_agent_and_tools(
            agent=agent,
            tools=toollist,
            verbose=agent_config.verbose,
            handle_parsing_errors=True,
        )

    def _build_tools(self) -> None:
        """Add tools to the agent"""
        memory_pool = self.memory  # memory 是一个 dict，包含多个 MemoryBank 实例

        for tool_config in self.config.tool_configs:
            if (
                    not self.config.enable_human_interaction
                    and tool_config.name == "human_interaction_tool"
            ):
                continue

            # 根据工具类型选择 memory 子模块
            if tool_config.name == "user_preference_tool":
                mem = memory_pool.get("user_profile", None)
            elif tool_config.name == "device_info_tool":
                mem = memory_pool.get("device_info", None)
            elif tool_config.name == "environment_info_tool":
                mem = memory_pool.get("environment_info", None)
            else:
                mem = None

            # 如果需要 memory 就注入
            if self.config.enable_memory_updating or mem is not None:
                tool = tool_config.instantiate(memory=mem)
            else:
                tool = tool_config.instantiate()

            self.tooldict[tool.name] = tool

    def update_tools(self, kwargs: dict[str, Any]) -> None:
        """Update the path from which the tool will read the json files for deviceInfo states and global states"""
        for tool_name, tool in self.tooldict.items():
            if tool_name in kwargs:
                tool.update(kwargs[tool_name])

    def update_memory(self, user_name: str, command: str) -> None:
        """Update the user memory."""
        self.memory.add_query(user_name, command, str(date.today()))
        for tool_config in self.config.tool_configs:
            if tool_config.name == "user_profile_tool":
                self.tooldict["user_profile_tool"] = tool_config.instantiate(
                    memory=self.memory
                )
                break

        # save a snapshot after snapshot_frequency interactions
        self.request_idx += 1
        if self.request_idx % self.config.snapshot_frequency == 0:
            self.memory.save_snapshot(
                os.path.join(self.config.output_dir, "memory_snapshots")
            )

    def _tool_desc(self) -> str:
        return "\n".join(f"{tool.name}: {tool.description}" for tool in self.tooldict.values())

    def execute(self, command: str) -> str:
        """Runs the agent with the provided command"""
        response = self.agent(
            {
                "input": command,
                "agent_scratchpad": "",  # 初始空白
                "tools": self._tool_desc(),  # 工具描述
                "prefix": self.config.agent_config.prefix,
                "suffix": self.config.agent_config.suffix,
            },
            callbacks=self.callbacks,
        )

        if self.config.enable_memory_updating:
            user_name, query = command.split(":")
            user_name = user_name.lower().strip()
            self.update_memory(user_name, query)

        return response