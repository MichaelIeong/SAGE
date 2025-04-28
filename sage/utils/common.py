"""
Common utility functions
"""

import json
import os
import time
from functools import lru_cache
from inspect import currentframe, getsource
from typing import Any, List

import numpy as np
import yaml
from box import Box
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.output_parsers.json import parse_json_markdown
from rich.console import Console

from sage.utils.embedding_utils import OllamaEmbeddingOnly

# ====== 常量定义 ======
SMARTHOME_ROOT = os.getenv("SMARTHOME_ROOT", "/home/nanachi/SAGE")
CONSOLE = Console(width=120)


# ====== 环境检查 ======
def check_env_vars(hf_setup: bool = False):
    """Check if the environment variables are set."""
    if SMARTHOME_ROOT is None:
        raise ValueError("Environment variable $SMARTHOME_ROOT is not set.")

    if hf_setup:
        HUGGINGFACEHUB_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")
        os.environ["CURL_CA_BUNDLE"] = ""  # Fix for Huggingface SSL issues

        if HUGGINGFACEHUB_API_TOKEN is None:
            raise ValueError("Environment variable $HUGGINGFACEHUB_API_TOKEN is not set.")


# ====== JSON / YAML 文件处理 ======
def parse_json(json_string: str) -> Any:
    """Try parsing a JSON string; fallback to parsing JSON inside markdown."""
    try:
        return json.loads(json_string)
    except json.JSONDecodeError:
        try:
            return parse_json_markdown(json_string)
        except Exception:
            return None


def read_config(config_path: str) -> Box:
    """Loads configuration from a YAML file."""
    with open(config_path, "r", encoding="utf-8") as ymlfile:
        return Box(yaml.safe_load(ymlfile))


def save_config(path: str, object_to_save: Any) -> None:
    """Saves an object to disk as a YAML file."""
    assert os.path.splitext(path)[1] == ".yaml", "Only .yaml files are supported."
    print(f"Writing metadata to {path}")
    with open(path, "w", encoding="utf-8") as fp:
        yaml.dump(object_to_save, fp)


def read_json(filename: str) -> dict[str, Any]:
    """Reads a JSON file."""
    with open(filename, "r", encoding="utf-8") as f:
        return json.load(f)


# ====== 其他实用函数 ======
def findall(long_string: str, short_string: str) -> List[int]:
    """Find all occurrences of short_string inside long_string."""
    out = []
    idx_start = 0

    while True:
        idx = long_string.find(short_string)
        if idx == -1:
            return out
        out.append(idx + idx_start)
        long_string = long_string[idx + len(short_string):]
        idx_start += idx + len(short_string)


def function2string(function_handle: Any, code_define: str) -> str:
    """Append the source code of a function to an existing code string."""
    return code_define + "\n" + getsource(function_handle)


# ====== 嵌入模型加载 ======
def load_embedding_model(model_name: str = "nomic-embed-text"):
    """Builds and caches an embedding model."""
    return OllamaEmbeddingOnly(model=model_name)


# ====== 时间线追踪工具 ======
class Timeliner:
    """
    Utility to measure elapsed time between operations.

    Usage:
        from sage.utils.common import tl
        tl.timeline()
        do_something()
        tl.timeline()
    """

    def __init__(self):
        self.t = time.time()
        self.last_nan_count = None

    def timeline(self, nan_count=None):
        cf = currentframe()
        tnew = time.time()
        print(
            "pid",
            os.getpid(),
            cf.f_back.f_code.co_filename,
            cf.f_back.f_lineno,
            tnew - self.t,
        )

        if nan_count is not None:
            n_nans = np.isnan(nan_count).sum()
            head = tail = ""

            if self.last_nan_count is not None and self.last_nan_count != n_nans:
                head = "\033[91m"
                tail = "\033[0m"

            self.last_nan_count = n_nans
            print(head, "    nan count", n_nans, tail)

        self.t = tnew


# 全局实例
tl = Timeliner()