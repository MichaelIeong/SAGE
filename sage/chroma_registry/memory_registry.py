# sage/chroma_registry/memory_registry.py

import os
import requests
import json

from sage.retrieval.memory_bank import MemoryBank
from sage.utils.common import SMARTHOME_ROOT


def fetch_device_info_from_api(project_id: int = 1) -> list[dict]:
    try:
        url = "http://localhost:8080/api/devices"
        params = {"project": project_id}
        resp = requests.get(url, params=params, timeout=5)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        print(f"[DeviceInfo API] Failed to fetch device info: {e}")
        return []


def fetch_env_info_from_api() -> list[dict]:
    try:
        resp = requests.get("http://localhost:8080/api/person", timeout=5)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        print(f"[EnvInfo API] Failed to fetch env info: {e}")
        return []


def convert_to_natural_language(data: list[dict], source: str) -> list[str]:
    if source == "device":
        return [
            f"Device '{d.get('deviceName', 'unknown')}' (ID: {d.get('deviceId', 'N/A')}) is located in space {d.get('spaceId', 'N/A')} and supports functions: {', '.join(f['functionName'] for f in d.get('functions', []))}."
            for d in data
        ]
    elif source == "env":
        return [
            f"Person '{d.get('personName', 'unknown')}' is currently located in space {d.get('spaceId', 'N/A')}."
            for d in data
        ]
    else:
        return [json.dumps(d) for d in data]


def ensure_json_file(path: str, data: list[dict], source: str):
    if not data:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for line in convert_to_natural_language(data, source):
            f.write(json.dumps({"instruction": line}, ensure_ascii=False) + "\n")


def init_shared_memory() -> MemoryBank:
    memory_data_root = os.path.join(SMARTHOME_ROOT, "memory_data")
    os.makedirs(memory_data_root, exist_ok=True)

    # ===== 1. 加载用户偏好 memory_bank.json =====
    memory = MemoryBank()
    user_profile_path = os.path.join(memory_data_root, "memory_bank.json")
    if os.path.exists(user_profile_path):
        memory.read_from_json(user_profile_path)
        if isinstance(memory.history, dict):
            memory.create_indexes("chroma", "sentence-transformers/all-MiniLM-L6-v2", load=True)
        else:
            print("[Warning] memory_bank.json format invalid. Skipped.")

    # ===== 2. 单独加载设备信息 device_info.json =====
    # device_info_path = os.path.join(memory_data_root, "device_info.json")
    # if os.path.exists(device_info_path):
    #     device_memory = MemoryBank()
    #     with open(device_info_path, "r", encoding="utf-8") as f:
    #         lines = [json.loads(line) for line in f if line.strip()]
    #     device_memory.history = lines
    #     device_memory.create_indexes("chroma_deviceinfo", "sentence-transformers/all-MiniLM-L6-v2", load=True)
    #
    device_info_path = os.path.join(memory_data_root, "device_info.json")
    if not os.path.exists(device_info_path) or os.stat(device_info_path).st_size == 0:
        device_info_data = fetch_device_info_from_api()
        ensure_json_file(device_info_path, device_info_data, source="device")
    memory.read_from_json(device_info_path)
    if isinstance(memory.history, list) and isinstance(memory.history[0], dict):
        memory.history = [item["instruction"] for item in memory.history if "instruction" in item]

    memory.create_indexes("chroma_deviceinfo", "sentence-transformers/all-MiniLM-L6-v2", load=True)

    # ===== 3. 单独加载环境信息 env_info.json =====
    # env_info_path = os.path.join(memory_data_root, "env_info.json")
    # if os.path.exists(env_info_path):
    #     env_memory = MemoryBank()
    #     with open(env_info_path, "r", encoding="utf-8") as f:
    #         lines = [json.loads(line) for line in f if line.strip()]
    #     env_memory.history = lines
    #     env_memory.create_indexes("chroma_environment", "sentence-transformers/all-MiniLM-L6-v2", load=True)

    env_info_path = os.path.join(memory_data_root, "env_info.json")
    if not os.path.exists(env_info_path) or os.stat(env_info_path).st_size == 0:
        env_info_data = fetch_env_info_from_api()
        ensure_json_file(env_info_path, env_info_data, source="env")
    memory.read_from_json(env_info_path)

    # 修复数据格式
    if isinstance(memory.history, list) and isinstance(memory.history[0], dict):
        memory.history = [item["instruction"] for item in memory.history if "instruction" in item]

    memory.create_indexes("chroma_environment", "sentence-transformers/all-MiniLM-L6-v2", load=True)
    return memory