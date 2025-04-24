# sage/utils/memory_registry.py

import os
import requests
import json

from sage.retrieval.memory_bank import MemoryBank
from sage.utils.common import SMARTHOME_ROOT


def fetch_device_info_from_api() -> list[dict]:
    try:
        # todo: 改API，加Token
        resp = requests.get("http://10.226.197.56:8080/smartcity/api/device/deviceInfoList", timeout=5)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        print(f"[DeviceInfo API] Failed to fetch device info: {e}")
        return []


def fetch_env_info_from_api() -> list[dict]:
    try:
        resp = requests.get("http://10.226.197.56:8080/smartcity/api/env", timeout=5)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        print(f"[EnvInfo API] Failed to fetch env info: {e}")
        return []


def fetch_device_function_from_api() -> list[dict]:
    try:
        resp = requests.get("http://10.226.197.56:8080/smartcity/api/devicefunction", timeout=5)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        print(f"[DeviceFunction API] Failed to fetch device functions: {e}")
        return []


def convert_to_natural_language(data: list[dict], source: str) -> list[str]:
    """根据 source 类型构造自然语言描述"""
    if source == "device":
        return [
            f"There is a device named {d['deviceName']} (ID: {d['deviceId']}) located in {d['SpaceId']}."
            for d in data
        ]
    elif source == "env":
        return [
            f"In {d['location']}, temperature is {d.get('temperature', 'unknown')}°C and humidity is {d.get('humidity', 'unknown')}%."
            for d in data
        ]
    elif source == "function":
        return [
            f"Device '{d['deviceName']}' (ID: {d['deviceId']}) supports the function '{d['functionName']}' via API '{d['api']}'."
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
    memory = MemoryBank()

    memory_data_root = os.path.join(SMARTHOME_ROOT, "memory_data")
    os.makedirs(memory_data_root, exist_ok=True)

    # 用户偏好
    user_profile_path = os.path.join(memory_data_root, "memory_bank.json")
    memory.read_from_json(user_profile_path)
    memory.create_indexes("chroma", "sentence-transformers/all-MiniLM-L6-v2", load=True)

    # 设备信息
    device_info_path = os.path.join(memory_data_root, "device_info.json")
    if not os.path.exists(device_info_path) or os.stat(device_info_path).st_size == 0:
        device_info_data = fetch_device_info_from_api()
        ensure_json_file(device_info_path, device_info_data, source="device")
    memory.read_from_json(device_info_path)
    memory.create_indexes("chroma_deviceinfo", "sentence-transformers/all-MiniLM-L6-v2", load=True)

    # 环境信息
    env_info_path = os.path.join(memory_data_root, "env_info.json")
    if not os.path.exists(env_info_path) or os.stat(env_info_path).st_size == 0:
        env_info_data = fetch_env_info_from_api()
        ensure_json_file(env_info_path, env_info_data, source="env")
    memory.read_from_json(env_info_path)
    memory.create_indexes("chroma_environment", "sentence-transformers/all-MiniLM-L6-v2", load=True)

    # 设备功能信息
    device_function_path = os.path.join(memory_data_root, "device_function.json")
    if not os.path.exists(device_function_path) or os.stat(device_function_path).st_size == 0:
        function_data = fetch_device_function_from_api()
        ensure_json_file(device_function_path, function_data, source="function")
    memory.read_from_json(device_function_path)
    memory.create_indexes("chroma_devicefunction", "sentence-transformers/all-MiniLM-L6-v2", load=True)

    return memory