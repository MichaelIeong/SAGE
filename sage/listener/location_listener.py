import json
from kafka import KafkaConsumer
from langchain.schema import Document
from sage.chroma_registry.memory_registry import init_shared_memory

# 初始化共享 memory（包含 chroma_environment 向量库）
memory = init_shared_memory()
index_name = "chroma_environment"

def human_location_to_nl(entry: dict) -> str:
    """
    将位置信息转换为自然语言句子。
    例如 {"name": "mmhu", "location": "Shanghai", "floor": "3"} ->
         "User mmhu is currently located at floor 3 in Shanghai."
    """
    name = entry.get("personName", "Unknown")
    location = entry.get("spaceId", "an unknown spaceId")
    return f"User {name} is currently located in {location}."

def main():
    consumer = KafkaConsumer(
        "env_update",
        bootstrap_servers=["10.192.48.114:9092"],
        auto_offset_reset="latest",
        enable_auto_commit=True,
        group_id="location_listener_group",
        value_deserializer=lambda x: json.loads(x.decode("utf-8")),
    )

    print("[Listener] Waiting for messages on topic 'human_location'...")

    for message in consumer:
        data = message.value
        try:
            sentence = human_location_to_nl(data)
            doc = Document(page_content=sentence, metadata=data)
            memory.indexes[index_name].add_documents([doc])
            print(f"[✓] Added to {index_name}: {sentence}")
        except Exception as e:
            print(f"[✗] Failed to process message: {e}")

if __name__ == "__main__":
    main()