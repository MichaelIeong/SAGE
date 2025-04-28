# sage_main.py

import langchain
from sage.coordinators.sage_coordinator import SAGECoordinatorConfig
from sage.chroma_registry.memory_registry import init_shared_memory

def main():
    langchain.verbose = True  # 可以看到推理过程日志

    # 初始化共享向量数据库（MemoryBank）
    memory = init_shared_memory()

    # 初始化协调器（大模型+工具）
    coordinator_config = SAGECoordinatorConfig()
    coordinator = coordinator_config.instantiate()

    # 把共享 memory 注入所有需要 memory 的工具
    for tool in coordinator.tooldict.values():
        if hasattr(tool, "memory") and tool.memory is None:
            tool.memory = memory

    print("✅ SAGE 系统初始化完成！现在可以输入指令了。\n")

    while True:
        try:
            user_input = input("\n🧠 请输入你的需求（输入 exit 退出）：\n> ")
            if user_input.lower() in ["exit", "quit"]:
                break

            result = coordinator.execute(user_input)
            print(f"\n🛎️  结果：{result}\n")

        except KeyboardInterrupt:
            print("\n⛔ 退出 SAGE 系统。")
            break

if __name__ == "__main__":
    main()