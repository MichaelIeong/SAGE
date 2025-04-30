device_info_prompt_template = """
You are given a list of device-related information retrieved from a smart space.
Your task is to help the user answer the question based on these context entries.

Context:
{context}

User: {username}
Question: {question}

Instructions:
- Answer clearly and concisely.
- Only mention devices relevant to the user's space or question.
- If no devices match, respond exactly with:
  [No matching devices found]
- Do NOT include "Thought", "Action", or "Final Answer".
- Do NOT invent answers not in the context.
"""