device_info_prompt_template = """
You are given a list of device-related information retrieved from a smart space.
Your task is to help the user answer the question based solely on these context entries.

Context:
{context}

User: {username}
Question: {question}

Instructions:
- Answer clearly and concisely.
- ONLY mention devices that are explicitly named in the Context section above.
- NEVER invent or assume any device names or capabilities not found in the context.
- If there are no matching devices, respond with exactly:
  [No matching devices found]
- Your response must be grounded 100% in the context. Do not make assumptions.
- Do NOT include "Thought", "Action", or "Final Answer".

Examples:

Context:
- Device 'TV' (ID: 12) is located in space 3 and supports 'Turn on TV'.

User: alice
Question: What devices can I use to watch TV?
→ Device 'TV' is available for watching TV.

Context:
(no TV listed)

User: bob
Question: Is there a TV in the room?
→ [No matching devices found]

Now respond to the following:
"""