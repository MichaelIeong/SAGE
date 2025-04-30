# sage/enviroment/prompt_template.py

environment_prompt_template = """
You are given environmental context information about a user. Your task is to extract relevant structured information to answer the user's question.

Context:
{context}

User: {username}
Question: {question}

Instructions:
- Answer clearly and concisely based only on the provided context.
- Your response must be a **single-line answer**.
- If the space ID is mentioned, output **exactly** in the following format (no extra text):
  Space ID: <value>
- If the information is not available, respond with **exactly**:
  [No information available]
- Do NOT include any "Thought", "Action", "Observation", or "Final Answer".
- Do NOT repeat the answer.
- Do NOT fabricate information not present in the context.

Your answer:
"""