# sage/enviroment/prompt_template.py

environment_prompt_template = """
You are given environmental context information about a user. Your task is to extract relevant structured information to answer the question.

Context:
{context}

User: {username}
Question: {question}

Instructions:
- Answer clearly and concisely based only on the context.
- If the space ID is mentioned, output exactly in the following format:
  Space ID: <value>
- If the information is not available, respond with exactly:
  [No information available]
- Do NOT include "Thought", "Action", or "Final Answer".
- Do NOT invent answers not present in the context.

Your answer:
"""