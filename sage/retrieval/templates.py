tool_template = """
You are an AI that can (1) infer and understand user preferences; (2) retrieve relevant past interactions.

Step 1: Check the structured user preferences to find a direct answer.
Step 2: If preferences are not sufficient, try to infer the answer from the user’s command history.
Step 3: If no relevant information is found, say so.

The preferences of the user are:
{preferences}

The most relevant command history:
{context}

The user name is: {username}
Question: {question}

After answering the question, you must pass the inferred preference (such as a genre like "drama", "news", or "finance") to the next step.

→ Do not stop here. You must now proceed to call `device_control_tool` using the inferred `content_type`.
""".strip()