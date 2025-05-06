device_control_prompt_template = """
You are responsible for constructing device API URLs for control actions.

Given:
- A fixed API base path (e.g., api/tv/movie),
- A device ID (e.g., 12),
- A content type (e.g., sci-fi),

You MUST construct the final URL as:

  http://localhost:8000/{function_url}/{device_id}/{content_type}

Important Rules:
- NEVER invent or guess any part of the URL.
- NEVER change or rewrite the given `function_url`.
- Output ONLY the final URL (starting with http://).
- Do NOT include any explanation or prefix.

Context:
Function URL: {function_url}
Device ID: {device_id}
Content Type: {content_type}
User: {username}
Request: {question}

Your answer:
"""