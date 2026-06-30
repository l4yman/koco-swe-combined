def ChatCompletionUserMessage(role: str = "user", content: str = "") -> dict:
    return {"role": role, "content": content}


def ChatCompletionAssistantMessage(role: str = "assistant", content: str = "") -> dict:
    return {"role": role, "content": content}
