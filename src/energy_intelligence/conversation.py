from typing import Literal, TypedDict

MAX_HISTORY_MESSAGES = 6
MAX_HISTORY_MESSAGE_CHARS = 4000


class ConversationTurn(TypedDict):
    role: Literal["user", "assistant"]
    content: str


def limit_history(history: list[ConversationTurn]) -> list[ConversationTurn]:
    return [
        {"role": turn["role"], "content": turn["content"][:MAX_HISTORY_MESSAGE_CHARS]}
        for turn in history[-MAX_HISTORY_MESSAGES:]
        if turn["content"].strip()
    ]
