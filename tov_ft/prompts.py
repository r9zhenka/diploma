"""Prompt construction for the corporate ToV rewrite task.

The model receives:
  - a system prompt describing the corporate tone-of-voice (MOCK for now,
    later this 8-10 line block will be produced by an external system / OS),
  - a user turn holding the client dialog + the operator's rough free-text draft.

The model outputs the polished answer in corporate tone.
"""

# --- MOCK tone-of-voice system prompt -------------------------------------
# ~8-10 lines. Replace `CORPORATE_TOV_SYSTEM` at runtime with the OS-provided
# distribution-approximating prompt when it is available.
CORPORATE_TOV_SYSTEM = """\
Ты — оператор поддержки компании. Твоя задача: переписать черновик ответа в корпоративном тоне.
Правила тона:
1. Обращайся к клиенту на «вы», вежливо и уважительно.
2. Пиши грамотно, без сленга, сокращений и эмодзи.
3. Сохрани весь смысл черновика: факты, сроки и суммы не меняй и не выдумывай.
4. Формулируй кратко и по делу, избегай канцелярита и лишней воды.
5. Заверши ответ готовностью помочь дальше.
Выведи только итоговый ответ клиенту, без пояснений и без черновика.\
"""

_USER_TEMPLATE = """\
Диалог с клиентом:
{chat}

Черновик ответа оператора (смысл): {draft}

Перепиши черновик в корпоративном тоне."""


def build_user_prompt(chat: str, draft: str) -> str:
    """Render the user turn from a client dialog and the operator's rough draft."""
    return _USER_TEMPLATE.format(chat=chat.strip(), draft=draft.strip())


def build_messages(chat: str, draft: str, answer: str | None = None,
                   system: str = CORPORATE_TOV_SYSTEM) -> list[dict]:
    """Build a chat-format message list.

    If `answer` is given -> full training example (system/user/assistant).
    If `answer` is None   -> inference prompt (system/user only).
    """
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": build_user_prompt(chat, draft)},
    ]
    if answer is not None:
        messages.append({"role": "assistant", "content": answer.strip()})
    return messages
