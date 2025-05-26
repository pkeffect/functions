"""
title: Context Clip Filter
author: open-webui
author_url: https://github.com/open-webui
funding_url: https://github.com/open-webui
version: 0.1
"""

from pydantic import BaseModel, Field
from typing import Optional


class Filter:
    class Valves(BaseModel):
        priority: int = Field(
            default=0, description="Priority level for the filter operations."
        )
        n_last_messages: int = Field(
            default=4, description="Number of last messages to retain."
        )
        pass

    class UserValves(BaseModel):
        pass

    def __init__(self):
        self.valves = self.Valves()
        pass

    def inlet(self, body: dict, __user__: Optional[dict] = None) -> dict:
        messages = body["messages"]
        # Ensure we always keep the system prompt
        system_prompt = next(
            (message for message in messages if message.get("role") == "system"), None
        )

        if system_prompt:
            messages = [
                message for message in messages if message.get("role") != "system"
            ]
            messages = messages[-self.valves.n_last_messages :]
            messages.insert(0, system_prompt)
        else:  # If no system prompt, simply truncate to the last n_last_messages
            messages = messages[-self.valves.n_last_messages :]

        body["messages"] = messages
        return body
