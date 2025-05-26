"""
title: Max Turns Filter
author: open-webui
author_url: https://github.com/open-webui
funding_url: https://github.com/open-webui
version: 0.1.1
"""

from pydantic import BaseModel, Field
from typing import Optional


class Filter:
    class Valves(BaseModel):
        priority: int = Field(
            default=0, description="Priority level for the filter operations."
        )
        max_turns_for_users: int = Field(
            default=8,
            description="Maximum allowable conversation turns for a non-admin user.",
        )
        max_turns_for_admins: int = Field(
            default=8,
            description="Maximum allowable conversation turns for an admin user.",
        )
        enabled_for_admins: bool = Field(
            default=True,
            description="Whether the max turns limit is enabled for admins.",
        )
        pass

    class UserValves(BaseModel):
        max_turns: int = Field(
            default=4, description="Maximum allowable conversation turns for a user."
        )
        pass

    def __init__(self):
        # Indicates custom file handling logic. This flag helps disengage default routines in favor of custom
        # implementations, informing the WebUI to defer file-related operations to designated methods within this class.
        # Alternatively, you can remove the files directly from the body in from the inlet hook
        # self.file_handler = True

        # Initialize 'valves' with specific configurations. Using 'Valves' instance helps encapsulate settings,
        # which ensures settings are managed cohesively and not confused with operational flags like 'file_handler'.
        self.valves = self.Valves()
        pass

    def inlet(self, body: dict, __user__: Optional[dict] = None) -> dict:
        # Modify the request body or validate it before processing by the chat completion API.
        # This function is the pre-processor for the API where various checks on the input can be performed.
        # It can also modify the request before sending it to the API.
        print(f"inlet:{__name__}")
        print(f"inlet:body:{body}")
        print(f"inlet:user:{__user__}")

        if __user__ is not None:
            messages = body.get("messages", [])
            if __user__.get("role") == "admin" and not self.valves.enabled_for_admins:
                max_turns = float("inf")
            else:
                max_turns = (
                    self.valves.max_turns_for_admins
                    if __user__.get("role") == "admin"
                    else self.valves.max_turns_for_users
                )
            current_turns = (
                len(messages) // 2
            )  # Each turn consists of a user message and an assistant response

            if current_turns >= max_turns:
                raise Exception(
                    f"Conversation turn limit exceeded. The maximum turns allowed is {max_turns}."
                )

        return body

    def outlet(self, body: dict, __user__: Optional[dict] = None) -> dict:
        # Modify or analyze the response body after processing by the API.
        # This function is the post-processor for the API, which can be used to modify the response
        # or perform additional checks and analytics.
        print(f"outlet:{__name__}")
        print(f"outlet:body:{body}")
        print(f"outlet:user:{__user__}")

        return body
