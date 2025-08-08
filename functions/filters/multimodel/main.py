"""
title: Multi-Model Conversation Filter - Enhanced Integration
description: Advanced multi-model orchestration with seamless Agent Hotswap integration and enhanced reasoning capabilities
author: assistant & pkeffect
version: 3.2.1
required_open_webui_version: 0.5.0
license: MIT
"""

import asyncio
import logging
import json
import hashlib
import re
from datetime import datetime
from typing import Dict, List, Optional, Any, Literal
from pydantic import BaseModel, Field

# OpenWebUI imports
from open_webui.utils.chat import generate_chat_completion
from open_webui.models.users import Users
from open_webui.models.models import Models

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def get_available_models() -> List[str]:
    """Get ALL models from OpenWebUI's app state."""
    # Use a set to automatically handle duplicates
    models = {"-- Select Model --"}

    try:
        from open_webui.main import app

        if hasattr(app, "state") and hasattr(app.state, "MODELS"):
            # Get all string keys from the MODELS dictionary, no filtering for local-only
            model_ids = {k for k in app.state.MODELS.keys() if isinstance(k, str)}
            if model_ids:
                models.update(model_ids)
                logger.info(
                    f"[MultiModel] Loaded {len(model_ids)} models from app state."
                )
            else:
                logger.warning("[MultiModel] app.state.MODELS was found but empty.")
        else:
            logger.warning("[MultiModel] Could not find app.state.MODELS.")
    except Exception as e:
        logger.error(f"[MultiModel] Error loading models from app state: {e}")

    # Return a sorted list for a consistent UI
    return sorted(list(models))


class AgentHotswapIntegration:
    """Enhanced integration with Agent Hotswap plugin"""

    @staticmethod
    def detect_persona_context(body: Dict) -> Dict:
        """Extract comprehensive persona context from Agent Hotswap"""
        filter_context = body.get("_filter_context", {})

        integration_info = {
            "has_agent_hotswap": bool(filter_context.get("agent_hotswap_active")),
            "hotswap_version": filter_context.get("agent_hotswap_version"),
            "persona_type": "none",
        }

        # Detect per-model personas
        if filter_context.get("per_model_active"):
            per_model_personas = {}
            for key, value in filter_context.items():
                if key.startswith("persona") and key[7:].isdigit():
                    model_num = int(key[7:])
                    per_model_personas[model_num] = value

            if per_model_personas:
                integration_info.update(
                    {
                        "persona_type": "per_model",
                        "per_model_personas": per_model_personas,
                        "total_assigned_models": filter_context.get(
                            "total_assigned_models", 0
                        ),
                        "assigned_model_numbers": filter_context.get(
                            "assigned_model_numbers", []
                        ),
                    }
                )

        # Detect single persona
        elif filter_context.get("single_persona_active"):
            integration_info.update(
                {
                    "persona_type": "single",
                    "active_persona": filter_context.get("active_persona"),
                    "active_persona_name": filter_context.get("active_persona_name"),
                    "active_persona_prompt": filter_context.get(
                        "active_persona_prompt"
                    ),
                }
            )

        # Detect multi-persona sequence
        elif filter_context.get("multi_persona_active"):
            integration_info.update(
                {
                    "persona_type": "multi",
                    "persona_sequence": filter_context.get("persona_sequence", []),
                    "total_personas": filter_context.get("total_personas", 0),
                }
            )

        return integration_info


class ReasoningCapabilities:
    """Enhanced reasoning capabilities for different model types"""

    @staticmethod
    def supports_native_reasoning(model_name: str) -> bool:
        """Check if model supports native reasoning tokens"""
        reasoning_models = [
            "claude",
            "deepseek-r1",
            "qwen-qwq",
            "gemini",
            "anthropic",
        ]
        model_lower = model_name.lower()
        return any(
            reasoning_model in model_lower for reasoning_model in reasoning_models
        )

    @staticmethod
    def get_reasoning_effort_tokens(effort: str) -> int:
        """Get reasoning token count based on effort level"""
        effort_map = {"low": 4000, "medium": 12000, "high": 32000}
        return effort_map.get(effort, 12000)

    @staticmethod
    def get_thinking_tags() -> List[tuple]:
        """Get list of thinking tag patterns to process"""
        return [
            ("think", "/think"),
            ("thinking", "/thinking"),
            ("reason", "/reason"),
            ("reasoning", "/reasoning"),
            ("thought", "/thought"),
            ("Thought", "/Thought"),
            ("|begin_of_thought|", "|end_of_thought|"),
        ]


class ModelIconManager:
    """Enhanced model icon management with persona awareness"""

    @staticmethod
    def get_model_icon(model_name: str, persona_info: Dict = None) -> str:
        """Get a unique icon for each model type, with persona override"""
        # If persona has specific icon, use it
        if persona_info and persona_info.get("name"):
            persona_name = persona_info["name"]
            if "🎓" in persona_name or "Teacher" in persona_name:
                return "🎓"
            elif "🎒" in persona_name or "Student" in persona_name:
                return "🎒"
            elif "🔬" in persona_name or "Scientist" in persona_name:
                return "🔬"
            elif (
                "🧭" in persona_name
                or "Ethicist" in persona_name
                or "Moral" in persona_name
            ):
                return "🧭"
            elif (
                "💻" in persona_name
                or "Tech" in persona_name
                or "Coder" in persona_name
            ):
                return "💻"
            elif (
                "📈" in persona_name
                or "Economist" in persona_name
                or "Market" in persona_name
            ):
                return "📈"
            elif "🏛️" in persona_name or "Policy" in persona_name:
                return "🏛️"
            elif "✍️" in persona_name or "Writer" in persona_name:
                return "✍️"
            elif "📊" in persona_name or "Analyst" in persona_name:
                return "📊"

        # Default model-based icons
        model_lower = model_name.lower()

        # Reasoning models get special icons
        if "deepseek-r1" in model_lower:
            return "🧠"
        elif "qwen-qwq" in model_lower:
            return "🤔"
        elif "claude" in model_lower:
            return "🎭"
        # Llama family
        elif "llama" in model_lower:
            return "🦙"
        # Qwen family
        elif "qwen" in model_lower:
            return "🧠"
        # Mistral family
        elif "mistral" in model_lower:
            return "🌪️"
        # Code models
        elif any(x in model_lower for x in ["code", "coder", "coding"]):
            return "💻"
        # Math/reasoning models
        elif any(x in model_lower for x in ["math", "reasoning", "think"]):
            return "🧮"
        # Phi family
        elif "phi" in model_lower:
            return "🔬"
        # Gemma family
        elif "gemma" in model_lower:
            return "💎"
        # Aya family
        elif "aya" in model_lower:
            return "🌟"
        # Yi family
        elif "yi" in model_lower:
            return "🎯"
        # DeepSeek family
        elif "deepseek" in model_lower:
            return "🤿"
        # Default for unknown models
        else:
            return "🤖"

    @staticmethod
    def get_model_display_name(model_name: str, persona_info: Dict = None) -> str:
        """Get a clean display name for the model with persona integration"""
        if persona_info and persona_info.get("name"):
            return persona_info["name"]

        # Remove :latest, :v1, etc.
        clean_name = model_name.split(":")[0]

        # Capitalize and format nicely
        parts = clean_name.split("-")
        formatted_parts = []

        for part in parts:
            if part.lower() in [
                "llama",
                "qwen",
                "mistral",
                "phi",
                "gemma",
                "aya",
                "yi",
                "deepseek",
            ]:
                formatted_parts.append(part.capitalize())
            elif part.isdigit() or any(c.isdigit() for c in part):
                formatted_parts.append(part.upper())
            else:
                formatted_parts.append(part.capitalize())

        return " ".join(formatted_parts)


class ThinkingProcessor:
    """Enhanced thinking processor with persona awareness"""

    def __init__(self):
        self.thinking_tags = ReasoningCapabilities.get_thinking_tags()
        self.thinking_pattern = self._build_thinking_pattern()

    def _build_thinking_pattern(self) -> re.Pattern:
        """Build regex pattern to match thinking blocks"""
        patterns = []
        for open_tag, close_tag in self.thinking_tags:
            # Escape special regex characters
            open_escaped = re.escape(f"<{open_tag}>")
            close_escaped = re.escape(f"<{close_tag}>")
            patterns.append(f"{open_escaped}(.*?){close_escaped}")

        combined_pattern = "|".join(patterns)
        return re.compile(combined_pattern, re.DOTALL | re.IGNORECASE)

    def extract_thinking_blocks(self, content: str) -> List[Dict[str, str]]:
        """Extract thinking blocks from content"""
        thinking_blocks = []
        matches = self.thinking_pattern.finditer(content)

        for match in matches:
            # Find which group matched (which tag pair)
            for i, group_content in enumerate(match.groups()):
                if group_content is not None:
                    thinking_blocks.append(
                        {
                            "content": group_content.strip(),
                            "start": match.start(),
                            "end": match.end(),
                            "tag_index": i,
                        }
                    )
                    break

        return thinking_blocks

    def format_thinking_for_display(
        self, thinking_content: str, persona_name: str = None
    ) -> str:
        """Format thinking content for display with persona context"""
        # If it's already formatted with details tags, return as-is
        if "<details>" in thinking_content and "<summary>" in thinking_content:
            formatted = thinking_content.replace("<strong>", "**").replace(
                "</strong>", "**"
            )
            return formatted

        # Format thinking content with blockquote and italics for visual distinction
        lines = thinking_content.split("\n")
        quoted_lines = [f"> *{line}*" if line.strip() else ">" for line in lines]
        quoted_content = "\n".join(quoted_lines)

        summary_text = (
            f"🤔 **{persona_name} Reasoning Process**"
            if persona_name
            else "🤔 **Model Reasoning Process**"
        )

        return f"""
<details>
<summary>{summary_text}</summary>

{quoted_content}

</details>
"""

    def remove_thinking_tags(self, content: str) -> str:
        """Remove thinking tags from content"""
        for open_tag, close_tag in self.thinking_tags:
            content = re.sub(
                f"<{re.escape(open_tag)}>.*?<{re.escape(close_tag)}>",
                "",
                content,
                flags=re.DOTALL | re.IGNORECASE,
            )
        return content.strip()

    def process_response_with_thinking(
        self, content: str, persona_name: str = None
    ) -> Dict[str, str]:
        """Process response and separate thinking from open_webui.main content"""
        thinking_blocks = self.extract_thinking_blocks(content)
        main_content = self.remove_thinking_tags(content)

        if thinking_blocks:
            # Combine all thinking blocks
            all_thinking = "\n\n---\n\n".join(
                [block["content"] for block in thinking_blocks]
            )
            formatted_thinking = self.format_thinking_for_display(
                all_thinking, persona_name
            )

            return {
                "has_thinking": True,
                "thinking": formatted_thinking,
                "main_content": main_content,
                "raw_content": content,
            }
        else:
            return {
                "has_thinking": False,
                "thinking": "",
                "main_content": content,
                "raw_content": content,
            }


class ConversationManager:
    """Enhanced conversation management with persona integration"""

    def __init__(self):
        self.conversations: Dict[str, Dict] = {}

    def detect_multi_command(self, content: str) -> tuple[Optional[str], Optional[str]]:
        if not content or not content.strip().lower().startswith("!multi"):
            return None, None

        content = content.strip()
        if content.lower() == "!multi":
            return "help", None
        elif content.lower() == "!multi test":
            return "test", None

        parts = content.split(None, 2)
        if len(parts) == 1:  # Just "!multi"
            return "help", None
        elif len(parts) == 2:
            return "general", parts[1]
        elif len(parts) >= 3:
            mode, topic = parts[1].lower(), " ".join(parts[2:])
            valid_modes = ["debate", "collab", "gen", "general"]
            if mode in valid_modes:
                return "general" if mode == "gen" else mode, topic
            else:
                return "general", " ".join(parts[1:])
        return None, None

    def create_conversation_id(self, user_id: str, topic: str) -> str:
        timestamp = str(int(datetime.now().timestamp()))
        content_hash = hashlib.md5(f"{user_id}_{topic}".encode()).hexdigest()[:8]
        return f"multi_{timestamp}_{content_hash}"

    def start_conversation(
        self,
        conv_id: str,
        topic: str,
        mode: str,
        models: List[str],
        user_id: str,
        persona_config: Dict,
        reasoning_config: Dict,
    ):
        active_models = [model for model in models if model != "-- Select Model --"]
        if len(active_models) < 2:
            raise ValueError("At least 2 models must be selected")

        self.conversations[conv_id] = {
            "topic": topic,
            "mode": mode,
            "models": active_models,
            "user_id": user_id,
            "current_turn": 0,
            "history": [],
            "started_at": datetime.now(),
            "status": "active",
            "persona_config": persona_config,
            "reasoning_config": reasoning_config,
        }

    def get_conversation(self, conv_id: str) -> Optional[Dict]:
        return self.conversations.get(conv_id)

    def add_turn(
        self, conv_id: str, model: str, response_data: Dict, persona_info: Dict
    ):
        if conv_id in self.conversations:
            self.conversations[conv_id]["history"].append(
                {
                    "model": model,
                    "response_data": response_data,
                    "timestamp": datetime.now(),
                    "turn": self.conversations[conv_id]["current_turn"],
                    "persona_info": persona_info,
                }
            )
            self.conversations[conv_id]["current_turn"] += 1


conversation_manager = ConversationManager()
thinking_processor = ThinkingProcessor()
AVAILABLE_MODELS = get_available_models()


class Filter:
    class Valves(BaseModel):
        priority: int = Field(default=10, description="Filter execution priority")

        model_1: str = Field(
            default=AVAILABLE_MODELS[0],
            description="First model for conversation",
            json_schema_extra={"enum": AVAILABLE_MODELS},
        )
        model_2: str = Field(
            default=(
                AVAILABLE_MODELS[1]
                if len(AVAILABLE_MODELS) > 1
                else AVAILABLE_MODELS[0]
            ),
            description="Second model for conversation",
            json_schema_extra={"enum": AVAILABLE_MODELS},
        )
        model_3: str = Field(
            default=(
                AVAILABLE_MODELS[2]
                if len(AVAILABLE_MODELS) > 2
                else AVAILABLE_MODELS[0]
            ),
            description="Third model for conversation (optional)",
            json_schema_extra={"enum": AVAILABLE_MODELS},
        )
        model_4: str = Field(
            default=(
                AVAILABLE_MODELS[3]
                if len(AVAILABLE_MODELS) > 3
                else AVAILABLE_MODELS[0]
            ),
            description="Fourth model for conversation (optional)",
            json_schema_extra={"enum": AVAILABLE_MODELS},
        )

        max_turns_per_model: int = Field(
            default=2, ge=1, le=5, description="Turns per model"
        )
        response_temperature: float = Field(default=0.7, ge=0.0, le=1.0)
        max_tokens: int = Field(default=500, ge=50, le=2000)
        show_persona_names: bool = Field(
            default=True, description="Show persona names in output"
        )
        conversation_pace: float = Field(
            default=1.0, ge=0.5, le=3.0, description="Seconds between responses"
        )
        response_timeout: int = Field(default=60, ge=30, le=120)
        enable_debug: bool = Field(default=True, description="Enable debug logging")

        # Enhanced reasoning settings
        enable_reasoning: bool = Field(
            default=True, description="Enable reasoning/thinking capabilities"
        )
        reasoning_effort: str = Field(
            default="medium",
            description="Reasoning effort level for capable models",
            json_schema_extra={"enum": ["low", "medium", "high"]},
        )
        show_thinking_process: bool = Field(
            default=True, description="Show thinking process in responses"
        )

        # Enhanced integration settings
        enable_agent_hotswap_integration: bool = Field(
            default=True, description="Enable Agent Hotswap integration"
        )
        integration_debug: bool = Field(
            default=False, description="Debug persona integration"
        )

    def __init__(self):
        self.valves = self.Valves()
        self.toggle = True
        self.stats = {
            "conversations_started": 0,
            "total_turns": 0,
            "successful_responses": 0,
            "errors": 0,
        }
        self.icon = """data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIGZpbGw9Im5vbmUiIHZpZXdCb3g9IjAgMCAyNCAyNCIgc3Ryb2tlLXdpZHRoPSIxLjUiIHN0cm9rZT0iY3VycmVudENvbG9yIj4KICA8cGF0aCBzdHJva2UtbGluZWNhcD0icm91bmQiIHN0cm9rZS1saW5lam9pbj0icm91bmQiIGQ9Im0zIDIxIDEuOS0xLjlhNS4wNzggNS4wNzggMCAwIDAgMS43LTUuNjdBNSA1IDAgMCAwIDEyIDNhNS4wNzggNS4wNzggMCAwIDAgNSA1IDUuMDc4IDUuMDc4IDAgMCAwIDUuNjcgMS43bDEuOS0xLjkiLz4KICA8cGF0aCBzdHJva2UtbGluZWNhcD0icm91bmQiIHN0cm9rZS1saW5lam9pbj0icm91bmQiIGQ9Im0xNCAzIDMuNjggMy42OGE1LjA3OCA1LjA3OCAwIDAgMSAwIDcuNjRMMTQgMTgiLz4KICA8cGF0aCBzdHJva2UtbGluZWNhcD0icm91bmQiIHN0cm9rZS1saW5lam9pbj0icm91bmQiIGQ9Im0yMSAxMC41LTQuNS00LjVhNS4wNzggNS4wNzggMCAwIDAtNy42NCAwTDEwIDgiLz4KPC9zdmc+"""

    def _safe_get_user_id(self, __user__: Optional[dict]) -> str:
        if not __user__:
            return "anonymous"
        return __user__.get("id", "anonymous")

    def _get_safe_user_obj(self, __user__: Optional[dict]):
        if not __user__:
            return None
        user_id = __user__.get("id", "unknown")
        try:
            return Users.get_user_by_id(user_id)
        except Exception as e:
            logger.warning(f"Could not get user object for {user_id}: {e}")
            return type(
                "User",
                (),
                {
                    "id": user_id,
                    "email": __user__.get("email", "unknown@example.com"),
                    "name": __user__.get("name", "Unknown User"),
                    "role": __user__.get("role", "user"),
                },
            )()

    def _debug_log(self, message: str):
        if self.valves.enable_debug:
            logger.info(f"[MultiModel] {message}")

    def _integration_debug(self, message: str):
        if self.valves.integration_debug:
            logger.info(f"[MultiModel:Integration] {message}")

    def _get_conversation_mode_info(self, mode: str) -> dict:
        mode_configs = {
            "debate": {
                "name": "Debate Mode",
                "description": "Models present opposing viewpoints and argue their positions",
                "system_template": "You are participating in a debate about: {topic}. Take a clear position, present strong arguments, and challenge opposing viewpoints respectfully.",
                "prompt_template": {
                    "first": "Begin the debate on {topic}. Present your opening position with clear arguments.",
                    "response": '{last_speaker} just argued: "{last_response}". Now present your counter-argument on {topic}.',
                },
            },
            "collab": {
                "name": "Collaboration Mode",
                "description": "Models work together to build comprehensive solutions",
                "system_template": "You are collaborating with other experts to address: {topic}. Build on each other's ideas and work toward a comprehensive solution.",
                "prompt_template": {
                    "first": "Start our collaboration on {topic}. Share your initial insights and approach.",
                    "response": '{last_speaker} contributed: "{last_response}". Build on this for {topic}. What can you add or improve?',
                },
            },
            "general": {
                "name": "General Discussion",
                "description": "Natural conversation with multiple perspectives",
                "system_template": "You are having a thoughtful discussion about: {topic}. Share your perspective and engage naturally with other participants.",
                "prompt_template": {
                    "first": "Let's discuss: {topic}. What are your thoughts on this?",
                    "response": '{last_speaker} said: "{last_response}". What\'s your perspective on this aspect of {topic}?',
                },
            },
        }
        return mode_configs.get(mode, mode_configs["general"])

    def _extract_persona_config(self, body: Dict) -> Dict:
        """Enhanced persona configuration extraction with Agent Hotswap integration"""
        if not self.valves.enable_agent_hotswap_integration:
            return {"type": "none"}

        # Use the enhanced integration class
        integration_info = AgentHotswapIntegration.detect_persona_context(body)

        if integration_info["has_agent_hotswap"]:
            self._integration_debug(
                f"Detected Agent Hotswap context: {integration_info['persona_type']}"
            )

            # Debug the full integration info
            if integration_info["persona_type"] == "per_model":
                per_model_personas = integration_info.get("per_model_personas", {})
                self._integration_debug(
                    f"Per-model personas detected: {list(per_model_personas.keys())}"
                )
                for model_num, persona_data in per_model_personas.items():
                    self._integration_debug(
                        f"  Model {model_num}: {persona_data.get('name', 'Unknown')} ({persona_data.get('key', 'no key')})"
                    )

        if integration_info["persona_type"] == "per_model":
            return {
                "type": "per_model",
                "personas": integration_info["per_model_personas"],
                "integration_info": integration_info,
            }
        elif integration_info["persona_type"] == "single":
            return {
                "type": "single",
                "persona": integration_info["active_persona"],
                "name": integration_info["active_persona_name"],
                "prompt": integration_info["active_persona_prompt"],
                "integration_info": integration_info,
            }
        elif integration_info["persona_type"] == "multi":
            return {
                "type": "multi",
                "sequence": integration_info["persona_sequence"],
                "integration_info": integration_info,
            }

        return {"type": "none", "integration_info": integration_info}

    def _extract_reasoning_config(self, body: Dict) -> Dict:
        """Extract reasoning configuration from request body"""
        reasoning_config = {
            "enabled": self.valves.enable_reasoning,
            "effort": self.valves.reasoning_effort,
            "show_thinking": self.valves.show_thinking_process,
        }

        # Override with request-specific reasoning effort if provided
        if "reasoning_effort" in body:
            reasoning_config["effort"] = body["reasoning_effort"]

        return reasoning_config

    def _create_system_message(
        self, topic: str, mode: str, persona_info: Dict, model_name: str
    ) -> str:
        """Enhanced system message creation with persona integration"""
        mode_info = self._get_conversation_mode_info(mode)
        base_instructions = mode_info["system_template"].format(topic=topic)

        # Use persona prompt if available, otherwise default
        if persona_info.get("prompt") and persona_info.get("prompt").strip():
            persona_context = persona_info["prompt"]
            persona_name = persona_info.get("name", model_name)

            # Create comprehensive system message with persona identity
            full_system = f"""{persona_context}

CONVERSATION CONTEXT: {base_instructions}

IMPORTANT: You are {persona_name}. Maintain this persona throughout the conversation. Draw upon your specialized knowledge and perspective as defined in your persona description above."""

            self._integration_debug(
                f"Created system message for {persona_name}: {len(full_system)} chars"
            )
            return full_system
        else:
            self._integration_debug(
                f"No persona found for {model_name}, using default system message"
            )
            return f"You are {model_name}. {base_instructions}"

    def _get_persona_for_model(
        self, model_index: int, model_name: str, persona_config: Dict
    ) -> Dict:
        """Enhanced persona retrieval for specific model with integration support"""
        default_info = {
            "prompt": "",
            "name": model_name,
            "key": "",
            "description": "",
            "capabilities": [],
        }

        self._integration_debug(
            f"Getting persona for model {model_index + 1} ({model_name}), config type: {persona_config.get('type')}"
        )

        if persona_config["type"] == "single":
            result = {
                "prompt": persona_config.get("prompt", ""),
                "name": persona_config.get("name", model_name),
                "key": persona_config.get("persona", ""),
                "description": persona_config.get("description", ""),
                "capabilities": persona_config.get("capabilities", []),
            }
            self._integration_debug(f"Single persona mode: {result['name']}")
            return result

        elif persona_config["type"] == "per_model":
            # Model numbers start from 1, list indices from 0
            model_persona_key = model_index + 1  # Convert to 1-based indexing
            self._integration_debug(
                f"Looking for persona{model_persona_key} in per-model config"
            )

            model_persona = persona_config["personas"].get(model_persona_key)
            if model_persona:
                result = {
                    "prompt": model_persona.get("prompt", ""),
                    "name": model_persona.get("name", model_name),
                    "key": model_persona.get("key", ""),
                    "description": model_persona.get("description", ""),
                    "capabilities": model_persona.get("capabilities", []),
                }
                self._integration_debug(
                    f"Found per-model persona for model {model_persona_key}: {result['name']}"
                )
                return result
            else:
                self._integration_debug(
                    f"No persona assignment found for model {model_persona_key}"
                )

        self._integration_debug(f"Using default info for {model_name}")
        return default_info

    def _get_active_models(self) -> List[str]:
        models = [getattr(self.valves, f"model_{i}", "") for i in range(1, 5)]
        return [m for m in models if m and m != "-- Select Model --"]

    async def _emit_status(
        self,
        emitter,
        description: str,
        progress: Optional[float] = None,
        done: bool = False,
    ):
        if emitter:
            await emitter(
                {
                    "type": "status",
                    "data": {
                        "description": description,
                        "done": done,
                        "progress": progress,
                    },
                }
            )

    async def _emit_message(self, emitter, content: str):
        if emitter:
            await emitter({"type": "message", "data": {"content": content}})

    def _prepare_model_body(
        self, model: str, messages: List[Dict], reasoning_config: Dict
    ) -> Dict:
        """Prepare request body with reasoning capabilities if supported"""
        body = {
            "model": model,
            "messages": messages,
            "temperature": self.valves.response_temperature,
            "max_tokens": self.valves.max_tokens,
            "stream": True,
        }

        # Add reasoning configuration for capable models
        if reasoning_config[
            "enabled"
        ] and ReasoningCapabilities.supports_native_reasoning(model):
            body["reasoning"] = {
                "max_tokens": ReasoningCapabilities.get_reasoning_effort_tokens(
                    reasoning_config["effort"]
                ),
                "exclude": False,
            }
            self._debug_log(
                f"Added reasoning tokens for {model}: {body['reasoning']['max_tokens']}"
            )

        return body

    async def _generate_model_response(
        self,
        model: str,
        messages: List[Dict],
        request_obj,
        user_obj,
        reasoning_config: Dict,
        persona_info: Dict = None,
    ) -> Dict:
        """Generate response from local model with reasoning and persona support"""
        try:
            persona_name = persona_info.get("name", model) if persona_info else model
            self._debug_log(f"Generating response from {model} as {persona_name}")

            # Prepare body with reasoning if supported
            body = self._prepare_model_body(model, messages, reasoning_config)

            response = await asyncio.wait_for(
                generate_chat_completion(request_obj, body, user_obj),
                timeout=self.valves.response_timeout,
            )

            # Handle response
            content = ""
            if hasattr(response, "body_iterator"):
                async for chunk in response.body_iterator:
                    if chunk:
                        try:
                            chunk_str = (
                                chunk.decode("utf-8")
                                if isinstance(chunk, bytes)
                                else str(chunk)
                            )
                            if chunk_str.startswith("data: "):
                                chunk_str = chunk_str[6:].strip()
                            if chunk_str == "[DONE]":
                                break
                            chunk_data = json.loads(chunk_str)
                            if (
                                chunk_data.get("choices")
                                and len(chunk_data["choices"]) > 0
                            ):
                                delta = chunk_data["choices"][0].get("delta", {})
                                if delta.get("content"):
                                    content += delta["content"]
                        except (
                            json.JSONDecodeError,
                            UnicodeDecodeError,
                            AttributeError,
                        ):
                            continue
            elif isinstance(response, dict):
                if response.get("choices") and len(response["choices"]) > 0:
                    message = response["choices"][0].get("message", {})
                    content = message.get("content", "")

            content = content.strip()
            if content and len(content) > 2:
                self.stats["successful_responses"] += 1
                self._debug_log(
                    f"Successfully got {len(content)} chars from {persona_name}"
                )

                # Process thinking/reasoning with persona context
                if reasoning_config["enabled"] and reasoning_config["show_thinking"]:
                    processed = thinking_processor.process_response_with_thinking(
                        content, persona_name
                    )
                else:
                    processed = {
                        "has_thinking": False,
                        "thinking": "",
                        "main_content": content,
                        "raw_content": content,
                    }

                return {"success": True, "content": processed, "error": None}
            else:
                self.stats["errors"] += 1
                return {
                    "success": False,
                    "content": None,
                    "error": f"Empty Response: {persona_name}",
                }

        except asyncio.TimeoutError:
            self.stats["errors"] += 1
            return {
                "success": False,
                "content": None,
                "error": f"Timeout: {persona_name} exceeded {self.valves.response_timeout}s",
            }
        except Exception as e:
            self.stats["errors"] += 1
            error_msg = str(e)
            self._debug_log(f"Error from {persona_name}: {error_msg}")
            return {
                "success": False,
                "content": None,
                "error": f"Error: {persona_name} - {error_msg[:50]}...",
            }

    def _create_enhanced_model_header(
        self,
        model: str,
        persona_info: Dict,
        turn_number: int,
        has_reasoning: bool = False,
    ) -> str:
        """Create enhanced visual header with persona integration"""
        icon = ModelIconManager.get_model_icon(model, persona_info)
        display_name = ModelIconManager.get_model_display_name(model, persona_info)
        reasoning_indicator = " 🧠" if has_reasoning else ""

        if self.valves.show_persona_names and persona_info.get("name") != model:
            # Show persona name prominently
            persona_name = persona_info.get("name", display_name)
            header = f"""
> {icon} **{persona_name}**{reasoning_indicator}  
> *Running on {display_name} • Turn {turn_number}*

"""
        else:
            # Show just model name
            header = f"""
> {icon} **{display_name}**{reasoning_indicator}  
> *Model • Turn {turn_number}*

"""

        return header

    async def _format_conversation_output(
        self, conv_id: str, conversation: Dict
    ) -> str:
        """Enhanced conversation output formatting with persona integration"""
        history = conversation["history"]
        mode_info = self._get_conversation_mode_info(
            conversation.get("mode", "general")
        )
        persona_config = conversation.get("persona_config", {})

        # Create header with enhanced information
        output = [
            f"## 🎭 Multi-Model {mode_info['name']}: {conversation['topic']}",
            "",
            f"> **Mode**: {mode_info['description']}  ",
        ]

        # Enhanced model list with persona information
        model_info = []
        for i, model in enumerate(conversation["models"]):
            persona_info = self._get_persona_for_model(i, model, persona_config)
            icon = ModelIconManager.get_model_icon(model, persona_info)
            display_name = ModelIconManager.get_model_display_name(model, persona_info)
            if persona_info.get("name") and persona_info["name"] != model:
                model_info.append(f"{icon} {persona_info['name']} ({display_name})")
            else:
                model_info.append(f"{icon} {display_name}")

        output.extend(
            [
                f"> **Models**: {', '.join(model_info)}  ",
                f"> **Exchanges**: {len(history)} responses",
                "",
                "---",
                "",
            ]
        )

        # Add each model response with enhanced formatting
        for turn in history:
            persona_info = turn.get("persona_info", {})
            turn_number = turn["turn"] + 1
            response_data = turn["response_data"]

            if response_data["success"]:
                content_data = response_data["content"]
                has_thinking = content_data["has_thinking"]

                # Create enhanced header with persona integration
                header = self._create_enhanced_model_header(
                    turn["model"], persona_info, turn_number, has_thinking
                )

                # Add thinking process if available
                if has_thinking and content_data["thinking"]:
                    output.append(header)
                    output.append(content_data["thinking"])
                    output.append(content_data["main_content"])
                else:
                    output.append(header)
                    output.append(content_data["main_content"])
            else:
                # Handle error responses with persona context
                icon = ModelIconManager.get_model_icon(turn["model"], persona_info)
                display_name = ModelIconManager.get_model_display_name(
                    turn["model"], persona_info
                )
                output.append(
                    f"""
> ❌ **{display_name}**  
> *{response_data["error"]}*
"""
                )

            output.append("")
            output.append("---")
            output.append("")

        # Enhanced footer with persona information
        persona_summary = ""
        if persona_config.get("type") == "per_model":
            persona_summary = " with persona assignments"
        elif persona_config.get("type") == "single":
            persona_summary = f" in {persona_config.get('name', 'persona')} mode"

        output.append(
            f"> ✅ **Conversation completed with {len(conversation['models'])} models{persona_summary}**"
        )

        return "\n".join(output)

    async def _orchestrate_conversation(
        self, conv_id: str, emitter, request_obj, user_obj
    ) -> str:
        """Enhanced conversation orchestration with persona integration"""
        conversation = conversation_manager.get_conversation(conv_id)
        if not conversation:
            return "Error: Conversation not found"

        topic, mode, models = (
            conversation["topic"],
            conversation["mode"],
            conversation["models"],
        )
        max_turns, persona_config, reasoning_config = (
            self.valves.max_turns_per_model,
            conversation["persona_config"],
            conversation["reasoning_config"],
        )

        # Enhanced status with persona information
        persona_info_text = ""
        if persona_config.get("type") == "per_model":
            persona_info_text = " with persona roles"
        elif persona_config.get("type") == "single":
            persona_info_text = f" in {persona_config.get('name', 'persona')} mode"

        await self._emit_status(
            emitter,
            f"🎭 Starting {mode} conversation with {len(models)} models{persona_info_text}...",
            5,
        )

        successful_responses = 0

        for round_num in range(max_turns):
            for model_idx, model in enumerate(models):
                # Get enhanced persona info for this specific model
                persona_info = self._get_persona_for_model(
                    model_idx, model, persona_config
                )

                # Debug persona assignment
                persona_name = persona_info.get("name", model)
                if persona_info.get("prompt"):
                    self._integration_debug(
                        f"Model {model_idx + 1} ({model}) assigned persona: {persona_name}"
                    )
                else:
                    self._integration_debug(
                        f"Model {model_idx + 1} ({model}) has no persona assignment"
                    )

                # Check if model supports reasoning
                supports_reasoning = ReasoningCapabilities.supports_native_reasoning(
                    model
                )
                thinking_indicator = " 🧠" if supports_reasoning else ""

                display_name = ModelIconManager.get_model_display_name(
                    model, persona_info
                )

                await self._emit_status(
                    emitter,
                    f"Round {round_num + 1}: {display_name}{thinking_indicator} thinking...",
                    (round_num * len(models) + model_idx + 1)
                    / (max_turns * len(models))
                    * 90,
                )

                # Create enhanced system message with persona integration
                system_message = self._create_system_message(
                    topic, mode, persona_info, model
                )
                messages = [{"role": "system", "content": system_message}]

                # Debug system message creation
                if persona_info.get("prompt"):
                    self._integration_debug(
                        f"System message for {persona_name}: {len(system_message)} characters"
                    )
                else:
                    self._integration_debug(
                        f"Default system message for {model}: {len(system_message)} characters"
                    )

                # Add conversation history (only successful responses)
                for past_turn in conversation["history"]:
                    if past_turn["response_data"]["success"]:
                        past_persona_info = past_turn.get("persona_info", {})
                        speaker_name = past_persona_info.get("name", past_turn["model"])
                        content_data = past_turn["response_data"]["content"]
                        # Use main content for context (without thinking blocks)
                        messages.append(
                            {
                                "role": "user",
                                "content": f"[{speaker_name}]: {content_data['main_content']}",
                            }
                        )

                # Create enhanced prompt for this turn
                mode_info = self._get_conversation_mode_info(mode)
                successful_history = [
                    h for h in conversation["history"] if h["response_data"]["success"]
                ]

                if not successful_history:
                    prompt = mode_info["prompt_template"]["first"].format(topic=topic)
                else:
                    last_turn = successful_history[-1]
                    last_persona_info = last_turn.get("persona_info", {})
                    last_speaker = last_persona_info.get("name", last_turn["model"])
                    last_content = last_turn["response_data"]["content"]["main_content"]
                    last_response_truncated = (
                        last_content[:150] + "..."
                        if len(last_content) > 150
                        else last_content
                    )
                    prompt = mode_info["prompt_template"]["response"].format(
                        last_speaker=last_speaker,
                        last_response=last_response_truncated,
                        topic=topic,
                    )

                messages.append({"role": "user", "content": prompt})

                # Generate response with persona integration
                response_data = await self._generate_model_response(
                    model,
                    messages,
                    request_obj,
                    user_obj,
                    reasoning_config,
                    persona_info,
                )

                # Count successful responses
                if response_data["success"]:
                    successful_responses += 1

                # Add to conversation with persona info
                conversation_manager.add_turn(
                    conv_id, model, response_data, persona_info
                )
                self.stats["total_turns"] += 1

                # Stream response with enhanced Markdown formatting
                if response_data["success"]:
                    content_data = response_data["content"]
                    has_thinking = content_data["has_thinking"]

                    header = self._create_enhanced_model_header(
                        model, persona_info, len(conversation["history"]), has_thinking
                    )

                    # Stream thinking process first if available
                    if (
                        has_thinking
                        and content_data["thinking"]
                        and reasoning_config["show_thinking"]
                    ):
                        # Stream header first
                        await self._emit_message(emitter, header)

                        # Then stream thinking content
                        await self._emit_message(emitter, content_data["thinking"])

                        # Then stream main content
                        main_content = f"""{content_data["main_content"]}

---
"""
                        await self._emit_message(emitter, main_content)
                    else:
                        # Stream normal response
                        styled_response = f"""{header}{content_data["main_content"]}

---
"""
                        await self._emit_message(emitter, styled_response)
                else:
                    # Show error with enhanced formatting
                    icon = ModelIconManager.get_model_icon(model, persona_info)
                    display_name = ModelIconManager.get_model_display_name(
                        model, persona_info
                    )
                    error_message = f"""
> ❌ **{display_name}**  
> *{response_data["error"]}*

"""
                    await self._emit_message(emitter, error_message)

                await asyncio.sleep(self.valves.conversation_pace)

        # Enhanced completion status
        if successful_responses == 0:
            await self._emit_status(emitter, "❌ All models failed", 100, done=True)
            return """## ❌ Multi-Model Conversation Failed

All selected models failed to generate responses.

### 🔧 Troubleshooting:
- Check if the models are pulled and available: `ollama list`
- Pull missing models: `ollama pull model-name`
- Try `!multi test` to diagnose issues
- Ensure the model server (e.g., Ollama) is running and accessible

### 💡 Common Working Models:
- **llama3.2:latest** - Good general purpose model
- **qwen2.5:latest** - Excellent reasoning capabilities
- **mistral:latest** - Fast and efficient
- **codellama:latest** - Great for technical discussions
- **deepseek-r1:latest** - Advanced reasoning model

### 🎭 Persona Integration:
- Try `!persona1 teacher !persona2 student !multi collab` for role-based conversations
- Use `!agent list` to see available personas"""

        await self._emit_status(
            emitter, "✅ Multi-model conversation complete!", 100, done=True
        )
        return await self._format_conversation_output(conv_id, conversation)

    async def inlet(
        self,
        body: dict,
        __event_emitter__=None,
        __user__: Optional[dict] = None,
        __request__=None,
    ) -> dict:
        if not self.toggle:
            return body

        messages = body.get("messages", [])
        if not messages:
            return body

        # Find last user message
        last_message = ""
        last_message_index = -1
        for i in range(len(messages) - 1, -1, -1):
            if messages[i].get("role") == "user":
                last_message = messages[i].get("content", "")
                last_message_index = i
                break

        if not last_message:
            return body

        mode, topic = conversation_manager.detect_multi_command(last_message)
        if mode is None:
            return body

        if mode == "help":
            if last_message_index >= 0:
                messages[last_message_index][
                    "content"
                ] = """# 🎭 Multi-Model Conversation Help - Enhanced Integration

## 📝 Quick Command Reference

### **Basic Commands**
- `!multi` - Show this help
- `!multi test` - Test your configuration
- `!multi <topic>` - Start general discussion
- `!multi collab <topic>` - Start collaboration mode
- `!multi debate <topic>` - Start debate mode

### **🎭 Persona Integration Commands**
Single persona for all models
!teacher !multi collab Explain quantum physics
Per-model persona assignments
!persona1 teacher !persona2 student !multi collab Explain quantum physics
!persona1 ethicist !persona2 technologist !persona3 economist !persona4 policymaker !multi debate AI regulation
Quick examples with personas
!scientist !multi debate Climate change solutions
!persona1 coder !persona2 analyst !multi collab Database optimization
code
Code
---

## 🎯 **Core Commands**
- `!multi <topic>` - **General Discussion** (default mode)
- `!multi collab <topic>` - **Collaboration Mode** 
- `!multi debate <topic>` - **Debate Mode**
- `!multi test` - **Configuration Test**

---

## 🧠 **Reasoning & Thinking Features**

### **Automatic Reasoning Detection**
Models with reasoning capabilities are automatically detected:
- 🧠 **DeepSeek-R1**: Advanced chain-of-thought reasoning
- 🤔 **Qwen-QwQ**: Question-answering with reasoning
- 🎭 **Claude**: Anthropic's reasoning models
- 💎 **Gemini**: Google's reasoning capabilities

### **Thinking Process Display**
- **Reasoning models** show their thinking process in collapsible sections
- **Persona-aware thinking**: Thinking processes show which persona is reasoning
- **Non-reasoning models** have their `<thinking>`, `<reason>`, `<thought>` tags processed
- Thinking content is separated from open_webui.main responses for clarity

---

## 🎭 **Enhanced Persona Integration**

### **Agent Hotswap Compatibility**
This filter seamlessly integrates with Agent Hotswap for advanced persona management:

**Per-Model Assignments:**
- `!persona1 teacher !persona2 student !multi collab` - Assign specific personas to each model
- Models 1-4 can have different personas for rich multi-perspective conversations
- Personas maintain their knowledge, tone, and capabilities throughout the conversation

**Single Persona Mode:**
- `!scientist !multi debate` - All models adopt the scientist persona
- Maintains consistency while still providing diverse model perspectives

**Visual Integration:**
- Model headers show persona names and icons
- Persona-specific thinking processes
- Enhanced conversation summaries with persona context

### **Available Personas** (use `!agent list` to see all)
- 🎓 **Teacher** - Educational expert and clear communicator
- 🎒 **Student** - Curious learner asking thoughtful questions  
- 🔬 **Scientist** - Evidence-based researcher and analyst
- 🧭 **Ethicist** - Moral philosopher examining ethical implications
- 💻 **Technologist** - Innovation-focused technical expert
- 📈 **Economist** - Market dynamics and economic analysis
- 🏛️ **Policymaker** - Governance and policy implementation
- ✍️ **Writer** - Creative content and communication specialist
- 📊 **Analyst** - Data analysis and insight generation

---

## 🤝 **Collaboration Mode** (`!multi collab`)

> **Enhanced with Personas**: Models work together as specialized experts

**How it works:**
- Models build upon each other's ideas with their persona expertise
- Each persona contributes their unique knowledge and perspective
- Collaborative, constructive tone with role-specific insights

**Examples:**
- `!persona1 teacher !persona2 scientist !multi collab Explain photosynthesis`
- `!economist !multi collab Market entry strategy for renewable energy`
- `!persona1 coder !persona2 analyst !persona3 writer !multi collab Create user documentation`

---

## ⚔️ **Debate Mode** (`!multi debate`)

> **Enhanced with Personas**: Specialized experts argue different positions

**How it works:**
- Each persona takes positions based on their expertise and worldview
- Present strong arguments with domain-specific evidence
- Challenge each other's reasoning from their professional perspectives

**Examples:**
- `!persona1 ethicist !persona2 technologist !multi debate AI development regulation`
- `!persona1 economist !persona2 scientist !multi debate Carbon tax effectiveness`
- `!teacher !multi debate Traditional vs digital learning methods`

---

## 💬 **General Mode** (`!multi` or `!multi gen`)

> **Enhanced with Personas**: Natural expert consultation

**Examples:**
- `!persona1 writer !persona2 teacher !multi What makes effective communication?`
- `!scientist !multi The future of space exploration`
- `!persona1 analyst !persona2 economist !multi gen Technology impact on employment`

---

## ⚙️ **Configuration & Setup**

### **Requirements:**
- At least **2 models** configured in filter settings
- **Agent Hotswap filter** for persona functionality
- No API keys needed

### **Integration Settings:**
- **Enable Agent Hotswap Integration**: `true` (recommended)
- **Integration Debug**: Enable for troubleshooting persona assignments
- **Show Persona Names**: Display persona names in conversation headers

### **Persona Setup:**
1. Ensure Agent Hotswap filter is installed and running (Priority 0)
2. Use `!agent list` to see available personas
3. Configure per-model assignments with `!persona1 name !persona2 name` syntax
4. Models automatically inherit persona knowledge and capabilities

---

## 🔧 **Advanced Settings**

Available in filter valves:
- **Max turns per model**: 1-5 (default: 2)
- **Temperature**: 0.0-1.0 (default: 0.7)
- **Max tokens**: 50-2000 (default: 500)
- **Conversation pace**: 0.5-3.0 seconds between responses
- **Enable reasoning**: Toggle reasoning capabilities
- **Reasoning effort**: Low/Medium/High token allocation
- **Show thinking process**: Display reasoning steps with persona context

---

## 🧪 **Testing & Troubleshooting**

### **Quick Test:**
!multi test
code
Code
Shows your configuration, available models, reasoning capabilities, and persona integration status.

### **Persona Integration Test:**
!persona1 teacher !persona2 student !multi test
code
Code
Tests persona assignment and integration.

### **Common Issues:**
1. **"Need at least 2 models"** → Configure more models in filter settings
2. **Personas not working** → Ensure Agent Hotswap filter is installed and running
3. **Models returning errors** → Check `ollama list` and `ollama pull model-name`
4. **No persona assignments** → Use exact syntax: `!persona1 name !persona2 name`

---

## 🚀 **Getting Started**

1. **Install dependencies**: Ensure Agent Hotswap filter is installed
2. **Test setup**: `!multi test`
3. **Try basic conversation**: `!multi What is artificial intelligence?`
4. **Add personas**: `!persona1 teacher !persona2 student !multi collab Explain machine learning`
5. **Explore modes**: `!persona1 ethicist !persona2 technologist !multi debate AI ethics`

> 💡 **Pro Tip**: Start with `!agent list` to explore available personas, then use `!persona1 name !persona2 name !multi test` to verify your setup before starting complex conversations!"""
            return body

        if mode == "test":
            active_models = self._get_active_models()

            # Enhanced persona integration status
            persona_integration_status = "❌ Not Available"
            persona_details = ""

            if self.valves.enable_agent_hotswap_integration:
                # Test for Agent Hotswap integration
                test_context = AgentHotswapIntegration.detect_persona_context(body)
                if test_context.get("has_agent_hotswap"):
                    persona_integration_status = (
                        f"✅ Active (v{test_context.get('hotswap_version', 'unknown')})"
                    )
                else:
                    persona_integration_status = (
                        "⚠️ Enabled but Agent Hotswap not detected"
                    )
                    persona_details = (
                        "\n> Install Agent Hotswap filter for persona functionality"
                    )

            # Create enhanced model status with persona awareness
            model_status = []
            reasoning_count = 0
            for i, model in enumerate(active_models):
                icon = ModelIconManager.get_model_icon(model)
                display_name = ModelIconManager.get_model_display_name(model)
                supports_reasoning = ReasoningCapabilities.supports_native_reasoning(
                    model
                )
                reasoning_indicator = " 🧠" if supports_reasoning else ""
                if supports_reasoning:
                    reasoning_count += 1
                model_status.append(
                    f"- {icon} **{display_name}** (`{model}`){reasoning_indicator} *→ Model {i+1}*"
                )

            test_result = f"""# 🧪 Multi-Model Configuration Test - Enhanced Integration

## 🤖 Available Models ({len(active_models)})

{chr(10).join(model_status) if model_status else "> ❌ **No models configured**"}

> 🧠 **{reasoning_count} model(s) support advanced reasoning**

---

## 🎭 Integration Status

- **Agent Hotswap Integration**: {persona_integration_status}{persona_details}
- **Persona Support**: {"✅ Per-model assignments available" if self.valves.enable_agent_hotswap_integration else "❌ Disabled"}
- **Integration Debug**: {"✅ Enabled" if self.valves.integration_debug else "❌ Disabled"}

---

## ⚙️ Current Settings

- **Max turns per model**: {self.valves.max_turns_per_model}
- **Temperature**: {self.valves.response_temperature}
- **Max tokens**: {self.valves.max_tokens}
- **Reasoning enabled**: {"✅ Yes" if self.valves.enable_reasoning else "❌ No"}
- **Reasoning effort**: {self.valves.reasoning_effort}
- **Show thinking process**: {"✅ Yes" if self.valves.show_thinking_process else "❌ No"}
- **Show persona names**: {"✅ Yes" if self.valves.show_persona_names else "❌ No"}

---

## 🎯 Status

> {"✅ **Ready for enhanced conversations!**" if len(active_models) >= 2 else "❌ **Need at least 2 models selected**"}

---

## 🚀 Quick Start Examples

### Basic Multi-Model:
!multi What is the future of AI?
code
Code
### With Single Persona:
!teacher !multi collab Explain quantum physics
code
Code
### With Per-Model Personas:
!persona1 teacher !persona2 student !multi collab Explain machine learning
!persona1 ethicist !persona2 technologist !multi debate AI regulation
code
Code
**🎭 Use `!agent list` to see all available personas!**
"""

            if last_message_index >= 0:
                messages[last_message_index]["content"] = test_result
            return body

        # Get active models
        active_models = self._get_active_models()
        if len(active_models) < 2:
            if last_message_index >= 0:
                messages[last_message_index][
                    "content"
                ] = f"""## ❌ Configuration Error

Need at least 2 models selected. Currently have {len(active_models)} model(s).

Please configure models in the filter settings.

### 🎭 Enhanced Features Available:
- **Persona Integration**: Use `!persona1 teacher !persona2 student` for role-based conversations
- **Reasoning Models**: DeepSeek-R1, Qwen-QwQ show thinking processes  
- **Agent Hotswap**: Full persona management with `!agent list`"""
            return body

        # Enhanced persona and reasoning configuration extraction
        persona_config = self._extract_persona_config(body)
        reasoning_config = self._extract_reasoning_config(body)

        user_id = self._safe_get_user_id(__user__)
        conv_id = conversation_manager.create_conversation_id(user_id, topic)

        try:
            conversation_manager.start_conversation(
                conv_id,
                topic,
                mode,
                active_models,
                user_id,
                persona_config,
                reasoning_config,
            )
            self.stats["conversations_started"] += 1

            user_obj = self._get_safe_user_obj(__user__)
            conversation_output = await self._orchestrate_conversation(
                conv_id, __event_emitter__, __request__, user_obj
            )

            if last_message_index >= 0:
                messages[last_message_index]["content"] = conversation_output

        except Exception as e:
            logger.error(f"Multi-model conversation failed: {e}")
            if last_message_index >= 0:
                messages[last_message_index][
                    "content"
                ] = f"""## ❌ Multi-Model Error

**Topic:** {topic}  
**Error:** {str(e)}

Please check your configuration and try again.

### 🎭 Troubleshooting:
- Ensure Agent Hotswap filter is installed for persona features
- Verify model configuration with `!multi test`
- Check model availability: `ollama list`"""

        return body

    async def stream(self, event: dict) -> dict:
        """Enhanced stream processing with persona-aware thinking tag processing"""
        if not self.valves.enable_reasoning:
            return event

        if event.get("choices") and event["choices"][0].get("delta"):
            delta = event["choices"][0]["delta"]
            if delta.get("content"):
                content = delta["content"]

                # Replace thinking tags with standardized thinking_block tags
                for open_tag, close_tag in thinking_processor.thinking_tags:
                    content = content.replace(f"<{open_tag}>", "<thinking_block>")
                    content = content.replace(f"<{close_tag}>", "</thinking_block>")

                # Handle HTML bold tags in streamed content
                content = content.replace("<strong>", "**").replace("</strong>", "**")

                delta["content"] = content

        return event

    async def outlet(
        self, body: dict, __event_emitter__=None, __user__: Optional[dict] = None
    ) -> dict:
        return body
