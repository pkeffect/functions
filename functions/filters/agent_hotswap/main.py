"""
title: Agent Hotswap
author: pkeffect & Claude AI
author_url: https://github.com/pkeffect
project_urls: https://github.com/pkeffect/functions/tree/main/functions/filters/agent_hotswap | https://github.com/open-webui/functions/tree/main/functions/filters/agent_hotswap | https://openwebui.com/f/pkeffect/agent_hotswap
funding_url: https://github.com/open-webui
version: 2.8.0
description: Universal AI persona switching with enhanced multi-plugin integration and robust per-model persona support.
requirements: pydantic>=2.0.0
"""


from pydantic import BaseModel, Field
from typing import Optional, Dict, List, Any
import re
import json
import time
import os
import asyncio
import aiofiles
import aiohttp
import urllib.parse
import shutil
import glob
from datetime import datetime
from pathlib import Path

# OpenWebUI Imports
try:
    from open_webui.models.users import Users
    from open_webui.models.models import Models

    NATIVE_DB_AVAILABLE = True
except ImportError:
    NATIVE_DB_AVAILABLE = False

# Configuration
CONFIG_FILE = "personas.json"
UI_FILE = "index.html"
DEFAULT_REPO = "https://raw.githubusercontent.com/pkeffect/functions/refs/heads/main/functions/filters/agent_hotswap/personas.json"
UI_REPO = "https://raw.githubusercontent.com/pkeffect/functions/refs/heads/main/functions/filters/agent_hotswap/ui/index.html"
TRUSTED_DOMAINS = ["github.com", "raw.githubusercontent.com"]

# Global cache for better performance
_GLOBAL_PERSONA_CACHE = {}
_CACHE_LOCK = asyncio.Lock()


class PluginIntegrationManager:
    """Manages integration with other plugins in the suite"""

    @staticmethod
    def create_integration_context(
        persona_data: Dict, command_type: str, **kwargs
    ) -> Dict:
        context = {
            "agent_hotswap_version": "2.8.0",
            "timestamp": time.time(),
            "command_type": command_type,
            "persona_data": persona_data,
        }
        context.update(kwargs)
        return context

    @staticmethod
    def prepare_per_model_context(
        assignments: Dict[int, Dict], personas_data: Dict
    ) -> Dict:
        context = {}
        for model_num, assignment in assignments.items():
            persona_key = assignment["key"]
            persona_info = personas_data.get(persona_key, {})
            context[f"persona{model_num}"] = {
                "key": persona_key,
                "name": assignment["name"],
                "prompt": persona_info.get("prompt", f"You are a {persona_key}."),
                "description": persona_info.get("description", ""),
                "capabilities": persona_info.get("capabilities", []),
                "model_num": model_num,
            }
        context["per_model_active"] = True
        context["total_assigned_models"] = len(assignments)
        context["assigned_model_numbers"] = list(assignments.keys())
        return context

    @staticmethod
    def prepare_single_persona_context(persona_key: str, personas_data: Dict) -> Dict:
        persona_info = personas_data.get(persona_key, {})
        return {
            "active_persona": persona_key,
            "active_persona_name": persona_info.get("name", persona_key.title()),
            "active_persona_prompt": persona_info.get(
                "prompt", f"You are a {persona_key}."
            ),
            "active_persona_description": persona_info.get("description", ""),
            "single_persona_active": True,
        }

    @staticmethod
    def prepare_multi_persona_context(
        sequence: List[Dict], personas_data: Dict
    ) -> Dict:
        sequence_data = []
        for step in sequence:
            persona_key = step["persona"]
            persona_info = personas_data.get(persona_key, {})
            sequence_data.append(
                {
                    "persona_key": persona_key,
                    "persona_name": persona_info.get("name", persona_key.title()),
                    "persona_prompt": persona_info.get(
                        "prompt", f"You are a {persona_key}."
                    ),
                    "task": step["task"],
                    "step_order": len(sequence_data) + 1,
                }
            )
        return {
            "multi_persona_active": True,
            "persona_sequence": sequence_data,
            "total_personas": len(sequence_data),
            "sequence_length": len(sequence),
        }


class PersonaStorage:
    @staticmethod
    async def store_persona(
        user_id: str,
        chat_id: str,
        persona: Optional[str],
        context: Optional[Dict] = None,
    ):
        if not NATIVE_DB_AVAILABLE or not user_id or not chat_id:
            return
        try:
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None,
                PersonaStorage._sync_store_persona,
                user_id,
                chat_id,
                persona,
                context,
            )
        except Exception as e:
            print(f"[PersonaStorage] Async store error: {e}")

    @staticmethod
    def _sync_store_persona(
        user_id: str,
        chat_id: str,
        persona: Optional[str],
        context: Optional[Dict] = None,
    ):
        try:
            user = Users.get_user_by_id(user_id)
            if user:
                metadata = user.info or {}
                persona_state = metadata.get("persona_state", {})
                if persona:
                    persona_state[chat_id] = {
                        "active_persona": persona,
                        "timestamp": time.time(),
                        "context": context or {},
                    }
                else:
                    persona_state.pop(chat_id, None)
                metadata["persona_state"] = persona_state
                Users.update_user_by_id(user_id, {"info": metadata})
        except Exception as e:
            print(f"[PersonaStorage] Sync store error: {e}")

    @staticmethod
    async def get_persona(
        user_id: str, chat_id: str
    ) -> tuple[Optional[str], Optional[Dict]]:
        if not NATIVE_DB_AVAILABLE or not user_id or not chat_id:
            return None, None
        try:
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(
                None, PersonaStorage._sync_get_persona, user_id, chat_id
            )
        except Exception as e:
            print(f"[PersonaStorage] Async get error: {e}")
        return None, None

    @staticmethod
    def _sync_get_persona(
        user_id: str, chat_id: str
    ) -> tuple[Optional[str], Optional[Dict]]:
        try:
            user = Users.get_user_by_id(user_id)
            if user and user.info:
                persona_data = user.info.get("persona_state", {}).get(chat_id, {})
                return persona_data.get("active_persona"), persona_data.get(
                    "context", {}
                )
        except Exception as e:
            print(f"[PersonaStorage] Sync get error: {e}")
        return None, None


class PersonaDownloader:
    def __init__(self, config_path_func):
        self.get_config_path = config_path_func
        self._session = None

    async def _get_session(self):
        if self._session is None or self._session.closed:
            timeout = aiohttp.ClientTimeout(total=30)
            self._session = aiohttp.ClientSession(
                timeout=timeout, headers={"User-Agent": "OpenWebUI-AgentHotswap/2.8.0"}
            )
        return self._session

    async def close_session(self):
        if self._session and not self._session.closed:
            await self._session.close()

    def is_trusted_domain(self, url: str) -> bool:
        try:
            return (
                urllib.parse.urlparse(url).scheme == "https"
                and urllib.parse.urlparse(url).netloc.lower() in TRUSTED_DOMAINS
            )
        except:
            return False

    def _get_ui_path(self) -> str:
        config_path = self.get_config_path()
        return os.path.join(os.path.dirname(config_path), UI_FILE)

    async def download_ui_file(
        self, url: str = None, force_download: bool = False
    ) -> Dict:
        download_url = url or UI_REPO
        if not self.is_trusted_domain(download_url):
            return {"success": False, "error": "Untrusted domain"}

        ui_path = self._get_ui_path()

        if not force_download and os.path.exists(ui_path):
            try:
                async with aiofiles.open(ui_path, "r", encoding="utf-8") as f:
                    content = await f.read()
                    if len(content) > 100:
                        return {
                            "success": True,
                            "skipped": True,
                            "reason": "file_exists",
                        }
            except Exception:
                pass

        try:
            session = await self._get_session()
            async with session.get(download_url) as response:
                if response.status != 200:
                    return {"success": False, "error": f"HTTP {response.status}"}
                content = await response.text()
                if len(content) > 1024 * 1024:
                    return {"success": False, "error": "UI file too large"}

                os.makedirs(os.path.dirname(ui_path), exist_ok=True)
                async with aiofiles.open(ui_path, "w", encoding="utf-8") as f:
                    await f.write(content)

                return {
                    "success": True,
                    "size": len(content),
                    "path": ui_path,
                }
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def download_personas(
        self,
        url: str = None,
        merge: bool = True,
        create_backup: bool = True,
        force_download: bool = False,
    ) -> Dict:
        download_url = url or DEFAULT_REPO
        if not self.is_trusted_domain(download_url):
            return {"success": False, "error": "Untrusted domain"}

        # Download UI file alongside personas
        ui_result = await self.download_ui_file(force_download=force_download)
        ui_success = ui_result.get("success", False)
        if not ui_success:
            print(
                f"[AGENT_HOTSWAP] UI download failed: {ui_result.get('error', 'unknown')}"
            )
        elif not ui_result.get("skipped"):
            print(
                f"[AGENT_HOTSWAP] UI file downloaded: {ui_result.get('size', 0)} bytes"
            )

        try:
            session = await self._get_session()
            async with session.get(download_url) as response:
                if response.status != 200:
                    return {"success": False, "error": f"HTTP {response.status}"}
                content = await response.text()
                if len(content) > 1024 * 1024 * 2:
                    return {"success": False, "error": "File too large"}
                remote_personas = json.loads(content)
                if not isinstance(remote_personas, dict):
                    return {"success": False, "error": "Invalid format"}

                config_path = self.get_config_path()
                os.makedirs(os.path.dirname(config_path), exist_ok=True)
                async with aiofiles.open(config_path, "w", encoding="utf-8") as f:
                    await f.write(
                        json.dumps(remote_personas, indent=2, ensure_ascii=False)
                    )

                count = len(
                    [k for k in remote_personas.keys() if not k.startswith("_")]
                )
                return {
                    "success": True,
                    "count": count,
                    "size": len(content),
                    "ui_downloaded": ui_success and not ui_result.get("skipped"),
                    "ui_status": "success" if ui_success else "failed",
                }
        except Exception as e:
            return {"success": False, "error": str(e)}


class UniversalPatternMatcher:
    def __init__(self, prefix: str = "!", case_sensitive: bool = False):
        self.prefix = prefix
        self.case_sensitive = case_sensitive
        self._patterns = {}
        self._compile_patterns()

    def _compile_patterns(self):
        flags = 0 if self.case_sensitive else re.IGNORECASE
        prefix_escaped = re.escape(self.prefix)
        self._patterns["persona"] = re.compile(
            rf"{prefix_escaped}([a-zA-Z][a-zA-Z0-9_]*)\b", flags
        )
        self._patterns["per_model"] = re.compile(
            rf"{prefix_escaped}persona(\d+)\s+([a-zA-Z][a-zA-Z0-9_]*)\b", flags
        )
        self._patterns["agent_list"] = re.compile(
            rf"{prefix_escaped}agent\s+list\b", flags
        )
        self._patterns["agent_base"] = re.compile(rf"{prefix_escaped}agent\b", flags)
        self._patterns["reset"] = re.compile(
            rf"{prefix_escaped}(?:reset|default|normal)\b", flags
        )
        self._patterns["multi"] = re.compile(rf"{prefix_escaped}multi\b", flags)

    async def detect_command(self, content: str) -> Dict[str, Any]:
        if not content:
            return {"type": "none"}
        if self._patterns["agent_list"].search(content):
            return {"type": "list"}
        if self._patterns["agent_base"].search(content):
            return {"type": "help"}
        if self._patterns["reset"].search(content):
            return {"type": "reset"}
        has_multi = bool(self._patterns["multi"].search(content))
        per_model_matches = {}
        for match in self._patterns["per_model"].finditer(content):
            model_num = int(match.group(1))
            persona_key = match.group(2)
            if not self.case_sensitive:
                persona_key = persona_key.lower()
            if 1 <= model_num <= 4:
                per_model_matches[model_num] = persona_key
        if per_model_matches:
            return {
                "type": "per_model",
                "personas": per_model_matches,
                "has_multi_command": has_multi,
            }
        matches = self._patterns["persona"].findall(content)
        if matches:
            command_keywords = {"agent", "list", "reset", "default", "normal", "multi"}
            personas = []
            for m in matches:
                persona_key = m if self.case_sensitive else m.lower()
                if (
                    persona_key.startswith("persona") and persona_key[7:].isdigit()
                ) or persona_key in command_keywords:
                    continue
                personas.append(persona_key)
            if personas:
                unique_personas = list(dict.fromkeys(personas))
                return (
                    {
                        "type": "single_persona",
                        "persona": unique_personas[0],
                        "has_multi_command": has_multi,
                    }
                    if len(unique_personas) == 1
                    else {
                        "type": "multi_persona",
                        "personas": unique_personas,
                        "has_multi_command": has_multi,
                    }
                )
        return {"type": "none"}

    def remove_commands(self, content: str) -> str:
        persona_commands_pattern = re.compile(
            rf"{re.escape(self.prefix)}(agent(\s+list)?|reset|default|normal|persona\d+\s+[a-zA-Z][a-zA-Z0-9_]*|(?!multi)[a-zA-Z][a-zA-Z0-9_]*)\b\s*",
            re.IGNORECASE,
        )
        return persona_commands_pattern.sub("", content).strip()


class PersonaCache:
    def __init__(self):
        self._cache = {}
        self._file_mtime = 0
        self._last_path = None

    async def get_personas(self, filepath: str) -> Dict:
        global _GLOBAL_PERSONA_CACHE, _CACHE_LOCK
        async with _CACHE_LOCK:
            try:
                if not os.path.exists(filepath):
                    return {}
                current_mtime = os.path.getmtime(filepath)
                cache_key = f"{filepath}:{current_mtime}"
                if cache_key in _GLOBAL_PERSONA_CACHE:
                    return _GLOBAL_PERSONA_CACHE[cache_key].copy()
                if (
                    filepath != self._last_path
                    or current_mtime > self._file_mtime
                    or not self._cache
                ):
                    async with aiofiles.open(filepath, "r", encoding="utf-8") as f:
                        content = await f.read()
                        self._cache = json.loads(content)
                    self._file_mtime = current_mtime
                    self._last_path = filepath
                    _GLOBAL_PERSONA_CACHE[cache_key] = self._cache.copy()
                    if len(_GLOBAL_PERSONA_CACHE) > 10:
                        oldest_key = min(_GLOBAL_PERSONA_CACHE.keys())
                        del _GLOBAL_PERSONA_CACHE[oldest_key]
                return self._cache.copy()
            except Exception as e:
                print(f"[PersonaCache] Error loading personas: {e}")
                return {}


class Filter:
    class Valves(BaseModel):
        priority: int = Field(default=0)
        keyword_prefix: str = Field(default="!")
        case_sensitive: bool = Field(default=False)
        show_persona_info: bool = Field(default=True)
        persistent_persona: bool = Field(default=True)
        auto_download_personas: bool = Field(default=True)
        merge_on_update: bool = Field(default=True)
        enable_debug: bool = Field(default=False)
        refresh_personas: bool = Field(default=False)
        multi_persona_transitions: bool = Field(default=True)
        enable_plugin_integration: bool = Field(
            default=True, description="Enable integration with other plugins"
        )
        integration_debug: bool = Field(
            default=False, description="Debug integration communications"
        )
        enable_automatic_backups: bool = Field(
            default=True, description="Create automatic backups before updates"
        )
        max_backup_files: int = Field(
            default=10, description="Maximum number of backup files to keep"
        )
        protect_custom_personas: bool = Field(
            default=True,
            description="Prevent auto-download from overwriting custom personas",
        )
        enable_global_cache: bool = Field(
            default=True, description="Use global persona cache for better performance"
        )

    def __init__(self):
        self.valves = self.Valves()
        self.toggle = True
        self.icon = "🎭"
        self.plugin_directory_name = self._get_plugin_directory_name()
        self.current_persona = None
        self.current_context = {}
        self.pattern_matcher = UniversalPatternMatcher(
            self.valves.keyword_prefix, self.valves.case_sensitive
        )
        self.persona_cache = PersonaCache()
        self.downloader = PersonaDownloader(self._get_config_path)
        self.integration_manager = PluginIntegrationManager()
        self._init_task = None

    def __del__(self):
        try:
            if hasattr(self, "downloader") and self.downloader._session:
                asyncio.create_task(self.downloader.close_session())
        except:
            pass

    def _get_plugin_directory_name(self) -> str:
        try:
            if __name__ != "__main__":
                module_parts = __name__.split(".")
                detected_name = module_parts[-1]
                if detected_name.startswith("function_"):
                    detected_name = detected_name[9:]
                cleaned_name = re.sub(r"[^a-zA-Z0-9_-]", "_", detected_name.lower())
                if cleaned_name and cleaned_name != "__main__":
                    print(f"[AGENT_HOTSWAP] Auto-detected plugin name: {cleaned_name}")
                    return cleaned_name
        except Exception as e:
            print(f"[AGENT_HOTSWAP] Method 1 (__name__) failed: {e}")

        fallback_name = "agent_hotswap"
        print(f"[AGENT_HOTSWAP] Using fallback plugin name: {fallback_name}")
        return fallback_name

    def _debug_log(self, message: str):
        if self.valves.enable_debug:
            print(f"[AGENT_HOTSWAP] {message}")

    def _integration_debug(self, message: str):
        if self.valves.integration_debug:
            print(f"[AGENT_HOTSWAP:INTEGRATION] {message}")

    def _get_config_path(self) -> str:
        data_dir = os.getenv("DATA_DIR") or (
            "/app/backend/data"
            if os.path.exists("/app/backend")
            else str(Path.home() / ".local/share/open-webui")
        )
        config_path = os.path.join(
            data_dir, "cache", "functions", self.plugin_directory_name, CONFIG_FILE
        )
        return config_path

    def _get_ui_path(self) -> str:
        config_path = self._get_config_path()
        return os.path.join(os.path.dirname(config_path), UI_FILE)

    def _get_relative_hub_url(self) -> str:
        try:
            relative_url = f"/cache/functions/{self.plugin_directory_name}/index.html"
            print(f"[Persona Browser URL] Generated relative URL: {relative_url}")
            return relative_url
        except Exception as e:
            print(f"[Persona Browser URL] Error generating URL: {e}")
            return f"/cache/functions/{self.plugin_directory_name}/index.html"

    async def _ensure_personas_available(self):
        config_path = self._get_config_path()
        if not os.path.exists(config_path):
            os.makedirs(os.path.dirname(config_path), exist_ok=True)
            await self._create_minimal_config(config_path)
            if self.valves.auto_download_personas:
                await self._download_personas_async()
        else:
            try:
                async with aiofiles.open(config_path, "r", encoding="utf-8") as f:
                    content = await f.read()
                    existing_data = json.loads(content)
                if not existing_data or len(existing_data) < 2:
                    if self.valves.auto_download_personas:
                        await self._download_personas_async()
            except (json.JSONDecodeError, Exception) as e:
                print(f"[AGENT_HOTSWAP] Error reading existing config: {e}")
                await self._create_minimal_config(config_path)
                if self.valves.auto_download_personas:
                    await self._download_personas_async()

    async def _create_minimal_config(self, config_path: str):
        try:
            minimal_config = {
                "_master_controller": {
                    "name": "🎛️ OpenWebUI Master Controller",
                    "hidden": True,
                    "always_active": True,
                    "prompt": """=== OPENWEBUI MASTER CONTROLLER ===
You operate in OpenWebUI with comprehensive native capabilities:

RENDERING: LaTeX ($formula$), Mermaid diagrams, HTML artifacts, SVG, enhanced Markdown
CODE EXECUTION: Python via Pyodide, Jupyter integration, interactive code blocks
FILE HANDLING: Multi-format extraction (PDF, Word, Excel, etc.), drag-drop upload
RAG: Document integration, web search, knowledge bases, citations
VOICE/AUDIO: STT/TTS, Voice Activity Detection, audio processing
INTEGRATIONS: OpenAPI tools, multi-API endpoints, WebSocket connections
UI/UX: Multi-model chat, message management, responsive design
ADMIN/SECURITY: User permissions, authentication, audit logging

Leverage these capabilities appropriately for the best user experience.
=== END MASTER CONTROLLER ===
""",
                },
                "coder": {
                    "name": "💻 Code Assistant",
                    "prompt": "You are the 💻 Code Assistant, an expert in programming and software development. Provide clean, efficient, well-documented code solutions with explanations of your approach and best practices.",
                    "description": "Expert programming assistance with best practices",
                    "capabilities": [
                        "programming",
                        "debugging",
                        "code_review",
                        "architecture",
                    ],
                },
                "writer": {
                    "name": "✍️ Creative Writer",
                    "prompt": "You are the ✍️ Creative Writer, a master of crafting engaging content. Help with all aspects of writing from structure and style to tone and clarity. Create compelling narratives and clear communication.",
                    "description": "Creative writing specialist and communication expert",
                    "capabilities": [
                        "writing",
                        "creativity",
                        "storytelling",
                        "communication",
                    ],
                },
                "_metadata": {
                    "version": "minimal_v2.8.0",
                    "last_updated": datetime.now().isoformat(),
                    "integration_ready": True,
                    "plugin_directory": self.plugin_directory_name,
                    "initial_setup": True,
                    "async_enabled": True,
                    "popup_ui": True,
                },
            }
            async with aiofiles.open(config_path, "w", encoding="utf-8") as f:
                await f.write(json.dumps(minimal_config, indent=4, ensure_ascii=False))
            print(
                f"[AGENT_HOTSWAP] Minimal config created in {self.plugin_directory_name}"
            )
        except Exception as e:
            print(f"[AGENT_HOTSWAP] Error creating minimal config: {e}")

    async def _download_personas_async(self, force_download: bool = False):
        try:
            result = await self.downloader.download_personas(
                merge=self.valves.merge_on_update,
                create_backup=self.valves.enable_automatic_backups,
                force_download=force_download,
            )
            if result["success"]:
                ui_msg = ""
                if result.get("ui_downloaded"):
                    ui_msg = " + UI downloaded"
                elif result.get("ui_status") == "failed":
                    ui_msg = " (UI download failed - will use fallback)"

                print(f"[AGENT_HOTSWAP] Downloaded {result['count']} personas{ui_msg}")
                global _GLOBAL_PERSONA_CACHE
                _GLOBAL_PERSONA_CACHE.clear()
            else:
                print(f"[AGENT_HOTSWAP] Download failed: {result['error']}")
        except Exception as e:
            print(f"[AGENT_HOTSWAP] Download error: {e}")

    async def _load_personas(self) -> Dict:
        return await self.persona_cache.get_personas(self._get_config_path()) or {}

    def _safe_get_ids(self, body: dict, user: Optional[dict]):
        chat_id = body.get("chat_id") or body.get("id") or f"chat_{int(time.time())}"
        user_id = user.get("id", "anonymous") if user else "anonymous"
        return str(chat_id), str(user_id)

    async def _emit_status(
        self,
        emitter,
        message: str,
        status_type: str = "in_progress",
        done: bool = False,
    ):
        if not emitter or not self.valves.show_persona_info:
            return
        try:
            await emitter(
                {
                    "type": "status",
                    "data": {
                        "description": message,
                        "status": status_type,
                        "done": done,
                        "hidden": done,
                    },
                }
            )
        except Exception as e:
            print(f"[Status Emit] Error: {e}")

    async def _update_personas_data_file(self, personas: Dict):
        """Create a separate personas.json file that the UI can load"""
        try:
            ui_dir = os.path.dirname(self._get_ui_path())
            personas_json_path = os.path.join(ui_dir, "personas.json")

            # Debug logging
            print(f"[AGENT_HOTSWAP] UI directory: {ui_dir}")
            print(f"[AGENT_HOTSWAP] Personas JSON path: {personas_json_path}")
            print(
                f"[AGENT_HOTSWAP] Personas count: {len([k for k, v in personas.items() if not k.startswith('_')])}"
            )

            # Ensure directory exists
            os.makedirs(ui_dir, exist_ok=True)

            # Write personas data as a separate JSON file
            async with aiofiles.open(personas_json_path, "w", encoding="utf-8") as f:
                await f.write(json.dumps(personas, indent=2, ensure_ascii=False))

            # Verify the file was written
            if os.path.exists(personas_json_path):
                file_size = os.path.getsize(personas_json_path)
                print(f"[AGENT_HOTSWAP] ✅ Updated personas.json ({file_size} bytes)")
            else:
                print(f"[AGENT_HOTSWAP] ❌ Failed to create personas.json")

        except Exception as e:
            print(f"[AGENT_HOTSWAP] ❌ Error updating personas.json: {e}")
            import traceback

            traceback.print_exc()

    def _generate_fallback_list(self, personas: Dict) -> str:
        display_personas = {k: v for k, v in personas.items() if not k.startswith("_")}
        if not display_personas:
            return "No personas available."

        lines = ["## 🎭 Available Personas\n"]
        for key, data in sorted(display_personas.items()):
            name = data.get("name", key.title())
            description = data.get("description", "")
            command = f"{self.valves.keyword_prefix}{key}"
            lines.append(f"**{name}** - `{command}`")
            if description:
                lines.append(f"_{description}_")
            lines.append("")

        return "\n".join(lines)

    def _generate_help_message(self) -> str:
        p = self.valves.keyword_prefix
        protection_status = "✅ ON" if self.valves.protect_custom_personas else "❌ OFF"
        return f"""### Agent Hotswap Commands

- **`{p}agent`**: Displays this help message.
- **`{p}agent list`**: Opens interactive persona browser in new window.
- **`{p}{{persona_name}}`**: Activates a specific persona (e.g., `{p}coder`).
- **`{p}reset`**: Resets to the default assistant.

**Multi-Persona Support:** Use multiple personas in one message!
Example: `{p}writer create a story {p}teacher explain techniques`

**Per-Model Personas:** Assign different personas to specific models!
Example: `{p}persona1 teacher {p}persona2 scientist {p}multi debate evolution`

**Performance Features:**
- **Async Operations:** ✅ ENABLED
- **Global Caching:** {"✅ ON" if self.valves.enable_global_cache else "❌ OFF"}
- **Custom Persona Protection:** {protection_status}
- **UI Mode:** Popup Window

**Integration Features:**
- Works seamlessly with Multi-Model conversations
- Supports conversation summarization with persona context
- Maintains persona state across long conversations

**Directory:** `{self.plugin_directory_name}`
**Version:** 2.8.0 (Popup Window UI)"""

    async def _create_system_message(self, persona_key: str) -> Dict:
        personas = await self._load_personas()
        if not personas:
            return {"role": "system", "content": "Error: Persona file not loaded."}
        master = personas.get("_master_controller", {})
        master_prompt = master.get("prompt", "")
        persona = personas.get(persona_key, {})
        persona_prompt = persona.get("prompt", f"You are the {persona_key} persona.")
        system_content = f"{master_prompt}\n\n{persona_prompt}"
        if self.valves.show_persona_info:
            persona_name = persona.get("name", persona_key.title())
            system_content += f"\n\n🎭 **Active Persona**: {persona_name}"
        return {"role": "system", "content": system_content}

    def _remove_persona_messages(self, messages: List[Dict]) -> List[Dict]:
        return [
            m
            for m in messages
            if not (
                m.get("role") == "system"
                and any(
                    marker in m.get("content", "")
                    for marker in [
                        "=== OPENWEBUI MASTER CONTROLLER ===",
                        "🎭 **Active Persona**",
                        "=== DYNAMIC MULTI-PERSONA MODE ===",
                    ]
                )
            )
        ]

    async def _create_integration_context(
        self, command_info: Dict, personas_data: Dict, **kwargs
    ) -> Dict:
        if not self.valves.enable_plugin_integration:
            return {}
        base_context = {
            "agent_hotswap_active": True,
            "agent_hotswap_version": "2.8.0",
            "command_info": command_info,
            "timestamp": time.time(),
            "plugin_directory": self.plugin_directory_name,
            "async_enabled": True,
            "popup_ui": True,
        }
        if command_info["type"] == "single_persona":
            integration_context = (
                self.integration_manager.prepare_single_persona_context(
                    command_info["persona"], personas_data
                )
            )
            base_context.update(integration_context)
        elif command_info["type"] == "per_model":
            assignments = {}
            for model_num, persona_key in command_info["personas"].items():
                if persona_key in personas_data:
                    persona_info = personas_data[persona_key]
                    assignments[model_num] = {
                        "key": persona_key,
                        "name": persona_info.get("name", persona_key.title()),
                        "prompt": persona_info.get(
                            "prompt", f"You are a {persona_key}."
                        ),
                        "description": persona_info.get("description", ""),
                        "capabilities": persona_info.get("capabilities", []),
                    }
            integration_context = self.integration_manager.prepare_per_model_context(
                assignments, personas_data
            )
            base_context.update(integration_context)
        base_context.update(kwargs)
        self._integration_debug(
            f"Created integration context: {list(base_context.keys())}"
        )
        return base_context

    async def _handle_list_command(self, body: dict, emitter, event_call=None) -> dict:
        try:
            await self._emit_status(
                emitter, "📋 Loading persona browser...", "in_progress"
            )

            personas = await self._load_personas()
            print(f"[AGENT_HOTSWAP] Loaded {len(personas)} total personas")
            display_personas = {
                k: v for k, v in personas.items() if not k.startswith("_")
            }
            print(f"[AGENT_HOTSWAP] Display personas: {list(display_personas.keys())}")

            ui_path = self._get_ui_path()
            print(f"[AGENT_HOTSWAP] UI path: {ui_path}")

            if not os.path.exists(ui_path):
                print(f"[AGENT_HOTSWAP] UI file not found, downloading...")
                result = await self.downloader.download_ui_file()
                if not result["success"]:
                    print(f"[AGENT_HOTSWAP] UI download failed: {result.get('error')}")
                    return await self._handle_list_command_fallback(body, emitter)
            else:
                print(f"[AGENT_HOTSWAP] UI file exists")

            await self._update_personas_data_file(personas)

            hub_url = self._get_relative_hub_url()
            print(f"[AGENT_HOTSWAP] Opening popup at: {hub_url}")

            popup_script = f"""
                console.log('Opening persona browser at: {hub_url}');
                window.open(
                    '{hub_url}', 
                    'personaBrowser_' + Date.now(), 
                    'width=' + Math.min(screen.availWidth, 1200) + ',height=' + Math.min(screen.availHeight, 800) + ',scrollbars=yes,resizable=yes,menubar=no,toolbar=no'
                );
            """

            if event_call:
                await event_call({"type": "execute", "data": {"code": popup_script}})

            await self._emit_status(emitter, "✅ Persona browser opened!", "complete")

            await asyncio.sleep(2)
            await self._emit_status(emitter, "", "complete", done=True)

            result = body.copy()
            result["_agent_hotswap_handled"] = True

            for msg in reversed(result.get("messages", [])):
                if msg.get("role") == "user":
                    msg["content"] = (
                        "Respond with exactly this message: 'Opening the 🎭 Persona Browser now. Browse and copy commands to switch between AI personas!'"
                    )
                    break

            return result

        except Exception as e:
            print(f"[AGENT_HOTSWAP] Error in list command: {e}")
            import traceback

            traceback.print_exc()
            return await self._handle_list_command_fallback(body, emitter)

    async def _handle_list_command_fallback(self, body: dict, emitter) -> dict:
        try:
            await self._emit_status(
                emitter, "⚠️ Using fallback persona list", "complete"
            )
            personas = await self._load_personas()
            fallback_content = self._generate_fallback_list(personas)

            if emitter:
                await emitter(
                    {"type": "message", "data": {"content": fallback_content}}
                )

            result = body.copy()
            result["messages"] = [{"role": "assistant", "content": fallback_content}]
            result["_agent_hotswap_handled"] = True
            return result

        except Exception as e:
            print(f"[AGENT_HOTSWAP] Error in fallback: {e}")
            error_message = f"Error loading persona list: {str(e)}"
            result = body.copy()
            result["messages"] = [{"role": "assistant", "content": error_message}]
            result["_agent_hotswap_handled"] = True
            return result

    async def _handle_help_command(self, body: dict, emitter) -> dict:
        try:
            help_content = self._generate_help_message()
            await self._emit_status(
                emitter, "ℹ️ Showing Agent Hotswap commands", "complete"
            )

            if emitter:
                await emitter({"type": "message", "data": {"content": help_content}})

            result = body.copy()
            result["messages"] = [{"role": "assistant", "content": help_content}]
            result["_agent_hotswap_handled"] = True
            return result

        except Exception as e:
            print(f"[AGENT_HOTSWAP] Error in help command: {e}")
            error_message = f"Error generating help: {str(e)}"
            if emitter:
                await emitter({"type": "message", "data": {"content": error_message}})
            result = body.copy()
            result["messages"] = [{"role": "assistant", "content": error_message}]
            result["_agent_hotswap_handled"] = True
            return result

    async def _handle_single_persona(
        self,
        persona_key: str,
        body: dict,
        messages: List[Dict],
        original_content: str,
        emitter,
        user_id: str,
        chat_id: str,
        command_info: Dict,
    ) -> dict:
        personas = await self._load_personas()
        if not personas or persona_key not in personas:
            return body
        self.current_persona = persona_key
        self.current_context = {"type": "single_persona", "persona": persona_key}
        await PersonaStorage.store_persona(
            user_id, chat_id, persona_key, self.current_context
        )
        clean_messages = self._remove_persona_messages(messages)
        system_msg = await self._create_system_message(persona_key)
        clean_messages.insert(0, system_msg)
        cleaned_content = self.pattern_matcher.remove_commands(original_content)
        persona_config = personas[persona_key]
        for msg in reversed(clean_messages):
            if msg.get("role") == "user":
                if not cleaned_content:
                    msg["content"] = (
                        "Please introduce yourself and explain what you can help with."
                    )
                else:
                    persona_name = persona_config.get("name", persona_key.title())
                    msg["content"] = (
                        f"Please briefly introduce yourself as {persona_name}. Then help with: {cleaned_content}"
                    )
                break
        body["messages"] = clean_messages
        if self.valves.enable_plugin_integration:
            body["_filter_context"] = await self._create_integration_context(
                command_info, personas, original_content=original_content
            )
        persona_name = persona_config.get("name", persona_key.title())
        await self._emit_status(emitter, f"🎭 Switched to {persona_name}", "complete")
        return body

    async def _apply_persistent_persona(self, body: dict, messages: List[Dict]) -> dict:
        if not self.valves.persistent_persona or not self.current_persona:
            return body
        if self.current_persona.startswith(("multi:", "per_model:")):
            return body
        personas = await self._load_personas()
        if not personas or self.current_persona not in personas:
            return body
        if any(
            "=== OPENWEBUI MASTER CONTROLLER ===" in m.get("content", "")
            for m in messages
            if m.get("role") == "system"
        ):
            return body
        clean_messages = self._remove_persona_messages(messages)
        system_msg = await self._create_system_message(self.current_persona)
        clean_messages.insert(0, system_msg)
        body["messages"] = clean_messages
        if self.valves.enable_plugin_integration and not body.get("_filter_context"):
            body["_filter_context"] = (
                self.integration_manager.prepare_single_persona_context(
                    self.current_persona, personas
                )
            )
        return body

    async def inlet(
        self,
        body: dict,
        __event_emitter__,
        __event_call__=None,
        __user__: Optional[dict] = None,
    ) -> dict:
        if not self.toggle:
            return body

        if body.get("_agent_hotswap_handled"):
            return body

        try:
            if not hasattr(self, "_initialized"):
                await self._ensure_personas_available()
                self._initialized = True
        except Exception as e:
            print(f"[AGENT_HOTSWAP] Initialization error: {e}")
            return body

        messages = body.get("messages", [])
        if not messages:
            return body

        chat_id, user_id = self._safe_get_ids(body, __user__)

        if (
            self.valves.persistent_persona
            and not self.current_persona
            and user_id != "anonymous"
        ):
            try:
                stored_persona, stored_context = await PersonaStorage.get_persona(
                    user_id, chat_id
                )
                if stored_persona:
                    self.current_persona = stored_persona
                    self.current_context = stored_context or {}
            except Exception as e:
                print(f"[AGENT_HOTSWAP] Error restoring persona: {e}")

        last_user_content = ""
        for msg in reversed(messages):
            if msg.get("role") == "user":
                last_user_content = msg.get("content", "")
                break

        if not last_user_content:
            return await self._apply_persistent_persona(body, messages)

        try:
            command_info = await self.pattern_matcher.detect_command(last_user_content)
        except Exception as e:
            print(f"[AGENT_HOTSWAP] Error detecting command: {e}")
            return await self._apply_persistent_persona(body, messages)

        try:
            if command_info["type"] == "help":
                return await self._handle_help_command(body, __event_emitter__)
            elif command_info["type"] == "list":
                return await self._handle_list_command(
                    body, __event_emitter__, __event_call__
                )
            elif command_info["type"] == "single_persona":
                return await self._handle_single_persona(
                    command_info["persona"],
                    body,
                    messages,
                    last_user_content,
                    __event_emitter__,
                    user_id,
                    chat_id,
                    command_info,
                )
            else:
                return await self._apply_persistent_persona(body, messages)
        except Exception as e:
            print(f"[AGENT_HOTSWAP] Error handling command {command_info['type']}: {e}")
            return await self._apply_persistent_persona(body, messages)

    async def outlet(
        self, body: dict, __event_emitter__, __user__: Optional[dict] = None
    ) -> dict:
        return body
