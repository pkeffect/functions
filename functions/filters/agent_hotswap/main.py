"""
title: Agent Hotswap - Enhanced Integration
author: pkeffect & Claude AI
author_url: https://github.com/pkeffect
project_urls: https://github.com/pkeffect/functions/tree/main/functions/filters/agent_hotswap | https://github.com/open-webui/functions/tree/main/functions/filters/agent_hotswap | https://openwebui.com/f/pkeffect/agent_hotswap
funding_url: https://github.com/open-webui
version: 2.5.0
description: Universal AI persona switching with enhanced multi-plugin integration and robust per-model persona support.
requirements: pydantic>=2.0.0
"""

from pydantic import BaseModel, Field
from typing import Optional, Dict, List, Any
import re
import json
import time
import os
import urllib.request
import urllib.parse
import shutil
import glob
from datetime import datetime
from pathlib import Path
from string import Template

# OpenWebUI Imports
try:
    from open_webui.models.users import Users
    from open_webui.models.models import Models

    NATIVE_DB_AVAILABLE = True
except ImportError:
    NATIVE_DB_AVAILABLE = False

# Configuration
CACHE_DIR = "agent_hotswap"
CONFIG_FILE = "personas.json"
DEFAULT_REPO = "https://raw.githubusercontent.com/open-webui/functions/refs/heads/main/functions/filters/agent_hotswap/personas/personas.json"
TRUSTED_DOMAINS = ["github.com", "raw.githubusercontent.com"]


class PluginIntegrationManager:
    """Manages integration with other plugins in the suite"""

    @staticmethod
    def create_integration_context(
        persona_data: Dict, command_type: str, **kwargs
    ) -> Dict:
        """Create standardized integration context for other plugins"""
        context = {
            "agent_hotswap_version": "2.5.0",
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
        """Prepare detailed per-model context for Multi-Model Filter"""
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

        # Add meta information
        context["per_model_active"] = True
        context["total_assigned_models"] = len(assignments)
        context["assigned_model_numbers"] = list(assignments.keys())

        return context

    @staticmethod
    def prepare_single_persona_context(persona_key: str, personas_data: Dict) -> Dict:
        """Prepare single persona context for other plugins"""
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
        """Prepare multi-persona sequence context"""
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
    def store_persona(
        user_id: str,
        chat_id: str,
        persona: Optional[str],
        context: Optional[Dict] = None,
    ):
        if not NATIVE_DB_AVAILABLE or not user_id or not chat_id:
            return
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
            print(f"[PersonaStorage] Store error: {e}")

    @staticmethod
    def get_persona(user_id: str, chat_id: str) -> tuple[Optional[str], Optional[Dict]]:
        if not NATIVE_DB_AVAILABLE or not user_id or not chat_id:
            return None, None
        try:
            user = Users.get_user_by_id(user_id)
            if user and user.info:
                persona_data = user.info.get("persona_state", {}).get(chat_id, {})
                return persona_data.get("active_persona"), persona_data.get(
                    "context", {}
                )
        except Exception as e:
            print(f"[PersonaStorage] Get error: {e}")
        return None, None


class PersonaDownloader:
    def __init__(self, config_path_func):
        self.get_config_path = config_path_func

    def is_trusted_domain(self, url: str) -> bool:
        try:
            return (
                urllib.parse.urlparse(url).scheme == "https"
                and urllib.parse.urlparse(url).netloc.lower() in TRUSTED_DOMAINS
            )
        except:
            return False

    def _get_backup_dir(self) -> str:
        """Get the backup directory path"""
        config_path = self.get_config_path()
        backup_dir = os.path.join(os.path.dirname(config_path), "backups")
        os.makedirs(backup_dir, exist_ok=True)
        return backup_dir

    def _create_backup(self, config_path: str) -> str:
        """Create a timestamped backup of the current personas.json"""
        if not os.path.exists(config_path):
            return None

        try:
            backup_dir = self._get_backup_dir()
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            backup_filename = f"personas_backup_{timestamp}.json"
            backup_path = os.path.join(backup_dir, backup_filename)

            shutil.copy2(config_path, backup_path)
            print(f"[AGENT_HOTSWAP] Backup created: {backup_filename}")
            return backup_path
        except Exception as e:
            print(f"[AGENT_HOTSWAP] Backup creation failed: {e}")
            return None

    def _cleanup_old_backups(self, max_backups: int = 10):
        """Remove old backup files, keeping only the most recent ones"""
        try:
            backup_dir = self._get_backup_dir()
            backup_pattern = os.path.join(backup_dir, "personas_backup_*.json")
            backup_files = glob.glob(backup_pattern)

            if len(backup_files) > max_backups:
                # Sort by modification time (oldest first)
                backup_files.sort(key=os.path.getmtime)
                files_to_remove = backup_files[:-max_backups]

                for file_path in files_to_remove:
                    try:
                        os.remove(file_path)
                        print(
                            f"[AGENT_HOTSWAP] Removed old backup: {os.path.basename(file_path)}"
                        )
                    except Exception as e:
                        print(
                            f"[AGENT_HOTSWAP] Failed to remove backup {file_path}: {e}"
                        )

        except Exception as e:
            print(f"[AGENT_HOTSWAP] Backup cleanup failed: {e}")

    def restore_from_backup(self, backup_filename: str = None) -> Dict:
        """Restore personas from a backup file"""
        try:
            backup_dir = self._get_backup_dir()

            if backup_filename:
                backup_path = os.path.join(backup_dir, backup_filename)
            else:
                # Use most recent backup
                backup_pattern = os.path.join(backup_dir, "personas_backup_*.json")
                backup_files = glob.glob(backup_pattern)
                if not backup_files:
                    return {"success": False, "error": "No backup files found"}
                backup_path = max(backup_files, key=os.path.getmtime)

            if not os.path.exists(backup_path):
                return {
                    "success": False,
                    "error": f"Backup file not found: {backup_filename or 'latest'}",
                }

            config_path = self.get_config_path()

            # Create backup of current state before restoring
            current_backup = self._create_backup(config_path)

            # Restore from backup
            shutil.copy2(backup_path, config_path)

            return {
                "success": True,
                "restored_from": os.path.basename(backup_path),
                "current_backup": (
                    os.path.basename(current_backup) if current_backup else None
                ),
            }

        except Exception as e:
            return {"success": False, "error": str(e)}

    def list_backups(self) -> List[Dict]:
        """List all available backup files with metadata"""
        try:
            backup_dir = self._get_backup_dir()
            backup_pattern = os.path.join(backup_dir, "personas_backup_*.json")
            backup_files = glob.glob(backup_pattern)

            backups = []
            for backup_path in sorted(backup_files, key=os.path.getmtime, reverse=True):
                try:
                    stat = os.stat(backup_path)
                    backups.append(
                        {
                            "filename": os.path.basename(backup_path),
                            "created": datetime.fromtimestamp(
                                stat.st_mtime
                            ).isoformat(),
                            "size": stat.st_size,
                            "path": backup_path,
                        }
                    )
                except Exception as e:
                    print(f"[AGENT_HOTSWAP] Error reading backup {backup_path}: {e}")

            return backups

        except Exception as e:
            print(f"[AGENT_HOTSWAP] Error listing backups: {e}")
            return []

    async def download_personas(
        self, url: str = None, merge: bool = True, create_backup: bool = True
    ) -> Dict:
        download_url = url or DEFAULT_REPO
        if not self.is_trusted_domain(download_url):
            return {"success": False, "error": "Untrusted domain"}

        try:
            req = urllib.request.Request(
                download_url, headers={"User-Agent": "OpenWebUI-AgentHotswap/2.5.0"}
            )
            with urllib.request.urlopen(req, timeout=30) as response:
                if response.status != 200:
                    return {"success": False, "error": f"HTTP {response.status}"}
                content = response.read().decode("utf-8")
                if len(content) > 1024 * 1024 * 2:
                    return {"success": False, "error": "File too large"}
                remote_personas = json.loads(content)
                if not isinstance(remote_personas, dict):
                    return {"success": False, "error": "Invalid format"}

                config_path = self.get_config_path()
                backup_path = None

                # CREATE BACKUP BEFORE MAKING CHANGES
                if create_backup:
                    backup_path = self._create_backup(config_path)

                final_personas = remote_personas
                if merge:
                    local_personas = {}
                    if os.path.exists(config_path):
                        try:
                            with open(config_path, "r", encoding="utf-8") as f:
                                local_personas = json.load(f)
                        except (json.JSONDecodeError, FileNotFoundError):
                            print(
                                "[AGENT_HOTSWAP] Warning: Could not parse local personas.json, will overwrite."
                            )

                    merged = remote_personas.copy()
                    if isinstance(local_personas, dict):
                        merged.update(local_personas)
                    final_personas = merged

                final_personas["_metadata"] = {
                    "last_updated": datetime.now().isoformat(),
                    "source_url": download_url,
                    "version": "auto-downloaded",
                    "merge_strategy": "local_priority" if merge else "overwrite",
                    "backup_created": backup_path if backup_path else None,
                }

                os.makedirs(os.path.dirname(config_path), exist_ok=True)
                with open(config_path, "w", encoding="utf-8") as f:
                    json.dump(final_personas, f, indent=2, ensure_ascii=False)

                # CLEANUP OLD BACKUPS
                if create_backup:
                    self._cleanup_old_backups()

                count = len([k for k in final_personas.keys() if not k.startswith("_")])
                return {
                    "success": True,
                    "count": count,
                    "size": len(content),
                    "backup_created": backup_path is not None,
                }

        except Exception as e:
            return {"success": False, "error": str(e)}


class UniversalPatternMatcher:
    def __init__(self, prefix: str = "!", case_sensitive: bool = False):
        self.prefix = prefix
        self.case_sensitive = case_sensitive
        self._compile_patterns()

    def _compile_patterns(self):
        flags = 0 if self.case_sensitive else re.IGNORECASE
        prefix_escaped = re.escape(self.prefix)

        # Universal persona detection
        self.persona_pattern = re.compile(
            rf"{prefix_escaped}([a-zA-Z][a-zA-Z0-9_]*)\b", flags
        )

        # Enhanced per-model persona pattern with better validation
        self.per_model_pattern = re.compile(
            rf"{prefix_escaped}persona(\d+)\s+([a-zA-Z][a-zA-Z0-9_]*)\b", flags
        )

        # Command patterns
        self.agent_list_pattern = re.compile(rf"{prefix_escaped}agent\s+list\b", flags)
        self.agent_base_pattern = re.compile(rf"{prefix_escaped}agent\b", flags)
        self.reset_pattern = re.compile(
            rf"{prefix_escaped}(?:reset|default|normal)\b", flags
        )

        # Multi-model pattern detection (for integration)
        self.multi_pattern = re.compile(rf"{prefix_escaped}multi\b", flags)

    def detect_command(self, content: str) -> Dict[str, Any]:
        if not content:
            return {"type": "none"}

        # Check special commands first
        if self.agent_list_pattern.search(content):
            return {"type": "list"}
        if self.agent_base_pattern.search(content):
            return {"type": "help"}
        if self.reset_pattern.search(content):
            return {"type": "reset"}

        # Check for multi-model commands (for integration awareness)
        has_multi = bool(self.multi_pattern.search(content))

        # Check per-model personas with enhanced validation
        per_model_matches = {}
        for match in self.per_model_pattern.finditer(content):
            model_num = int(match.group(1))
            persona_key = match.group(2)
            if not self.case_sensitive:
                persona_key = persona_key.lower()

            # Validate model number (1-4 is reasonable range)
            if 1 <= model_num <= 4:
                per_model_matches[model_num] = persona_key

        if per_model_matches:
            return {
                "type": "per_model",
                "personas": per_model_matches,
                "has_multi_command": has_multi,
            }

        # Check regular persona commands
        matches = self.persona_pattern.findall(content)
        if matches:
            command_keywords = {"agent", "list", "reset", "default", "normal", "multi"}
            personas = []
            for m in matches:
                persona_key = m if self.case_sensitive else m.lower()
                # Skip per-model commands and reserved keywords
                if (
                    persona_key.startswith("persona") and persona_key[7:].isdigit()
                ) or persona_key in command_keywords:
                    continue
                personas.append(persona_key)

            if personas:
                unique_personas = list(
                    dict.fromkeys(personas)
                )  # Remove duplicates while preserving order
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

    def parse_multi_persona_sequence(self, content: str) -> Dict:
        """Parse content with multiple persona switches into structured sequence."""
        if not content:
            return {"is_multi_persona": False}

        persona_matches = []
        for match in self.persona_pattern.finditer(content):
            persona_key = match.group(1)
            if not self.case_sensitive:
                persona_key = persona_key.lower()

            # Skip per-model commands and reserved keywords
            command_keywords = {"agent", "list", "reset", "default", "normal", "multi"}
            if (
                persona_key.startswith("persona") and persona_key[7:].isdigit()
            ) or persona_key in command_keywords:
                continue

            persona_matches.append(
                {"persona": persona_key, "start": match.start(), "end": match.end()}
            )

        if len(persona_matches) < 1:
            return {"is_multi_persona": False}

        sequence = []
        for i, match in enumerate(persona_matches):
            task_start = match["end"]
            task_end = (
                persona_matches[i + 1]["start"]
                if i + 1 < len(persona_matches)
                else len(content)
            )
            task_content = content[task_start:task_end].strip()

            if i + 1 < len(persona_matches):
                next_persona_cmd = f"{self.prefix}{persona_matches[i + 1]['persona']}"
                if task_content.endswith(next_persona_cmd):
                    task_content = task_content[: -len(next_persona_cmd)].strip()

            sequence.append(
                {
                    "persona": match["persona"],
                    "task": (
                        task_content
                        if task_content
                        else "Please introduce yourself and explain your capabilities."
                    ),
                }
            )

        requested_personas = list(
            dict.fromkeys(match["persona"] for match in persona_matches)
        )

        return {
            "is_multi_persona": len(sequence) > 0,
            "is_single_persona": len(sequence) == 1,
            "sequence": sequence,
            "requested_personas": requested_personas,
        }

    def remove_commands(self, content: str) -> str:
        """Enhanced command removal that preserves multi-model commands"""
        # Remove persona commands but preserve !multi commands
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

    def get_personas(self, filepath: str) -> Dict:
        try:
            if not os.path.exists(filepath):
                return {}
            current_mtime = os.path.getmtime(filepath)
            if (
                filepath != self._last_path
                or current_mtime > self._file_mtime
                or not self._cache
            ):
                with open(filepath, "r", encoding="utf-8") as f:
                    self._cache = json.load(f)
                self._file_mtime = current_mtime
                self._last_path = filepath
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

        # Enhanced integration settings
        enable_plugin_integration: bool = Field(
            default=True, description="Enable integration with other plugins"
        )
        integration_debug: bool = Field(
            default=False, description="Debug integration communications"
        )

        # Backup settings
        enable_automatic_backups: bool = Field(
            default=True, description="Create automatic backups before updates"
        )
        max_backup_files: int = Field(
            default=10, description="Maximum number of backup files to keep"
        )

    def __init__(self):
        self.valves = self.Valves()
        self.toggle = True
        self.icon = "🎭"
        self.current_persona = None
        self.current_context = {}
        self.pattern_matcher = UniversalPatternMatcher(
            self.valves.keyword_prefix, self.valves.case_sensitive
        )
        self.persona_cache = PersonaCache()
        self.downloader = PersonaDownloader(self._get_config_path)
        self.integration_manager = PluginIntegrationManager()
        self._ensure_personas_available()
        self._handle_refresh()

    def _debug_log(self, message: str):
        if self.valves.enable_debug:
            print(f"[AGENT_HOTSWAP] {message}")

    def _integration_debug(self, message: str):
        if self.valves.integration_debug:
            print(f"[AGENT_HOTSWAP:INTEGRATION] {message}")

    def _handle_refresh(self):
        if getattr(self.valves, "refresh_personas", False):
            self._download_personas_async()
            self.valves.refresh_personas = False

    def _get_config_path(self) -> str:
        data_dir = os.getenv("DATA_DIR") or (
            "/app/backend/data"
            if os.path.exists("/app/backend")
            else str(Path.home() / ".local/share/open-webui")
        )
        return os.path.join(data_dir, "cache", "functions", CACHE_DIR, CONFIG_FILE)

    def _ensure_personas_available(self):
        config_path = self._get_config_path()
        if not os.path.exists(config_path) or os.path.getsize(config_path) < 5:
            os.makedirs(os.path.dirname(config_path), exist_ok=True)
            self._create_minimal_config(config_path)
        if self.valves.auto_download_personas:
            self._download_personas_async()

    def _create_minimal_config(self, config_path: str):
        """Creates a minimal config with master controller and only 2 essential personas."""
        try:
            minimal_config = {
                "_master_controller": {
                    "name": "🎛️ OpenWebUI Master Controller",
                    "hidden": True,
                    "always_active": True,
                    "prompt": """=== OPENWEBUI MASTER CONTROLLER ===
You operate in OpenWebUI with comprehensive native capabilities:

RENDERING: LaTeX ($$formula$$), Mermaid diagrams, HTML artifacts, SVG, enhanced Markdown
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
                # Only 2 essential personas - the rest will be downloaded automatically
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
                    "version": "minimal_v2.5.0",
                    "last_updated": datetime.now().isoformat(),
                    "integration_ready": True,
                    "auto_download_pending": True,
                },
            }

            with open(config_path, "w", encoding="utf-8") as f:
                json.dump(minimal_config, f, indent=4, ensure_ascii=False)
            print(
                "[AGENT_HOTSWAP] Minimal config created with 2 personas - downloading full collection..."
            )
        except Exception as e:
            print(f"[AGENT_HOTSWAP] Error creating minimal config: {e}")

    def _download_personas_async(self):
        try:
            import threading

            def download():
                try:
                    import asyncio

                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    result = loop.run_until_complete(
                        self.downloader.download_personas(
                            merge=self.valves.merge_on_update,
                            create_backup=self.valves.enable_automatic_backups,
                        )
                    )
                    if result["success"]:
                        backup_msg = (
                            " (backup created)" if result.get("backup_created") else ""
                        )
                        print(
                            f"[AGENT_HOTSWAP] Downloaded {result['count']} personas{backup_msg}"
                        )
                        self.persona_cache._cache = {}
                    else:
                        print(f"[AGENT_HOTSWAP] Download failed: {result['error']}")
                    loop.close()
                except Exception as e:
                    print(f"[AGENT_HOTSWAP] Download thread error: {e}")

            threading.Thread(target=download, daemon=True).start()
        except Exception as e:
            print(f"[AGENT_HOTSWAP] Could not start download: {e}")

    def _load_personas(self) -> Dict:
        return self.persona_cache.get_personas(self._get_config_path()) or {}

    def _safe_get_ids(self, body: dict, user: Optional[dict]):
        chat_id = body.get("chat_id") or body.get("id") or f"chat_{int(time.time())}"
        user_id = user.get("id", "anonymous") if user else "anonymous"
        return str(chat_id), str(user_id)

    def _create_system_message(self, persona_key: str) -> Dict:
        personas = self._load_personas()
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

    def _create_multi_persona_system(self, requested_personas: List[str]) -> Dict:
        """Create system message for multi-persona mode."""
        personas = self._load_personas()
        master = personas.get("_master_controller", {})
        system_content = master.get("prompt", "")

        persona_definitions = []
        valid_personas = []

        for persona_key in requested_personas:
            if persona_key in personas:
                persona_data = personas[persona_key]
                persona_name = persona_data.get("name", persona_key.title())
                persona_prompt = persona_data.get("prompt", "")
                valid_personas.append(persona_key)

                persona_definitions.append(
                    f"""
=== {persona_name.upper()} PERSONA ===
Activation Command: !{persona_key}
{persona_prompt}
=== END {persona_name.upper()} ===
"""
                )

        transition_instruction = ""
        if self.valves.multi_persona_transitions and self.valves.show_persona_info:
            transition_instruction = '3. Announce switches: "🎭 **[Persona Name]**"'
        else:
            transition_instruction = (
                "3. Switch personas seamlessly without announcements"
            )

        multi_persona_instructions = f"""

=== DYNAMIC MULTI-PERSONA MODE ===
Active Personas: {len(valid_personas)} loaded

{(''.join(persona_definitions))}

EXECUTION FRAMEWORK:
1. Parse user's persona sequence from their original message
2. When you encounter !{{persona}}, switch to that persona immediately
{transition_instruction}
4. Execute the task following each !command until the next !command
5. Maintain context flow between all switches
6. Available commands: {', '.join([f'!{p}' for p in valid_personas])}

Execute the user's multi-persona sequence seamlessly.
=== END DYNAMIC MULTI-PERSONA MODE ===
"""

        return {
            "role": "system",
            "content": system_content + multi_persona_instructions,
            "valid_personas": valid_personas,
        }

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

    def _generate_persona_list_html(self, personas: Dict) -> str:
        if not personas or not isinstance(personas, dict):
            return "<h3>Error: Personas file is empty or invalid.</h3>"

        display_personas = {k: v for k, v in personas.items() if not k.startswith("_")}
        if not display_personas:
            return "<h3>No personas available.</h3>"

        persona_count = len(display_personas)

        sorted_keys = sorted(display_personas.keys())
        grouped_personas = {}
        for key in sorted_keys:
            first_letter = key[0].upper()
            if first_letter not in grouped_personas:
                grouped_personas[first_letter] = []
            grouped_personas[first_letter].append(key)

        persona_grid_html = ""
        for letter in sorted(grouped_personas.keys()):
            persona_grid_html += f'<div class="card-full letter-group" id="group-{letter}"><h2>{letter}</h2></div>'
            for key in grouped_personas[letter]:
                name = display_personas[key].get("name", key.title())
                description = display_personas[key].get(
                    "description", "No description available."
                )
                command = f"{self.valves.keyword_prefix}{key}"
                safe_name = name.replace('"', '"')
                persona_grid_html += f"""<div class="card persona-card" data-id="{key.lower()}" data-name="{safe_name.lower()}"><h2>{name}</h2><p class="text-muted">{description}</p><div style="margin-top:auto;display:flex;justify-content:space-between;align-items:center;border-top:1px solid var(--border-light);padding-top:1rem"><code>{command}</code><button class="btn btn-secondary copy-btn" data-command="{command}">Copy</button></div></div>"""

        html_template = Template(
            """<!DOCTYPE html><html lang="en"><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1.0"><title>${persona_count} Available Personas</title><style>@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600&display=swap');:root{--font-primary:'Inter',sans-serif;--bg-color:#f8f9fa;--surface-color:#fff;--primary-text:#212529;--secondary-text:#495057;--accent:#0d6efd;--border:#dee2e6;--border-light:#e9ecef;--success:#198754;--code-bg:#e9ecef;--shadow:0 1px 3px rgba(0,0,0,.1)}html.dark{--bg-color:#111315;--surface-color:#1a1d21;--primary-text:#e4e4e7;--secondary-text:#a0a0a9;--accent:#3b82f6;--border:#363b42;--border-light:#2a2f36;--success:#22c55e;--code-bg:#1e2124;--shadow:0 1px 3px rgba(0,0,0,.2)}*{box-sizing:border-box;margin:0;padding:0}body{font-family:var(--font-primary);background:var(--bg-color);color:var(--primary-text);line-height:1.6;padding:1rem}.container{max-width:1200px;margin:0 auto}header{text-align:center;padding:2rem 0;border-bottom:1px solid var(--border);margin-bottom:2rem}.title{font-size:clamp(2.5rem,5vw,4rem);font-weight:600;margin-bottom:.5rem}.subtitle{color:var(--secondary-text);font-size:1.1rem;margin:0}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(350px,1fr));gap:1.5rem}.card{display:none;background:var(--surface-color);border:1px solid var(--border);border-radius:12px;padding:1.5rem;box-shadow:var(--shadow);transition:all .2s cubic-bezier(.4,0,.2,1);flex-direction:column}.card:hover{box-shadow:var(--shadow-lg);transform:translateY(-2px)}.card h2{font-size:1.25rem;font-weight:600;margin-bottom:1rem;padding-bottom:.5rem;border-bottom:1px solid var(--border-light)}.text-muted{color:var(--secondary-text);flex-grow:1}code{background:var(--code-bg);border:1px solid var(--border-light);border-radius:4px;padding:.2em .4em;font-family:monospace;font-size:.9em}input{width:100%;padding:.75rem;margin-bottom:1rem;background:var(--surface-color);border:1px solid var(--border);border-radius:6px;font-family:var(--font-primary);font-size:1rem;color:var(--primary-text)}html.dark input{background:#252830}.btn{display:inline-flex;align-items:center;gap:.5rem;padding:.5rem 1rem;border:none;border-radius:6px;font-size:.9rem;font-weight:600;cursor:pointer;transition:all .2s}.btn-secondary{background:#f8f9fa;color:var(--primary-text);border:1px solid var(--border)}html.dark .btn-secondary{background:#252830}.btn-secondary:hover{background:var(--surface-color)}.copy-btn.copied{background:var(--success);color:#fff;border-color:var(--success)}.card-full{grid-column:1/-1;background:#f8f9fa;border:1px solid var(--border);border-radius:12px;padding:.5rem 1.5rem;box-shadow:var(--shadow);display:none}html.dark .card-full{background:#252830}</style></head><body><div class=container><header><h1 class=title>${persona_count} Available Personas</h1><p class=subtitle>Search and copy commands to activate an AI persona.</p></header><input type=text id=search-bar placeholder="Search by name or command (e.g., 'coder', 'writer')..."><div class=grid>$persona_grid_html</div></div><script>document.addEventListener('DOMContentLoaded',function(){try{const e=parent.document.documentElement.classList.contains('dark');e&&document.documentElement.classList.add('dark')}catch(e){const t=localStorage.getItem('theme')||(window.matchMedia('(prefers-color-scheme: dark)').matches?'dark':'light');'dark'===t&&document.documentElement.classList.add('dark')}const e=document.getElementById('search-bar'),t=document.querySelectorAll('.persona-card'),o=document.querySelectorAll('.letter-group');t.forEach(e=>e.style.display='flex'),o.forEach(e=>e.style.display='block'),e.addEventListener('input',e=>{const c=e.target.value.toLowerCase();o.forEach(e=>{let o=0;let n=e.nextElementSibling;for(;n&&n.classList.contains('persona-card');){const t=n.dataset.id.includes(c)||n.dataset.name.includes(c);n.style.display=t?'flex':'none',t&&o++,n=n.nextElementSibling}e.style.display=o>0?'block':'none'})}),document.querySelectorAll('.copy-btn').forEach(e=>{e.addEventListener('click',()=>{const t=e.dataset.command;navigator.clipboard.writeText(t).then(()=>{e.textContent='Copied!',e.classList.add('copied'),'vibrate'in navigator&&navigator.vibrate(50),setTimeout(()=>{e.textContent='Copy',e.classList.remove('copied')},1500)})})})})</script></body></html>"""
        )

        return html_template.substitute(
            persona_grid_html=persona_grid_html, persona_count=persona_count
        )

    def _generate_help_message(self) -> str:
        p = self.valves.keyword_prefix
        return f"""### Agent Hotswap Commands

- **`{p}agent`**: Displays this help message.
- **`{p}agent list`**: Shows an interactive list of all available personas.
- **`{p}{{persona_name}}`**: Activates a specific persona (e.g., `{p}coder`).
- **`{p}reset`**: Resets to the default assistant.

**Multi-Persona Support:** Use multiple personas in one message!
Example: `{p}writer create a story {p}teacher explain techniques`

**Per-Model Personas:** Assign different personas to specific models!
Example: `{p}persona1 teacher {p}persona2 scientist {p}multi debate evolution`

**Integration Features:**
- Works seamlessly with Multi-Model conversations
- Supports conversation summarization with persona context
- Maintains persona state across long conversations"""

    async def _emit_status(self, emitter, message: str):
        if emitter and self.valves.show_persona_info:
            await emitter(
                {
                    "type": "status",
                    "data": {"description": message, "done": True, "timeout": 3000},
                }
            )

    def _create_integration_context(
        self, command_info: Dict, personas_data: Dict, **kwargs
    ) -> Dict:
        """Create comprehensive integration context for other plugins"""
        if not self.valves.enable_plugin_integration:
            return {}

        base_context = {
            "agent_hotswap_active": True,
            "agent_hotswap_version": "2.5.0",
            "command_info": command_info,
            "timestamp": time.time(),
        }

        if command_info["type"] == "per_model":
            # Enhanced per-model context
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

        elif command_info["type"] == "single_persona":
            integration_context = (
                self.integration_manager.prepare_single_persona_context(
                    command_info["persona"], personas_data
                )
            )
            base_context.update(integration_context)

        elif command_info["type"] == "multi_persona":
            sequence_data = self.pattern_matcher.parse_multi_persona_sequence(
                kwargs.get("original_content", "")
            )
            if sequence_data["is_multi_persona"]:
                integration_context = (
                    self.integration_manager.prepare_multi_persona_context(
                        sequence_data["sequence"], personas_data
                    )
                )
                base_context.update(integration_context)

        base_context.update(kwargs)

        self._integration_debug(
            f"Created integration context: {list(base_context.keys())}"
        )
        return base_context

    async def _handle_list_command(self, body: dict, emitter) -> dict:
        await self._emit_status(emitter, "📋 Generating interactive persona list...")
        personas = self._load_personas()
        html_content = self._generate_persona_list_html(personas)
        final_content = f"```html\n{html_content}\n```"

        # Send the artifact content using the streaming pattern
        await emitter(
            {"type": "message", "data": {"content": final_content, "done": False}}
        )

        # Send final completion chunk
        await emitter({"type": "message", "data": {"content": "", "done": True}})

        # Return body with empty messages to prevent LLM processing
        result = body.copy()
        result["messages"] = []
        result["_stop_processing"] = True
        return result

    async def _handle_help_command(self, body: dict, emitter) -> dict:
        help_content = self._generate_help_message()
        await self._emit_status(emitter, "ℹ️ Showing Agent Hotswap commands")

        # Send the help content using the streaming pattern
        await emitter(
            {"type": "message", "data": {"content": help_content, "done": False}}
        )

        # Send final completion chunk
        await emitter({"type": "message", "data": {"content": "", "done": True}})

        # Return body with empty messages to prevent LLM processing
        result = body.copy()
        result["messages"] = []
        result["_stop_processing"] = True
        return result

    async def _handle_reset_command(
        self,
        body: dict,
        messages: List[Dict],
        original_content: str,
        emitter,
        user_id: str,
        chat_id: str,
    ) -> dict:
        self.current_persona = None
        self.current_context = {}
        PersonaStorage.store_persona(user_id, chat_id, None)
        clean_messages = self._remove_persona_messages(messages)
        cleaned_content = self.pattern_matcher.remove_commands(original_content)

        reset_prompt = (
            "You have been reset. Please confirm you are in default assistant mode."
        )

        for msg in reversed(clean_messages):
            if msg.get("role") == "user":
                msg["content"] = (
                    f"{reset_prompt} Then help with: {cleaned_content}"
                    if cleaned_content
                    else reset_prompt
                )
                break

        body["messages"] = clean_messages

        # Clear integration context
        if self.valves.enable_plugin_integration:
            body["_filter_context"] = {"agent_hotswap_reset": True}

        await self._emit_status(emitter, "🔄 Reset to default assistant")
        return body

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
        personas = self._load_personas()
        if not personas or persona_key not in personas:
            return body

        self.current_persona = persona_key
        self.current_context = {"type": "single_persona", "persona": persona_key}
        PersonaStorage.store_persona(
            user_id, chat_id, persona_key, self.current_context
        )

        clean_messages = self._remove_persona_messages(messages)
        system_msg = self._create_system_message(persona_key)
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

        # Enhanced integration context
        if self.valves.enable_plugin_integration:
            body["_filter_context"] = self._create_integration_context(
                command_info, personas, original_content=original_content
            )

        persona_name = persona_config.get("name", persona_key.title())
        await self._emit_status(emitter, f"🎭 Switched to {persona_name}")
        return body

    async def _handle_multi_persona(
        self,
        personas_list: List[str],
        body: dict,
        messages: List[Dict],
        original_content: str,
        emitter,
        user_id: str,
        chat_id: str,
        command_info: Dict,
    ) -> dict:
        """Handle multi-persona sequences."""
        sequence_data = self.pattern_matcher.parse_multi_persona_sequence(
            original_content
        )

        if not sequence_data["is_multi_persona"]:
            return body

        personas_data = self._load_personas()
        valid_personas = [
            p for p in sequence_data["requested_personas"] if p in personas_data
        ]

        if not valid_personas:
            return body

        clean_messages = self._remove_persona_messages(messages)
        system_msg = self._create_multi_persona_system(valid_personas)
        clean_messages.insert(0, system_msg)

        # Build instructions for the sequence
        instructions = ["Execute this multi-persona sequence:\n"]
        for i, step in enumerate(sequence_data["sequence"], 1):
            persona_key = step["persona"]
            task = step["task"]
            if persona_key in personas_data:
                persona_name = personas_data[persona_key].get(
                    "name", persona_key.title()
                )
                instructions.append(f"**Step {i} - {persona_name}:**\n{task}\n")

        instructions.append(
            "\nExecute each step in sequence, following the persona switching framework."
        )

        for msg in reversed(clean_messages):
            if msg.get("role") == "user":
                msg["content"] = "\n".join(instructions)
                break

        body["messages"] = clean_messages

        # Update current persona state
        self.current_persona = f"multi:{':'.join(valid_personas)}"
        self.current_context = {
            "type": "multi_persona",
            "personas": valid_personas,
            "sequence": sequence_data["sequence"],
        }
        PersonaStorage.store_persona(
            user_id, chat_id, self.current_persona, self.current_context
        )

        # Enhanced integration context
        if self.valves.enable_plugin_integration:
            body["_filter_context"] = self._create_integration_context(
                command_info, personas_data, original_content=original_content
            )

        persona_names = [
            personas_data.get(p, {}).get("name", p.title()) for p in valid_personas
        ]
        await self._emit_status(
            emitter, f"🎭 Multi-persona: {' → '.join(persona_names)}"
        )
        return body

    async def _handle_per_model_personas(
        self,
        per_model_dict: Dict[int, str],
        body: dict,
        messages: List[Dict],
        original_content: str,
        emitter,
        user_id: str,
        chat_id: str,
        command_info: Dict,
    ) -> dict:
        """Enhanced per-model persona assignments with robust integration."""
        personas_data = self._load_personas()
        valid_assignments = {}

        for model_num, persona_key in per_model_dict.items():
            if persona_key in personas_data:
                valid_assignments[model_num] = {
                    "key": persona_key,
                    "name": personas_data[persona_key].get("name", persona_key.title()),
                }

        if not valid_assignments:
            await self._emit_status(emitter, "❌ No valid persona assignments found")
            return body

        # Clean message content (remove persona commands but preserve !multi)
        cleaned_content = self.pattern_matcher.remove_commands(original_content)

        for msg in reversed(messages):
            if msg.get("role") == "user":
                if cleaned_content:
                    msg["content"] = cleaned_content
                else:
                    persona_list = ", ".join(
                        [p["name"] for p in valid_assignments.values()]
                    )
                    msg["content"] = (
                        f"Please have each assigned persona ({persona_list}) introduce themselves."
                    )
                break

        # Enhanced integration context creation
        if self.valves.enable_plugin_integration:
            body["_filter_context"] = self._create_integration_context(
                command_info,
                personas_data,
                original_content=original_content,
                valid_assignments=valid_assignments,
            )

            self._integration_debug(
                f"Per-model context created with {len(valid_assignments)} assignments"
            )

        # Update persistence
        self.current_persona = (
            f"per_model:{':'.join([a['key'] for a in valid_assignments.values()])}"
        )
        self.current_context = {
            "type": "per_model",
            "assignments": valid_assignments,
            "has_multi_command": command_info.get("has_multi_command", False),
        }
        PersonaStorage.store_persona(
            user_id, chat_id, self.current_persona, self.current_context
        )

        assignments_text = []
        for model_num in sorted(valid_assignments.keys()):
            assignments_text.append(
                f"Model {model_num}: {valid_assignments[model_num]['name']}"
            )

        await self._emit_status(emitter, f"🎭 Per-model: {', '.join(assignments_text)}")
        return body

    def _apply_persistent_persona(self, body: dict, messages: List[Dict]) -> dict:
        if not self.valves.persistent_persona or not self.current_persona:
            return body

        # Skip if multi-persona or per-model mode
        if self.current_persona.startswith(("multi:", "per_model:")):
            return body

        personas = self._load_personas()
        if not personas or self.current_persona not in personas:
            return body

        # Check if system message already exists
        if any(
            "=== OPENWEBUI MASTER CONTROLLER ===" in m.get("content", "")
            for m in messages
            if m.get("role") == "system"
        ):
            return body

        clean_messages = self._remove_persona_messages(messages)
        system_msg = self._create_system_message(self.current_persona)
        clean_messages.insert(0, system_msg)
        body["messages"] = clean_messages

        # Add persistent persona context
        if self.valves.enable_plugin_integration and not body.get("_filter_context"):
            body["_filter_context"] = (
                self.integration_manager.prepare_single_persona_context(
                    self.current_persona, personas
                )
            )

        return body

    async def inlet(
        self, body: dict, __event_emitter__, __user__: Optional[dict] = None
    ) -> dict:
        if not self.toggle:
            return body

        messages = body.get("messages", [])
        if not messages:
            return body

        chat_id, user_id = self._safe_get_ids(body, __user__)

        # Restore persistent persona
        if (
            self.valves.persistent_persona
            and not self.current_persona
            and user_id != "anonymous"
        ):
            stored_persona, stored_context = PersonaStorage.get_persona(
                user_id, chat_id
            )
            if stored_persona:
                self.current_persona = stored_persona
                self.current_context = stored_context or {}

        # Find last user message
        last_user_content = ""
        for msg in reversed(messages):
            if msg.get("role") == "user":
                last_user_content = msg.get("content", "")
                break

        if not last_user_content:
            return self._apply_persistent_persona(body, messages)

        # Detect command type
        command_info = self.pattern_matcher.detect_command(last_user_content)

        # Handle special commands first (these need to stop processing completely)
        if command_info["type"] == "help":
            result = await self._handle_help_command(body, __event_emitter__)
            return result
        elif command_info["type"] == "list":
            result = await self._handle_list_command(body, __event_emitter__)
            return result

        # Handle other command types
        if command_info["type"] == "reset":
            return await self._handle_reset_command(
                body, messages, last_user_content, __event_emitter__, user_id, chat_id
            )
        elif command_info["type"] == "per_model":
            return await self._handle_per_model_personas(
                command_info["personas"],
                body,
                messages,
                last_user_content,
                __event_emitter__,
                user_id,
                chat_id,
                command_info,
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
        elif command_info["type"] == "multi_persona":
            return await self._handle_multi_persona(
                command_info["personas"],
                body,
                messages,
                last_user_content,
                __event_emitter__,
                user_id,
                chat_id,
                command_info,
            )
        else:
            return self._apply_persistent_persona(body, messages)

    async def outlet(
        self, body: dict, __event_emitter__, __user__: Optional[dict] = None
    ) -> dict:
        return body
