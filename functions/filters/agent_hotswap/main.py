"""
title: Agent Hotswap
description: Universal AI persona switching.
author: pkeffect
author_url: https://github.com/pkeffect/
project_url: https://github.com/pkeffect/functions/tree/main/functions/filters/agent_hotswap
funding_url: https://github.com/open-webui
required_open_webui_version: 0.6.0+
version: 2.9.0
date: 2025-08-02
license: MIT
requirements: pydantic>=2.0.0, aiofiles>=23.0.0, aiohttp>=3.8.0
"""

from pydantic import BaseModel, Field
from typing import Optional, Dict, List, Any, Union
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
import logging
import hashlib
from datetime import datetime
from pathlib import Path
from enum import Enum

# OpenWebUI Imports
try:
    from open_webui.models.users import Users
    from open_webui.models.models import Models

    NATIVE_DB_AVAILABLE = True
except ImportError:
    NATIVE_DB_AVAILABLE = False

# Configuration
CONFIG_FILE = "personas.json"
VERSIONS_FILE = "persona_versions.json"
UI_FILE = "index.html"
DEFAULT_REPO = "https://raw.githubusercontent.com/pkeffect/functions/refs/heads/main/functions/filters/agent_hotswap/personas.json"
UI_REPO = "https://raw.githubusercontent.com/pkeffect/functions/refs/heads/main/functions/filters/agent_hotswap/index.html"
TRUSTED_DOMAINS = ["github.com", "raw.githubusercontent.com"]

# Global cache for better performance
_GLOBAL_PERSONA_CACHE = {}
_CACHE_LOCK = asyncio.Lock()


class LogLevel(Enum):
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"


class StructuredLogger:
    """Enhanced logging system with structured output and filtering"""

    def __init__(self, name: str = "AGENT_HOTSWAP", enabled: bool = True):
        self.name = name
        self.enabled = enabled
        self.start_time = time.time()

    def _format_message(self, level: LogLevel, message: str, **kwargs) -> str:
        """Format structured log message"""
        timestamp = datetime.now().isoformat()
        duration = round((time.time() - self.start_time) * 1000, 2)

        log_data = {
            "timestamp": timestamp,
            "level": level.value,
            "component": self.name,
            "duration_ms": duration,
            "message": message,
            **kwargs,
        }

        # For console output, use readable format
        extra_info = []
        for key, value in kwargs.items():
            if key not in ["user_id", "chat_id", "action", "performance"]:
                continue
            extra_info.append(f"{key}={value}")

        extra_str = f" [{', '.join(extra_info)}]" if extra_info else ""
        return f"[{self.name}:{level.value}] {message}{extra_str}"

    def debug(self, message: str, **kwargs):
        if self.enabled:
            print(self._format_message(LogLevel.DEBUG, message, **kwargs))

    def info(self, message: str, **kwargs):
        if self.enabled:
            print(self._format_message(LogLevel.INFO, message, **kwargs))

    def warning(self, message: str, **kwargs):
        if self.enabled:
            print(self._format_message(LogLevel.WARNING, message, **kwargs))

    def error(self, message: str, **kwargs):
        if self.enabled:
            print(self._format_message(LogLevel.ERROR, message, **kwargs))


class PersonaVersion:
    """Manages persona versioning and change tracking"""

    def __init__(
        self, version: str = "1.0.0", description: str = "", author: str = "system"
    ):
        self.version = version
        self.description = description
        self.author = author
        self.timestamp = time.time()
        self.hash = None

    def calculate_hash(self, persona_data: Dict) -> str:
        """Calculate hash of persona content for change detection"""
        content = json.dumps(persona_data, sort_keys=True)
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    @staticmethod
    def increment_version(current: str, change_type: str = "patch") -> str:
        """Increment version number based on change type"""
        try:
            major, minor, patch = map(int, current.split("."))

            if change_type == "major":
                return f"{major + 1}.0.0"
            elif change_type == "minor":
                return f"{major}.{minor + 1}.0"
            else:  # patch
                return f"{major}.{minor}.{patch + 1}"
        except:
            return "1.0.0"


class PersonaVersionManager:
    """Manages persona version history and migrations"""

    def __init__(self, config_path_func, logger: StructuredLogger):
        self.get_config_path = config_path_func
        self.logger = logger

    def _get_versions_path(self) -> str:
        config_path = self.get_config_path()
        return os.path.join(os.path.dirname(config_path), VERSIONS_FILE)

    async def load_version_history(self) -> Dict:
        """Load version history from disk"""
        versions_path = self._get_versions_path()
        try:
            if os.path.exists(versions_path):
                async with aiofiles.open(versions_path, "r", encoding="utf-8") as f:
                    content = await f.read()
                    return json.loads(content)
        except Exception as e:
            self.logger.error(f"Failed to load version history: {e}")
        return {}

    async def save_version_history(self, history: Dict):
        """Save version history to disk"""
        versions_path = self._get_versions_path()
        try:
            os.makedirs(os.path.dirname(versions_path), exist_ok=True)
            async with aiofiles.open(versions_path, "w", encoding="utf-8") as f:
                await f.write(json.dumps(history, indent=2, ensure_ascii=False))
        except Exception as e:
            self.logger.error(f"Failed to save version history: {e}")

    async def create_version(
        self,
        persona_key: str,
        persona_data: Dict,
        change_type: str = "patch",
        description: str = "",
    ) -> str:
        """Create new version of a persona"""
        history = await self.load_version_history()

        if persona_key not in history:
            history[persona_key] = {"versions": [], "current": "1.0.0"}

        current_version = history[persona_key]["current"]
        new_version = PersonaVersion.increment_version(current_version, change_type)

        version_obj = PersonaVersion(new_version, description)
        version_obj.hash = version_obj.calculate_hash(persona_data)

        version_entry = {
            "version": new_version,
            "description": description,
            "author": "system",
            "timestamp": version_obj.timestamp,
            "hash": version_obj.hash,
            "data": persona_data,
        }

        history[persona_key]["versions"].append(version_entry)
        history[persona_key]["current"] = new_version

        await self.save_version_history(history)
        self.logger.info(
            f"Created persona version {new_version}",
            persona=persona_key,
            change_type=change_type,
        )
        return new_version

    async def get_persona_version(
        self, persona_key: str, version: str = None
    ) -> Optional[Dict]:
        """Get specific version of a persona"""
        history = await self.load_version_history()

        if persona_key not in history:
            return None

        target_version = version or history[persona_key]["current"]

        for version_entry in history[persona_key]["versions"]:
            if version_entry["version"] == target_version:
                return version_entry["data"]

        return None

    async def list_persona_versions(self, persona_key: str) -> List[Dict]:
        """List all versions of a persona"""
        history = await self.load_version_history()

        if persona_key not in history:
            return []

        return sorted(
            history[persona_key]["versions"], key=lambda x: x["timestamp"], reverse=True
        )


class StreamPersonaManager:
    """Manages real-time persona switching during streaming responses"""

    def __init__(self, pattern_matcher, logger: StructuredLogger):
        self.pattern_matcher = pattern_matcher
        self.logger = logger
        self.active_streams = {}

    async def detect_stream_transitions(self, user_message: str) -> List[Dict]:
        """Detect persona transition points in user message"""
        transitions = []

        # Look for multiple persona commands in sequence
        matches = list(self.pattern_matcher._patterns["persona"].finditer(user_message))

        if len(matches) > 1:
            for i, match in enumerate(matches):
                persona_key = match.group(1).lower()
                start_pos = match.start()
                end_pos = (
                    matches[i + 1].start()
                    if i + 1 < len(matches)
                    else len(user_message)
                )

                # Extract the text segment for this persona
                segment = user_message[start_pos:end_pos].strip()
                segment = self.pattern_matcher.remove_commands(segment).strip()

                transitions.append(
                    {
                        "persona": persona_key,
                        "segment": segment,
                        "order": i,
                        "start_pos": start_pos,
                        "end_pos": end_pos,
                    }
                )

        return transitions

    async def should_transition(self, current_content: str, target_length: int) -> bool:
        """Determine if we should transition to next persona based on content analysis"""
        # Simple heuristic: transition when we've generated roughly the expected amount
        # or when we detect natural completion markers

        completion_markers = [
            ". Now,",
            ". Next,",
            ". Moving on,",
            ". In conclusion,",
            ".\n\n",
            "?\n\n",
            "!\n\n",
        ]

        if len(current_content) >= target_length * 0.8:  # 80% of expected content
            return True

        for marker in completion_markers:
            if marker in current_content[-100:]:  # Check last 100 chars
                return True

        return False


class BackendCompatibility:
    """Handles backend compatibility detection and fallbacks"""

    def __init__(self):
        self._custom_fields_supported = None
        self._test_performed = False

    def supports_custom_fields(self) -> bool:
        """Detect if backend accepts _filter_context"""
        if self._test_performed:
            return self._custom_fields_supported

        # Simple heuristic based on environment
        # Docker containers typically support custom fields
        self._custom_fields_supported = os.path.exists(
            "/app/backend"
        ) or os.path.exists("/app/frontend")
        self._test_performed = True
        return self._custom_fields_supported

    def get_compatibility_info(self) -> Dict[str, Any]:
        """Get detailed compatibility information"""
        return {
            "custom_fields_supported": self.supports_custom_fields(),
            "native_db_available": NATIVE_DB_AVAILABLE,
            "docker_environment": os.path.exists("/app"),
            "data_dir": os.getenv("DATA_DIR", "not_set"),
        }


class PluginIntegrationManager:
    """Manages integration with other plugins in the suite"""

    @staticmethod
    def create_integration_context(
        persona_data: Dict, command_type: str, **kwargs
    ) -> Dict:
        context = {
            "agent_hotswap_version": "2.9.0",
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

    @staticmethod
    async def store_context(user_id: str, chat_id: str, context: Dict):
        """Store context in user metadata as fallback when _filter_context isn't supported"""
        if not NATIVE_DB_AVAILABLE or not user_id or not chat_id:
            return
        try:
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None,
                PersonaStorage._sync_store_context,
                user_id,
                chat_id,
                context,
            )
        except Exception as e:
            print(f"[PersonaStorage] Context store error: {e}")

    @staticmethod
    def _sync_store_context(user_id: str, chat_id: str, context: Dict):
        try:
            user = Users.get_user_by_id(user_id)
            if user:
                metadata = user.info or {}
                context_state = metadata.get("filter_context", {})
                context_state[chat_id] = {
                    "context": context,
                    "timestamp": time.time(),
                }
                metadata["filter_context"] = context_state
                Users.update_user_by_id(user_id, {"info": metadata})
        except Exception as e:
            print(f"[PersonaStorage] Sync context store error: {e}")


class PersonaDownloader:
    def __init__(self, config_path_func, logger: StructuredLogger):
        self.get_config_path = config_path_func
        self.logger = logger
        self._session = None

    async def _get_session(self):
        if self._session is None or self._session.closed:
            timeout = aiohttp.ClientTimeout(total=30)
            self._session = aiohttp.ClientSession(
                timeout=timeout, headers={"User-Agent": "OpenWebUI-AgentHotswap/2.9.0"}
            )
        return self._session

    async def close_session(self):
        if self._session and not self._session.closed:
            await self._session.close()

    def is_trusted_domain(self, url: str) -> bool:
        try:
            parsed = urllib.parse.urlparse(url)
            return parsed.scheme == "https" and parsed.netloc.lower() in TRUSTED_DOMAINS
        except Exception as e:
            self.logger.error(f"URL validation error: {e}")
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
            self.logger.warning(
                f"UI download failed: {ui_result.get('error', 'unknown')}"
            )
        elif not ui_result.get("skipped"):
            self.logger.info(f"UI file downloaded: {ui_result.get('size', 0)} bytes")

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

                # Create backup if requested and file exists
                if create_backup and os.path.exists(config_path):
                    try:
                        backup_path = f"{config_path}.backup.{int(time.time())}"
                        shutil.copy2(config_path, backup_path)
                        self.logger.info(f"Backup created: {backup_path}")
                    except Exception as e:
                        self.logger.error(f"Backup failed: {e}")

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

        # More robust pattern compilation with better error handling
        try:
            self._patterns["persona"] = re.compile(
                rf"{prefix_escaped}([a-zA-Z][a-zA-Z0-9_]*)\b", flags
            )
            self._patterns["per_model"] = re.compile(
                rf"{prefix_escaped}persona(\d+)\s+([a-zA-Z][a-zA-Z0-9_]*)\b", flags
            )
            self._patterns["agent_list"] = re.compile(
                rf"{prefix_escaped}agent\s+list\b", flags
            )
            self._patterns["agent_base"] = re.compile(
                rf"{prefix_escaped}agent\b", flags
            )
            self._patterns["reset"] = re.compile(
                rf"{prefix_escaped}(?:reset|default|normal)\b", flags
            )
            self._patterns["multi"] = re.compile(rf"{prefix_escaped}multi\b", flags)
        except re.error as e:
            print(f"[PatternMatcher] Regex compilation error: {e}")
            # Fallback to simple patterns
            self._patterns = {
                "persona": re.compile(rf"{prefix_escaped}(\w+)", flags),
                "agent_list": re.compile(rf"{prefix_escaped}agent\s+list", flags),
                "agent_base": re.compile(rf"{prefix_escaped}agent", flags),
                "reset": re.compile(rf"{prefix_escaped}reset", flags),
                "multi": re.compile(rf"{prefix_escaped}multi", flags),
                "per_model": re.compile(
                    rf"{prefix_escaped}persona(\d+)\s+(\w+)", flags
                ),
            }

    async def detect_command(self, content: str) -> Dict[str, Any]:
        if not content or not isinstance(content, str):
            return {"type": "none"}

        try:
            if self._patterns["agent_list"].search(content):
                return {"type": "list"}
            if self._patterns["agent_base"].search(content):
                return {"type": "help"}
            if self._patterns["reset"].search(content):
                return {"type": "reset"}

            has_multi = bool(self._patterns["multi"].search(content))
            per_model_matches = {}

            for match in self._patterns["per_model"].finditer(content):
                try:
                    model_num = int(match.group(1))
                    persona_key = match.group(2)
                    if not self.case_sensitive:
                        persona_key = persona_key.lower()
                    if 1 <= model_num <= 4:
                        per_model_matches[model_num] = persona_key
                except (ValueError, IndexError) as e:
                    print(f"[PatternMatcher] Per-model parsing error: {e}")
                    continue

            if per_model_matches:
                return {
                    "type": "per_model",
                    "personas": per_model_matches,
                    "has_multi_command": has_multi,
                }

            matches = self._patterns["persona"].findall(content)
            if matches:
                command_keywords = {
                    "agent",
                    "list",
                    "reset",
                    "default",
                    "normal",
                    "multi",
                }
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
        except Exception as e:
            print(f"[PatternMatcher] Command detection error: {e}")
            return {"type": "none"}

    def remove_commands(self, content: str) -> str:
        if not content or not isinstance(content, str):
            return ""

        try:
            persona_commands_pattern = re.compile(
                rf"{re.escape(self.prefix)}(agent(\s+list)?|reset|default|normal|persona\d+\s+[a-zA-Z][a-zA-Z0-9_]*|(?!multi)[a-zA-Z][a-zA-Z0-9_]*)\b\s*",
                re.IGNORECASE,
            )
            return persona_commands_pattern.sub("", content).strip()
        except Exception as e:
            print(f"[PatternMatcher] Command removal error: {e}")
            return content


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

                    # Store in global cache with size management
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
        priority: int = Field(default=0, title="Priority")
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
        force_legacy_mode: bool = Field(
            default=False,
            description="Force legacy compatibility mode for older OpenWebUI versions",
        )
        # Enhanced logging options
        log_level: str = Field(
            default="INFO", description="Logging level: DEBUG, INFO, WARNING, ERROR"
        )
        enable_structured_logging: bool = Field(
            default=True, description="Enable structured logging with metadata"
        )
        # Persona versioning options
        enable_persona_versioning: bool = Field(
            default=True, description="Track persona versions and changes"
        )
        auto_version_on_update: bool = Field(
            default=True,
            description="Automatically create versions when personas are updated",
        )
        # Enhanced streaming enhancements
        enable_stream_switching: bool = Field(
            default=False,
            description="Enable real-time persona switching during streaming (experimental)",
        )
        stream_transition_threshold: float = Field(
            default=0.8,
            description="Content completion threshold for stream transitions (0.0-1.0)",
        )

    def __init__(self):
        # First, create the valves instance
        self.valves = self.Valves()

        # Initialize structured logger
        self.logger = StructuredLogger(
            "AGENT_HOTSWAP", enabled=self.valves.enable_structured_logging
        )

        # Get version info
        self._openwebui_version = self._get_openwebui_version()

        # The rest of the original __init__ method follows
        self.toggle = True
        self.icon = "🎭"
        self.plugin_directory_name = self._get_plugin_directory_name()
        self.current_persona = None
        self.current_context = {}
        self.pattern_matcher = UniversalPatternMatcher(
            self.valves.keyword_prefix, self.valves.case_sensitive
        )
        self.persona_cache = PersonaCache()
        self.downloader = PersonaDownloader(self._get_config_path, self.logger)
        self.integration_manager = PluginIntegrationManager()
        self.compatibility = BackendCompatibility()

        # Enhanced components
        self.version_manager = PersonaVersionManager(self._get_config_path, self.logger)
        self.stream_manager = StreamPersonaManager(self.pattern_matcher, self.logger)

        self._init_task = None
        self._initialized = False

        self.logger.info(
            "Agent Hotswap v2.9.0 initialized",
            plugin_dir=self.plugin_directory_name,
            openwebui_version=self._openwebui_version,
            stream_switching=self.valves.enable_stream_switching,
        )

    def __del__(self):
        try:
            if hasattr(self, "downloader") and self.downloader._session:
                asyncio.create_task(self.downloader.close_session())
        except:
            pass

    def _get_openwebui_version(self) -> str:
        """
        Tries to find and parse the OpenWebUI version from its package.json.
        This is most reliable in standard Docker installations.
        """
        try:
            # Try multiple possible paths for package.json
            possible_paths = [
                "/app/frontend/package.json",
                "/app/package.json",
                "./frontend/package.json",
                "./package.json",
            ]

            for package_json_path in possible_paths:
                if os.path.exists(package_json_path):
                    try:
                        with open(package_json_path, "r", encoding="utf-8") as f:
                            data = json.load(f)

                        version = data.get("version")
                        if version:
                            return f"v{version}"
                    except (json.JSONDecodeError, IOError) as e:
                        self.logger.debug(f"Error reading {package_json_path}: {e}")
                        continue

            # If no package.json found, try to get from environment or other sources
            return "Not Detected"

        except Exception as e:
            self.logger.error(f"Error in version detection: {e}")
            return "Error"

    def _get_plugin_directory_name(self) -> str:
        try:
            if __name__ != "__main__":
                module_parts = __name__.split(".")
                detected_name = module_parts[-1]
                if detected_name.startswith("function_"):
                    detected_name = detected_name[9:]
                cleaned_name = re.sub(r"[^a-zA-Z0-9_-]", "_", detected_name.lower())
                if cleaned_name and cleaned_name != "__main__":
                    self.logger.debug(f"Auto-detected plugin name: {cleaned_name}")
                    return cleaned_name
        except Exception as e:
            self.logger.error(f"Method 1 (__name__) failed: {e}")

        fallback_name = "agent_hotswap"
        self.logger.debug(f"Using fallback plugin name: {fallback_name}")
        return fallback_name

    def _debug_log(self, message: str, **kwargs):
        if self.valves.enable_debug:
            self.logger.debug(message, **kwargs)

    def _integration_debug(self, message: str, **kwargs):
        if self.valves.integration_debug:
            self.logger.debug(f"INTEGRATION: {message}", **kwargs)

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
            self.logger.debug(f"Generated relative URL: {relative_url}")
            return relative_url
        except Exception as e:
            self.logger.error(f"Error generating URL: {e}")
            return f"/cache/functions/{self.plugin_directory_name}/index.html"

    async def _initialize_plugin(self):
        """Consolidated initialization with proper error handling"""
        if self._initialized:
            return

        try:
            start_time = time.time()
            self.logger.info("Starting plugin initialization...")

            # Update logger settings
            if hasattr(self.valves, "log_level"):
                # Update logger configuration based on valves
                pass  # Logger level handling could be enhanced here

            # Ensure personas are available
            await self._ensure_personas_available()

            # Update pattern matcher if prefix changed
            if self.pattern_matcher.prefix != self.valves.keyword_prefix:
                self.pattern_matcher = UniversalPatternMatcher(
                    self.valves.keyword_prefix, self.valves.case_sensitive
                )

            # Clean old backup files if enabled
            if self.valves.enable_automatic_backups:
                await self._cleanup_old_backups()
                # Check if we need a periodic backup
                await self._periodic_backup_check()

            init_time = round((time.time() - start_time) * 1000, 2)
            self._initialized = True
            self.logger.info(
                "Plugin initialization completed",
                duration_ms=init_time,
                versioning_enabled=self.valves.enable_persona_versioning,
                streaming_enabled=self.valves.enable_stream_switching,
            )

        except Exception as e:
            self.logger.error(f"Initialization error: {e}")
            # Don't prevent the plugin from working if initialization fails
            self._initialized = True

    async def _create_backup(self, reason: str = "manual") -> bool:
        """Create a backup of the current personas file"""
        try:
            config_path = self._get_config_path()
            if not os.path.exists(config_path):
                self.logger.debug("No personas file to backup")
                return False

            timestamp = int(time.time())
            backup_path = f"{config_path}.backup.{timestamp}"

            # Create backup
            shutil.copy2(config_path, backup_path)

            self.logger.info(
                f"Backup created: {reason}",
                backup_path=backup_path,
                original_size=os.path.getsize(config_path),
            )

            # Clean up old backups
            await self._cleanup_old_backups()

            return True

        except Exception as e:
            self.logger.error(f"Backup creation failed: {e}", reason=reason)
            return False

    async def _should_create_backup(self) -> bool:
        """Check if we should create a backup based on time and changes"""
        try:
            config_path = self._get_config_path()
            if not os.path.exists(config_path):
                return False

            # Check if we have any recent backups
            backup_pattern = f"{config_path}.backup.*"
            backup_files = glob.glob(backup_pattern)

            if not backup_files:
                self.logger.info("No existing backups found, creating initial backup")
                return True

            # Get the most recent backup
            backup_files.sort(key=os.path.getmtime, reverse=True)
            latest_backup = backup_files[0]

            # Check if current file is different from latest backup
            try:
                async with aiofiles.open(config_path, "r", encoding="utf-8") as f:
                    current_content = await f.read()
                async with aiofiles.open(latest_backup, "r", encoding="utf-8") as f:
                    backup_content = await f.read()

                if current_content != backup_content:
                    self.logger.info("Personas file changed, backup needed")
                    return True

            except Exception as e:
                self.logger.warning(f"Could not compare with latest backup: {e}")
                return True

            # Check if latest backup is older than 7 days
            backup_age = time.time() - os.path.getmtime(latest_backup)
            if backup_age > (7 * 24 * 60 * 60):  # 7 days
                self.logger.info(
                    "Latest backup is older than 7 days, creating new backup"
                )
                return True

            return False

        except Exception as e:
            self.logger.error(f"Error checking backup status: {e}")
            return True  # When in doubt, backup
        """Clean up old backup files to prevent disk space issues"""
        try:
            config_path = self._get_config_path()
            backup_pattern = f"{config_path}.backup.*"
            backup_files = glob.glob(backup_pattern)

            if len(backup_files) > self.valves.max_backup_files:
                # Sort by modification time and remove oldest
                backup_files.sort(key=os.path.getmtime)
                files_to_remove = backup_files[: -self.valves.max_backup_files]
                for backup_file in files_to_remove:
                    try:
                        os.remove(backup_file)
                        self.logger.debug(f"Removed old backup: {backup_file}")
                    except Exception as e:
                        self.logger.error(f"Failed to remove backup {backup_file}: {e}")
        except Exception as e:
            self.logger.error(f"Backup cleanup error: {e}")

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
                self.logger.error(f"Error reading existing config: {e}")
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
                    "version": "minimal_v2.9.0",
                    "last_updated": datetime.now().isoformat(),
                    "integration_ready": True,
                    "plugin_directory": self.plugin_directory_name,
                    "initial_setup": True,
                    "async_enabled": True,
                    "popup_ui": True,
                    "enhanced_features": {
                        "streaming": True,
                        "versioning": True,
                        "structured_logging": True,
                    },
                },
            }
            async with aiofiles.open(config_path, "w", encoding="utf-8") as f:
                await f.write(json.dumps(minimal_config, indent=4, ensure_ascii=False))

            self.logger.info(
                f"Minimal config created",
                plugin_dir=self.plugin_directory_name,
                personas_count=2,
            )

            # Create initial versions if versioning is enabled
            if self.valves.enable_persona_versioning:
                for key, data in minimal_config.items():
                    if not key.startswith("_"):
                        await self.version_manager.create_version(
                            key, data, "major", "Initial persona creation"
                        )

        except Exception as e:
            self.logger.error(f"Error creating minimal config: {e}")

    async def _download_personas_async(self, force_download: bool = False):
        try:
            start_time = time.time()
            result = await self.downloader.download_personas(
                merge=self.valves.merge_on_update,
                create_backup=self.valves.enable_automatic_backups,
                force_download=force_download,
            )

            download_time = round((time.time() - start_time) * 1000, 2)

            if result["success"]:
                ui_msg = ""
                if result.get("ui_downloaded"):
                    ui_msg = " + UI downloaded"
                elif result.get("ui_status") == "failed":
                    ui_msg = " (UI download failed - will use fallback)"

                self.logger.info(
                    f"Downloaded {result['count']} personas{ui_msg}",
                    duration_ms=download_time,
                    size_bytes=result.get("size", 0),
                )
                global _GLOBAL_PERSONA_CACHE
                _GLOBAL_PERSONA_CACHE.clear()

                # Auto-version new personas if enabled
                if (
                    self.valves.enable_persona_versioning
                    and self.valves.auto_version_on_update
                ):
                    await self._auto_version_updated_personas()

            else:
                self.logger.error(f"Download failed: {result['error']}")
        except Exception as e:
            self.logger.error(f"Download error: {e}")

    async def _auto_version_updated_personas(self):
        """Automatically create versions for updated personas"""
        try:
            personas = await self._load_personas()
            for key, data in personas.items():
                if not key.startswith("_"):
                    await self.version_manager.create_version(
                        key, data, "minor", "Auto-update from repository"
                    )
        except Exception as e:
            self.logger.error(f"Auto-versioning error: {e}")

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
            self.logger.error(f"Status emit error: {e}")

    async def _update_personas_data_file(self, personas: Dict):
        """Create a separate personas.json file that the UI can load"""
        try:
            ui_dir = os.path.dirname(self._get_ui_path())
            personas_json_path = os.path.join(ui_dir, "personas.json")

            # Ensure directory exists
            os.makedirs(ui_dir, exist_ok=True)

            # Write personas data as a separate JSON file
            async with aiofiles.open(personas_json_path, "w", encoding="utf-8") as f:
                await f.write(json.dumps(personas, indent=2, ensure_ascii=False))

            # Verify the file was written
            if os.path.exists(personas_json_path):
                file_size = os.path.getsize(personas_json_path)
                self.logger.debug(f"Updated personas.json ({file_size} bytes)")
            else:
                self.logger.error("Failed to create personas.json")

        except Exception as e:
            self.logger.error(f"Error updating personas.json: {e}")

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
        compatibility_info = self.compatibility.get_compatibility_info()

        return f"""### Agent Hotswap Commands

- **`{p}agent`**: Displays this help message.
- **`{p}agent list`**: Opens interactive persona browser in new window.
- **`{p}{{persona_name}}`**: Activates a specific persona (e.g., `{p}coder`).
- **`{p}reset`**: Resets to the default assistant.

**Multi-Persona Support:** Use multiple personas in one message!
Example: `{p}writer create a story {p}teacher explain techniques`

**Per-Model Personas:** Assign different personas to specific models!
Example: `{p}persona1 teacher {p}persona2 scientist {p}multi debate evolution`

**Enhanced Features:**
- **Stream Switching:** {"✅ ON" if self.valves.enable_stream_switching else "❌ OFF (Classic Mode)"}
- **Persona Versioning:** {"✅ ON" if self.valves.enable_persona_versioning else "❌ OFF"}
- **Structured Logging:** {"✅ ON" if self.valves.enable_structured_logging else "❌ OFF"}
- **Global Caching:** {"✅ ON" if self.valves.enable_global_cache else "❌ OFF"}
- **Custom Persona Protection:** {protection_status}

**Performance Features:**
- **Async Operations:** ✅ ENABLED
- **UI Mode:** Popup Window
- **Backend Compatibility:** {"✅ MODERN" if compatibility_info['custom_fields_supported'] else "⚠️ LEGACY"}

**Integration Features:**
- Works seamlessly with Multi-Model conversations
- Supports conversation summarization with persona context
- Maintains persona state across long conversations

**System Info:**
- **OpenWebUI Version:** {self._openwebui_version}
- **Plugin Version:** 2.9.0
- **Directory:** `{self.plugin_directory_name}`
- **Environment:** {"🐳 Docker" if compatibility_info['docker_environment'] else "💻 Local"}"""

    async def _safe_set_context(
        self, body: dict, context: dict, user_id: str, chat_id: str
    ):
        """Safely set context with fallback strategies"""
        try:
            # Check if we should force legacy mode or if backend doesn't support custom fields
            if (
                self.valves.force_legacy_mode
                or not self.compatibility.supports_custom_fields()
            ):
                self._integration_debug(
                    "Using legacy mode - storing context in metadata",
                    user_id=user_id,
                    chat_id=chat_id,
                )
                await PersonaStorage.store_context(user_id, chat_id, context)
                body["_context_stored"] = True
                return True
            else:
                self._integration_debug(
                    "Using modern mode - setting _filter_context",
                    user_id=user_id,
                    chat_id=chat_id,
                )
                body["_filter_context"] = context
                return True
        except Exception as e:
            self.logger.error(
                f"Context setting failed: {e}", user_id=user_id, chat_id=chat_id
            )
            # Fallback to metadata storage
            try:
                await PersonaStorage.store_context(user_id, chat_id, context)
                body["_context_stored"] = True
                return True
            except Exception as fallback_error:
                self.logger.error(f"Context fallback also failed: {fallback_error}")
                return False

    async def _create_system_message(self, persona_key: str) -> Dict:
        personas = await self._load_personas()
        if not personas:
            return {"role": "system", "content": "Error: Persona file not loaded."}

        master = personas.get("_master_controller", {})
        master_prompt = master.get("prompt", "")

        # Check for versioned persona if versioning is enabled
        if self.valves.enable_persona_versioning:
            persona_data = await self.version_manager.get_persona_version(persona_key)
            if not persona_data:
                persona_data = personas.get(persona_key, {})
        else:
            persona_data = personas.get(persona_key, {})

        persona_prompt = persona_data.get(
            "prompt", f"You are the {persona_key} persona."
        )
        system_content = f"{master_prompt}\n\n{persona_prompt}"

        if self.valves.show_persona_info:
            persona_name = persona_data.get("name", persona_key.title())
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
            "agent_hotswap_version": "2.9.0",
            "command_info": command_info,
            "timestamp": time.time(),
            "plugin_directory": self.plugin_directory_name,
            "async_enabled": True,
            "popup_ui": True,
            "enhanced_features": {
                "streaming": self.valves.enable_stream_switching,
                "versioning": self.valves.enable_persona_versioning,
                "structured_logging": self.valves.enable_structured_logging,
            },
            "compatibility_mode": (
                "legacy" if self.valves.force_legacy_mode else "modern"
            ),
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
            self.logger.debug(f"Loaded {len(personas)} total personas")
            display_personas = {
                k: v for k, v in personas.items() if not k.startswith("_")
            }

            ui_path = self._get_ui_path()

            if not os.path.exists(ui_path):
                self.logger.debug("UI file not found, downloading...")
                result = await self.downloader.download_ui_file()
                if not result["success"]:
                    self.logger.warning(f"UI download failed: {result.get('error')}")
                    return await self._handle_list_command_fallback(body, emitter)
            else:
                self.logger.debug("UI file exists")

            await self._update_personas_data_file(personas)

            hub_url = self._get_relative_hub_url()

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
            self.logger.error(f"Error in list command: {e}")
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
            self.logger.error(f"Error in fallback: {e}")
            error_message = f"Error loading persona list: {str(e)}"
            result = body.copy()
            result["messages"] = [{"role": "assistant", "content": error_message}]
            result["_agent_hotswap_handled"] = True
            return result

    async def _handle_help_command(self, body: dict, emitter) -> dict:
        try:
            help_content = self._generate_help_message()
            await self._emit_status(
                emitter, "ℹ️ Showing Agent Hotswap commands", "complete", done=True
            )

            if emitter:
                await emitter({"type": "message", "data": {"content": help_content}})

            result = body.copy()
            result["messages"] = [{"role": "assistant", "content": help_content}]
            result["_agent_hotswap_handled"] = True
            return result

        except Exception as e:
            self.logger.error(f"Error in help command: {e}")
            error_message = f"Error generating help: {str(e)}"
            if emitter:
                await emitter({"type": "message", "data": {"content": error_message}})
            result = body.copy()
            result["messages"] = [{"role": "assistant", "content": error_message}]
            result["_agent_hotswap_handled"] = True
            return result

    async def _handle_reset_command(
        self, body: dict, emitter, user_id: str, chat_id: str
    ) -> dict:
        try:
            self.current_persona = None
            self.current_context = {}

            # Clear stored persona
            await PersonaStorage.store_persona(user_id, chat_id, None, None)

            await self._emit_status(
                emitter, "🔄 Reset to default assistant", "complete"
            )

            self.logger.info(
                "Persona reset to default", user_id=user_id, chat_id=chat_id
            )

            result = body.copy()
            # Remove any existing persona messages
            clean_messages = self._remove_persona_messages(result.get("messages", []))

            # Update the user's message to request a fresh introduction
            for msg in reversed(clean_messages):
                if msg.get("role") == "user":
                    msg["content"] = (
                        "Please introduce yourself as the default OpenWebUI assistant."
                    )
                    break

            result["messages"] = clean_messages
            result["_agent_hotswap_handled"] = True
            return result

        except Exception as e:
            self.logger.error(f"Error in reset command: {e}")
            error_message = "Reset to default assistant."
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
            await self._emit_status(
                emitter, f"❌ Persona '{persona_key}' not found", "error"
            )
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

        # Use safe context setting
        if self.valves.enable_plugin_integration:
            context = await self._create_integration_context(
                command_info, personas, original_content=original_content
            )
            await self._safe_set_context(body, context, user_id, chat_id)

        persona_name = persona_config.get("name", persona_key.title())
        await self._emit_status(emitter, f"🎭 Switched to {persona_name}", "complete")

        self.logger.info(
            f"Persona switched to {persona_key}",
            user_id=user_id,
            chat_id=chat_id,
            persona_name=persona_name,
        )
        return body

    async def _handle_multi_persona_streaming(
        self,
        body: dict,
        messages: List[Dict],
        original_content: str,
        emitter,
        user_id: str,
        chat_id: str,
    ) -> dict:
        """Handle multi-persona streaming transitions with fallback to classic mode"""
        try:
            # Detect transition points
            transitions = await self.stream_manager.detect_stream_transitions(
                original_content
            )

            if len(transitions) <= 1:
                return body  # Not a multi-persona request

            self.logger.info(
                f"Multi-persona sequence detected with {len(transitions)} transitions",
                user_id=user_id,
                chat_id=chat_id,
            )

            # Check if streaming is enabled and supported
            if self.valves.enable_stream_switching:
                # Try streaming approach
                streaming_context = {
                    "type": "multi_persona_streaming",
                    "transitions": transitions,
                    "current_transition": 0,
                    "accumulated_content": "",
                    "threshold": self.valves.stream_transition_threshold,
                }

                # Set special context for streaming
                await self._safe_set_context(body, streaming_context, user_id, chat_id)

                # Start with the first persona
                first_persona = transitions[0]["persona"]
                system_msg = await self._create_system_message(first_persona)

                clean_messages = self._remove_persona_messages(messages)
                clean_messages.insert(0, system_msg)

                # Update user message to only include first segment
                for msg in reversed(clean_messages):
                    if msg.get("role") == "user":
                        msg["content"] = transitions[0]["segment"]
                        break

                body["messages"] = clean_messages
                body["_streaming_personas"] = streaming_context

                await self._emit_status(
                    emitter, f"🎭 Starting multi-persona sequence", "in_progress"
                )

                return body
            else:
                # Use classic multi-persona approach (immediate sequential response)
                return await self._handle_multi_persona_classic(
                    transitions, body, messages, emitter, user_id, chat_id
                )

        except Exception as e:
            self.logger.error(f"Error in multi-persona processing: {e}")
            return body

    async def _handle_multi_persona_classic(
        self,
        transitions: List[Dict],
        body: dict,
        messages: List[Dict],
        emitter,
        user_id: str,
        chat_id: str,
    ) -> dict:
        """Handle multi-persona requests with classic immediate sequential approach"""
        try:
            await self._emit_status(
                emitter, "🎭 Processing multi-persona sequence", "in_progress"
            )

            personas = await self._load_personas()
            if not personas:
                return body

            # Build the combined response
            response_parts = []
            valid_transitions = []

            for i, transition in enumerate(transitions):
                persona_key = transition["persona"]
                segment = transition["segment"]

                if persona_key not in personas:
                    self.logger.warning(
                        f"Persona '{persona_key}' not found, skipping",
                        user_id=user_id,
                        chat_id=chat_id,
                    )
                    continue

                persona_info = personas[persona_key]
                persona_name = persona_info.get("name", persona_key.title())

                # Create a structured prompt for this persona
                persona_prompt = f"""You are {persona_name}. {persona_info.get('prompt', f'You are a {persona_key}.')}

Task: {segment}

Provide a focused response as {persona_name}. Be authentic to your persona."""

                response_parts.append(
                    {
                        "persona": persona_key,
                        "persona_name": persona_name,
                        "segment": segment,
                        "prompt": persona_prompt,
                        "order": i,
                    }
                )
                valid_transitions.append(transition)

            if not response_parts:
                return body

            # Create master system message with all personas
            master = personas.get("_master_controller", {})
            master_prompt = master.get("prompt", "")

            # Build the combined system prompt
            system_content = f"{master_prompt}\n\n=== DYNAMIC MULTI-PERSONA MODE ===\n"
            system_content += "You will respond as multiple personas in sequence. For each persona:\n\n"

            for part in response_parts:
                system_content += (
                    f"**{part['order'] + 1}. {part['persona_name']} Response:**\n"
                )
                system_content += f"Persona: {part['persona_name']}\n"
                system_content += f"Task: {part['segment']}\n"
                system_content += f"Instructions: {part['prompt']}\n\n"

            system_content += "Provide all responses in order, clearly marking each persona transition."

            if self.valves.show_persona_info:
                system_content += f"\n\n🎭 **Multi-Persona Mode**: {len(response_parts)} personas active"

            # Create the combined user message
            user_content = "Respond as the requested personas in sequence:\n\n"
            for i, part in enumerate(response_parts):
                user_content += (
                    f"{i + 1}. **{part['persona_name']}**: {part['segment']}\n"
                )

            # Build clean messages
            clean_messages = self._remove_persona_messages(messages)
            clean_messages.insert(0, {"role": "system", "content": system_content})

            # Update the user's message
            for msg in reversed(clean_messages):
                if msg.get("role") == "user":
                    msg["content"] = user_content
                    break

            body["messages"] = clean_messages

            # Set integration context
            if self.valves.enable_plugin_integration:
                context = await self._create_integration_context(
                    {
                        "type": "multi_persona",
                        "personas": [t["persona"] for t in valid_transitions],
                    },
                    personas,
                    transitions=valid_transitions,
                )
                await self._safe_set_context(body, context, user_id, chat_id)

            # Update current state
            self.current_persona = (
                f"multi:{','.join([t['persona'] for t in valid_transitions])}"
            )
            self.current_context = {
                "type": "multi_persona",
                "personas": valid_transitions,
                "count": len(valid_transitions),
            }

            await PersonaStorage.store_persona(
                user_id, chat_id, self.current_persona, self.current_context
            )

            await self._emit_status(
                emitter,
                f"🎭 Multi-persona sequence ready ({len(valid_transitions)} personas)",
                "complete",
            )

            return body

        except Exception as e:
            self.logger.error(f"Error in classic multi-persona handling: {e}")
            return body

    async def _apply_persistent_persona(
        self, body: dict, messages: List[Dict], user_id: str, chat_id: str
    ) -> dict:
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

        if (
            self.valves.enable_plugin_integration
            and not body.get("_filter_context")
            and not body.get("_context_stored")
        ):
            context = self.integration_manager.prepare_single_persona_context(
                self.current_persona, personas
            )
            await self._safe_set_context(body, context, user_id, chat_id)

        return body

    # New stream method for real-time persona switching
    async def stream(self, event: dict) -> dict:
        """Handle streaming responses with potential persona transitions"""
        if not self.valves.enable_stream_switching:
            return event

        try:
            # Check if this is a multi-persona streaming session
            streaming_context = getattr(self, "_current_streaming_context", None)

            if (
                not streaming_context
                or streaming_context.get("type") != "multi_persona_streaming"
            ):
                return event

            # Extract content from the event
            content_chunk = ""
            if event.get("type") == "content" and "data" in event:
                content_chunk = event["data"].get("content", "")

            if not content_chunk:
                return event

            # Accumulate content
            streaming_context["accumulated_content"] += content_chunk
            current_transition = streaming_context["current_transition"]
            transitions = streaming_context["transitions"]

            if current_transition >= len(transitions) - 1:
                return event  # No more transitions

            # Check if we should transition to next persona
            current_segment = transitions[current_transition]["segment"]
            expected_length = len(current_segment) * 2  # Rough estimate

            should_transition = await self.stream_manager.should_transition(
                streaming_context["accumulated_content"], expected_length
            )

            if should_transition:
                # Transition to next persona
                next_transition = current_transition + 1
                next_persona = transitions[next_transition]["persona"]

                self.logger.info(
                    f"Stream transition {current_transition} -> {next_transition}",
                    persona_from=transitions[current_transition]["persona"],
                    persona_to=next_persona,
                )

                # Update streaming context
                streaming_context["current_transition"] = next_transition
                streaming_context["accumulated_content"] = ""

                # Add transition marker to content
                transition_marker = (
                    f"\n\n[Transitioning to {next_persona} persona...]\n\n"
                )
                event["data"]["content"] = content_chunk + transition_marker

            return event

        except Exception as e:
            self.logger.error(f"Error in stream processing: {e}")
            return event

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

        # Initialize plugin if needed
        if not self._initialized:
            await self._initialize_plugin()

        messages = body.get("messages", [])
        if not messages:
            return body

        chat_id, user_id = self._safe_get_ids(body, __user__)

        # Restore persistent persona if available
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
                    self.logger.debug(
                        f"Restored persona: {stored_persona}",
                        user_id=user_id,
                        chat_id=chat_id,
                    )
            except Exception as e:
                self.logger.error(f"Error restoring persona: {e}")

        last_user_content = ""
        for msg in reversed(messages):
            if msg.get("role") == "user":
                last_user_content = msg.get("content", "")
                break

        if not last_user_content:
            return await self._apply_persistent_persona(
                body, messages, user_id, chat_id
            )

        try:
            command_info = await self.pattern_matcher.detect_command(last_user_content)
        except Exception as e:
            self.logger.error(f"Error detecting command: {e}")
            return await self._apply_persistent_persona(
                body, messages, user_id, chat_id
            )

        try:
            if command_info["type"] == "help":
                return await self._handle_help_command(body, __event_emitter__)
            elif command_info["type"] == "list":
                return await self._handle_list_command(
                    body, __event_emitter__, __event_call__
                )
            elif command_info["type"] == "reset":
                return await self._handle_reset_command(
                    body, __event_emitter__, user_id, chat_id
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
                # Handle multi-persona with both streaming and classic modes
                return await self._handle_multi_persona_streaming(
                    body,
                    messages,
                    last_user_content,
                    __event_emitter__,
                    user_id,
                    chat_id,
                )
            else:
                return await self._apply_persistent_persona(
                    body, messages, user_id, chat_id
                )
        except Exception as e:
            self.logger.error(f"Error handling command {command_info['type']}: {e}")
            return await self._apply_persistent_persona(
                body, messages, user_id, chat_id
            )

    async def outlet(
        self, body: dict, __event_emitter__, __user__: Optional[dict] = None
    ) -> dict:
        # Handle streaming context cleanup and final processing
        try:
            if body.get("_streaming_personas"):
                # Store streaming context for stream method
                self._current_streaming_context = body["_streaming_personas"]

                # Clean up the body
                del body["_streaming_personas"]

        except Exception as e:
            self.logger.error(f"Error in outlet processing: {e}")

        return body
