"""
title: Agent Hotswap
author: pkeffect & Claude AI
author_url: https://github.com/pkeffect
project_urls: https://github.com/pkeffect/functions/tree/main/functions/filters/agent_hotswap | https://github.com/open-webui/functions/tree/main/functions/filters/agent_hotswap | https://openwebui.com/f/pkeffect/agent_hotswap
funding_url: https://github.com/open-webui
date: 2025-06-15
version: 0.2.0
description: Universal AI persona switching with dynamic multi-persona support. Features: mid-prompt persona switching, universal persona detection, smart caching, auto-download, and modular architecture. Commands: !list, !reset, !coder, !writer, plus unlimited combinations.
"""

from pydantic import BaseModel, Field
from typing import Optional, Dict, List, Callable, Any
import re
import json
import asyncio
import time
import os
import traceback
import urllib.request
import urllib.parse
import urllib.error
from datetime import datetime
from pathlib import Path


CACHE_DIRECTORY_NAME = "agent_hotswap"
CONFIG_FILENAME = "personas.json"
BACKUP_COUNT = 5
DEFAULT_PERSONAS_REPO = "https://raw.githubusercontent.com/open-webui/functions/refs/heads/main/functions/filters/agent_hotswap/personas/personas.json"
TRUSTED_DOMAINS = ["github.com", "raw.githubusercontent.com", "gitlab.com"]
DOWNLOAD_TIMEOUT = 30


class PersonaDownloadManager:
    """Manages downloading persona configurations from remote repositories."""

    def __init__(self, get_config_filepath_func):
        self.get_config_filepath = get_config_filepath_func

    def is_trusted_domain(self, url: str) -> bool:
        """Check if URL domain is in the trusted whitelist."""
        try:
            parsed = urllib.parse.urlparse(url)
            if not parsed.scheme or parsed.scheme.lower() not in ["https"]:
                return False
            return parsed.netloc.lower() in TRUSTED_DOMAINS
        except Exception:
            return False

    async def download_personas(self, url: str = None) -> Dict:
        """Download personas from remote repository with validation."""
        download_url = url or DEFAULT_PERSONAS_REPO

        if not self.is_trusted_domain(download_url):
            return {"success": False, "error": f"Untrusted domain"}

        try:
            req = urllib.request.Request(
                download_url, headers={"User-Agent": "OpenWebUI-AgentHotswap/0.2.0"}
            )

            with urllib.request.urlopen(req, timeout=DOWNLOAD_TIMEOUT) as response:
                if response.status != 200:
                    return {"success": False, "error": f"HTTP {response.status}"}

                content = response.read().decode("utf-8")
                content_size = len(content)

                if content_size > 1024 * 1024:  # 1MB limit
                    return {
                        "success": False,
                        "error": f"File too large: {content_size} bytes",
                    }

                try:
                    remote_personas = json.loads(content)
                except json.JSONDecodeError as e:
                    return {"success": False, "error": f"Invalid JSON: {str(e)}"}

                # Basic validation
                if not isinstance(remote_personas, dict) or not remote_personas:
                    return {"success": False, "error": "Invalid personas format"}

                return {
                    "success": True,
                    "personas": remote_personas,
                    "size": content_size,
                    "count": len(remote_personas),
                }

        except Exception as e:
            return {"success": False, "error": str(e)}

    def create_backup(self, current_personas: Dict) -> str:
        """Create a timestamped backup of current personas configuration."""
        try:
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            backup_filename = f"personas_backup_{timestamp}.json"

            config_dir = os.path.dirname(self.get_config_filepath())
            backup_dir = os.path.join(config_dir, "backups")
            os.makedirs(backup_dir, exist_ok=True)

            backup_path = os.path.join(backup_dir, backup_filename)

            with open(backup_path, "w", encoding="utf-8") as f:
                json.dump(current_personas, f, indent=4, ensure_ascii=False)

            # Auto-cleanup old backups
            self._cleanup_old_backups(backup_dir)
            return backup_filename

        except Exception as e:
            return f"Error: {str(e)}"

    def _cleanup_old_backups(self, backup_dir: str):
        """Remove old backup files, keeping only the most recent ones."""
        try:
            backup_files = []
            for filename in os.listdir(backup_dir):
                if filename.startswith("personas_backup_") and filename.endswith(
                    ".json"
                ):
                    filepath = os.path.join(backup_dir, filename)
                    mtime = os.path.getmtime(filepath)
                    backup_files.append((mtime, filepath, filename))

            backup_files.sort(reverse=True)
            files_to_remove = backup_files[BACKUP_COUNT:]
            for _, filepath, filename in files_to_remove:
                os.remove(filepath)

        except Exception:
            pass  # Fail silently for cleanup

    async def download_and_apply_personas(self, url: str = None) -> Dict:
        """Download personas and apply them immediately with backup."""
        download_result = await self.download_personas(url)
        if not download_result["success"]:
            return download_result

        try:
            remote_personas = download_result["personas"]

            # Read current personas for backup
            current_personas = self._read_current_personas()

            # Create backup if we have existing data
            backup_name = None
            if current_personas:
                backup_name = self.create_backup(current_personas)

            # Add metadata to track when this was updated
            remote_personas["_metadata"] = {
                "last_updated": datetime.now().isoformat(),
                "source_url": url or DEFAULT_PERSONAS_REPO,
                "version": "auto-downloaded",
                "persona_count": len(
                    [k for k in remote_personas.keys() if not k.startswith("_")]
                ),
            }

            # Write the new configuration
            config_path = self.get_config_filepath()
            os.makedirs(os.path.dirname(config_path), exist_ok=True)

            with open(config_path, "w", encoding="utf-8") as f:
                json.dump(remote_personas, f, indent=4, ensure_ascii=False)

            print(
                f"[PERSONA INIT] Downloaded and applied {len(remote_personas)} personas"
            )

            return {
                "success": True,
                "backup_created": backup_name,
                "personas_count": len(remote_personas),
                "size": download_result["size"],
            }

        except Exception as e:
            return {"success": False, "error": f"Failed to apply download: {str(e)}"}

    def _read_current_personas(self) -> Dict:
        """Read current personas configuration from file."""
        try:
            config_path = self.get_config_filepath()
            if not os.path.exists(config_path):
                return {}

            with open(config_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}


class UniversalPatternCompiler:
    """Enhanced pattern compiler with universal persona detection capabilities."""

    def __init__(self, config_valves):
        self.valves = config_valves
        self.persona_patterns = {}
        self.reset_pattern = None
        self.list_pattern = None
        self.universal_persona_pattern = None
        self._last_compiled_config = None
        self._compile_patterns()

    def _compile_patterns(self):
        """Compile all regex patterns once for reuse, including universal detection."""
        try:
            current_config = {
                "prefix": self.valves.keyword_prefix,
                "reset_keywords": self.valves.reset_keywords,
                "list_keyword": self.valves.list_command_keyword,
                "case_sensitive": self.valves.case_sensitive,
            }

            if current_config == self._last_compiled_config:
                return

            prefix_escaped = re.escape(self.valves.keyword_prefix)
            flags = 0 if self.valves.case_sensitive else re.IGNORECASE

            # Compile universal persona detection pattern
            # Matches: !{word} where word starts with letter, followed by letters/numbers/underscores
            self.universal_persona_pattern = re.compile(
                rf"{prefix_escaped}([a-zA-Z][a-zA-Z0-9_]*)\b", flags
            )

            # Compile list command pattern
            list_cmd = self.valves.list_command_keyword
            if not self.valves.case_sensitive:
                list_cmd = list_cmd.lower()
            self.list_pattern = re.compile(
                rf"{prefix_escaped}{re.escape(list_cmd)}\b", flags
            )

            # Compile reset patterns
            reset_keywords = [
                word.strip() for word in self.valves.reset_keywords.split(",")
            ]
            reset_pattern_parts = []
            for keyword in reset_keywords:
                if not self.valves.case_sensitive:
                    keyword = keyword.lower()
                reset_pattern_parts.append(re.escape(keyword))

            reset_pattern_str = (
                rf"{prefix_escaped}(?:{'|'.join(reset_pattern_parts)})\b"
            )
            self.reset_pattern = re.compile(reset_pattern_str, flags)

            # Clear old persona patterns
            self.persona_patterns.clear()
            self._last_compiled_config = current_config

        except Exception as e:
            print(f"[PATTERN COMPILER] Error compiling patterns: {e}")

    def discover_all_persona_commands(self, message_content: str) -> List[str]:
        """
        Dynamically discover ALL persona commands in content.
        Works with current 50+ personas AND any future additions.
        """
        if not message_content:
            return []

        self._compile_patterns()

        if not self.universal_persona_pattern:
            return []

        content_to_check = (
            message_content if self.valves.case_sensitive else message_content.lower()
        )

        # Find all persona commands
        matches = self.universal_persona_pattern.findall(content_to_check)

        # Remove duplicates while preserving order
        seen = set()
        unique_personas = []
        for persona in matches:
            persona_key = persona if self.valves.case_sensitive else persona.lower()
            if persona_key not in seen:
                seen.add(persona_key)
                unique_personas.append(persona_key)

        return unique_personas

    def detect_special_commands(self, message_content: str) -> Optional[str]:
        """Detect special commands (list, reset) that take precedence."""
        if not message_content:
            return None

        self._compile_patterns()

        content_to_check = (
            message_content if self.valves.case_sensitive else message_content.lower()
        )

        # Check list command
        if self.list_pattern and self.list_pattern.search(content_to_check):
            return "list_personas"

        # Check reset commands
        if self.reset_pattern and self.reset_pattern.search(content_to_check):
            return "reset"

        return None

    def parse_multi_persona_sequence(self, content: str) -> Dict:
        """
        Parse content with multiple persona switches into structured sequence.

        Input: "!writer do X !teacher do Y !physicist do Z"

        Output: {
            'is_multi_persona': True,
            'sequence': [
                {'persona': 'writer', 'task': 'do X'},
                {'persona': 'teacher', 'task': 'do Y'},
                {'persona': 'physicist', 'task': 'do Z'}
            ],
            'requested_personas': ['writer', 'teacher', 'physicist']
        }
        """
        if not content:
            return {"is_multi_persona": False}

        self._compile_patterns()

        if not self.universal_persona_pattern:
            return {"is_multi_persona": False}

        # Find all persona commands and their positions
        persona_matches = []
        for match in self.universal_persona_pattern.finditer(content):
            persona_key = match.group(1)
            if not self.valves.case_sensitive:
                persona_key = persona_key.lower()

            persona_matches.append(
                {"persona": persona_key, "start": match.start(), "end": match.end()}
            )

        if len(persona_matches) < 1:
            return {"is_multi_persona": False}

        # Extract content segments between persona commands
        sequence = []
        for i, match in enumerate(persona_matches):
            # Get content from end of current command to start of next command
            # (or end of string for last command)
            task_start = match["end"]
            task_end = (
                persona_matches[i + 1]["start"]
                if i + 1 < len(persona_matches)
                else len(content)
            )
            task_content = content[task_start:task_end].strip()

            # Clean up task content by removing any leading persona commands from next segment
            if i + 1 < len(persona_matches):
                # Remove the next persona command if it bleeds into this task
                next_persona_cmd = (
                    f"{self.valves.keyword_prefix}{persona_matches[i + 1]['persona']}"
                )
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

        # Get unique personas requested
        requested_personas = list(
            dict.fromkeys(match["persona"] for match in persona_matches)
        )

        return {
            "is_multi_persona": len(sequence) > 0,
            "is_single_persona": len(sequence) == 1,
            "sequence": sequence,
            "requested_personas": requested_personas,
        }


class SmartPersonaCache:
    """Intelligent caching system for persona configurations."""

    def __init__(self):
        self._cache = {}
        self._file_mtime = 0
        self._last_filepath = None

    def get_personas(self, filepath: str, force_reload: bool = False) -> Dict:
        """Get personas with smart caching - only reload if file changed."""
        try:
            if not os.path.exists(filepath):
                return {}

            current_mtime = os.path.getmtime(filepath)
            filepath_changed = filepath != self._last_filepath
            file_modified = current_mtime > self._file_mtime

            if force_reload or filepath_changed or file_modified or not self._cache:
                with open(filepath, "r", encoding="utf-8") as f:
                    loaded_data = json.load(f)

                self._cache = loaded_data
                self._file_mtime = current_mtime
                self._last_filepath = filepath

            return self._cache.copy()

        except Exception:
            return {}

    def invalidate_cache(self):
        """Force cache invalidation on next access."""
        self._cache.clear()
        self._file_mtime = 0
        self._last_filepath = None


class Filter:
    class Valves(BaseModel):
        keyword_prefix: str = Field(
            default="!",
            description="Prefix character(s) that trigger persona switching (e.g., '!coder')",
        )
        reset_keywords: str = Field(
            default="reset,default,normal",
            description="Comma-separated keywords to reset to default behavior",
        )
        list_command_keyword: str = Field(
            default="list",
            description="Keyword (without prefix) to trigger listing available personas. Prefix will be added (e.g., '!list').",
        )
        case_sensitive: bool = Field(
            default=False, description="Whether keyword matching is case-sensitive"
        )
        show_persona_info: bool = Field(
            default=True,
            description="Show persona information when switching (UI status messages)",
        )
        persistent_persona: bool = Field(
            default=True,
            description="Keep persona active across messages until changed",
        )
        status_message_auto_close_delay_ms: int = Field(
            default=5000,
            description="Delay in milliseconds before attempting to auto-close UI status messages.",
        )
        debug_performance: bool = Field(
            default=False,
            description="Enable performance debugging - logs timing information",
        )
        multi_persona_transitions: bool = Field(
            default=True,
            description="Show transition announcements in multi-persona responses (🎭 **Persona Name**)",
        )

    def __init__(self):
        self.valves = self.Valves()
        self.toggle = True
        self.icon = """data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIGZpbGw9Im5vbmUiIHZpZXdCb3g9IjAgMCAyNCAyNCIgc3Ryb2tlLXdpZHRoPSIxLjUiIHN0cm9rZT0iY3VycmVudENvbG9yIj4KICA8cGF0aCBzdHJva2UtbGluZWNhcD0icm91bmQiIHN0cm9rZS1saW5lam9pbj0icm91bmQiIGQ9Ik0xNS43NSA1QzE1Ljc1IDMuMzQzIDE0LjQwNyAyIDEyLjc1IDJTOS43NSAzLjM0MyA5Ljc1IDV2MC41QTMuNzUgMy43NSAwIDAgMCAxMy41IDkuMjVjMi4xIDAgMy44MS0xLjc2NyAzLjc1LTMuODZWNVoiLz4KICA8cGF0aCBzdHJva2UtbGluZWNhcD0icm91bmQiIHN0cm9rZS1saW5lam9pbj0icm91bmQiIGQ9Ik04LjI1IDV2LjVhMy43NSAzLjc1IDAgMCAwIDMuNzUgMy43NWMuNzE0IDAgMS4zODUtLjIgMS45Ni0uNTU2QTMuNzUgMy43NSAwIDAgMCAxNy4yNSA1djAuNUMxNy4yNSAzLjM0MyAxNS45MDcgMiAxNC4yNSAyczMuNzUgMS4zNDMgMy43NSAzdjAuNUEzLjc1IDMuNzUgMCAwIDAgMjEuNzUgOWMuNzE0IDAgMS4zODUtLjIgMS45Ni0uNTU2QTMuNzUgMy43NSAwIDAgMCAyMS4yNSA1djAuNSIvPgo8L3N2Zz4="""

        # State management
        self.current_persona = None
        self.was_toggled_off_last_call = False
        self.active_status_message_id = None
        self.event_emitter_for_close_task = None

        # Performance optimization components
        self.pattern_compiler = UniversalPatternCompiler(self.valves)
        self.persona_cache = SmartPersonaCache()

        # Download system
        self.download_manager = PersonaDownloadManager(self._get_config_filepath)

        # Initialize config file and auto-download personas
        self._ensure_personas_available()

    @property
    def config_filepath(self):
        """Dynamic property to get the current config file path."""
        return self._get_config_filepath()

    def _get_config_filepath(self):
        """Constructs the config file path using DATA_DIR environment variable with fallbacks."""
        data_dir = os.getenv("DATA_DIR")

        if not data_dir:
            if os.path.exists("/app/backend"):
                data_dir = "/app/backend/data"  # Docker installation
            else:
                home_dir = Path.home()
                data_dir = str(
                    home_dir / ".local" / "share" / "open-webui"
                )  # Native installation

        target_dir = os.path.join(data_dir, "cache", "functions", CACHE_DIRECTORY_NAME)
        filepath = os.path.join(target_dir, CONFIG_FILENAME)
        return filepath

    def get_master_controller_persona(self) -> Dict:
        """Returns the master controller persona - always active foundation."""
        return {
            "_master_controller": {
                "name": "🎛️ OpenWebUI Master Controller",
                "hidden": True,
                "always_active": True,
                "priority": 0,
                "version": "0.6.5+",
                "rules": [
                    "1. This is the foundational system context for OpenWebUI environment",
                    "2. Always active beneath any selected persona",
                    "3. Provides comprehensive native capabilities and rendering context",
                    "4. Transparent to user - no status messages about master controller",
                    "5. Only deactivated on reset/default commands or system toggle off",
                ],
                "prompt": """=== OPENWEBUI MASTER CONTROLLER ===
You operate in OpenWebUI with these native capabilities:

RENDERING: LaTeX ($$formula$$), Mermaid diagrams (```mermaid blocks), HTML artifacts (complete webpages, ThreeJS, D3.js), SVG (pan/zoom, downloadable), enhanced Markdown with alerts, collapsible code blocks, client-side PDF generation

CODE EXECUTION: Python via Pyodide (pandas, matplotlib, numpy included), Jupyter integration for persistent contexts, interactive code blocks with Run buttons, sandbox execution, multiple tool calls, configurable timeouts

FILE HANDLING: Multi-format extraction (PDF, Word, Excel, PowerPoint, CSV, JSON, images, audio), multiple engines (Tika, Docling), encoding detection, drag-drop upload, bypass embedding mode

RAG: Local/remote document integration (#syntax), web search (multiple providers), knowledge bases, YouTube transcripts, Google Drive/OneDrive, vector databases (ChromaDB, Redis, Elasticsearch), hybrid search (BM25+embedding), citations, full context mode

VOICE/AUDIO: STT/TTS (browser/external APIs, OpenAI, Azure), Voice Activity Detection, SpeechT5, audio processing, granular permissions, mobile haptic feedback

INTEGRATIONS: OpenAPI tool servers, MCP support via MCPO, multi-API endpoints, WebSocket with auto-reconnection, load balancing, HTTP/S proxy, Redis caching

UI/UX: Multi-model chat, temporary chats, message management (edit/delete/continue), formatted copying, responsive mobile design, PWA support, widescreen mode, tag system, 20+ languages with RTL support

ADMIN/SECURITY: Granular user permissions, LDAP/OAuth/OIDC auth, access controls, audit logging, enterprise features, resource management

DEPLOYMENT: Docker/Kubernetes/Podman, high availability, OpenTelemetry monitoring, scalable architecture, extensive environment configuration

Leverage these capabilities appropriately - use LaTeX for math, Mermaid for diagrams, artifacts for interactive content, code execution for analysis, RAG for document context, voice features when beneficial. Be direct and maximize OpenWebUI's native functionality.
=== END MASTER CONTROLLER ===
""",
                "description": "Lean OpenWebUI environment context providing complete native capabilities: rendering (LaTeX, Mermaid, HTML artifacts, SVG), code execution (Python/Jupyter), file handling, RAG, voice/audio, integrations, UI/UX, admin/security, internationalization, and deployment features.",
            }
        }

    def _ensure_personas_available(self):
        """Ensures personas are available, downloading them automatically on first run or when outdated."""
        config_path = self.config_filepath

        # Always create directory structure first
        os.makedirs(os.path.dirname(config_path), exist_ok=True)

        # Check if we need to initialize or update
        needs_initialization = not os.path.exists(config_path)
        needs_update = False

        if not needs_initialization:
            # Check if existing config needs updating
            needs_update = self._should_update_personas(config_path)

        if needs_initialization:
            print("[PERSONA INIT] First time setup - creating initial config...")

            # Step 1: Create minimal working config first
            self._create_minimal_config(config_path)
            print("[PERSONA INIT] Minimal config created successfully")

            # Step 2: Now try to download and replace with full collection
            print(
                "[PERSONA INIT] Attempting to download complete persona collection..."
            )
            self._download_full_collection_async()

        elif needs_update:
            print("[PERSONA INIT] Existing config found but needs updating...")
            print("[PERSONA INIT] Downloading latest persona collection...")
            self._download_full_collection_async()

        else:
            print(
                f"[PERSONA INIT] Personas config found and up-to-date at: {config_path}"
            )

    def _should_update_personas(self, config_path: str) -> bool:
        """Determines if the existing personas config should be updated."""
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                config = json.load(f)

            # Check metadata if available
            metadata = config.get("_metadata", {})

            if metadata:
                # Check if last updated more than 7 days ago
                last_updated_str = metadata.get("last_updated")
                if last_updated_str:
                    try:
                        last_updated = datetime.fromisoformat(
                            last_updated_str.replace("Z", "+00:00")
                        )
                        age_days = (datetime.now() - last_updated).days
                        if age_days > 7:
                            print(
                                f"[PERSONA INIT] Config is {age_days} days old, will update"
                            )
                            return True
                    except Exception:
                        pass  # If parsing fails, continue with other checks
            else:
                # No metadata, check file age
                file_age = time.time() - os.path.getmtime(config_path)
                if file_age > (7 * 24 * 60 * 60):  # 7 days in seconds
                    print(
                        "[PERSONA INIT] Config is older than 7 days and has no metadata, will update"
                    )
                    return True

            # Check persona count - if less than 20, probably needs updating
            display_personas = {
                k: v for k, v in config.items() if not k.startswith("_")
            }
            if len(display_personas) < 20:
                print(
                    f"[PERSONA INIT] Only {len(display_personas)} personas found, will update to get full collection"
                )
                return True

            print(
                f"[PERSONA INIT] Config is up-to-date with {len(display_personas)} personas"
            )
            return False

        except Exception as e:
            print(f"[PERSONA INIT] Error checking config update status: {e}")
            return True  # Update on error to be safe

    def _download_full_collection_async(self):
        """Attempts to download the full persona collection asynchronously."""
        try:
            import asyncio
            import threading

            def download_in_thread():
                """Run the download in a separate thread to avoid event loop conflicts."""
                try:
                    # Small delay to let initialization complete
                    time.sleep(1)

                    # Create new event loop for this thread
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)

                    # Download personas
                    download_result = loop.run_until_complete(
                        self.download_manager.download_and_apply_personas()
                    )

                    if download_result["success"]:
                        print(
                            f"[PERSONA INIT] Successfully downloaded {download_result['personas_count']} personas!"
                        )
                        # Invalidate cache so the new personas are loaded
                        self.persona_cache.invalidate_cache()
                    else:
                        print(
                            f"[PERSONA INIT] Download failed: {download_result['error']}"
                        )
                        print("[PERSONA INIT] Will continue with minimal config")

                    loop.close()

                except Exception as e:
                    print(f"[PERSONA INIT] Download thread failed: {e}")
                    print("[PERSONA INIT] Will continue with minimal config")

            # Start download in background thread
            thread = threading.Thread(target=download_in_thread, daemon=True)
            thread.start()
            print("[PERSONA INIT] Download started in background...")

        except Exception as e:
            print(f"[PERSONA INIT] Could not start background download: {e}")
            print("[PERSONA INIT] Will continue with minimal config")

    def _create_minimal_config(self, config_path: str):
        """Creates a minimal config with master controller and a few basic personas as fallback."""
        try:
            # Start with master controller
            minimal_config = self.get_master_controller_persona()

            # Add a few essential personas so users have something to work with immediately
            minimal_config.update(
                {
                    "coder": {
                        "name": "💻 Code Assistant",
                        "prompt": "You are the 💻 Code Assistant, an expert in programming and software development. Provide clean, efficient, well-documented code solutions and explain your reasoning clearly.",
                        "description": "Expert programming and development assistance.",
                        "rules": [
                            "Prioritize clean, efficient code",
                            "Explain reasoning clearly",
                            "Consider security and maintainability",
                        ],
                    },
                    "writer": {
                        "name": "✍️ Creative Writer",
                        "prompt": "You are the ✍️ Creative Writer, a master of crafting engaging, well-structured content. Help with writing projects from brainstorming to final polish.",
                        "description": "Creative writing and content creation specialist.",
                        "rules": [
                            "Craft engaging content",
                            "Assist with all writing stages",
                            "Focus on clarity and impact",
                        ],
                    },
                    "analyst": {
                        "name": "📊 Data Analyst",
                        "prompt": "You are the 📊 Data Analyst, expert in transforming complex data into clear, actionable insights. Create meaningful visualizations and explain findings clearly.",
                        "description": "Data analysis and business intelligence expert.",
                        "rules": [
                            "Provide clear insights",
                            "Create understandable visualizations",
                            "Focus on actionable recommendations",
                        ],
                    },
                }
            )

            # Add metadata to track this is a minimal config
            minimal_config["_metadata"] = {
                "last_updated": datetime.now().isoformat(),
                "source_url": "minimal_config",
                "version": "minimal",
                "persona_count": len(
                    [k for k in minimal_config.keys() if not k.startswith("_")]
                ),
            }

            with open(config_path, "w", encoding="utf-8") as f:
                json.dump(minimal_config, f, indent=4, ensure_ascii=False)
            print(
                f"[PERSONA INIT] Minimal config created with {len([k for k in minimal_config.keys() if not k.startswith('_')])} personas"
            )
        except Exception as e:
            print(f"[PERSONA INIT] Error creating minimal config: {e}")

    def _debug_log(self, message: str):
        """Log debug information if performance debugging is enabled."""
        if self.valves.debug_performance:
            print(f"[PERFORMANCE DEBUG] {message}")

    def _load_personas(self) -> Dict:
        """Loads personas from the external JSON config file with smart caching."""
        start_time = time.time() if self.valves.debug_performance else 0

        try:
            loaded_personas = self.persona_cache.get_personas(self.config_filepath)

            if not loaded_personas:
                print(
                    "[PERSONA CONFIG] No personas loaded, using master controller only"
                )
                loaded_personas = self.get_master_controller_persona()

            if self.valves.debug_performance:
                elapsed = (time.time() - start_time) * 1000
                self._debug_log(
                    f"_load_personas completed in {elapsed:.2f}ms ({len(loaded_personas)} personas)"
                )

            return loaded_personas

        except Exception as e:
            print(f"[PERSONA CONFIG] Error loading personas: {e}")
            return self.get_master_controller_persona()

    def _load_requested_personas_only(self, requested_personas: List[str]) -> Dict:
        """
        Load ONLY the personas actually requested in the prompt.
        Includes validation and graceful error handling.
        """
        # Load the full personas database once
        all_available_personas = self._load_personas()

        # Always include Master Controller
        result = {
            "_master_controller": all_available_personas.get("_master_controller", {})
        }

        # Track what we found vs what was requested
        found_personas = []
        missing_personas = []

        for persona_key in requested_personas:
            if persona_key in all_available_personas:
                result[persona_key] = all_available_personas[persona_key]
                found_personas.append(persona_key)
            else:
                missing_personas.append(persona_key)

        # Add metadata about the loading process
        result["_loading_info"] = {
            "requested": requested_personas,
            "found": found_personas,
            "missing": missing_personas,
            "total_loaded": len(found_personas),
        }

        return result

    def _create_persona_system_message(self, persona_key: str) -> Dict:
        """Enhanced system message that ALWAYS includes master controller + selected persona."""
        personas = self._load_personas()

        # ALWAYS start with master controller
        master_controller = personas.get("_master_controller", {})
        master_prompt = master_controller.get("prompt", "")

        # Add selected persona prompt
        persona = personas.get(persona_key, {})
        persona_prompt = persona.get(
            "prompt", f"You are acting as the {persona_key} persona."
        )

        # Combine: Master Controller + Selected Persona
        system_content = f"{master_prompt}\n\n{persona_prompt}"

        # Add persona indicator (but NOT for master controller)
        if self.valves.show_persona_info and persona_key != "_master_controller":
            persona_name = persona.get("name", persona_key.title())
            system_content += f"\n\n🎭 **Active Persona**: {persona_name}"

        return {"role": "system", "content": system_content}

    def _create_dynamic_multi_persona_system(
        self, requested_personas: List[str]
    ) -> Dict:
        """
        Build dynamic system message with Master Controller + requested personas.
        Works with ANY persona combination, current or future.
        """
        # Load only the requested personas
        loaded_personas = self._load_requested_personas_only(requested_personas)
        loading_info = loaded_personas.pop("_loading_info")

        # Build system message with Master Controller + requested personas
        master_controller = loaded_personas.get("_master_controller", {})
        system_content = master_controller.get("prompt", "")

        # Add each successfully loaded persona
        persona_definitions = []
        for persona_key in loading_info["found"]:
            persona_data = loaded_personas[persona_key]
            persona_name = persona_data.get("name", persona_key.title())
            persona_prompt = persona_data.get("prompt", "")

            persona_definitions.append(
                f"""
=== {persona_name.upper()} PERSONA ===
Activation Command: !{persona_key}
{persona_prompt}
=== END {persona_name.upper()} ===
"""
            )

        # Create execution instructions
        transition_instruction = ""
        if self.valves.multi_persona_transitions and self.valves.show_persona_info:
            transition_instruction = '3. Announce switches: "🎭 **[Persona Name]**"'
        else:
            transition_instruction = (
                "3. Switch personas seamlessly without announcements"
            )

        multi_persona_instructions = f"""

=== DYNAMIC MULTI-PERSONA MODE ===
Active Personas: {len(loading_info['found'])} loaded on-demand

{(''.join(persona_definitions))}

EXECUTION FRAMEWORK:
1. Parse user's persona sequence from their original message
2. When you encounter !{{persona}}, switch to that persona immediately
{transition_instruction}
4. Execute the task following each !command until the next !command
5. Maintain context flow between all switches
6. Available commands in this session: {', '.join([f'!{p}' for p in loading_info['found']])}

{f"⚠️ Unrecognized commands (will be ignored): {', '.join([f'!{p}' for p in loading_info['missing']])}" if loading_info['missing'] else ""}

Execute the user's multi-persona sequence seamlessly.
=== END DYNAMIC MULTI-PERSONA MODE ===
"""

        return {
            "role": "system",
            "content": system_content + multi_persona_instructions,
            "loading_info": loading_info,  # For status messages
        }

    def _remove_keyword_from_message(self, content: str, keyword_found: str) -> str:
        """Remove persona command keywords from message content."""
        prefix = re.escape(self.valves.keyword_prefix)
        flags = 0 if self.valves.case_sensitive else re.IGNORECASE

        if keyword_found == "reset":
            reset_keywords_list = [
                word.strip() for word in self.valves.reset_keywords.split(",")
            ]
            for r_keyword in reset_keywords_list:
                pattern_to_remove = rf"{prefix}{re.escape(r_keyword)}\b\s*"
                content = re.sub(pattern_to_remove, "", content, flags=flags)
        elif keyword_found == "list_personas":
            list_cmd_keyword_to_remove = self.valves.list_command_keyword
            pattern_to_remove = rf"{prefix}{re.escape(list_cmd_keyword_to_remove)}\b\s*"
            content = re.sub(pattern_to_remove, "", content, flags=flags)
        else:
            # Handle persona switching commands
            keyword_to_remove_escaped = re.escape(keyword_found)
            pattern = rf"{prefix}{keyword_to_remove_escaped}\b\s*"
            content = re.sub(pattern, "", content, flags=flags)

        return content.strip()

    def _build_multi_persona_instructions(
        self, sequence: List[Dict], personas_data: Dict
    ) -> str:
        """
        Convert parsed sequence into clear LLM instructions.
        """
        if not sequence:
            return "No valid persona sequence found."

        instructions = ["Execute this multi-persona sequence:\n"]

        for i, step in enumerate(sequence, 1):
            persona_key = step["persona"]
            task = step["task"]
            persona_name = personas_data.get(persona_key, {}).get(
                "name", persona_key.title()
            )

            instructions.append(
                f"""
**Step {i} - {persona_name}:**
{task}
"""
            )

        instructions.append(
            """
\nExecute each step in sequence, following the persona switching framework provided."""
        )

        return "\n".join(instructions)

    async def _emit_and_schedule_close(
        self,
        emitter: Callable[[dict], Any],
        description: str,
        status_type: str = "in_progress",
    ):
        """Emit status message and schedule auto-close."""
        if not emitter or not self.valves.show_persona_info:
            return

        message_id = f"persona_status_{int(time.time() * 1000)}_{hash(description)}"
        self.active_status_message_id = message_id
        self.event_emitter_for_close_task = emitter

        status_message = {
            "type": "status",
            "message_id": message_id,
            "data": {
                "status": status_type,
                "description": description,
                "done": False,
                "hidden": False,
                "message_id": message_id,
                "timeout": self.valves.status_message_auto_close_delay_ms,
            },
        }
        await emitter(status_message)
        asyncio.create_task(self._try_close_message_after_delay(message_id))

    async def _try_close_message_after_delay(self, message_id_to_close: str):
        """Auto-close status message after configured delay."""
        await asyncio.sleep(self.valves.status_message_auto_close_delay_ms / 1000.0)
        if (
            self.event_emitter_for_close_task
            and self.active_status_message_id == message_id_to_close
        ):
            update_message = {
                "type": "status",
                "message_id": message_id_to_close,
                "data": {
                    "message_id": message_id_to_close,
                    "description": "",
                    "done": True,
                    "close": True,
                    "hidden": True,
                },
            }
            try:
                await self.event_emitter_for_close_task(update_message)
            except Exception:
                pass
            self.active_status_message_id = None
            self.event_emitter_for_close_task = None

    def _find_last_user_message(self, messages: List[Dict]) -> tuple[int, str]:
        """Find the last user message in the conversation."""
        for i in range(len(messages) - 1, -1, -1):
            if messages[i].get("role") == "user":
                return i, messages[i].get("content", "")
        return -1, ""

    def _remove_persona_system_messages(self, messages: List[Dict]) -> List[Dict]:
        """Remove existing persona system messages (including master controller)."""
        return [
            msg
            for msg in messages
            if not (
                msg.get("role") == "system"
                and (
                    "🎭 **Active Persona**" in msg.get("content", "")
                    or "=== OPENWEBUI MASTER CONTROLLER ===" in msg.get("content", "")
                    or "=== DYNAMIC MULTI-PERSONA MODE ===" in msg.get("content", "")
                )
            )
        ]

    def _generate_persona_table(self, personas: Dict) -> str:
        """Generate instructions for LLM to create persona table (excludes master controller and metadata)."""
        # Filter out master controller and metadata from display
        display_personas = {
            k: v
            for k, v in personas.items()
            if not k.startswith("_")  # Excludes _master_controller and _metadata
        }

        sorted_persona_keys = sorted(display_personas.keys())
        table_rows_str_list = []
        items_per_row_pair = 2

        for i in range(0, len(sorted_persona_keys), items_per_row_pair):
            row_cells = []
            for j in range(items_per_row_pair):
                if i + j < len(sorted_persona_keys):
                    key = sorted_persona_keys[i + j]
                    data = display_personas[key]
                    command = f"`{self.valves.keyword_prefix}{key}`"
                    name = data.get("name", key.title())
                    row_cells.extend([command, name])
                else:
                    row_cells.extend([" ", " "])  # Empty cells for better rendering
            table_rows_str_list.append(f"| {' | '.join(row_cells)} |")

        table_data_str = "\n".join(table_rows_str_list)
        headers = " | ".join(["Command", "Name"] * items_per_row_pair)
        separators = " | ".join(["---|---"] * items_per_row_pair)

        # Prepare reset commands string
        reset_cmds_formatted = [
            f"`{self.valves.keyword_prefix}{rk.strip()}`"
            for rk in self.valves.reset_keywords.split(",")
        ]
        reset_cmds_str = ", ".join(reset_cmds_formatted)

        # Return instructions for the LLM to present the table
        return (
            f"Please present the following information. First, a Markdown table of available persona commands, "
            f"titled '**Available Personas**'. The table should have columns for 'Command' and 'Name', "
            f"displaying two pairs of these per row.\n\n"
            f"**Available Personas**\n"
            f"| {headers} |\n"
            f"| {separators} |\n"
            f"{table_data_str}\n\n"
            f"After the table, please add the following explanation on a new line:\n"
            f"To revert to the default assistant, use one of these commands: {reset_cmds_str}\n\n"
            f"**Multi-Persona Support:** You can now use multiple personas in a single message! "
            f"Example: `{self.valves.keyword_prefix}writer create a story {self.valves.keyword_prefix}teacher explain the literary techniques {self.valves.keyword_prefix}artist describe visuals`\n\n"
            f"Ensure the output is properly formatted Markdown."
        )

    async def _handle_toggle_off_state(
        self, body: Dict, __event_emitter__: Callable[[dict], Any]
    ) -> Dict:
        """Handle behavior when filter is toggled off."""
        messages = body.get("messages", [])
        if messages is None:
            messages = []

        if self.current_persona is not None or not self.was_toggled_off_last_call:
            persona_was_active_before_toggle_off = self.current_persona is not None
            self.current_persona = None
            if messages:
                body["messages"] = self._remove_persona_system_messages(messages)
            if persona_was_active_before_toggle_off:
                await self._emit_and_schedule_close(
                    __event_emitter__,
                    "ℹ️ Persona Switcher is OFF. Assistant reverted to default.",
                    status_type="complete",
                )
        self.was_toggled_off_last_call = True
        return body

    async def _handle_list_personas_command(
        self,
        body: Dict,
        messages: List[Dict],
        last_message_idx: int,
        __event_emitter__: Callable[[dict], Any],
    ) -> Dict:
        """Handle !list command - generates persona table."""
        personas = self._load_personas()

        # Filter out internal entries for counting
        display_personas = {k: v for k, v in personas.items() if not k.startswith("_")}

        if not personas or len(display_personas) == 0:
            list_prompt_content = "No personas are currently available. The system may still be initializing."
        else:
            list_prompt_content = self._generate_persona_table(personas)

        messages[last_message_idx]["content"] = list_prompt_content
        await self._emit_and_schedule_close(
            __event_emitter__,
            "📋 Preparing persona list...",
            status_type="complete",
        )
        return body

    async def _handle_reset_command(
        self,
        body: Dict,
        messages: List[Dict],
        last_message_idx: int,
        original_content: str,
        __event_emitter__: Callable[[dict], Any],
    ) -> Dict:
        """Handle !reset command - clears current persona."""
        self.current_persona = None
        temp_messages = []
        user_message_updated = False

        for msg_dict in messages:
            msg = dict(msg_dict)
            if msg.get("role") == "system" and (
                "🎭 **Active Persona**" in msg.get("content", "")
                or "=== DYNAMIC MULTI-PERSONA MODE ===" in msg.get("content", "")
            ):
                continue
            if (
                not user_message_updated
                and msg.get("role") == "user"
                and msg.get("content", "") == original_content
            ):
                cleaned_content = self._remove_keyword_from_message(
                    original_content, "reset"
                )
                reset_confirmation_prompt = "You have been reset from any specialized persona. Please confirm you are now operating in your default/standard assistant mode."
                if cleaned_content.strip():
                    msg["content"] = (
                        f"{reset_confirmation_prompt} Then, please address the following: {cleaned_content}"
                    )
                else:
                    msg["content"] = reset_confirmation_prompt
                user_message_updated = True
            temp_messages.append(msg)

        body["messages"] = temp_messages
        await self._emit_and_schedule_close(
            __event_emitter__,
            "🔄 Reset to default. LLM will confirm.",
            status_type="complete",
        )
        return body

    async def _handle_single_persona_command(
        self,
        persona_key: str,
        body: Dict,
        messages: List[Dict],
        last_message_idx: int,
        original_content: str,
        __event_emitter__: Callable[[dict], Any],
    ) -> Dict:
        """Handle single persona switching commands like !coder, !writer, etc."""
        personas_data = self._load_personas()
        if persona_key not in personas_data:
            return body

        self.current_persona = persona_key
        persona_config = personas_data[persona_key]
        temp_messages = []
        user_message_modified = False

        for msg_dict in messages:
            msg = dict(msg_dict)
            if msg.get("role") == "system" and (
                "🎭 **Active Persona**" in msg.get("content", "")
                or "=== DYNAMIC MULTI-PERSONA MODE ===" in msg.get("content", "")
            ):
                continue
            if (
                not user_message_modified
                and msg.get("role") == "user"
                and msg.get("content", "") == original_content
            ):
                cleaned_content = self._remove_keyword_from_message(
                    original_content, persona_key
                )
                intro_request_default = (
                    "Please introduce yourself and explain what you can help me with."
                )

                if persona_config.get("prompt"):
                    intro_marker = "When introducing yourself,"
                    if intro_marker in persona_config["prompt"]:
                        try:
                            prompt_intro_segment = (
                                persona_config["prompt"]
                                .split(intro_marker, 1)[1]
                                .split(".", 1)[0]
                                .strip()
                            )
                            if prompt_intro_segment:
                                intro_request_default = f"Please introduce yourself, {prompt_intro_segment}, and then explain what you can help me with."
                        except IndexError:
                            pass

                if not cleaned_content.strip():
                    msg["content"] = intro_request_default
                else:
                    persona_name_for_prompt = persona_config.get(
                        "name", persona_key.title()
                    )
                    msg["content"] = (
                        f"Please briefly introduce yourself as {persona_name_for_prompt}. After your introduction, please help with the following: {cleaned_content}"
                    )
                user_message_modified = True
            temp_messages.append(msg)

        persona_system_msg = self._create_persona_system_message(persona_key)
        temp_messages.insert(0, persona_system_msg)
        body["messages"] = temp_messages

        persona_display_name = persona_config.get("name", persona_key.title())
        await self._emit_and_schedule_close(
            __event_emitter__,
            f"🎭 Switched to {persona_display_name}",
            status_type="complete",
        )
        return body

    async def _handle_multi_persona_command(
        self,
        sequence_data: Dict,
        body: Dict,
        messages: List[Dict],
        last_message_idx: int,
        original_content: str,
        __event_emitter__: Callable[[dict], Any],
    ) -> Dict:
        """
        Handle complex multi-persona sequences.
        """
        requested_personas = sequence_data["requested_personas"]
        sequence = sequence_data["sequence"]

        # Build dynamic system message
        dynamic_system_result = self._create_dynamic_multi_persona_system(
            requested_personas
        )
        loading_info = dynamic_system_result.pop("loading_info")

        # Remove old persona messages
        temp_messages = self._remove_persona_system_messages(messages)
        temp_messages.insert(0, dynamic_system_result)

        # Build instruction content from the sequence
        all_personas = self._load_personas()
        instruction_content = self._build_multi_persona_instructions(
            sequence, all_personas
        )

        # Update user message with structured instructions
        temp_messages[last_message_idx + 1][
            "content"
        ] = instruction_content  # +1 because we inserted system message

        body["messages"] = temp_messages

        # Update current state for multi-persona
        if len(loading_info["found"]) == 1:
            self.current_persona = loading_info["found"][0]
        else:
            self.current_persona = f"multi:{':'.join(loading_info['found'])}"

        # Status message
        if loading_info["found"]:
            persona_names = []
            for p in loading_info["found"]:
                name = all_personas.get(p, {}).get("name", p.title())
                persona_names.append(name)

            status_msg = f"🎭 Multi-persona sequence: {' → '.join(persona_names)}"
            if loading_info["missing"]:
                status_msg += f" | ⚠️ Unknown: {', '.join([f'!{p}' for p in loading_info['missing']])}"

            await self._emit_and_schedule_close(
                __event_emitter__, status_msg, "complete"
            )

        return body

    def _apply_persistent_persona(self, body: Dict, messages: List[Dict]) -> Dict:
        """Apply current persona to messages when no command detected (ALWAYS includes master controller)."""
        if not self.valves.persistent_persona:
            return body

        personas = self._load_personas()
        target_persona = self.current_persona if self.current_persona else None

        # Handle multi-persona persistent state
        if target_persona and target_persona.startswith("multi:"):
            # For multi-persona, we don't persist - user needs to issue new commands
            return body

        if not target_persona or target_persona not in personas:
            return body

        # Check if correct persona system message exists
        expected_persona_name = personas[target_persona].get(
            "name", target_persona.title()
        )
        master_controller_expected = "=== OPENWEBUI MASTER CONTROLLER ==="

        correct_system_msg_found = False
        temp_messages = []

        for msg_dict in messages:
            msg = dict(msg_dict)
            is_system_msg = msg.get("role") == "system"

            if is_system_msg:
                content = msg.get("content", "")
                has_master_controller = master_controller_expected in content
                has_correct_persona = (
                    f"🎭 **Active Persona**: {expected_persona_name}" in content
                )

                if has_master_controller and (
                    not self.valves.show_persona_info or has_correct_persona
                ):
                    correct_system_msg_found = True
                    temp_messages.append(msg)
            else:
                temp_messages.append(msg)

        # Add system message if not found
        if not correct_system_msg_found:
            system_msg = self._create_persona_system_message(target_persona)
            temp_messages.insert(0, system_msg)

        body["messages"] = temp_messages
        return body

    async def inlet(
        self,
        body: dict,
        __event_emitter__: Callable[[dict], Any],
        __user__: Optional[dict] = None,
    ) -> dict:
        """Main entry point - orchestrates the universal persona switching flow."""
        messages = body.get("messages", [])
        if messages is None:
            messages = []

        # Handle toggle off state
        if not self.toggle:
            return await self._handle_toggle_off_state(body, __event_emitter__)

        # Update toggle state tracking
        if self.toggle and self.was_toggled_off_last_call:
            self.was_toggled_off_last_call = False

        # Handle empty messages
        if not messages:
            return body

        # Find last user message
        last_message_idx, original_content_of_last_user_msg = (
            self._find_last_user_message(messages)
        )

        # Handle non-user messages (apply persistent persona)
        if last_message_idx == -1:
            return self._apply_persistent_persona(body, messages)

        # Check for special commands first (they take precedence)
        special_command = self.pattern_compiler.detect_special_commands(
            original_content_of_last_user_msg
        )

        if special_command:
            if special_command == "list_personas":
                return await self._handle_list_personas_command(
                    body, messages, last_message_idx, __event_emitter__
                )
            elif special_command == "reset":
                return await self._handle_reset_command(
                    body,
                    messages,
                    last_message_idx,
                    original_content_of_last_user_msg,
                    __event_emitter__,
                )

        # Parse for persona sequence (universal detection)
        sequence_data = self.pattern_compiler.parse_multi_persona_sequence(
            original_content_of_last_user_msg
        )

        if sequence_data["is_multi_persona"]:
            if sequence_data["is_single_persona"]:
                # Single persona command
                persona_key = sequence_data["sequence"][0]["persona"]
                return await self._handle_single_persona_command(
                    persona_key,
                    body,
                    messages,
                    last_message_idx,
                    original_content_of_last_user_msg,
                    __event_emitter__,
                )
            else:
                # Multi-persona sequence
                return await self._handle_multi_persona_command(
                    sequence_data,
                    body,
                    messages,
                    last_message_idx,
                    original_content_of_last_user_msg,
                    __event_emitter__,
                )
        else:
            # No persona commands detected, apply persistent persona if active
            return self._apply_persistent_persona(body, messages)

    async def outlet(
        self, body: dict, __event_emitter__, __user__: Optional[dict] = None
    ) -> dict:
        return body

    def get_persona_list(self) -> str:
        """Get formatted list of available personas for API/external use."""
        personas = self._load_personas()

        # Filter out master controller and metadata from user-facing list
        display_personas = {k: v for k, v in personas.items() if not k.startswith("_")}

        persona_list_items = []
        for keyword in sorted(display_personas.keys()):
            data = display_personas[keyword]
            name = data.get("name", keyword.title())
            desc = data.get("description", "No description available.")
            persona_list_items.append(
                f"• `{self.valves.keyword_prefix}{keyword}` - {name}: {desc}"
            )

        reset_keywords_display = ", ".join(
            [
                f"`{self.valves.keyword_prefix}{rk.strip()}`"
                for rk in self.valves.reset_keywords.split(",")
            ]
        )

        list_command_display = (
            f"`{self.valves.keyword_prefix}{self.valves.list_command_keyword}`"
        )

        command_info = (
            f"\n\n**System Commands:**\n"
            f"• {list_command_display} - Lists persona commands and names in a multi-column Markdown table.\n"
            f"• {reset_keywords_display} - Reset to default assistant behavior (LLM will confirm).\n\n"
            f"**Multi-Persona Support:** Use multiple personas in one message!\n"
            f"Example: `{self.valves.keyword_prefix}writer story {self.valves.keyword_prefix}teacher explain {self.valves.keyword_prefix}artist visuals`"
        )

        if not persona_list_items:
            main_list_str = "No personas configured."
        else:
            main_list_str = "\n".join(persona_list_items)

        return "Available Personas:\n" + main_list_str + command_info
