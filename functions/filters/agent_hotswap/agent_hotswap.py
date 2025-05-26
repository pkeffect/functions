"""
title: Agent Hotswap
author: pkeffect
author_url: https://github.com/pkeffect
project_url: https://github.com/pkeffect/agent_hotswap
funding_url: https://github.com/open-webui
version: 0.1.0
description: Switch between AI personas with optimized performance. Features: external config, pre-compiled regex patterns, smart caching, validation, and modular architecture. Commands: !list, !reset, !coder, !writer, etc.
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
import difflib


class PersonaDownloadManager:
    """Manages downloading and applying persona configurations from remote repositories."""

    def __init__(self, valves, get_config_filepath_func):
        self.valves = valves
        self.get_config_filepath = get_config_filepath_func

    def is_trusted_domain(self, url: str) -> bool:
        """Check if URL domain is in the trusted whitelist."""
        try:
            print(f"[DOMAIN DEBUG] Checking URL: {url}")
            parsed = urllib.parse.urlparse(url)
            print(
                f"[DOMAIN DEBUG] Parsed URL - scheme: {parsed.scheme}, netloc: {parsed.netloc}"
            )

            if not parsed.scheme or parsed.scheme.lower() not in ["https"]:
                print(f"[DOMAIN DEBUG] Scheme check failed - scheme: '{parsed.scheme}'")
                return False

            print(f"[DOMAIN DEBUG] Scheme check passed")

            trusted_domains_raw = self.valves.trusted_domains
            trusted_domains = [
                d.strip().lower() for d in trusted_domains_raw.split(",")
            ]
            print(f"[DOMAIN DEBUG] Trusted domains raw: '{trusted_domains_raw}'")
            print(f"[DOMAIN DEBUG] Trusted domains processed: {trusted_domains}")
            print(f"[DOMAIN DEBUG] URL netloc (lowercase): '{parsed.netloc.lower()}'")

            is_trusted = parsed.netloc.lower() in trusted_domains
            print(f"[DOMAIN DEBUG] Domain trusted check result: {is_trusted}")

            return is_trusted

        except Exception as e:
            print(f"[DOMAIN DEBUG] Exception in domain check: {e}")
            traceback.print_exc()
            return False

    async def download_personas(self, url: str = None) -> Dict:
        """Download personas from remote repository with validation."""
        download_url = url or self.valves.default_personas_repo

        print(f"[DOWNLOAD DEBUG] Starting download from: {download_url}")

        # Validate URL
        print(f"[DOWNLOAD DEBUG] Checking if domain is trusted...")
        if not self.is_trusted_domain(download_url):
            error_msg = (
                f"Untrusted domain. Allowed domains: {self.valves.trusted_domains}"
            )
            print(f"[DOWNLOAD DEBUG] Domain check failed: {error_msg}")
            return {"success": False, "error": error_msg, "url": download_url}

        print(f"[DOWNLOAD DEBUG] Domain check passed")

        try:
            # Download with timeout
            print(f"[DOWNLOAD DEBUG] Creating HTTP request...")
            req = urllib.request.Request(
                download_url, headers={"User-Agent": "OpenWebUI-AgentHotswap/3.1"}
            )
            print(f"[DOWNLOAD DEBUG] Request created, opening connection...")

            with urllib.request.urlopen(
                req, timeout=self.valves.download_timeout
            ) as response:
                print(f"[DOWNLOAD DEBUG] Connection opened, status: {response.status}")
                print(f"[DOWNLOAD DEBUG] Response headers: {dict(response.headers)}")

                if response.status != 200:
                    error_msg = f"HTTP {response.status}: {response.reason}"
                    print(f"[DOWNLOAD DEBUG] HTTP error: {error_msg}")
                    return {"success": False, "error": error_msg, "url": download_url}

                print(f"[DOWNLOAD DEBUG] Reading response content...")
                content = response.read().decode("utf-8")
                content_size = len(content)
                print(
                    f"[DOWNLOAD DEBUG] Content read successfully: {content_size} bytes"
                )
                print(
                    f"[DOWNLOAD DEBUG] Content preview (first 200 chars): {content[:200]}"
                )

                # Basic size check (prevent huge files)
                if content_size > 1024 * 1024:  # 1MB limit
                    error_msg = f"File too large: {content_size} bytes (max 1MB)"
                    print(f"[DOWNLOAD DEBUG] Size check failed: {error_msg}")
                    return {"success": False, "error": error_msg, "url": download_url}

                print(f"[DOWNLOAD DEBUG] Size check passed")

                # Parse JSON
                print(f"[DOWNLOAD DEBUG] Parsing JSON...")
                try:
                    remote_personas = json.loads(content)
                    print(
                        f"[DOWNLOAD DEBUG] JSON parsed successfully, {len(remote_personas)} items found"
                    )
                    print(
                        f"[DOWNLOAD DEBUG] Top-level keys: {list(remote_personas.keys())[:5]}"
                    )
                except json.JSONDecodeError as e:
                    error_msg = f"Invalid JSON: {str(e)}"
                    print(f"[DOWNLOAD DEBUG] JSON parsing failed: {error_msg}")
                    print(
                        f"[DOWNLOAD DEBUG] Content that failed parsing: {content[:500]}"
                    )
                    return {"success": False, "error": error_msg, "url": download_url}

                # Validate structure
                print(f"[DOWNLOAD DEBUG] Validating persona structure...")
                validation_errors = PersonaValidator.validate_personas_config(
                    remote_personas
                )
                if validation_errors:
                    error_msg = f"Validation failed: {'; '.join(validation_errors[:3])}"
                    print(f"[DOWNLOAD DEBUG] Validation failed: {validation_errors}")
                    return {
                        "success": False,
                        "error": error_msg,
                        "url": download_url,
                        "validation_errors": validation_errors,
                    }

                print(
                    f"[DOWNLOAD DEBUG] Validation passed - {len(remote_personas)} personas"
                )

                return {
                    "success": True,
                    "personas": remote_personas,
                    "url": download_url,
                    "size": content_size,
                    "count": len(remote_personas),
                }

        except urllib.error.URLError as e:
            error_msg = f"Download failed: {str(e)}"
            print(f"[DOWNLOAD DEBUG] URLError: {error_msg}")
            print(f"[DOWNLOAD DEBUG] URLError details: {type(e).__name__}: {e}")
            return {"success": False, "error": error_msg, "url": download_url}
        except Exception as e:
            error_msg = f"Unexpected error: {str(e)}"
            print(f"[DOWNLOAD DEBUG] Unexpected error: {error_msg}")
            print(f"[DOWNLOAD DEBUG] Exception type: {type(e).__name__}")
            traceback.print_exc()
            return {"success": False, "error": error_msg, "url": download_url}

    def analyze_differences(self, remote_personas: Dict, local_personas: Dict) -> Dict:
        """Analyze differences between remote and local persona configurations."""
        analysis = {
            "new_personas": [],
            "updated_personas": [],
            "conflicts": [],
            "unchanged_personas": [],
            "summary": {},
        }

        # Analyze each remote persona
        for persona_key, remote_persona in remote_personas.items():
            if persona_key not in local_personas:
                # New persona
                analysis["new_personas"].append(
                    {
                        "key": persona_key,
                        "name": remote_persona.get("name", persona_key.title()),
                        "description": remote_persona.get(
                            "description", "No description"
                        ),
                        "prompt_length": len(remote_persona.get("prompt", "")),
                    }
                )
            else:
                # Existing persona - check for differences
                local_persona = local_personas[persona_key]
                differences = []

                # Compare key fields
                for field in ["name", "description", "prompt"]:
                    local_val = local_persona.get(field, "")
                    remote_val = remote_persona.get(field, "")
                    if local_val != remote_val:
                        differences.append(field)

                if differences:
                    analysis["conflicts"].append(
                        {
                            "key": persona_key,
                            "local": local_persona,
                            "remote": remote_persona,
                            "differences": differences,
                        }
                    )
                else:
                    analysis["unchanged_personas"].append(persona_key)

        # Generate summary
        analysis["summary"] = {
            "new_count": len(analysis["new_personas"]),
            "conflict_count": len(analysis["conflicts"]),
            "unchanged_count": len(analysis["unchanged_personas"]),
            "total_remote": len(remote_personas),
            "total_local": len(local_personas),
        }

        return analysis

    def generate_diff_view(
        self, local_persona: Dict, remote_persona: Dict, persona_key: str
    ) -> str:
        """Generate a detailed diff view for a specific persona conflict."""
        diff_lines = []

        # Compare key fields
        for field in ["name", "description", "prompt"]:
            local_val = local_persona.get(field, "")
            remote_val = remote_persona.get(field, "")

            if local_val != remote_val:
                diff_lines.append(f"\n**{field.upper()}:**")
                diff_lines.append("```diff")

                if field == "prompt":
                    # For long prompts, show character count and first/last lines
                    local_preview = (
                        f"{local_val[:100]}..." if len(local_val) > 100 else local_val
                    )
                    remote_preview = (
                        f"{remote_val[:100]}..."
                        if len(remote_val) > 100
                        else remote_val
                    )
                    diff_lines.append(
                        f"- LOCAL ({len(local_val)} chars): {local_preview}"
                    )
                    diff_lines.append(
                        f"+ REMOTE ({len(remote_val)} chars): {remote_preview}"
                    )
                else:
                    diff_lines.append(f"- {local_val}")
                    diff_lines.append(f"+ {remote_val}")

                diff_lines.append("```")

        return "\n".join(diff_lines)

    def create_backup(self, current_personas: Dict) -> str:
        """Create a timestamped backup of current personas configuration."""
        try:
            # Generate backup filename with timestamp
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            backup_filename = f"personas_backup_{timestamp}.json"

            # Create backups directory
            config_dir = os.path.dirname(self.get_config_filepath())
            backup_dir = os.path.join(config_dir, "backups")
            os.makedirs(backup_dir, exist_ok=True)

            backup_path = os.path.join(backup_dir, backup_filename)

            # Write backup
            with open(backup_path, "w", encoding="utf-8") as f:
                json.dump(current_personas, f, indent=4, ensure_ascii=False)

            print(f"[BACKUP] Created backup: {backup_path}")

            # Auto-cleanup old backups
            self._cleanup_old_backups(backup_dir)

            return backup_filename

        except Exception as e:
            print(f"[BACKUP] Error creating backup: {e}")
            return f"Error: {str(e)}"

    def _cleanup_old_backups(self, backup_dir: str):
        """Remove old backup files, keeping only the most recent ones."""
        try:
            # Get all backup files
            backup_files = []
            for filename in os.listdir(backup_dir):
                if filename.startswith("personas_backup_") and filename.endswith(
                    ".json"
                ):
                    filepath = os.path.join(backup_dir, filename)
                    mtime = os.path.getmtime(filepath)
                    backup_files.append((mtime, filepath, filename))

            # Sort by modification time (newest first)
            backup_files.sort(reverse=True)

            # Remove old backups beyond the limit
            files_to_remove = backup_files[self.valves.backup_count :]
            for _, filepath, filename in files_to_remove:
                os.remove(filepath)
                print(f"[BACKUP] Removed old backup: {filename}")

        except Exception as e:
            print(f"[BACKUP] Error during cleanup: {e}")

    async def download_and_apply_personas(
        self, url: str = None, merge_strategy: str = "merge"
    ) -> Dict:
        """Download personas and apply them immediately with backup."""
        # Download first
        download_result = await self.download_personas(url)
        if not download_result["success"]:
            return download_result

        try:
            remote_personas = download_result["personas"]

            # Load current personas for backup and analysis
            current_personas = self._read_current_personas()

            # Create backup first
            backup_name = self.create_backup(current_personas)
            print(f"[DOWNLOAD APPLY] Backup created: {backup_name}")

            # Analyze differences for reporting
            analysis = self.analyze_differences(remote_personas, current_personas)

            # Apply merge strategy
            if merge_strategy == "replace":
                # Replace entire configuration
                final_personas = remote_personas.copy()
            else:
                # Merge strategy (default)
                final_personas = current_personas.copy()

                # Add new personas
                for new_persona in analysis["new_personas"]:
                    key = new_persona["key"]
                    final_personas[key] = remote_personas[key]

                # For conflicts, use remote version (simple strategy)
                for conflict in analysis["conflicts"]:
                    key = conflict["key"]
                    final_personas[key] = conflict["remote"]

            # Write the final configuration
            config_path = self.get_config_filepath()
            with open(config_path, "w", encoding="utf-8") as f:
                json.dump(final_personas, f, indent=4, ensure_ascii=False)

            print(
                f"[DOWNLOAD APPLY] Applied configuration - {len(final_personas)} personas"
            )

            return {
                "success": True,
                "backup_created": backup_name,
                "personas_count": len(final_personas),
                "changes_applied": {
                    "new_added": len(analysis["new_personas"]),
                    "conflicts_resolved": len(analysis["conflicts"]),
                    "total_downloaded": len(remote_personas),
                },
                "analysis": analysis,
                "url": download_result["url"],
                "size": download_result["size"],
            }

        except Exception as e:
            print(f"[DOWNLOAD APPLY] Error applying download: {e}")
            traceback.print_exc()
            return {"success": False, "error": f"Failed to apply download: {str(e)}"}

    def _read_current_personas(self) -> Dict:
        """Read current personas configuration from file."""
        try:
            config_path = self.get_config_filepath()
            if not os.path.exists(config_path):
                return {}

            with open(config_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            print(f"[DOWNLOAD APPLY] Error reading current personas: {e}")
            return {}


class PersonaValidator:
    """Validates persona configuration structure."""

    @staticmethod
    def validate_persona_config(persona: Dict) -> List[str]:
        """Validate a single persona configuration.

        Returns:
            List of error messages, empty if valid
        """
        errors = []
        required_fields = ["name", "prompt", "description"]

        for field in required_fields:
            if field not in persona:
                errors.append(f"Missing required field: {field}")
            elif not isinstance(persona[field], str):
                errors.append(f"Field '{field}' must be a string")
            elif not persona[field].strip():
                errors.append(f"Field '{field}' cannot be empty")

        # Validate optional fields
        if "rules" in persona and not isinstance(persona["rules"], list):
            errors.append("Field 'rules' must be a list")

        return errors

    @staticmethod
    def validate_personas_config(personas: Dict) -> List[str]:
        """Validate entire personas configuration.

        Returns:
            List of error messages, empty if valid
        """
        all_errors = []

        if not isinstance(personas, dict):
            return ["Personas config must be a dictionary"]

        if not personas:
            return ["Personas config cannot be empty"]

        for persona_key, persona_data in personas.items():
            if not isinstance(persona_key, str) or not persona_key.strip():
                all_errors.append(f"Invalid persona key: {persona_key}")
                continue

            if not isinstance(persona_data, dict):
                all_errors.append(f"Persona '{persona_key}' must be a dictionary")
                continue

            persona_errors = PersonaValidator.validate_persona_config(persona_data)
            for error in persona_errors:
                all_errors.append(f"Persona '{persona_key}': {error}")

        return all_errors


class PatternCompiler:
    """Pre-compiles and manages regex patterns for efficient persona detection."""

    def __init__(self, config_valves):
        self.valves = config_valves
        self.persona_patterns = {}
        self.reset_pattern = None
        self.list_pattern = None
        self._last_compiled_config = None
        self._compile_patterns()

    def _compile_patterns(self):
        """Compile all regex patterns once for reuse."""
        try:
            # Get current config state for change detection
            current_config = {
                "prefix": self.valves.keyword_prefix,
                "reset_keywords": self.valves.reset_keywords,
                "list_keyword": self.valves.list_command_keyword,
                "case_sensitive": self.valves.case_sensitive,
            }

            # Only recompile if config changed
            if current_config == self._last_compiled_config:
                return

            print(
                f"[PATTERN COMPILER] Compiling patterns for prefix '{self.valves.keyword_prefix}'"
            )

            # Compile base patterns
            prefix_escaped = re.escape(self.valves.keyword_prefix)
            flags = 0 if self.valves.case_sensitive else re.IGNORECASE

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

            # Compile download command pattern
            self.download_pattern = re.compile(
                rf"{prefix_escaped}download_personas\b", flags
            )

            # Clear old persona patterns - they'll be compiled on demand
            self.persona_patterns.clear()

            self._last_compiled_config = current_config
            print(f"[PATTERN COMPILER] Patterns compiled successfully")

        except Exception as e:
            print(f"[PATTERN COMPILER] Error compiling patterns: {e}")
            traceback.print_exc()

    def get_persona_pattern(self, persona_key: str):
        """Get or compile a pattern for a specific persona."""
        if persona_key not in self.persona_patterns:
            try:
                prefix_escaped = re.escape(self.valves.keyword_prefix)
                keyword_check = (
                    persona_key if self.valves.case_sensitive else persona_key.lower()
                )
                flags = 0 if self.valves.case_sensitive else re.IGNORECASE
                pattern_str = rf"{prefix_escaped}{re.escape(keyword_check)}\b"
                self.persona_patterns[persona_key] = re.compile(pattern_str, flags)
            except Exception as e:
                print(
                    f"[PATTERN COMPILER] Error compiling pattern for '{persona_key}': {e}"
                )
                return None

        return self.persona_patterns[persona_key]

    def detect_keyword(
        self, message_content: str, available_personas: Dict
    ) -> Optional[str]:
        """Efficiently detect persona keywords using pre-compiled patterns."""
        if not message_content:
            return None

        # Ensure patterns are up to date
        self._compile_patterns()

        content_to_check = (
            message_content if self.valves.case_sensitive else message_content.lower()
        )

        # Check list command (fastest check first)
        if self.list_pattern and self.list_pattern.search(content_to_check):
            return "list_personas"

        # Check reset commands
        if self.reset_pattern and self.reset_pattern.search(content_to_check):
            return "reset"

        # Check download command
        if self.download_pattern and self.download_pattern.search(content_to_check):
            return "download_personas"

        # Check persona commands
        for persona_key in available_personas.keys():
            pattern = self.get_persona_pattern(persona_key)
            if pattern and pattern.search(content_to_check):
                return persona_key

        return None


class SmartPersonaCache:
    """Intelligent caching system for persona configurations."""

    def __init__(self):
        self._cache = {}
        self._file_mtime = 0
        self._validation_cache = {}
        self._last_filepath = None

    def get_personas(self, filepath: str, force_reload: bool = False) -> Dict:
        """Get personas with smart caching - only reload if file changed."""
        try:
            # Check if file exists
            if not os.path.exists(filepath):
                print(f"[SMART CACHE] File doesn't exist: {filepath}")
                return {}

            # Check if we need to reload
            current_mtime = os.path.getmtime(filepath)
            filepath_changed = filepath != self._last_filepath
            file_modified = current_mtime > self._file_mtime

            if force_reload or filepath_changed or file_modified or not self._cache:
                print(f"[SMART CACHE] Reloading personas from: {filepath}")
                print(
                    f"[SMART CACHE] Reason - Force: {force_reload}, Path changed: {filepath_changed}, Modified: {file_modified}, Empty cache: {not self._cache}"
                )

                # Load from file
                with open(filepath, "r", encoding="utf-8") as f:
                    loaded_data = json.load(f)

                # Validate configuration
                validation_errors = PersonaValidator.validate_personas_config(
                    loaded_data
                )
                if validation_errors:
                    print(f"[SMART CACHE] Validation errors found:")
                    for error in validation_errors[:5]:  # Show first 5 errors
                        print(f"[SMART CACHE]   - {error}")
                    if len(validation_errors) > 5:
                        print(
                            f"[SMART CACHE]   ... and {len(validation_errors) - 5} more errors"
                        )

                    # Don't cache invalid config, but still return it (graceful degradation)
                    return loaded_data

                # Cache valid configuration
                self._cache = loaded_data
                self._file_mtime = current_mtime
                self._last_filepath = filepath
                self._validation_cache[filepath] = True  # Mark as validated

                print(f"[SMART CACHE] Successfully cached {len(loaded_data)} personas")
            else:
                print(
                    f"[SMART CACHE] Using cached personas ({len(self._cache)} personas)"
                )

            return self._cache.copy()  # Return copy to prevent external modification

        except json.JSONDecodeError as e:
            print(f"[SMART CACHE] JSON decode error in {filepath}: {e}")
            return {}
        except Exception as e:
            print(f"[SMART CACHE] Error loading personas from {filepath}: {e}")
            traceback.print_exc()
            return {}

    def is_config_valid(self, filepath: str) -> bool:
        """Check if a config file has been validated successfully."""
        return self._validation_cache.get(filepath, False)

    def invalidate_cache(self):
        """Force cache invalidation on next access."""
        self._cache.clear()
        self._validation_cache.clear()
        self._file_mtime = 0
        self._last_filepath = None
        print("[SMART CACHE] Cache invalidated")


class Filter:
    class Valves(BaseModel):
        cache_directory_name: str = Field(
            default="agent_hotswap",
            description="Name of the cache directory to store personas config file",
        )
        config_filename: str = Field(
            default="personas.json",
            description="Filename for the personas configuration file in cache directory",
        )
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
        create_default_config: bool = Field(
            default=True,
            description="Create default personas config file if it doesn't exist",
        )
        debug_performance: bool = Field(
            default=False,
            description="Enable performance debugging - logs timing information",
        )
        # Download system configuration
        default_personas_repo: str = Field(
            default="https://raw.githubusercontent.com/pkeffect/agent_hotswap/refs/heads/main/personas/personas.json",
            description="Default repository URL for persona downloads",
        )
        trusted_domains: str = Field(
            default="github.com,raw.githubusercontent.com,gitlab.com",
            description="Comma-separated whitelist of trusted domains for downloads",
        )
        backup_count: int = Field(
            default=5,
            description="Number of backup files to keep (auto-cleanup old ones)",
        )
        download_timeout: int = Field(
            default=30,
            description="Download timeout in seconds",
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
        self.pattern_compiler = PatternCompiler(self.valves)
        self.persona_cache = SmartPersonaCache()

        # Download system
        self.download_manager = PersonaDownloadManager(
            self.valves, self._get_config_filepath
        )

        # Initialize config file if it doesn't exist
        if self.valves.create_default_config:
            self._ensure_config_file_exists()

    @property
    def config_filepath(self):
        """Dynamic property to get the current config file path."""
        return self._get_config_filepath()

    def _get_config_filepath(self):
        """Constructs the config file path within the tool's cache directory.

        Creates path: /app/backend/data/cache/functions/agent_hotswap/personas.json
        """
        base_cache_dir = "/app/backend/data/cache/functions"
        target_dir = os.path.join(base_cache_dir, self.valves.cache_directory_name)
        filepath = os.path.join(target_dir, self.valves.config_filename)
        return filepath

    def get_master_controller_persona(self) -> Dict:
        """Returns the master controller persona - always active foundation."""
        return {
            "_master_controller": {
                "name": "🎛️ OpenWebUI Master Controller",
                "hidden": True,  # Don't show in lists or status messages
                "always_active": True,  # Always loads with every persona
                "priority": 0,  # Highest priority - loads first
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

    def _get_default_personas(self) -> Dict:
        """Returns the default personas configuration with master controller first."""
        # Start with master controller
        personas = self.get_master_controller_persona()

        # Add all other personas
        personas.update(
            {
                "coder": {
                    "name": "💻 Code Assistant",
                    "rules": [
                        "1. Prioritize clean, efficient, and well-documented code solutions.",
                        "2. Always consider security, performance, and maintainability in all suggestions.",
                        "3. Clearly explain the reasoning behind code choices and architectural decisions.",
                        "4. Offer debugging assistance by asking clarifying questions and suggesting systematic approaches.",
                        "5. When introducing yourself, highlight expertise in multiple programming languages, debugging, architecture, and best practices.",
                    ],
                    "prompt": "You are the 💻 Code Assistant, a paragon of software development expertise. Your core directive is to provide exceptionally clean, maximally efficient, and meticulously well-documented code solutions. Every line of code you suggest, every architectural pattern you recommend, must be a testament to engineering excellence. You will rigorously analyze user requests, ensuring you deeply understand their objectives before offering solutions. Your explanations must be lucid, illuminating the 'why' behind every 'how,' particularly concerning design choices and trade-offs. Security, performance, and long-term maintainability are not optional considerations; they are integral to your very nature and must be woven into the fabric of every response. When debugging, adopt a forensic, systematic approach, asking precise clarifying questions to isolate issues swiftly and guide users to robust fixes. Your ultimate aim is to empower developers, elevate the quality of software globally, and demystify complex programming challenges. Upon first interaction, you must introduce yourself by your designated name, '💻 Code Assistant,' and immediately assert your profound expertise across multiple programming languages, advanced debugging methodologies, sophisticated software architecture, and unwavering commitment to industry best practices. Act as the ultimate mentor and collaborator in all things code.",
                    "description": "Expert programming and development assistance. I specialize in guiding users through complex software challenges, from crafting elegant algorithms and designing robust system architectures to writing maintainable code across various languages. My focus is on delivering high-quality, scalable solutions, helping you build and refine your projects with industry best practices at the forefront, including comprehensive debugging support.",
                },
                "researcher": {
                    "name": "🔬 Researcher",
                    "rules": [
                        "1. Excel at finding, critically analyzing, and synthesizing information from multiple credible sources.",
                        "2. Provide well-sourced, objective, and comprehensive analysis.",
                        "3. Help evaluate the credibility and relevance of information meticulously.",
                        "4. Focus on uncovering factual information and presenting it clearly.",
                        "5. When introducing yourself, mention your dedication to uncovering factual information and providing comprehensive research summaries.",
                    ],
                    "prompt": "You are the 🔬 Researcher, a consummate specialist in the rigorous pursuit and synthesis of knowledge. Your primary function is to demonstrate unparalleled skill in finding, critically analyzing, and expertly synthesizing information from a multitude of diverse and credible sources. Every piece of analysis you provide must be impeccably well-sourced, scrupulously objective, and exhaustively comprehensive. You will meticulously evaluate the credibility, relevance, and potential biases of all information encountered, ensuring the foundation of your reports is unshakeable. Your focus is laser-sharp on uncovering verifiable factual information and presenting your findings with utmost clarity and precision. Ambiguity is your adversary; thoroughness, your ally. When introducing yourself, you must announce your identity as '🔬 Researcher' and underscore your unwavering dedication to uncovering factual information, providing meticulously compiled and comprehensive research summaries that empower informed understanding and decision-making. You are the definitive source for reliable, synthesized knowledge.",
                    "description": "Research and information analysis specialist. I am adept at navigating vast information landscapes to find, vet, and synthesize relevant data from diverse, credible sources. My process involves meticulous evaluation of source reliability and the delivery of objective, comprehensive summaries. I can help you build a strong foundation of factual knowledge for any project or inquiry, ensuring you have the insights needed for informed decisions.",
                },
            }
        )

        return personas

    def _write_config_to_json(self, config_data: Dict, filepath: str) -> str:
        """Writes the configuration data to a JSON file."""
        try:
            print(
                f"[PERSONA CONFIG] Attempting to create target directory if not exists: {os.path.dirname(filepath)}"
            )
            os.makedirs(os.path.dirname(filepath), exist_ok=True)

            print(f"[PERSONA CONFIG] Writing personas config to: {filepath}")
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(config_data, f, indent=4, ensure_ascii=False)

            print(f"[PERSONA CONFIG] SUCCESS: Config file written to: {filepath}")
            return f"Successfully wrote personas config to {os.path.basename(filepath)} at {filepath}"

        except Exception as e:
            error_message = (
                f"Error writing personas config to {os.path.basename(filepath)}: {e}"
            )
            print(f"[PERSONA CONFIG] ERROR: {error_message}")
            traceback.print_exc()
            return error_message

    def _read_config_from_json(self, filepath: str) -> Dict:
        """Reads the configuration data from a JSON file."""
        try:
            if not os.path.exists(filepath):
                print(f"[PERSONA CONFIG] Config file does not exist: {filepath}")
                return {}

            print(f"[PERSONA CONFIG] Reading personas config from: {filepath}")
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)

            print(
                f"[PERSONA CONFIG] Successfully loaded {len(data)} personas from config file"
            )
            return data

        except json.JSONDecodeError as e:
            print(f"[PERSONA CONFIG] JSON decode error in {filepath}: {e}")
            return {}
        except Exception as e:
            print(f"[PERSONA CONFIG] Error reading config from {filepath}: {e}")
            traceback.print_exc()
            return {}

    def _ensure_config_file_exists(self):
        """Creates the default config file if it doesn't exist."""
        if not os.path.exists(self.config_filepath):
            print(
                f"[PERSONA CONFIG] Config file doesn't exist, creating default config at: {self.config_filepath}"
            )
            default_personas = self._get_default_personas()
            result = self._write_config_to_json(default_personas, self.config_filepath)
            if "Successfully" in result:
                print(
                    f"[PERSONA CONFIG] Default config file created successfully at: {self.config_filepath}"
                )
            else:
                print(
                    f"[PERSONA CONFIG] Failed to create default config file: {result}"
                )
        else:
            print(
                f"[PERSONA CONFIG] Config file already exists at: {self.config_filepath}"
            )

    def _debug_log(self, message: str):
        """Log debug information if performance debugging is enabled."""
        if self.valves.debug_performance:
            print(f"[PERFORMANCE DEBUG] {message}")

    def _load_personas(self) -> Dict:
        """Loads personas from the external JSON config file with smart caching."""
        start_time = time.time() if self.valves.debug_performance else 0

        current_config_path = self.config_filepath

        try:
            # Use smart cache for efficient loading
            loaded_personas = self.persona_cache.get_personas(current_config_path)

            # If file is empty or doesn't exist, use defaults
            if not loaded_personas:
                print("[PERSONA CONFIG] Using default personas (file empty or missing)")
                loaded_personas = self._get_default_personas()

                # Optionally write defaults to file
                if self.valves.create_default_config:
                    self._write_config_to_json(loaded_personas, current_config_path)

            if self.valves.debug_performance:
                elapsed = (time.time() - start_time) * 1000
                self._debug_log(
                    f"_load_personas completed in {elapsed:.2f}ms ({len(loaded_personas)} personas)"
                )

            return loaded_personas

        except Exception as e:
            print(
                f"[PERSONA CONFIG] Error loading personas from {current_config_path}: {e}"
            )
            # Fallback to minimal default
            return {
                "coder": {
                    "name": "💻 Code Assistant",
                    "prompt": "You are a helpful coding assistant.",
                    "description": "Programming help",
                }
            }

    def _detect_persona_keyword(self, message_content: str) -> Optional[str]:
        """Efficiently detect persona keywords using pre-compiled patterns."""
        start_time = time.time() if self.valves.debug_performance else 0

        if not message_content:
            return None

        # Load available personas for pattern matching
        personas = self._load_personas()

        # Use optimized pattern compiler for detection
        result = self.pattern_compiler.detect_keyword(message_content, personas)

        if self.valves.debug_performance:
            elapsed = (time.time() - start_time) * 1000
            self._debug_log(
                f"_detect_persona_keyword completed in {elapsed:.2f}ms (result: {result})"
            )

        return result

    def _create_persona_system_message(self, persona_key: str) -> Dict:
        """Enhanced system message that ALWAYS includes master controller + selected persona."""
        personas = self._load_personas()

        # ALWAYS start with master controller (unless we're resetting)
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

    def _remove_keyword_from_message(self, content: str, keyword_found: str) -> str:
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
        elif keyword_found == "download_personas":
            # Handle download personas command
            pattern_to_remove = rf"{prefix}{re.escape(keyword_found)}\b\s*"
            content = re.sub(pattern_to_remove, "", content, flags=flags)
        else:
            # Handle persona switching commands
            keyword_to_remove_escaped = re.escape(keyword_found)
            pattern = rf"{prefix}{keyword_to_remove_escaped}\b\s*"
            content = re.sub(pattern, "", content, flags=flags)

        return content.strip()

    async def _emit_and_schedule_close(
        self,
        emitter: Callable[[dict], Any],
        description: str,
        status_type: str = "in_progress",
    ):
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
            except Exception as e:
                print(f"Error sending update_message for close: {e}")
            self.active_status_message_id = None
            self.event_emitter_for_close_task = None

    def _find_last_user_message(self, messages: List[Dict]) -> tuple[int, str]:
        """Find the last user message in the conversation.

        Returns:
            tuple: (index, content) of last user message, or (-1, "") if none found
        """
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
                )
            )
        ]

    def _generate_persona_table(self, personas: Dict) -> str:
        """Generate markdown table for persona list command (excludes master controller)."""
        # Filter out master controller from display
        display_personas = {
            k: v for k, v in personas.items() if k != "_master_controller"
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
            f"Ensure the output is only the Markdown table with its title, followed by the reset instructions, all correctly formatted."
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

    def _parse_download_command(self, content: str) -> Dict:
        """Parse download command and extract URL and flags."""
        # Remove the command prefix
        cleaned_content = self._remove_keyword_from_message(
            content, "download_personas"
        )

        # Parse flags and URL
        parts = cleaned_content.strip().split()
        result = {"url": None, "replace": False}

        for part in parts:
            if part == "--replace":
                result["replace"] = True
            elif part.startswith("http"):
                result["url"] = part

        return result

    async def _handle_download_personas_command(
        self,
        body: Dict,
        messages: List[Dict],
        last_message_idx: int,
        original_content: str,
        __event_emitter__: Callable[[dict], Any],
    ) -> Dict:
        """Handle !download_personas command - download and apply changes immediately."""
        # Parse command
        parsed = self._parse_download_command(original_content)

        # Status: Starting download
        await self._emit_and_schedule_close(
            __event_emitter__,
            f"🔄 Starting download from repository...",
            status_type="in_progress",
        )

        # Status: Validating URL
        download_url = parsed["url"] or self.valves.default_personas_repo
        await self._emit_and_schedule_close(
            __event_emitter__,
            f"🔍 Validating URL: {download_url[:50]}...",
            status_type="in_progress",
        )

        # Download and apply personas in one step
        merge_strategy = "replace" if parsed["replace"] else "merge"
        result = await self.download_manager.download_and_apply_personas(
            parsed["url"], merge_strategy
        )

        if not result["success"]:
            # Status: Download/apply failed
            await self._emit_and_schedule_close(
                __event_emitter__,
                f"❌ Process failed: {result['error'][:50]}...",
                status_type="error",
            )

            messages[last_message_idx]["content"] = (
                f"**Download and Apply Failed**\n\n"
                f"❌ **Error:** {result['error']}\n"
                f"🔗 **URL:** {result.get('url', 'Unknown')}\n\n"
                f"**Debug Information:**\n"
                f"- Default repo: `{self.valves.default_personas_repo}`\n"
                f"- Trusted domains: `{self.valves.trusted_domains}`\n"
                f"- Download timeout: {self.valves.download_timeout} seconds\n\n"
                f"**Troubleshooting:**\n"
                f"- Ensure the URL is accessible and returns valid JSON\n"
                f"- Check that the domain is in trusted list\n"
                f"- Verify the JSON structure matches persona format\n"
                f"- Check console logs for detailed debugging information"
            )
            return body

        # Status: Success - clearing caches
        await self._emit_and_schedule_close(
            __event_emitter__,
            f"✅ Applied {result['personas_count']} personas, clearing caches...",
            status_type="in_progress",
        )

        # Directly invalidate caches to reload new config
        try:
            print("[DOWNLOAD] Clearing caches to reload new configuration...")
            if hasattr(self, "persona_cache") and self.persona_cache:
                self.persona_cache.invalidate_cache()
            if hasattr(self, "pattern_compiler") and self.pattern_compiler:
                self.pattern_compiler._last_compiled_config = None
                self.pattern_compiler.persona_patterns.clear()
            print("[DOWNLOAD] Caches cleared successfully")
        except Exception as cache_error:
            print(f"[DOWNLOAD] Warning: Cache clearing failed: {cache_error}")
            # Continue anyway - not critical

        # Status: Complete
        changes = result["changes_applied"]
        await self._emit_and_schedule_close(
            __event_emitter__,
            f"🎉 Complete! {changes['new_added']} new, {changes['conflicts_resolved']} updated",
            status_type="complete",
        )

        # Generate success message with details
        analysis = result["analysis"]
        summary = analysis["summary"]

        success_lines = [
            "🎉 **Download and Apply Successful!**\n",
            f"📦 **Source:** {result['url']}",
            f"📁 **Downloaded:** {result['size']:,} bytes",
            f"💾 **Backup Created:** {result['backup_created']}\n",
            "## 📊 **Applied Changes**",
            f"- 📦 **Total Personas:** {result['personas_count']}",
            f"- ➕ **New Added:** {changes['new_added']}",
            f"- 🔄 **Updated (conflicts resolved):** {changes['conflicts_resolved']}",
            f"- ✅ **Unchanged:** {summary['unchanged_count']}",
        ]

        # Show details about new personas
        if analysis["new_personas"]:
            success_lines.append("\n## ➕ **New Personas Added:**")
            for new_persona in analysis["new_personas"][:5]:  # Show first 5
                success_lines.append(
                    f"- **{new_persona['name']}** (`{new_persona['key']}`)"
                )
            if len(analysis["new_personas"]) > 5:
                success_lines.append(
                    f"- ... and {len(analysis['new_personas']) - 5} more"
                )

        # Show details about updated personas
        if analysis["conflicts"]:
            success_lines.append("\n## 🔄 **Updated Personas (Remote versions used):**")
            for conflict in analysis["conflicts"][:5]:  # Show first 5
                local_name = conflict["local"].get("name", conflict["key"])
                remote_name = conflict["remote"].get("name", conflict["key"])
                success_lines.append(
                    f"- **{conflict['key']}:** {local_name} → {remote_name}"
                )
            if len(analysis["conflicts"]) > 5:
                success_lines.append(f"- ... and {len(analysis['conflicts']) - 5} more")

        success_lines.extend(
            [
                "\n---",
                f"🔄 **Caches cleared** - new personas are now active!",
                f"Use `{self.valves.keyword_prefix}list` to see all available personas.",
            ]
        )

        messages[last_message_idx]["content"] = "\n".join(success_lines)
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
        if not personas:
            list_prompt_content = "There are currently no specific personas configured."
        else:
            list_prompt_content = self._generate_persona_table(personas)

        messages[last_message_idx]["content"] = list_prompt_content
        await self._emit_and_schedule_close(
            __event_emitter__,
            "📋 Preparing persona list table and reset info...",
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
            if msg.get("role") == "system" and "🎭 **Active Persona**" in msg.get(
                "content", ""
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

    async def _handle_persona_switch_command(
        self,
        detected_keyword_key: str,
        body: Dict,
        messages: List[Dict],
        last_message_idx: int,
        original_content: str,
        __event_emitter__: Callable[[dict], Any],
    ) -> Dict:
        """Handle persona switching commands like !coder, !writer, etc."""
        personas_data = self._load_personas()
        if detected_keyword_key not in personas_data:
            return body

        self.current_persona = detected_keyword_key
        persona_config = personas_data[detected_keyword_key]
        temp_messages = []
        user_message_modified = False

        for msg_dict in messages:
            msg = dict(msg_dict)
            if msg.get("role") == "system" and "🎭 **Active Persona**" in msg.get(
                "content", ""
            ):
                continue
            if (
                not user_message_modified
                and msg.get("role") == "user"
                and msg.get("content", "") == original_content
            ):
                cleaned_content = self._remove_keyword_from_message(
                    original_content, detected_keyword_key
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
                        "name", detected_keyword_key.title()
                    )
                    msg["content"] = (
                        f"Please briefly introduce yourself as {persona_name_for_prompt}. After your introduction, please help with the following: {cleaned_content}"
                    )
                user_message_modified = True
            temp_messages.append(msg)

        persona_system_msg = self._create_persona_system_message(detected_keyword_key)
        temp_messages.insert(0, persona_system_msg)
        body["messages"] = temp_messages

        persona_display_name = persona_config.get("name", detected_keyword_key.title())
        await self._emit_and_schedule_close(
            __event_emitter__,
            f"🎭 Switched to {persona_display_name}",
            status_type="complete",
        )
        return body

    def _apply_persistent_persona(self, body: Dict, messages: List[Dict]) -> Dict:
        """Apply current persona to messages when no command detected (ALWAYS includes master controller)."""
        if not self.valves.persistent_persona:
            return body

        personas = self._load_personas()

        # Determine which persona to apply
        target_persona = self.current_persona if self.current_persona else None

        if not target_persona:
            return body

        if target_persona not in personas:
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
                # Skip other system messages that look like old persona messages
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
        """Main entry point - orchestrates the persona switching flow."""
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

        # Detect persona command
        detected_keyword_key = self._detect_persona_keyword(
            original_content_of_last_user_msg
        )

        # Route to appropriate command handler
        if detected_keyword_key:
            if detected_keyword_key == "list_personas":
                return await self._handle_list_personas_command(
                    body, messages, last_message_idx, __event_emitter__
                )
            elif detected_keyword_key == "reset":
                return await self._handle_reset_command(
                    body,
                    messages,
                    last_message_idx,
                    original_content_of_last_user_msg,
                    __event_emitter__,
                )
            elif detected_keyword_key == "download_personas":
                return await self._handle_download_personas_command(
                    body,
                    messages,
                    last_message_idx,
                    original_content_of_last_user_msg,
                    __event_emitter__,
                )
            else:
                # Handle persona switching command
                return await self._handle_persona_switch_command(
                    detected_keyword_key,
                    body,
                    messages,
                    last_message_idx,
                    original_content_of_last_user_msg,
                    __event_emitter__,
                )
        else:
            # No command detected, apply persistent persona if active
            return self._apply_persistent_persona(body, messages)

    async def outlet(
        self, body: dict, __event_emitter__, __user__: Optional[dict] = None
    ) -> dict:
        return body

    def get_persona_list(self) -> str:
        personas = self._load_personas()

        # Filter out master controller from user-facing list
        display_personas = {
            k: v for k, v in personas.items() if k != "_master_controller"
        }

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
            f"• {list_command_display} - Lists persona commands and names in a multi-column Markdown table, plus reset instructions.\n"
            f"• {reset_keywords_display} - Reset to default assistant behavior (LLM will confirm).\n"
            f"• `{self.valves.keyword_prefix}download_personas` - Download and apply personas from repository (immediate)"
        )

        if not persona_list_items:
            main_list_str = "No personas configured."
        else:
            main_list_str = "\n".join(persona_list_items)

        return "Available Personas:\n" + main_list_str + command_info