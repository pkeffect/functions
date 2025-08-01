"""
title: Conversation Summarizer - Popup Window
author: assistant & pkeffect
author_url: https://github.com/pkeffect
version: 2.3.3
required_open_webui_version: 0.5.0
license: MIT
description: Advanced conversation summarizer with popup window UI, full Agent Hotswap and Multi-Model integration
requirements: pydantic>=2.0.0, aiofiles>=23.0.0, aiohttp>=3.8.0
"""

from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
import json
import time
import logging
import re
import os
import asyncio
import aiofiles
import aiohttp
import urllib.parse
from datetime import datetime
from pathlib import Path

# OpenWebUI Native Imports
try:
    from open_webui.models.chats import Chats
    from open_webui.models.users import Users
    from open_webui.internal.db import get_db
    from open_webui.utils.chat import generate_chat_completion

    NATIVE_DB_AVAILABLE = True
except ImportError:
    NATIVE_DB_AVAILABLE = False
    logging.warning("Native OpenWebUI database models not available")

logger = logging.getLogger(__name__)

# Configuration
CONFIG_FILE = "config.json"
UI_FILE = "index.html"
SUMMARY_DATA_FILE = "summary.json"
DEFAULT_REPO = "https://raw.githubusercontent.com/pkeffect/functions/refs/heads/main/functions/filters/summarizer/config.json"
UI_REPO = "https://raw.githubusercontent.com/pkeffect/functions/refs/heads/main/functions/filters/summarizer/index.html"
TRUSTED_DOMAINS = ["github.com", "raw.githubusercontent.com"]

# Global cache for better performance
_GLOBAL_CACHE = {}
_CACHE_LOCK = asyncio.Lock()


class SummarizerDownloader:
    """Downloads config and UI files - EXACT same pattern as Agent Hotswap PersonaDownloader"""
    
    def __init__(self, config_path_func):
        self.get_config_path = config_path_func
        self._session = None

    async def _get_session(self):
        if self._session is None or self._session.closed:
            timeout = aiohttp.ClientTimeout(total=30)
            self._session = aiohttp.ClientSession(
                timeout=timeout,
                headers={"User-Agent": "OpenWebUI-ConversationSummarizer/2.3.3"},
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

    async def download_config_and_ui(
        self,
        url: str = None,
        force_download: bool = False,
    ) -> Dict:
        """Download config and UI files - EXACT same pattern as Agent Hotswap download_personas"""
        download_url = url or DEFAULT_REPO
        if not self.is_trusted_domain(download_url):
            return {"success": False, "error": "Untrusted domain"}

        # Download UI file alongside config - EXACT Agent Hotswap pattern
        ui_result = await self.download_ui_file(force_download=force_download)
        ui_success = ui_result.get("success", False)
        if not ui_success:
            print(
                f"[SUMMARIZER] UI download failed: {ui_result.get('error', 'unknown')}"
            )
        elif not ui_result.get("skipped"):
            print(
                f"[SUMMARIZER] UI file downloaded: {ui_result.get('size', 0)} bytes"
            )

        try:
            session = await self._get_session()
            async with session.get(download_url) as response:
                if response.status != 200:
                    return {"success": False, "error": f"HTTP {response.status}"}
                content = await response.text()
                if len(content) > 1024 * 1024 * 2:
                    return {"success": False, "error": "File too large"}
                
                # Try to parse as JSON, or create minimal config if it fails
                try:
                    remote_config = json.loads(content)
                    if not isinstance(remote_config, dict):
                        remote_config = self._create_default_config()
                except json.JSONDecodeError:
                    print("[SUMMARIZER] Downloaded content not valid JSON, creating default config")
                    remote_config = self._create_default_config()

                config_path = self.get_config_path()
                os.makedirs(os.path.dirname(config_path), exist_ok=True)
                async with aiofiles.open(config_path, "w", encoding="utf-8") as f:
                    await f.write(
                        json.dumps(remote_config, indent=2, ensure_ascii=False)
                    )

                return {
                    "success": True,
                    "size": len(content),
                    "ui_downloaded": ui_success and not ui_result.get("skipped"),
                    "ui_status": "success" if ui_success else "failed",
                }
        except Exception as e:
            return {"success": False, "error": str(e)}

    def _create_default_config(self) -> Dict:
        """Create default config with summary templates - like Agent Hotswap personas"""
        return {
            "summary_templates": {
                "simple": {
                    "name": "📋 Simple Summary",
                    "prompt": """You are a conversation analyst specializing in AI conversations. Write a concise 1-2 paragraph summary of this conversation.

Guidelines:
- Focus on main topics and key conclusions
- Write in prose, not bullet points  
- Be objective and capture the conversation flow
- Note any special features (personas, reasoning, multi-model interactions)
- Start directly without preamble

Provide only the summary content.""",
                    "description": "Quick overview of conversation highlights",
                    "max_tokens": 600
                },
                "complete": {
                    "name": "🚀 Intelligence Report",
                    "prompt": """You are a comprehensive conversation analyst specializing in AI conversations. Create a thorough analysis of this conversation including:

1. HIGHLIGHTS: 5-7 key bullet points of the most important topics and insights
2. PROS & CONS: What worked well and what could be improved in the conversation
3. DETAILED SUMMARY: 3-4 paragraphs providing comprehensive analysis of the conversation flow, key decisions, insights, and outcomes

Guidelines:
- Be thorough and analytical
- Include specific examples and quotes when relevant
- Note conversation dynamics and effectiveness
- Analyze the quality of responses and engagement
- Consider the educational or practical value
- Be objective but insightful

Format your response as:

**HIGHLIGHTS:**
• [Key point 1]
• [Key point 2]
• [etc.]

**PROS & CONS:**

*What Worked Well:*
• [Positive aspect 1]
• [Positive aspect 2]

*Areas for Improvement:*
• [Improvement area 1]
• [Improvement area 2]

**COMPREHENSIVE ANALYSIS:**

[3-4 detailed paragraphs analyzing the conversation thoroughly]""",
                    "description": "Comprehensive conversation analysis with detailed insights",
                    "max_tokens": 1200
                },
                "technical": {
                    "name": "🔧 Technical Analysis",
                    "prompt": """You are a technical conversation analyst specializing in development discussions. Analyze this conversation focusing on:

1. TECHNICAL TOPICS: Key technologies, frameworks, and tools discussed
2. CODE QUALITY: Analysis of any code snippets, solutions, or approaches
3. IMPLEMENTATION: Practical steps, decisions, and technical outcomes
4. LEARNING OUTCOMES: Technical knowledge gained or shared

Guidelines:
- Focus on technical accuracy and implementation details
- Highlight code examples, debugging processes, and solutions
- Note any architectural decisions or design patterns
- Assess the technical depth and educational value
- Write in a clear, technical but accessible style

Provide a comprehensive technical summary.""",
                    "description": "Technical conversation breakdown focusing on code and implementation",
                    "max_tokens": 800
                }
            },
            
            "integration_settings": {
                "persona_detection": True,
                "multi_model_analysis": True,
                "reasoning_extraction": True,
                "complexity_scoring": True,
                "auto_categorization": True
            },
            
            "ui_settings": {
                "default_theme": "auto",
                "show_metadata": True,
                "show_integration_badges": True,
                "enable_export": True,
                "auto_scroll": True,
                "compact_mode": False
            },
            
            "prompt_enhancements": {
                "persona_context": "This conversation involved AI personas. Note which personas participated and how they contributed their expertise.",
                "multi_model_context": "This was a multi-model conversation with {model_count} AI models. Summarize how different models contributed different perspectives.",
                "reasoning_context": "This conversation included {reasoning_blocks} reasoning processes. Summarize key insights from the thinking processes."
            },
            
            "_metadata": {
                "version": "2.3.3",
                "last_updated": datetime.now().isoformat(),
                "plugin_directory": "conversation_summarizer",
                "popup_ui": True,
                "integration_ready": True,
                "initial_setup": True,
                "async_enabled": True,
                "template_count": 3,
                "supports_customization": True
            }
        }


class PluginIntegrationDetector:
    """Detects and extracts context from other plugins in the suite"""

    @staticmethod
    def detect_agent_hotswap_context(filter_context: Dict) -> Dict:
        """Extract Agent Hotswap persona context"""
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

    @staticmethod
    def detect_multi_model_context(body: Dict) -> Dict:
        """Detect Multi-Model conversation context"""
        messages = body.get("messages", [])

        # Check for multi-model conversation indicators
        has_multi_model = False
        model_count = 0
        reasoning_indicators = 0

        for message in messages:
            content = message.get("content", "")

            # Look for multi-model conversation patterns
            if "Multi-Model" in content or "Running on" in content:
                has_multi_model = True

            # Count model icons/indicators
            model_indicators = [
                "🦙",
                "🧠",
                "🌪️",
                "💻",
                "🔬",
                "💎",
                "🌟",
                "🎯",
                "🤿",
                "🤖",
            ]
            for indicator in model_indicators:
                if indicator in content:
                    model_count += 1
                    break

            # Count reasoning processes
            if "🤔 **" in content and "Reasoning Process" in content:
                reasoning_indicators += 1

        return {
            "has_multi_model": has_multi_model,
            "estimated_model_count": model_count,
            "reasoning_responses": reasoning_indicators,
            "conversation_type": "multi_model" if has_multi_model else "single_model",
        }


class ModelManager:
    """Enhanced model management with integration awareness"""

    @staticmethod
    def get_available_models() -> List[str]:
        """Get available models with simple fallback"""
        try:
            from open_webui.main import app

            if hasattr(app, "state") and hasattr(app.state, "MODELS"):
                models = list(app.state.MODELS.keys())
                return ["auto"] + sorted(models) if models else ["auto"]
        except:
            pass

        # Enhanced fallback models including reasoning models
        return [
            "auto",
            "llama3.2:latest",
            "qwen2.5:latest",
            "deepseek-r1:latest",
            "mistral:latest",
            "gpt-4o",
            "claude-3-sonnet",
        ]

    @staticmethod
    def get_model_dropdown() -> Dict:
        """Get schema for model dropdown"""
        return {"enum": ModelManager.get_available_models()}


class ConversationManager:
    """Enhanced conversation history management with plugin integration awareness"""

    def __init__(self):
        self.cache = {}
        self.cache_ttl = 300  # 5 minutes

    async def get_conversation_history(
        self, chat_id: str, user_id: str, current_messages: List[Dict] = None
    ):
        """Get conversation history with enhanced metadata"""
        cache_key = f"{chat_id}_{user_id}"

        # Check cache
        if cache_key in self.cache:
            cached_data = self.cache[cache_key]
            if time.time() - cached_data["timestamp"] < self.cache_ttl:
                return cached_data["messages"], cached_data["metadata"]

        # Try database
        if NATIVE_DB_AVAILABLE:
            try:
                with get_db() as db:
                    chat = (
                        db.query(Chats)
                        .filter(Chats.id == chat_id)
                        .filter(Chats.user_id == user_id)
                        .first()
                    )

                    if chat and hasattr(chat, "chat") and chat.chat:
                        messages = chat.chat.get("messages", [])

                        # Enhanced metadata with integration detection
                        metadata = {
                            "source": "database",
                            "total_messages": len(messages),
                            "conversation_messages": len(
                                [
                                    m
                                    for m in messages
                                    if m.get("role") in ["user", "assistant"]
                                ]
                            ),
                            "user_messages": len(
                                [m for m in messages if m.get("role") == "user"]
                            ),
                            "assistant_messages": len(
                                [m for m in messages if m.get("role") == "assistant"]
                            ),
                            "title": getattr(chat, "title", None)
                            or chat.chat.get("title", "Untitled Chat"),
                            "created_at": getattr(chat, "created_at", None),
                            "updated_at": getattr(chat, "updated_at", None),
                        }

                        # Analyze conversation for plugin usage
                        analysis = self._analyze_conversation_patterns(messages)
                        metadata.update(analysis)

                        # Calculate turns and model interactions
                        metadata.update(self._calculate_turn_metrics(messages))

                        # Cache the result
                        self.cache[cache_key] = {
                            "messages": messages,
                            "metadata": metadata,
                            "timestamp": time.time(),
                        }

                        return messages, metadata
            except Exception as e:
                logger.debug(f"Database access failed: {e}")

        # Fallback to current messages with enhanced analysis
        current_messages = current_messages or []
        conv_messages = [
            m for m in current_messages if m.get("role") in ["user", "assistant"]
        ]

        metadata = {
            "source": "current_request",
            "total_messages": len(current_messages),
            "conversation_messages": len(conv_messages),
            "user_messages": len(
                [m for m in current_messages if m.get("role") == "user"]
            ),
            "assistant_messages": len(
                [m for m in current_messages if m.get("role") == "assistant"]
            ),
            "title": "Current Session",
            "warning": "Using current messages only - full history unavailable",
        }

        # Analyze current messages for patterns
        analysis = self._analyze_conversation_patterns(current_messages)
        metadata.update(analysis)

        # Calculate turns and model interactions
        metadata.update(self._calculate_turn_metrics(current_messages))

        return current_messages, metadata

    def _calculate_turn_metrics(self, messages: List[Dict]) -> Dict:
        """Calculate detailed turn and interaction metrics"""
        metrics = {
            "conversation_turns": 0,
            "total_model_responses": 0,
            "unique_models": set(),
            "reasoning_turns": 0,
            "persona_switches": 0,
        }

        # Track conversation turns (user message + assistant response = 1 turn)
        user_messages = [m for m in messages if m.get("role") == "user"]
        assistant_messages = [m for m in messages if m.get("role") == "assistant"]
        metrics["conversation_turns"] = min(len(user_messages), len(assistant_messages))

        # Track multi-model specific metrics
        for msg in messages:
            content = msg.get("content", "")

            # Count model responses and extract model info
            if "Running on" in content:
                metrics["total_model_responses"] += 1
                model_match = re.search(r"Running on ([^•]+)", content)
                if model_match:
                    model_name = model_match.group(1).strip()
                    metrics["unique_models"].add(model_name)

            # Count reasoning processes
            if "🤔 **" in content and "Reasoning Process" in content:
                metrics["reasoning_turns"] += 1

            # Count persona switches
            if "🎭 **" in content:
                metrics["persona_switches"] += content.count("🎭 **")

        # Convert set to count
        metrics["unique_models"] = len(metrics["unique_models"])

        return metrics

    def _analyze_conversation_patterns(self, messages: List[Dict]) -> Dict:
        """Analyze conversation for plugin usage patterns"""
        analysis = {
            "has_personas": False,
            "has_multi_model": False,
            "has_reasoning": False,
            "persona_switches": 0,
            "model_count": 0,
            "reasoning_blocks": 0,
            "conversation_complexity": "simple",
        }

        persona_indicators = [
            "🎭",
            "Active Persona",
            "persona1",
            "persona2",
            "persona3",
            "persona4",
        ]
        multi_model_indicators = ["Multi-Model", "Running on", "Turn 1", "Turn 2"]
        reasoning_indicators = ["Reasoning Process", "🤔", "<details>", "<summary>"]

        for message in messages:
            content = message.get("content", "")

            # Check for personas
            for indicator in persona_indicators:
                if indicator in content:
                    analysis["has_personas"] = True
                    if "🎭" in content:
                        analysis["persona_switches"] += content.count("🎭")
                    break

            # Check for multi-model conversations
            for indicator in multi_model_indicators:
                if indicator in content:
                    analysis["has_multi_model"] = True
                    if "Running on" in content:
                        analysis["model_count"] += 1
                    break

            # Check for reasoning
            for indicator in reasoning_indicators:
                if indicator in content:
                    analysis["has_reasoning"] = True
                    if "Reasoning Process" in content:
                        analysis["reasoning_blocks"] += 1
                    break

        # Determine complexity
        complexity_score = 0
        if analysis["has_personas"]:
            complexity_score += 1
        if analysis["has_multi_model"]:
            complexity_score += 2
        if analysis["has_reasoning"]:
            complexity_score += 1
        if analysis["model_count"] > 2:
            complexity_score += 1

        if complexity_score >= 4:
            analysis["conversation_complexity"] = "highly_complex"
        elif complexity_score >= 2:
            analysis["conversation_complexity"] = "complex"
        elif complexity_score >= 1:
            analysis["conversation_complexity"] = "moderate"

        return analysis


class SummarizationEngine:
    """Enhanced summarization with persona and multi-model awareness"""

    def __init__(self, valves, config_loader_func):
        self.valves = valves
        self.get_config = config_loader_func
        self.fallback_prompt = """You are a conversation analyst specializing in AI conversations. Write a concise 1-2 paragraph summary of this conversation.

Guidelines:
- Focus on main topics and key conclusions
- Write in prose, not bullet points  
- Be objective and capture the conversation flow
- Note any special features (personas, reasoning, multi-model interactions)
- Start directly without preamble

Provide only the summary content."""

    async def _load_summary_templates(self) -> Dict:
        """Load summary templates from config file"""
        try:
            config = await self.get_config()
            return config.get("summary_templates", {})
        except Exception as e:
            print(f"[SUMMARIZER] Error loading templates: {e}")
            return {}

    def _get_template_prompt(self, template_type: str, templates: Dict) -> str:
        """Get prompt for specific template type"""
        template_map = {
            "simple": "simple",
            "complete": "complete", 
            "technical": "technical"
        }
        
        template_key = template_map.get(template_type, "simple")
        template = templates.get(template_key, {})
        return template.get("prompt", self.fallback_prompt)

    def _get_max_tokens(self, template_type: str, templates: Dict) -> int:
        """Get max tokens for template type"""
        template_map = {
            "simple": "simple",
            "complete": "complete",
            "technical": "technical"
        }
        
        template_key = template_map.get(template_type, "simple")
        template = templates.get(template_key, {})
        return template.get("max_tokens", 600)

    async def _create_enhanced_prompt(
        self, metadata: Dict, integration_context: Dict, is_complete: bool = False, is_technical: bool = False
    ) -> str:
        """Create enhanced prompt based on conversation context and templates"""
        
        # Load templates from config
        templates = await self._load_summary_templates()
        
        # Determine template type
        if is_technical:
            template_type = "technical"
        elif is_complete:
            template_type = "complete" 
        else:
            template_type = "simple"
            
        # Get base prompt from template
        base_prompt = self._get_template_prompt(template_type, templates)
        
        # Load prompt enhancements from config
        try:
            config = await self.get_config()
            enhancements = config.get("prompt_enhancements", {})
        except:
            enhancements = {}

        # Add persona-specific instructions
        if integration_context.get("has_agent_hotswap") or metadata.get("has_personas"):
            persona_context = enhancements.get("persona_context", 
                "This conversation involved AI personas. Note which personas participated and how they contributed their expertise.")
            base_prompt += f"\n\nSPECIAL FOCUS: {persona_context}"

        # Add multi-model instructions
        if metadata.get("has_multi_model"):
            model_count = metadata.get('model_count', 'multiple')
            multi_model_context = enhancements.get("multi_model_context", 
                "This was a multi-model conversation with {model_count} AI models. Summarize how different models contributed different perspectives.")
            context_text = multi_model_context.format(model_count=model_count)
            base_prompt += f"\n\nMULTI-MODEL CONTEXT: {context_text}"

        # Add reasoning instructions
        if metadata.get("has_reasoning"):
            reasoning_blocks = metadata.get('reasoning_blocks', 'several')
            reasoning_context = enhancements.get("reasoning_context",
                "This conversation included {reasoning_blocks} reasoning processes. Summarize key insights from the thinking processes.")
            context_text = reasoning_context.format(reasoning_blocks=reasoning_blocks)
            base_prompt += f"\n\nREASONING ANALYSIS: {context_text}"

        return base_prompt

    def _clean_content_for_summary(self, content: str) -> str:
        """Clean content for better summarization"""
        # Remove markdown formatting
        content = re.sub(r"\*\*(.*?)\*\*", r"\1", content)  # Bold
        content = re.sub(r"\*(.*?)\*", r"\1", content)  # Italic
        content = re.sub(r"`(.*?)`", r"\1", content)  # Code

        # Remove system messages and formatting artifacts
        content = re.sub(r"> \*Running on.*?\*\n", "", content)
        content = re.sub(r"> .*?\n", "", content)  # Remove quotes
        content = re.sub(r"---\n", "", content)  # Remove separators

        # Remove reasoning blocks for main summary
        content = re.sub(r"<details>.*?</details>", "", content, flags=re.DOTALL)

        # Clean up extra whitespace
        content = re.sub(r"\n+", " ", content)
        content = re.sub(r"\s+", " ", content)

        return content.strip()

    async def generate_summary(
        self,
        messages: List[Dict],
        model: str,
        request,
        user,
        metadata: Dict = None,
        integration_context: Dict = None,
        is_complete: bool = False,
        is_technical: bool = False,
    ) -> str:
        """Generate enhanced LLM-powered summary with template-based prompts"""
        if not user or not request:
            return "[Error: Missing context for LLM call]"

        metadata = metadata or {}
        integration_context = integration_context or {}

        # Filter conversation messages
        conv_messages = [m for m in messages if m.get("role") in ["user", "assistant"]]
        if not conv_messages:
            return "No conversation content to summarize."

        # Build enhanced transcript with context awareness
        transcript_parts = []

        for i, msg in enumerate(conv_messages):
            content = msg.get("content", "")
            role = msg["role"].upper()

            # Enhanced message processing for integration features
            if metadata.get("has_personas") and "🎭" in content:
                # Extract persona information
                persona_match = re.search(r"🎭 \*\*([^*]+)\*\*", content)
                if persona_match:
                    persona_name = persona_match.group(1)
                    role = f"{role} ({persona_name})"

            elif metadata.get("has_multi_model") and "Running on" in content:
                # Extract model information
                model_match = re.search(r"Running on ([^•]+)", content)
                if model_match:
                    model_name = model_match.group(1).strip()
                    role = f"{role} ({model_name})"

            # Clean content of formatting for transcript
            clean_content = self._clean_content_for_summary(content)
            transcript_parts.append(f"[{role}]: {clean_content}")

        transcript = "\n\n".join(transcript_parts)

        # Create enhanced prompt using templates
        enhanced_prompt = await self._create_enhanced_prompt(
            metadata, integration_context, is_complete, is_technical
        )

        # Get max tokens from template
        templates = await self._load_summary_templates()
        template_type = "technical" if is_technical else ("complete" if is_complete else "simple")
        max_tokens = self._get_max_tokens(template_type, templates)

        # Prepare LLM request
        llm_messages = [
            {"role": "system", "content": enhanced_prompt},
            {"role": "user", "content": f"Conversation to summarize:\n\n{transcript}"},
        ]

        try:
            body = {
                "model": model,
                "messages": llm_messages,
                "temperature": 0.5,
                "max_tokens": max_tokens,
                "stream": False,
            }

            response = await generate_chat_completion(request, body, user)

            if isinstance(response, dict) and response.get("choices"):
                content = response["choices"][0].get("message", {}).get("content", "")
                return content.strip() if content else "[Error: Empty response]"

            return "[Error: Invalid response format]"

        except Exception as e:
            logger.error(f"LLM summarization failed: {e}")
            return f"[Error: {str(e)[:100]}]"


class Filter:
    class Valves(BaseModel):
        priority: int = Field(
            default=5,
            description="Filter priority (Agent Hotswap: 0, Summarizer: 5, Multi-Model: 10)",
        )
        summary_trigger_turns: int = Field(
            default=8,
            ge=4,
            le=20,
            description="Number of turns before auto-summarization",
        )
        preserve_recent_turns: int = Field(
            default=4,
            ge=2,
            le=10,
            description="Recent turns to preserve during summarization",
        )
        summary_model: str = Field(
            default="auto",
            description="Model for summaries ('auto' uses current model)",
            json_schema_extra=ModelManager.get_model_dropdown(),
        )
        enable_session_awareness: bool = Field(
            default=True, description="Access complete conversation history"
        )
        enable_debug: bool = Field(default=False, description="Enable debug logging")
        force_summarize_next: bool = Field(
            default=False, description="Force summarization on next message (testing)"
        )
        refresh_models: bool = Field(
            default=False, description="🔄 Refresh model list (auto-resets)"
        )

        # Enhanced integration settings
        enable_plugin_integration: bool = Field(
            default=True,
            description="Enable integration with Agent Hotswap and Multi-Model filters",
        )
        integration_debug: bool = Field(
            default=False, description="Debug plugin integration"
        )
        persona_aware_summaries: bool = Field(
            default=True, description="Include persona context in summaries"
        )
        multi_model_analysis: bool = Field(
            default=True, description="Analyze multi-model conversation patterns"
        )

        # Popup window settings
        use_popup_ui: bool = Field(
            default=True,
            description="Use popup window for summaries (like Agent Hotswap)",
        )
        auto_download_config: bool = Field(
            default=True, description="Auto-download config and UI files from repository"
        )
        show_popup_placeholder: bool = Field(
            default=True,
            description="Show placeholder text when using popup",
        )

    def __init__(self):
        self.valves = self.Valves()
        self.toggle = True
        self.icon = "📋"
        
        # EXACT same directory detection as Agent Hotswap
        self.plugin_directory_name = self._get_plugin_directory_name()
        
        # Initialize enhanced components
        self.conversation_manager = ConversationManager()
        self.summarization_engine = SummarizationEngine(self.valves, self._load_config)
        self.integration_detector = PluginIntegrationDetector()
        self.downloader = SummarizerDownloader(self._get_config_path)

        # Track summarization with enhanced context
        self.last_summary_counts = {}
        self._config_cache = None
        self._config_cache_time = 0

        # Handle model refresh
        self._handle_model_refresh()

    async def _load_config(self) -> Dict:
        """Load configuration from file with caching"""
        current_time = time.time()
        
        # Use cache if recent (5 minutes)
        if self._config_cache and (current_time - self._config_cache_time) < 300:
            return self._config_cache
            
        try:
            config_path = self._get_config_path()
            if os.path.exists(config_path):
                async with aiofiles.open(config_path, "r", encoding="utf-8") as f:
                    content = await f.read()
                    config = json.loads(content)
                    self._config_cache = config
                    self._config_cache_time = current_time
                    return config
        except Exception as e:
            print(f"[SUMMARIZER] Error loading config: {e}")
            
        # Return default if loading fails
        return self.downloader._create_default_config()

    def __del__(self):
        try:
            if hasattr(self, "downloader") and self.downloader._session:
                asyncio.create_task(self.downloader.close_session())
        except:
            pass

    def _get_plugin_directory_name(self) -> str:
        """Auto-detect plugin directory name - EXACT same as Agent Hotswap"""
        try:
            if __name__ != "__main__":
                module_parts = __name__.split(".")
                detected_name = module_parts[-1]
                if detected_name.startswith("function_"):
                    detected_name = detected_name[9:]
                cleaned_name = re.sub(r"[^a-zA-Z0-9_-]", "_", detected_name.lower())
                if cleaned_name and cleaned_name != "__main__":
                    print(f"[SUMMARIZER] Auto-detected plugin name: {cleaned_name}")
                    return cleaned_name
        except Exception as e:
            print(f"[SUMMARIZER] Method 1 (__name__) failed: {e}")

        fallback_name = "conversation_summarizer"
        print(f"[SUMMARIZER] Using fallback plugin name: {fallback_name}")
        return fallback_name

    def _get_config_path(self) -> str:
        """Get configuration path - EXACT same as Agent Hotswap"""
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
        """Get UI file path"""
        config_path = self._get_config_path()
        return os.path.join(os.path.dirname(config_path), UI_FILE)

    def _get_summary_data_path(self) -> str:
        """Get summary data file path"""
        config_path = self._get_config_path()
        return os.path.join(os.path.dirname(config_path), SUMMARY_DATA_FILE)

    def _get_relative_ui_url(self) -> str:
        """Generate relative URL for popup window"""
        try:
            relative_url = f"/cache/functions/{self.plugin_directory_name}/index.html"
            print(f"[SUMMARIZER] Generated relative URL: {relative_url}")
            return relative_url
        except Exception as e:
            print(f"[SUMMARIZER] Error generating URL: {e}")
            return f"/cache/functions/{self.plugin_directory_name}/index.html"

    def _debug_log(self, message: str):
        """Debug logging helper"""
        if self.valves.enable_debug:
            print(f"[SUMMARIZER] {message}")

    def _integration_debug(self, message: str):
        """Integration debug logging"""
        if self.valves.integration_debug:
            print(f"[SUMMARIZER:INTEGRATION] {message}")

    def _handle_model_refresh(self):
        """Handle model refresh toggle"""
        if getattr(self.valves, "refresh_models", False):
            logger.info("Model refresh triggered")
            field = self.Valves.model_fields["summary_model"]
            if not hasattr(field, "json_schema_extra"):
                field.json_schema_extra = {}
            field.json_schema_extra.update(ModelManager.get_model_dropdown())
            self.valves.refresh_models = False

    def _get_summary_model(self, current_model: str) -> str:
        """Get model for summarization"""
        if self.valves.summary_model == "auto":
            return current_model
        return self.valves.summary_model

    def _safe_get_ids(self, body: dict, user: Optional[dict]):
        """Safely extract chat and user IDs"""
        chat_id = body.get("chat_id") or body.get("id") or f"chat_{int(time.time())}"
        user_id = user.get("id", "anonymous") if user else "anonymous"
        return str(chat_id), str(user_id)

    def _get_user_obj(self, user_dict: Optional[dict]):
        """Get user object for LLM calls"""
        if not user_dict:
            return None

        # Handle both dict and UserModel objects
        if hasattr(user_dict, "get"):  # Dictionary-like object
            user_id = user_dict.get("id")
            user_email = user_dict.get("email", "unknown@example.com")
            user_name = user_dict.get("name", "Unknown")
        elif hasattr(user_dict, "id"):  # UserModel object
            user_id = getattr(user_dict, "id", None)
            user_email = getattr(user_dict, "email", "unknown@example.com")
            user_name = getattr(user_dict, "name", "Unknown")
        else:
            return None

        if not user_id:
            return None

        try:
            if NATIVE_DB_AVAILABLE:
                from open_webui.models.users import Users
                return Users.get_user_by_id(user_id)
        except:
            pass

        # Mock user object for API compatibility
        return type(
            "User", (), {"id": user_id, "email": user_email, "name": user_name}
        )()

    def _detect_command(self, content: str) -> Optional[str]:
        """Detect summarization commands"""
        if not content or not content.strip().startswith("!summarize"):
            return None

        content_lower = content.lower().strip()

        if "help" in content_lower:
            return "help"
        elif "complete" in content_lower:
            return "complete"
        elif content_lower == "!summarize" or "simple" in content_lower:
            return "simple"

        return None

    def _extract_integration_context(self, body: Dict) -> Dict:
        """Extract integration context from other plugins"""
        if not self.valves.enable_plugin_integration:
            return {}

        filter_context = body.get("_filter_context", {})

        # Detect Agent Hotswap integration
        agent_hotswap_context = self.integration_detector.detect_agent_hotswap_context(
            filter_context
        )

        # Detect Multi-Model integration
        multi_model_context = self.integration_detector.detect_multi_model_context(body)

        integration_context = {
            "agent_hotswap": agent_hotswap_context,
            "multi_model": multi_model_context,
            "has_integrations": (
                agent_hotswap_context.get("has_agent_hotswap")
                or multi_model_context.get("has_multi_model")
            ),
        }

        # Flatten for easier access
        integration_context.update(agent_hotswap_context)
        integration_context.update(multi_model_context)

        if integration_context["has_integrations"]:
            self._integration_debug(
                f"Detected integrations: Agent Hotswap={agent_hotswap_context.get('has_agent_hotswap')}, Multi-Model={multi_model_context.get('has_multi_model')}"
            )

        return integration_context

    async def _ensure_config_available(self):
        """Ensure config and UI files are available - EXACT pattern as Agent Hotswap _ensure_personas_available"""
        config_path = self._get_config_path()
        
        print(f"[SUMMARIZER] Checking config path: {config_path}")
        
        if not os.path.exists(config_path):
            print(f"[SUMMARIZER] Config doesn't exist, creating directory and downloading...")
            os.makedirs(os.path.dirname(config_path), exist_ok=True)
            await self._create_minimal_config(config_path)
            if self.valves.auto_download_config:
                print(f"[SUMMARIZER] Auto-downloading config and UI...")
                await self._download_config_async()
        else:
            try:
                async with aiofiles.open(config_path, "r", encoding="utf-8") as f:
                    content = await f.read()
                    existing_data = json.loads(content)
                if not existing_data or len(existing_data) < 2:
                    print(f"[SUMMARIZER] Config incomplete, downloading...")
                    if self.valves.auto_download_config:
                        await self._download_config_async()
            except (json.JSONDecodeError, Exception) as e:
                print(f"[SUMMARIZER] Error reading existing config: {e}")
                await self._create_minimal_config(config_path)
                if self.valves.auto_download_config:
                    await self._download_config_async()

    async def _create_minimal_config(self, config_path: str):
        """Create minimal config file - same pattern as Agent Hotswap"""
        try:
            minimal_config = {
                "_metadata": {
                    "version": "2.3.3",
                    "last_updated": datetime.now().isoformat(),
                    "plugin_directory": self.plugin_directory_name,
                    "popup_ui": True,
                    "auto_download_enabled": self.valves.auto_download_config,
                    "integration_ready": True,
                    "initial_setup": True,
                    "async_enabled": True,
                },
                "settings": {
                    "summary_trigger_turns": self.valves.summary_trigger_turns,
                    "preserve_recent_turns": self.valves.preserve_recent_turns,
                    "use_popup_ui": self.valves.use_popup_ui,
                },
            }
            
            async with aiofiles.open(config_path, "w", encoding="utf-8") as f:
                await f.write(json.dumps(minimal_config, indent=4, ensure_ascii=False))
            
            print(f"[SUMMARIZER] Minimal config created in {self.plugin_directory_name}")
            
        except Exception as e:
            print(f"[SUMMARIZER] Error creating minimal config: {e}")

    async def _download_config_async(self, force_download: bool = False):
        """Download config and UI files asynchronously - EXACT pattern as Agent Hotswap _download_personas_async"""
        try:
            print(f"[SUMMARIZER] Starting config+UI download, force={force_download}")
            
            result = await self.downloader.download_config_and_ui(
                force_download=force_download
            )
            
            print(f"[SUMMARIZER] Download result: {result}")
            
            if result["success"]:
                ui_msg = ""
                if result.get("ui_downloaded"):
                    ui_msg = " + UI downloaded"
                elif result.get("ui_status") == "failed":
                    ui_msg = " (UI download failed - will use fallback)"

                print(f"[SUMMARIZER] ✅ Downloaded config{ui_msg}")
                
                # Clear global cache like Agent Hotswap
                global _GLOBAL_CACHE
                _GLOBAL_CACHE.clear()
            else:
                print(f"[SUMMARIZER] ❌ Download failed: {result.get('error', 'Unknown error')}")
                
        except Exception as e:
            print(f"[SUMMARIZER] ❌ Download error: {e}")
            import traceback
            traceback.print_exc()

    async def _update_summary_data_file(self, summary_data: Dict):
        """Create summary data file for UI to load - EXACT pattern as Agent Hotswap _update_personas_data_file"""
        # Create background task to avoid blocking response
        asyncio.create_task(self._write_summary_data_background(summary_data))

    async def _write_summary_data_background(self, summary_data: Dict):
        """Background task to write summary data file"""
        try:
            ui_dir = os.path.dirname(self._get_ui_path())
            summary_json_path = self._get_summary_data_path()

            print(f"[SUMMARIZER] UI directory: {ui_dir}")
            print(f"[SUMMARIZER] Summary JSON path: {summary_json_path}")

            # Ensure directory exists
            os.makedirs(ui_dir, exist_ok=True)

            # Write summary data as JSON file
            async with aiofiles.open(summary_json_path, "w", encoding="utf-8") as f:
                await f.write(json.dumps(summary_data, indent=2, ensure_ascii=False))

            # Verify the file was written
            if os.path.exists(summary_json_path):
                file_size = os.path.getsize(summary_json_path)
                print(f"[SUMMARIZER] ✅ Updated summary.json ({file_size} bytes)")
            else:
                print(f"[SUMMARIZER] ❌ Failed to create summary.json")

        except Exception as e:
            print(f"[SUMMARIZER] ❌ Error writing summary.json: {e}")

    async def _emit_status(self, emitter, message: str, status_type: str = "in_progress", done: bool = False):
        """Emit status message - same as Agent Hotswap"""
        if not emitter:
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
            print(f"[SUMMARIZER] Status emit error: {e}")

    def _generate_help_message(self) -> str:
        """Generate help message"""
        return f"""### 📋 Conversation Summarizer Commands

- **`!summarize`**: Opens popup window with intelligent conversation summary.
- **`!summarize complete`**: Opens popup window with comprehensive intelligence report.
- **`!summarize help`**: Displays this help message.

**Popup Window Features:**
- Beautiful full-window summary display
- Dark/Light theme synchronization
- Integration badges for personas and multi-model conversations
- Auto-download UI from repository

**Integration Features:**
- Works seamlessly with Agent Hotswap personas
- Detects multi-model conversations
- Includes reasoning process analysis

**Directory:** `{self.plugin_directory_name}`
**Version:** 2.3.3 (Fixed Download Pattern)"""

    async def _handle_help_command(self, body: dict, emitter) -> dict:
        """Handle help command"""
        try:
            help_content = self._generate_help_message()
            await self._emit_status(emitter, "ℹ️ Showing summarizer help", "complete")

            if emitter:
                await emitter({"type": "message", "data": {"content": help_content}})

            result = body.copy()
            result["messages"] = [{"role": "assistant", "content": help_content}]
            result["_summarizer_handled"] = True
            return result

        except Exception as e:
            print(f"[SUMMARIZER] Error in help command: {e}")
            error_message = f"Error generating help: {str(e)}"
            if emitter:
                await emitter({"type": "message", "data": {"content": error_message}})
            result = body.copy()
            result["messages"] = [{"role": "assistant", "content": error_message}]
            result["_summarizer_handled"] = True
            return result

    async def _handle_popup_command(
        self, body: dict, command: str, emitter, event_call, request, user_obj
    ) -> dict:
        """Handle popup window command - EXACT pattern as Agent Hotswap _handle_list_command"""
        try:
            await self._emit_status(emitter, "📋 Loading summary window...", "in_progress")

            # Extract IDs and integration context
            chat_id, user_id = self._safe_get_ids(body, user_obj)
            integration_context = self._extract_integration_context(body)

            # Get enhanced conversation history
            if self.valves.enable_session_awareness:
                messages, metadata = (
                    await self.conversation_manager.get_conversation_history(
                        chat_id, user_id, body.get("messages", [])
                    )
                )
            else:
                messages = body.get("messages", [])
                metadata = {
                    "source": "current_request",
                    "total_messages": len(messages),
                }

            # Add integration analysis to metadata
            if integration_context.get("has_integrations"):
                metadata.update(
                    {
                        "has_integrations": True,
                        "integration_features": integration_context,
                    }
                )

            await self._emit_status(emitter, "🧠 Analyzing conversation...", "in_progress")

            # Generate enhanced summary
            model = self._get_summary_model(body.get("model", "auto"))
            summary = await self.summarization_engine.generate_summary(
                messages,
                model,
                request,
                user_obj,
                metadata,
                integration_context,
                is_complete=(command == "complete"),
            )

            if summary.startswith("[Error:"):
                await self._emit_status(emitter, "❌ Summary failed", "complete")
                return await self._handle_popup_fallback(body, summary, emitter)

            # Prepare summary data for popup
            summary_data = {
                "type": command,
                "content": summary,
                "metadata": metadata,
                "integration": integration_context,
                "timestamp": datetime.now().strftime("%B %d, %Y at %I:%M %p"),
            }

            # Update summary data file (background task)
            await self._update_summary_data_file(summary_data)

            # Ensure UI file exists - EXACT Agent Hotswap pattern
            ui_path = self._get_ui_path()
            print(f"[SUMMARIZER] UI path: {ui_path}")

            if not os.path.exists(ui_path):
                print(f"[SUMMARIZER] UI file not found, downloading...")
                result = await self.downloader.download_ui_file()
                if not result["success"]:
                    print(f"[SUMMARIZER] UI download failed: {result.get('error')}")
                    return await self._handle_popup_fallback(body, "UI file not available", emitter)
            else:
                print(f"[SUMMARIZER] UI file exists")

            # Open popup window - EXACT same script as Agent Hotswap
            hub_url = self._get_relative_ui_url()
            print(f"[SUMMARIZER] Opening popup at: {hub_url}")

            popup_script = f"""
                console.log('Opening summary window at: {hub_url}');
                window.open(
                    '{hub_url}', 
                    'summaryWindow_' + Date.now(), 
                    'width=' + Math.min(screen.availWidth, 1000) + ',height=' + Math.min(screen.availHeight, 800) + ',scrollbars=yes,resizable=yes,menubar=no,toolbar=no'
                );
            """

            if event_call:
                await event_call({"type": "execute", "data": {"code": popup_script}})

            await self._emit_status(emitter, "✅ Summary window opened!", "complete")

            await asyncio.sleep(2)
            await self._emit_status(emitter, "", "complete", done=True)

            result = body.copy()
            result["_summarizer_handled"] = True

            # Show placeholder message if enabled - same pattern as Agent Hotswap
            if self.valves.show_popup_placeholder:
                report_type = (
                    "Intelligence Report" if command == "complete" else "Summary"
                )
                placeholder_msg = f"Opening the 📋 {report_type} window now. View your detailed conversation analysis in the popup!"
                
                for msg in reversed(result.get("messages", [])):
                    if msg.get("role") == "user":
                        msg["content"] = f"Respond with exactly this message: '{placeholder_msg}'"
                        break

            return result

        except Exception as e:
            print(f"[SUMMARIZER] Error in popup command: {e}")
            import traceback
            traceback.print_exc()
            return await self._handle_popup_fallback(body, str(e), emitter)

    async def _handle_popup_fallback(self, body: dict, error_msg: str, emitter) -> dict:
        """Fallback when popup fails - EXACT Agent Hotswap pattern"""
        try:
            await self._emit_status(emitter, "⚠️ Using fallback display", "complete")

            fallback_content = f"## ❌ Summary Generation Failed\n\n{error_msg}\n\n*Please try again or check plugin configuration.*"

            result = body.copy()
            result["messages"] = [{"role": "assistant", "content": fallback_content}]
            result["_summarizer_handled"] = True
            return result

        except Exception as e:
            print(f"[SUMMARIZER] Error in fallback: {e}")
            error_message = f"Critical error: {str(e)}"
            result = body.copy()
            result["messages"] = [{"role": "assistant", "content": error_message}]
            result["_summarizer_handled"] = True
            return result

    def _should_auto_summarize(
        self, messages: List[Dict], chat_id: str, metadata: Dict = None
    ) -> bool:
        """Enhanced auto-summarization logic with integration awareness"""
        if self.valves.force_summarize_next:
            self.valves.force_summarize_next = False
            return True

        conv_messages = [m for m in messages if m.get("role") in ["user", "assistant"]]
        valid_count = len([m for m in conv_messages if len(m.get("content", "")) > 20])

        # Base threshold
        base_threshold = self.valves.summary_trigger_turns

        # Adjust threshold based on conversation complexity
        if metadata:
            complexity = metadata.get("conversation_complexity", "simple")
            if complexity == "highly_complex":
                base_threshold = max(6, base_threshold - 2)
            elif complexity == "complex":
                base_threshold = max(7, base_threshold - 1)

        if valid_count < base_threshold:
            return False

        # Check if we summarized recently
        last_count = self.last_summary_counts.get(chat_id, 0)
        if valid_count - last_count < base_threshold // 2:
            return False

        return True

    async def _perform_auto_summary(
        self,
        body: dict,
        chat_id: str,
        request,
        user_obj,
        integration_context: Dict = None,
    ) -> dict:
        """Enhanced automatic summarization - always uses chat format for auto-summaries"""
        messages = body.get("messages", [])
        preserve_count = self.valves.preserve_recent_turns

        if len(messages) <= preserve_count:
            return body

        # Split messages
        to_summarize = messages[:-preserve_count]
        to_preserve = messages[-preserve_count:]

        conv_messages = [
            m for m in to_summarize if m.get("role") in ["user", "assistant"]
        ]
        if not conv_messages:
            return body

        # Create metadata for auto-summary
        metadata = {
            "source": "auto_summarization",
            "total_messages": len(conv_messages),
            "auto_summary": True,
        }

        # Add integration analysis
        if integration_context:
            metadata.update(integration_context)

        # Generate enhanced summary
        model = self._get_summary_model(body.get("model", "auto"))
        summary = await self.summarization_engine.generate_summary(
            conv_messages,
            model,
            request,
            user_obj,
            metadata,
            integration_context,
            is_complete=False,
        )

        if summary and not summary.startswith("[Error:"):
            # Create enhanced summary message with integration context
            integration_note = ""
            if integration_context and integration_context.get("has_integrations"):
                features = []
                if integration_context.get("has_agent_hotswap"):
                    features.append("🎭 Persona-enhanced")
                if integration_context.get("has_multi_model"):
                    features.append("🤖 Multi-model")
                if integration_context.get("has_reasoning"):
                    features.append("🧠 Reasoning-enabled")

                if features:
                    integration_note = f" ({' • '.join(features)})"

            summary_msg = {
                "role": "system",
                "content": f"## 📋 Enhanced Conversation Summary{integration_note}\n\n{summary}\n\n---\n*Auto-summary covers {len(conv_messages)} earlier messages with integration analysis. Recent messages continue below.*",
            }

            # Rebuild message list
            body["messages"] = [summary_msg] + to_preserve
            self.last_summary_counts[chat_id] = len(conv_messages)

            self._debug_log(
                f"Auto-summarized {len(conv_messages)} messages with integration context"
            )

        return body

    async def inlet(
        self,
        body: dict,
        __event_emitter__,
        __event_call__=None,
        __user__: Optional[dict] = None,
        __request__=None,
    ) -> dict:
        """Enhanced main filter entry point with popup window support - EXACT initialization pattern as Agent Hotswap"""
        if not self.toggle:
            return body

        # Skip if already handled
        if body.get("_summarizer_handled"):
            return body

        # Initialize if needed - EXACT same pattern as Agent Hotswap
        try:
            if not hasattr(self, "_initialized"):
                print(f"[SUMMARIZER] Initializing plugin...")
                await self._ensure_config_available()
                self._initialized = True
                print(f"[SUMMARIZER] ✅ Plugin initialized successfully")
        except Exception as e:
            print(f"[SUMMARIZER] ❌ Initialization error: {e}")
            return body

        messages = body.get("messages", [])
        if not messages:
            return body

        # Extract integration context early
        integration_context = self._extract_integration_context(body)

        # Find last user message
        last_user_content = ""
        for msg in reversed(messages):
            if msg.get("role") == "user":
                last_user_content = msg.get("content", "")
                break

        # Check for command
        command = self._detect_command(last_user_content)
        if command:
            user_obj = self._get_user_obj(__user__)

            if command == "help":
                return await self._handle_help_command(body, __event_emitter__)
            elif self.valves.use_popup_ui and __event_call__:
                # Use popup for summary commands - same as Agent Hotswap list command
                return await self._handle_popup_command(
                    body,
                    command,
                    __event_emitter__,
                    __event_call__,
                    __request__,
                    user_obj,
                )
            else:
                # Fallback to chat format if popup disabled or event_call unavailable
                return await self._handle_chat_command(
                    body, command, __event_emitter__, __request__, user_obj
                )

        # Check for auto-summarization with enhanced logic
        chat_id, user_id = self._safe_get_ids(body, __user__)

        if self.valves.enable_session_awareness:
            complete_messages, metadata = (
                await self.conversation_manager.get_conversation_history(
                    chat_id, user_id, messages
                )
            )

            # Enhanced auto-summarization decision
            if self._should_auto_summarize(complete_messages, chat_id, metadata):
                user_obj = self._get_user_obj(__user__)
                body["messages"] = complete_messages

                # Include integration context in auto-summary
                return await self._perform_auto_summary(
                    body, chat_id, __request__, user_obj, integration_context
                )

        return body

    async def _handle_chat_command(
        self, body: dict, command: str, emitter, request, user_obj
    ) -> dict:
        """Fallback chat-based command handling"""
        try:
            # Extract IDs and integration context
            chat_id, user_id = self._safe_get_ids(body, user_obj)
            integration_context = self._extract_integration_context(body)

            # Get enhanced conversation history
            if self.valves.enable_session_awareness:
                messages, metadata = (
                    await self.conversation_manager.get_conversation_history(
                        chat_id, user_id, body.get("messages", [])
                    )
                )
            else:
                messages = body.get("messages", [])
                metadata = {
                    "source": "current_request",
                    "total_messages": len(messages),
                }

            # Add integration analysis to metadata
            if integration_context.get("has_integrations"):
                metadata.update(
                    {
                        "has_integrations": True,
                        "integration_features": integration_context,
                    }
                )

            await self._emit_status(emitter, "🧠 Analyzing conversation...", "in_progress")

            # Generate enhanced summary
            model = self._get_summary_model(body.get("model", "auto"))
            summary = await self.summarization_engine.generate_summary(
                messages,
                model,
                request,
                user_obj,
                metadata,
                integration_context,
                is_complete=(command == "complete"),
            )

            if summary.startswith("[Error:"):
                await self._emit_status(emitter, "❌ Summary failed", "complete")
                content = f"## ❌ Summarization Failed\n\n{summary}"
            else:
                await self._emit_status(emitter, "✅ Summary complete", "complete")

                # Create simple chat format summary
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                # Add integration context if available
                context_note = ""
                if metadata:
                    features = []
                    if metadata.get("has_personas"):
                        features.append("🎭 Persona-enhanced")
                    if metadata.get("has_multi_model"):
                        features.append(
                            f"🤖 Multi-model ({metadata.get('model_count', 'multiple')} models)"
                        )
                    if metadata.get("has_reasoning"):
                        features.append("🧠 Reasoning-enabled")

                    if features:
                        context_note = f"\n\n*{' • '.join(features)}*"

                report_type = (
                    "Intelligence Report"
                    if command == "complete"
                    else "Conversation Summary"
                )
                content = f"## 📋 {report_type}\n\n{summary}{context_note}\n\n---\n*Generated on {timestamp}*"

            await emitter({"type": "message", "data": {"content": content}})

            # Clear body messages to prevent LLM processing
            result = body.copy()
            result["messages"] = []
            result["_summarizer_handled"] = True
            return result

        except Exception as e:
            print(f"[SUMMARIZER] Error in chat command: {e}")
            error_content = f"## ❌ Summarization Error\n\n{str(e)}"
            await emitter({"type": "message", "data": {"content": error_content}})

            result = body.copy()
            result["messages"] = []
            result["_summarizer_handled"] = True
            return result

    async def outlet(
        self, body: dict, __event_emitter__, __user__: Optional[dict] = None
    ) -> dict:
        """Enhanced post-process output with integration awareness"""
        return body

    # Enhanced integration points for other plugins
    def get_summary_model(self) -> str:
        """Get currently configured summary model"""
        return self.valves.summary_model

    def set_summary_model(self, model_id: str):
        """Set summary model from external plugin"""
        available = ModelManager.get_available_models()
        if model_id in available:
            self.valves.summary_model = model_id
            return True
        return False

    def trigger_summary(self, force: bool = True):
        """External trigger for summarization"""
        if force:
            self.valves.force_summarize_next = True

    def get_integration_status(self) -> Dict:
        """Get current integration status for other plugins"""
        return {
            "plugin_integration_enabled": self.valves.enable_plugin_integration,
            "persona_aware_summaries": self.valves.persona_aware_summaries,
            "multi_model_analysis": self.valves.multi_model_analysis,
            "integration_debug": self.valves.integration_debug,
            "use_popup_ui": self.valves.use_popup_ui,
            "version": "2.3.3",
        }

    async def analyze_conversation_patterns(self, messages: List[Dict]) -> Dict:
        """External API for other plugins to analyze conversation patterns"""
        return self.conversation_manager._analyze_conversation_patterns(messages)