"""
title: OpenRouter Universal Image Filter
version: 1.0.0
author: pkeffect & Claude Sonnet 4.5
author_url: https://github.com/pkeffect/
project_url: https://github.com/pkeffect/functions/tree/main/functions/filters/openrouter_universal_image_filter
funding_url: https://github.com/open-webui
required_open_webui_version: 0.6.0+
version: 0.0.1
date: 2025-10-03
license: MIT
description: Complete production filter for OpenRouter vision and image generation models with retry logic, validation, and comprehensive error handling
requirements: httpx, pillow
"""

from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any, Callable, Awaitable
import base64
import io
import asyncio
import random
import httpx
from PIL import Image
from datetime import datetime, timedelta
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("OpenRouterImageFilter")


# ========================================
# CIRCUIT BREAKER PATTERN
# ========================================

class CircuitBreaker:
    """Circuit breaker pattern for API resilience"""
    
    def __init__(self, failure_threshold: int = 5, timeout_seconds: int = 60):
        self.failure_count = 0
        self.last_failure_time = None
        self.threshold = failure_threshold
        self.timeout = timeout_seconds
        self.is_open = False
    
    def record_success(self):
        """Reset on successful call"""
        self.failure_count = 0
        self.is_open = False
    
    def record_failure(self):
        """Increment failure count"""
        self.failure_count += 1
        self.last_failure_time = datetime.now()
        
        if self.failure_count >= self.threshold:
            self.is_open = True
            logger.warning(f"Circuit breaker OPENED after {self.failure_count} failures")
    
    def can_attempt(self) -> bool:
        """Check if we can attempt a call"""
        if not self.is_open:
            return True
        
        if self.last_failure_time:
            elapsed = (datetime.now() - self.last_failure_time).total_seconds()
            if elapsed > self.timeout:
                logger.info("Circuit breaker attempting to close after timeout")
                self.is_open = False
                self.failure_count = 0
                return True
        
        return False


# ========================================
# MAIN FILTER CLASS
# ========================================

class Filter:
    
    # ========================================
    # CONFIGURATION (VALVES)
    # ========================================
    
    class Valves(BaseModel):
        """Admin-only settings (system-wide)"""
        
        priority: int = Field(
            default=0,
            description="Filter priority"
        )
        
        # Admin API Key (fallback if user doesn't have their own)
        OPENROUTER_API_KEY: str = Field(
            default="",
            description="[ADMIN] Fallback OpenRouter API key (used if user doesn't provide their own)"
        )
        
        OPENROUTER_BASE_URL: str = Field(
            default="https://openrouter.ai/api/v1",
            description="[ADMIN] OpenRouter API base URL"
        )
        
        # System Limits
        max_image_size_mb: float = Field(
            default=5.0,
            description="[ADMIN] Max image size in MB (enforced limit)"
        )
        
        request_timeout: int = Field(
            default=120,
            description="[ADMIN] API request timeout in seconds"
        )
        
        max_retries: int = Field(
            default=3,
            description="[ADMIN] Maximum retry attempts for failed requests"
        )
        
        enable_circuit_breaker: bool = Field(
            default=True,
            description="[ADMIN] Enable circuit breaker for API resilience"
        )
        
        # Tracking
        app_name: str = Field(
            default="Open WebUI",
            description="[ADMIN] Application name for OpenRouter tracking"
        )
        
        site_url: str = Field(
            default="https://openwebui.com",
            description="[ADMIN] Site URL for OpenRouter tracking"
        )
        
        pipelines: List[str] = ["*"]
    
    class UserValves(BaseModel):
        """User-configurable settings (per-user preferences)"""
        
        # User's Personal API Key (optional - overrides admin key)
        OPENROUTER_API_KEY: str = Field(
            default="",
            description="Your personal OpenRouter API key (optional - leave empty to use system default)"
        )
        
        # Model Selection
        vision_model: str = Field(
            default="qwen/qwen-2-vl-7b-instruct",
            description="Model for image analysis/editing"
        )
        
        generation_model: str = Field(
            default="google/gemini-2.5-flash-image-preview",
            description="Model for image generation"
        )
        
        auto_model_selection: bool = Field(
            default=True,
            description="Auto-switch to vision model when images detected"
        )
        
        # Operation Mode
        operation_mode: str = Field(
            default="auto",
            description="Operation mode",
            json_schema_extra={"enum": ["auto", "vision_only", "generation_only"]}
        )
        
        # Image Processing
        max_image_dimension: int = Field(
            default=1568,
            description="Max image dimension (Claude optimal: 1568px)"
        )
        
        auto_resize: bool = Field(
            default=True,
            description="Automatically resize large images"
        )
        
        jpeg_quality: int = Field(
            default=85,
            description="JPEG compression quality (1-100)"
        )
        
        # Generation Parameters
        temperature: float = Field(
            default=0.7,
            description="Sampling temperature for generation (0.0-2.0)"
        )
        
        max_tokens: int = Field(
            default=4096,
            description="Maximum tokens in response"
        )
        
        # UI Preferences
        enable_logging: bool = Field(
            default=False,
            description="Enable debug logging for your requests"
        )
        
        show_status_updates: bool = Field(
            default=True,
            description="Show real-time status updates in chat"
        )
    
    # ========================================
    # INITIALIZATION
    # ========================================
    
    def __init__(self):
        self.type = "filter"
        self.name = "OpenRouter Image Filter"
        self.valves = self.Valves()
        self.user_valves = self.UserValves()
        self.circuit_breaker = CircuitBreaker()
        
        # State tracking
        self.last_generation_id = None
        self.processing_images = False
    
    # ========================================
    # HELPER METHODS - API KEY RESOLUTION
    # ========================================
    
    def get_api_key(self, __user__: Optional[dict] = None) -> Optional[str]:
        """
        Get API key with priority:
        1. User's personal API key (if set)
        2. User's OpenRouter connection in Open WebUI (if exists)
        3. Admin fallback key
        """
        # Try user's personal key first
        if __user__ and "valves" in __user__:
            user_valves = __user__["valves"]
            if hasattr(user_valves, "OPENROUTER_API_KEY") and user_valves.OPENROUTER_API_KEY:
                return user_valves.OPENROUTER_API_KEY
        
        # Try to get from Open WebUI's model connection
        # Open WebUI stores user's model connections with API keys
        if __user__ and "model" in __user__:
            model_info = __user__.get("model", {})
            # Check if user has OpenRouter configured in their models
            if "api_key" in model_info:
                return model_info.get("api_key")
        
        # Check if user has OpenRouter in their model providers
        if __user__ and "settings" in __user__:
            settings = __user__.get("settings", {})
            models = settings.get("models", {})
            
            # Look for OpenRouter in user's configured models
            for model_id, model_config in models.items():
                if "openrouter" in model_id.lower():
                    if "api_key" in model_config:
                        return model_config.get("api_key")
        
        # Fallback to admin key
        return self.valves.OPENROUTER_API_KEY if self.valves.OPENROUTER_API_KEY else None
    
    # ========================================
    # HELPER METHODS - LOGGING & UI
    # ========================================
    
    def log(self, message: str, level: str = "info", user_valves: Optional[Any] = None):
        """Conditional logging based on user preference"""
        if user_valves and hasattr(user_valves, 'enable_logging'):
            if not user_valves.enable_logging:
                return
        elif not self.user_valves.enable_logging:
            return
        
        log_func = getattr(logger, level, logger.info)
        log_func(f"{message}")
    
    async def emit_status(
        self,
        event_emitter: Optional[Callable],
        description: str,
        done: bool = False,
        user_valves: Optional[Any] = None
    ):
        """Emit status update to UI (respects user preference)"""
        show_status = True
        if user_valves and hasattr(user_valves, 'show_status_updates'):
            show_status = user_valves.show_status_updates
        elif hasattr(self.user_valves, 'show_status_updates'):
            show_status = self.user_valves.show_status_updates
        
        if event_emitter and show_status:
            await event_emitter({
                "type": "status",
                "data": {
                    "description": description,
                    "done": done
                }
            })
    
    async def emit_message(
        self,
        event_emitter: Optional[Callable],
        message: str
    ):
        """Emit message to chat (always shown - for errors/important info)"""
        if event_emitter:
            await event_emitter({
                "type": "message",
                "data": {"content": message}
            })
    
    # ========================================
    # HELPER METHODS - IMAGE DETECTION
    # ========================================
    
    def _detect_images(self, messages: List[dict]) -> bool:
        """Detect if messages contain images"""
        for msg in messages:
            content = msg.get("content", "")
            if isinstance(content, list):
                for item in content:
                    if isinstance(item, dict) and item.get("type") == "image_url":
                        return True
            
            if "images" in msg and msg["images"]:
                return True
        
        return False
    
    def _is_vision_model(self, model: str) -> bool:
        """Check if model supports vision"""
        vision_keywords = ["vision", "vl", "multimodal", "gpt-4o", "claude", "gemini", "qwen"]
        return any(keyword in model.lower() for keyword in vision_keywords)
    
    def _is_generation_request(self, messages: List[dict]) -> bool:
        """Detect if user is requesting image generation"""
        generation_keywords = [
            "generate", "create", "make", "draw", "paint", "render",
            "produce", "design", "illustrate", "sketch"
        ]
        
        for msg in reversed(messages):
            if msg.get("role") == "user":
                content = msg.get("content", "")
                if isinstance(content, str):
                    content_lower = content.lower()
                    has_generation = any(kw in content_lower for kw in generation_keywords)
                    has_image_ref = any(word in content_lower for word in ["image", "picture", "photo", "artwork"])
                    return has_generation and has_image_ref
                break
        
        return False
    
    # ========================================
    # HELPER METHODS - IMAGE VALIDATION
    # ========================================
    
    def validate_image_data(self, base64_string: str, user_valves: Optional[Any] = None) -> tuple[bool, Optional[str]]:
        """Validate base64 image data"""
        try:
            if "," in base64_string:
                base64_string = base64_string.split(",", 1)[1]
            
            base64_string = self._fix_base64_padding(base64_string)
            decoded = base64.b64decode(base64_string)
            
            img = Image.open(io.BytesIO(decoded))
            img.verify()
            
            size_mb = len(decoded) / (1024 * 1024)
            auto_resize = user_valves.auto_resize if user_valves else self.user_valves.auto_resize
            
            if size_mb > self.valves.max_image_size_mb and not auto_resize:
                return False, f"Image too large: {size_mb:.2f}MB (admin max: {self.valves.max_image_size_mb}MB)"
            
            self.log(f"✓ Valid image: {img.format}, {img.size}, {size_mb:.2f}MB", user_valves=user_valves)
            return True, None
            
        except Exception as e:
            return False, f"Invalid image data: {str(e)}"
    
    # ========================================
    # HELPER METHODS - BASE64 PROCESSING
    # ========================================
    
    def _fix_base64_padding(self, s: str) -> str:
        """Add missing padding to base64 string"""
        s = "".join(s.split())
        
        if "-" in s or "_" in s:
            s = s.replace("-", "+").replace("_", "/")
        
        missing_padding = len(s) % 4
        if missing_padding:
            s += "=" * (4 - missing_padding)
        
        return s
    
    async def _optimize_image(
        self,
        image_data: str,
        max_dimension: int,
        jpeg_quality: int,
        user_valves: Optional[Any] = None
    ) -> str:
        """Resize and compress image for API transmission"""
        try:
            if image_data.startswith("data:"):
                header, encoded = image_data.split(",", 1)
            else:
                encoded = image_data
                header = "data:image/png;base64"
            
            encoded = self._fix_base64_padding(encoded)
            decoded = base64.b64decode(encoded)
            original_size = len(decoded)
            
            img = Image.open(io.BytesIO(decoded))
            
            needs_resize = max(img.size) > max_dimension
            size_mb = original_size / (1024 * 1024)
            needs_compression = size_mb > self.valves.max_image_size_mb
            
            if not needs_resize and not needs_compression:
                self.log(f"Image already optimal: {img.size}, {size_mb:.2f}MB", user_valves=user_valves)
                return image_data
            
            if needs_resize:
                ratio = max_dimension / max(img.size)
                new_size = tuple(int(dim * ratio) for dim in img.size)
                img = img.resize(new_size, Image.LANCZOS)
                self.log(f"Resized: {img.size} → {new_size}", user_valves=user_valves)
            
            if img.mode not in ("RGB", "L"):
                if img.mode == "RGBA":
                    background = Image.new("RGB", img.size, (255, 255, 255))
                    background.paste(img, mask=img.split()[3])
                    img = background
                else:
                    img = img.convert("RGB")
            
            buffer = io.BytesIO()
            img.save(buffer, format="JPEG", quality=jpeg_quality, optimize=True)
            new_size = len(buffer.getvalue())
            
            self.log(
                f"Compressed: {original_size/1024:.1f}KB → {new_size/1024:.1f}KB "
                f"({(1-new_size/original_size)*100:.1f}% reduction)",
                user_valves=user_valves
            )
            
            optimized_b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
            return f"data:image/jpeg;base64,{optimized_b64}"
            
        except Exception as e:
            self.log(f"Optimization failed: {e}", "error", user_valves)
            return image_data
    
    # ========================================
    # HELPER METHODS - MESSAGE PROCESSING
    # ========================================
    
    async def _process_images(
        self,
        messages: List[dict],
        user_valves: Optional[Any] = None
    ) -> List[dict]:
        """Convert format and optimize images"""
        processed = []
        
        auto_resize = user_valves.auto_resize if user_valves else self.user_valves.auto_resize
        max_dimension = user_valves.max_image_dimension if user_valves else self.user_valves.max_image_dimension
        jpeg_quality = user_valves.jpeg_quality if user_valves else self.user_valves.jpeg_quality
        
        for msg in messages:
            if "images" in msg and msg["images"]:
                content = []
                
                text_content = msg.get("content", "")
                if text_content:
                    content.append({"type": "text", "text": text_content})
                
                for img_data in msg["images"]:
                    is_valid, error_msg = self.validate_image_data(img_data, user_valves)
                    if not is_valid:
                        self.log(f"Skipping invalid image: {error_msg}", "warning", user_valves)
                        continue
                    
                    if auto_resize:
                        img_data = await self._optimize_image(
                            img_data,
                            max_dimension,
                            jpeg_quality,
                            user_valves
                        )
                    
                    if not img_data.startswith("data:"):
                        img_data = f"data:image/png;base64,{img_data}"
                    
                    content.append({
                        "type": "image_url",
                        "image_url": {"url": img_data}
                    })
                
                processed.append({
                    "role": msg["role"],
                    "content": content
                })
            
            elif isinstance(msg.get("content"), list):
                content = []
                for item in msg["content"]:
                    if item.get("type") == "image_url":
                        img_url = item["image_url"]["url"]
                        
                        is_valid, error_msg = self.validate_image_data(img_url, user_valves)
                        if not is_valid:
                            self.log(f"Skipping invalid image: {error_msg}", "warning", user_valves)
                            continue
                        
                        if auto_resize:
                            optimized = await self._optimize_image(
                                img_url,
                                max_dimension,
                                jpeg_quality,
                                user_valves
                            )
                            content.append({
                                "type": "image_url",
                                "image_url": {"url": optimized}
                            })
                        else:
                            content.append(item)
                    else:
                        content.append(item)
                
                processed.append({
                    "role": msg["role"],
                    "content": content
                })
            else:
                processed.append(msg)
        
        return processed
    
    # ========================================
    # OPENROUTER API INTEGRATION
    # ========================================
    
    async def call_openrouter_api(
        self,
        model: str,
        messages: List[dict],
        api_key: str,
        event_emitter: Optional[Callable] = None,
        is_generation: bool = False,
        user_valves: Optional[Any] = None
    ) -> Optional[dict]:
        """Make API request to OpenRouter with retry logic"""
        
        if self.valves.enable_circuit_breaker and not self.circuit_breaker.can_attempt():
            error_msg = "Circuit breaker is OPEN - too many recent failures"
            self.log(error_msg, "error", user_valves)
            await self.emit_message(event_emitter, f"⚠️ {error_msg}")
            return None
        
        url = f"{self.valves.OPENROUTER_BASE_URL}/chat/completions"
        
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": self.valves.site_url,
            "X-Title": self.valves.app_name
        }
        
        temperature = user_valves.temperature if user_valves else self.user_valves.temperature
        max_tokens = user_valves.max_tokens if user_valves else self.user_valves.max_tokens
        
        body = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens
        }
        
        if is_generation:
            body["modalities"] = ["image", "text"]
        
        for attempt in range(self.valves.max_retries):
            try:
                async with httpx.AsyncClient() as client:
                    response = await client.post(
                        url,
                        headers=headers,
                        json=body,
                        timeout=self.valves.request_timeout
                    )
                    
                    if response.status_code == 200:
                        result = response.json()
                        self.circuit_breaker.record_success()
                        
                        if "id" in result:
                            self.last_generation_id = result["id"]
                            self.log(f"✓ Generation ID: {self.last_generation_id}", user_valves=user_valves)
                        
                        return result
                    
                    elif response.status_code == 400:
                        error_data = response.json()
                        error_msg = error_data.get("error", {}).get("message", "Bad request")
                        self.log(f"API error 400: {error_msg}", "error", user_valves)
                        await self.emit_message(event_emitter, f"❌ Invalid request: {error_msg}")
                        self.circuit_breaker.record_failure()
                        return None
                    
                    elif response.status_code == 401:
                        error_msg = "Authentication failed - check your OpenRouter API key"
                        self.log(error_msg, "error", user_valves)
                        await self.emit_message(event_emitter, f"❌ {error_msg}")
                        self.circuit_breaker.record_failure()
                        return None
                    
                    elif response.status_code == 402:
                        error_msg = "Insufficient credits on OpenRouter account"
                        self.log(error_msg, "error", user_valves)
                        await self.emit_message(event_emitter, f"❌ {error_msg}")
                        self.circuit_breaker.record_failure()
                        return None
                    
                    elif response.status_code == 429:
                        if attempt < self.valves.max_retries - 1:
                            wait_time = (2 ** attempt) + (random.random() * 0.5)
                            self.log(
                                f"Rate limited, retrying in {wait_time:.1f}s "
                                f"(attempt {attempt+1}/{self.valves.max_retries})",
                                user_valves=user_valves
                            )
                            await self.emit_status(
                                event_emitter,
                                f"Rate limited, retrying in {wait_time:.1f}s...",
                                user_valves=user_valves
                            )
                            await asyncio.sleep(wait_time)
                            continue
                        else:
                            error_msg = "Rate limit exceeded after retries"
                            self.log(error_msg, "error", user_valves)
                            await self.emit_message(event_emitter, f"❌ {error_msg}")
                            self.circuit_breaker.record_failure()
                            return None
                    
                    elif response.status_code in [500, 502, 503, 504]:
                        if attempt < self.valves.max_retries - 1:
                            wait_time = 2 ** attempt
                            self.log(
                                f"Server error {response.status_code}, retrying in {wait_time}s",
                                user_valves=user_valves
                            )
                            await self.emit_status(
                                event_emitter,
                                f"Server error, retrying in {wait_time}s...",
                                user_valves=user_valves
                            )
                            await asyncio.sleep(wait_time)
                            continue
                        else:
                            error_msg = f"OpenRouter service error {response.status_code}"
                            self.log(error_msg, "error", user_valves)
                            await self.emit_message(event_emitter, f"❌ {error_msg}")
                            self.circuit_breaker.record_failure()
                            return None
                    
                    else:
                        error_msg = f"Unexpected status code: {response.status_code}"
                        self.log(error_msg, "error", user_valves)
                        await self.emit_message(event_emitter, f"❌ {error_msg}")
                        self.circuit_breaker.record_failure()
                        return None
            
            except httpx.TimeoutException:
                if attempt < self.valves.max_retries - 1:
                    wait_time = 2 ** attempt
                    self.log(f"Request timeout, retrying in {wait_time}s", user_valves=user_valves)
                    await self.emit_status(
                        event_emitter,
                        f"Timeout, retrying in {wait_time}s...",
                        user_valves=user_valves
                    )
                    await asyncio.sleep(wait_time)
                    continue
                else:
                    error_msg = f"Request timed out after {self.valves.request_timeout}s"
                    self.log(error_msg, "error", user_valves)
                    await self.emit_message(event_emitter, f"❌ {error_msg}")
                    self.circuit_breaker.record_failure()
                    return None
            
            except httpx.ConnectError as e:
                error_msg = f"Connection failed: {str(e)}"
                self.log(error_msg, "error", user_valves)
                await self.emit_message(event_emitter, f"❌ {error_msg}")
                self.circuit_breaker.record_failure()
                return None
            
            except Exception as e:
                self.log(f"Unexpected error: {str(e)}", "error", user_valves)
                if attempt < self.valves.max_retries - 1:
                    await asyncio.sleep(2 ** attempt)
                    continue
                else:
                    await self.emit_message(event_emitter, f"❌ Unexpected error: {str(e)}")
                    self.circuit_breaker.record_failure()
                    return None
        
        return None
    
    # ========================================
    # FILTER INTERFACE - INLET
    # ========================================
    
    async def inlet(
        self,
        body: dict,
        __user__: Optional[dict] = None,
        __event_emitter__: Optional[Callable[[dict], Awaitable[None]]] = None
    ) -> dict:
        """Pre-process requests"""
        
        user_valves = __user__.get("valves") if __user__ else None
        if not user_valves:
            user_valves = self.user_valves
        
        # Get API key (user's personal key, or from OI connection, or admin fallback)
        api_key = self.get_api_key(__user__)
        
        if not api_key:
            self.log("No OpenRouter API key available", "error", user_valves)
            await self.emit_message(
                __event_emitter__,
                "⚠️ No OpenRouter API key configured. Please either:\n"
                "1. Set your personal API key in filter settings, OR\n"
                "2. Configure OpenRouter in your model connections, OR\n"
                "3. Ask admin to set a system-wide fallback key"
            )
            return body
        
        try:
            messages = body.get("messages", [])
            
            has_images = self._detect_images(messages)
            is_generation = self._is_generation_request(messages)
            
            self.log(
                f"Request analysis: has_images={has_images}, is_generation={is_generation}",
                user_valves=user_valves
            )
            
            mode = user_valves.operation_mode if hasattr(user_valves, 'operation_mode') else self.user_valves.operation_mode
            
            if mode == "auto":
                if is_generation and not has_images:
                    mode = "generation_only"
                elif has_images:
                    mode = "vision_only"
            
            self.log(f"Operation mode: {mode}", user_valves=user_valves)
            
            if mode == "vision_only" and has_images:
                await self.emit_status(
                    __event_emitter__,
                    "Processing images for vision analysis...",
                    user_valves=user_valves
                )
                
                body["messages"] = await self._process_images(messages, user_valves)
                
                auto_select = user_valves.auto_model_selection if hasattr(user_valves, 'auto_model_selection') else self.user_valves.auto_model_selection
                
                if auto_select:
                    if not self._is_vision_model(body.get("model", "")):
                        original_model = body.get("model")
                        vision_model = user_valves.vision_model if hasattr(user_valves, 'vision_model') else self.user_valves.vision_model
                        body["model"] = vision_model
                        self.log(
                            f"Switched to vision model: {original_model} → {body['model']}",
                            user_valves=user_valves
                        )
                
                await self.emit_status(
                    __event_emitter__,
                    "Images ready for analysis",
                    done=True,
                    user_valves=user_valves
                )
                self.processing_images = True
            
            elif mode == "generation_only":
                await self.emit_status(
                    __event_emitter__,
                    "Preparing image generation request...",
                    user_valves=user_valves
                )
                
                gen_model = user_valves.generation_model if hasattr(user_valves, 'generation_model') else self.user_valves.generation_model
                body["model"] = gen_model
                
                if "modalities" not in body:
                    body["modalities"] = ["image", "text"]
                
                await self.emit_status(
                    __event_emitter__,
                    "Generation request ready",
                    done=True,
                    user_valves=user_valves
                )
                self.processing_images = False
            
            return body
            
        except Exception as e:
            self.log(f"Inlet error: {str(e)}", "error", user_valves)
            await self.emit_message(__event_emitter__, f"⚠️ Processing error: {str(e)}")
            return body
    
    # ========================================
    # FILTER INTERFACE - OUTLET
    # ========================================
    
    async def outlet(
        self,
        body: dict,
        __user__: Optional[dict] = None,
        __event_emitter__: Optional[Callable[[dict], Awaitable[None]]] = None
    ) -> dict:
        """Post-process responses"""
        
        user_valves = __user__.get("valves") if __user__ else None
        if not user_valves:
            user_valves = self.user_valves
        
        try:
            if "id" in body:
                self.log(f"Response ID: {body['id']}", user_valves=user_valves)
            
            if "choices" in body:
                for choice in body["choices"]:
                    message = choice.get("message", {})
                    
                    if "images" in message and message["images"]:
                        num_images = len(message["images"])
                        self.log(
                            f"✓ Response contains {num_images} generated image(s)",
                            user_valves=user_valves
                        )
                        
                        content = message.get("content", "")
                        if not content or len(content) < 50:
                            image_links = "\n\n".join([
                                f"![Generated Image {i+1}]({img})"
                                for i, img in enumerate(message["images"])
                            ])
                            message["content"] = f"Here are the generated images:\n\n{image_links}"
            
            return body
            
        except Exception as e:
            self.log(f"Outlet error: {str(e)}", "error", user_valves)
            return body