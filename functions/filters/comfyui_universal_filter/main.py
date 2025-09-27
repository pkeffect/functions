"""
title: ComfyUI Universal Filter - Combined
version: 4.1.0
author: pkeffect & therezz (reptar)
author_url: https://github.com/pkeffect/
project_url: https://github.com/pkeffect/functions/tree/main/functions/filters/comfyui_universal_filter
funding_url: https://github.com/open-webui
required_open_webui_version: 0.6.0
date: 2025-09-27
license: MIT
description: Advanced ComfyUI filter with node:field mapping, robust base64 handling, VRAM management, and dual response modes
"""

import json
import uuid
import aiohttp
import asyncio
import random
import base64
import requests
from typing import List, Dict, Optional, Callable, Awaitable, Tuple
from pydantic import BaseModel, Field
from io import BytesIO
from PIL import Image

import logging

# Setup comprehensive logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("ComfyUIUniversalFilter")


# --- OLLAMA VRAM Management Functions ---
def get_loaded_models(api_url: str = "http://localhost:11434") -> list:
    """Get list of currently loaded Ollama models"""
    try:
        response = requests.get(f"{api_url.rstrip('/')}/api/ps", timeout=5)
        response.raise_for_status()
        return response.json().get("models", [])
    except requests.RequestException as e:
        logger.error(f"Error fetching loaded Ollama models: {e}")
        return []


def unload_all_models(api_url: str = "http://localhost:11434"):
    """Unload all Ollama models from VRAM"""
    try:
        for model in get_loaded_models(api_url):
            model_name = model.get("name")
            if model_name:
                requests.post(
                    f"{api_url.rstrip('/')}/api/generate",
                    json={"model": model_name, "keep_alive": 0},
                    timeout=10,
                )
        logger.info("All Ollama models unloaded")
    except requests.RequestException as e:
        logger.error(f"Error unloading Ollama models: {e}")


# --- Base64 Helper Functions ---
def _fix_base64_padding(s: str) -> str:
    """Remove whitespace, convert URL-safe alphabet, add '=' padding."""
    s = "".join(s.split())
    if "-" in s or "_" in s:
        s = s.replace("-", "+").replace("_", "/")
    missing = len(s) % 4
    if missing:
        s += "=" * (4 - missing)
    return s


def _decode_base64_robust(s: str) -> bytes:
    """Robust base64 decoder: supports data URLs, missing padding, URL-safe alphabet."""
    if isinstance(s, bytes):
        return s
    if not isinstance(s, str):
        raise TypeError("image must be data URL string, raw base64 string, or bytes")

    if s.startswith("data:image"):
        s = s.split(",", 1)[1]
    try:
        return base64.b64decode(s, validate=False)
    except Exception:
        fixed = _fix_base64_padding(s)
        try:
            return base64.b64decode(fixed, validate=False)
        except Exception as e:
            raise ValueError(f"Invalid base64 image: {e}")


def _normalize_base64_png(image_input) -> str:
    """Convert input (data URL / raw base64 / bytes) to PNG base64 (NO data URL prefix)."""
    raw = _decode_base64_robust(image_input)
    img = Image.open(BytesIO(raw))
    buf = BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


# --- Mapping Helper Functions ---
def _parse_mapping(mapping: str) -> Tuple[Optional[str], Optional[str]]:
    """Parse 'nodeID:fieldName' into ('nodeID', 'fieldName'). Returns (None, None) if invalid."""
    if not mapping or ":" not in mapping:
        return None, None
    node_id, field = mapping.split(":", 1)
    node_id = node_id.strip()
    field = field.strip()
    if not node_id or not field:
        return None, None
    return node_id, field


def _set_field(workflow: dict, mapping: str, value, name: str) -> bool:
    """Set workflow[node_id]['inputs'][field] = value using a mapping string."""
    node_id, field = _parse_mapping(mapping)
    if not node_id or not field:
        logger.debug(f"[map] {name}: mapping not configured")
        return False
    node = workflow.get(node_id)
    if not node:
        logger.warning(f"[map] {name}: node {node_id} not found")
        return False
    if "inputs" not in node:
        logger.warning(f"[map] {name}: node {node_id} missing 'inputs'")
        return False
    node["inputs"][field] = value
    logger.debug(
        f"[map] {name}: node {node_id} field '{field}' set -> {str(value)[:200]}"
    )
    return True


class Filter:
    class Valves(BaseModel):
        priority: int = Field(
            default=0, description="Priority level for the filter function."
        )

        # ComfyUI Connection
        ComfyUI_Address: str = Field(
            default="http://host.docker.internal:8188",
            description="Address of the running ComfyUI server. Use 'host.docker.internal' if Open WebUI runs in Docker.",
        )
        ComfyUI_API_Key: str = Field(
            default="",
            description="Optional Bearer token for ComfyUI authentication",
        )

        # Workflow Configuration
        ComfyUI_Workflow_JSON: str = Field(
            default="{}",
            description="The entire ComfyUI workflow in JSON format (API format).",
            extra={"type": "textarea"},
        )

        # Node:Field Mappings (Advanced)
        Prompt_Mapping: str = Field(
            default="76:prompt",
            description="node:field for the text prompt (e.g., '76:prompt')",
        )
        Image_Mapping: str = Field(
            default="78:image",
            description="node:field for the input image (e.g., '78:image')",
        )
        Seed_Mapping: str = Field(
            default="3:seed",
            description="node:field for the seed (e.g., '3:seed')",
        )
        Negative_Mapping: str = Field(
            default="",
            description="node:field for negative prompt (optional, e.g., '77:prompt')",
        )
        Steps_Mapping: str = Field(
            default="", description="node:field for steps (optional, e.g., '3:steps')"
        )
        CFG_Mapping: str = Field(
            default="", description="node:field for cfg (optional, e.g., '3:cfg')"
        )
        Denoise_Mapping: str = Field(
            default="",
            description="node:field for denoise (optional, e.g., '3:denoise')",
        )
        Model_Mapping: str = Field(
            default="", description="node:field for model name (optional)"
        )
        Sampler_Mapping: str = Field(
            default="", description="node:field for sampler name (optional)"
        )
        Scheduler_Mapping: str = Field(
            default="", description="node:field for scheduler (optional)"
        )

        # Default Values
        Default_Steps: int = Field(default=20, description="Default steps value")
        Default_CFG: float = Field(default=7.0, description="Default cfg value")
        Default_Denoise: float = Field(
            default=0.75, description="Default denoise value"
        )
        Default_Negative: str = Field(default="", description="Default negative prompt")
        Default_Model: str = Field(default="", description="Default model name")
        Default_Sampler: str = Field(default="euler", description="Default sampler")
        Default_Scheduler: str = Field(
            default="normal", description="Default scheduler"
        )

        # Response Mode
        Response_Mode: str = Field(
            default="llm_instruction",
            description="Response mode: 'direct_injection' (add to messages) or 'llm_instruction' (tell LLM to output markdown)",
        )

        # VRAM Management
        unload_ollama_models: bool = Field(
            default=False,
            description="Unload all Ollama models from VRAM before running ComfyUI.",
        )
        ollama_url: str = Field(
            default="http://host.docker.internal:11434",
            description="Ollama API URL for unloading models.",
        )

        # Timing
        max_wait_time: int = Field(
            default=300, description="Max wait time for generation (seconds)."
        )
        poll_interval: float = Field(
            default=1.0, description="Polling interval for /history (seconds)"
        )

        # UI Settings
        show_detailed_progress: bool = Field(
            default=False, description="Show detailed step-by-step progress updates"
        )

    def __init__(self):
        self.valves = self.Valves()
        self.client_id = str(uuid.uuid4())

        # Response tracking for outlet method
        self.processed_image_url = None  # For direct injection mode
        self.last_result_url = None  # For llm_instruction mode success
        self.last_processing_failed = False  # For llm_instruction mode failure

        # UI Toggle and Icon for Open WebUI filter button
        self.toggle = True
        self.icon = "data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIyNCIgaGVpZ2h0PSIyNCIgdmlld0JveD0iMCAwIDI0IDI0IiBmaWxsPSJub25lIiBzdHJva2U9ImN1cnJlbnRDb2xvciIgc3Ryb2tlLXdpZHRoPSIyIiBzdHJva2UtbGluZWNhcD0icm91bmQiIHN0cm9rZS1saW5lam9pbj0icm91bmQiPjxwYXRoIGQ9Im0xOCAzIDQgNEw0IDIyIDMgMjEgMjEgMyIvPjxwYXRoIGQ9Im0yMiAzLTEgMSIvPjwvc3ZnPg=="

    # --- Message Processing ---
    def get_image_from_messages(self, messages: List[Dict]) -> Optional[str]:
        """Extract base64 image from messages (latest user message)"""
        logger.debug(f"Scanning {len(messages)} messages for images")
        for message in reversed(messages):
            if message.get("role") != "user":
                continue

            # Check content field
            content = message.get("content")
            if isinstance(content, list):
                for item in content:
                    if item.get("type") == "image_url":
                        url = item.get("image_url", {}).get("url", "")
                        if "base64," in url:
                            base64_data = url.split("base64,", 1)[1]
                            logger.info(f"Found base64 image: {len(base64_data)} chars")
                            return base64_data

            # Check images field (Open WebUI specific)
            if "images" in message:
                for img_url in message["images"]:
                    if "base64," in img_url:
                        base64_data = img_url.split("base64,", 1)[1]
                        logger.info(f"Found base64 image: {len(base64_data)} chars")
                        return base64_data

            break  # Only check latest user message

        return None

    def get_prompt_from_messages(self, messages: List[Dict]) -> str:
        """Extract text prompt from messages (latest user message)"""
        for message in reversed(messages):
            if message.get("role") != "user":
                continue

            content = message.get("content", "")
            if isinstance(content, list):
                text_parts = [
                    item.get("text", "")
                    for item in content
                    if item.get("type") == "text"
                ]
                return " ".join(text_parts).strip()
            elif isinstance(content, str):
                return content.strip()

            break
        return ""

    def strip_images_for_llm(self, messages: List[Dict]) -> List[Dict]:
        """Remove image payloads so the LLM never sees base64 images."""
        scrubbed = []
        for msg in messages:
            m = dict(msg)
            if m.get("role") == "user":
                if isinstance(m.get("content"), list):
                    m["content"] = [
                        it for it in m["content"] if it.get("type") != "image_url"
                    ]
                    if not m["content"]:
                        m["content"] = ""
                if "images" in m:
                    m.pop("images", None)
            scrubbed.append(m)
        return scrubbed

    # --- ComfyUI Integration ---
    async def upload_image_to_comfyui(self, base64_data: str) -> Optional[str]:
        """Upload base64 image to ComfyUI and return filename"""
        try:
            # Normalize to PNG base64
            png_b64 = _normalize_base64_png(base64_data)
            image_bytes = base64.b64decode(png_b64)
            filename = f"openwebui_upload_{uuid.uuid4().hex}.png"

            base_url = self.valves.ComfyUI_Address.rstrip("/")
            headers = {}
            if self.valves.ComfyUI_API_Key:
                headers["Authorization"] = f"Bearer {self.valves.ComfyUI_API_Key}"

            async with aiohttp.ClientSession(headers=headers) as session:
                # Create form data for upload
                data = aiohttp.FormData()
                data.add_field(
                    "image", image_bytes, filename=filename, content_type="image/png"
                )
                data.add_field("overwrite", "true")
                data.add_field("type", "input")

                async with session.post(
                    f"{base_url}/upload/image", data=data, timeout=60
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        uploaded_filename = result.get("name", filename)
                        logger.info(f"Successfully uploaded image: {uploaded_filename}")
                        return uploaded_filename
                    else:
                        error_text = await response.text()
                        logger.error(f"Upload failed: {response.status} - {error_text}")
                        return None

        except Exception as e:
            logger.error(f"Image upload error: {e}", exc_info=True)
            return None

    def configure_workflow(self, workflow: dict, prompt: str, image_ref: str) -> dict:
        """Configure workflow using node:field mappings"""
        v = self.valves

        # Required mappings
        _set_field(workflow, v.Prompt_Mapping, prompt, "prompt")
        _set_field(workflow, v.Image_Mapping, image_ref, "image")
        _set_field(workflow, v.Seed_Mapping, random.randint(0, 2**32 - 1), "seed")

        # Optional mappings
        if v.Negative_Mapping:
            _set_field(
                workflow, v.Negative_Mapping, v.Default_Negative, "negative_prompt"
            )
        if v.Steps_Mapping:
            _set_field(workflow, v.Steps_Mapping, v.Default_Steps, "steps")
        if v.CFG_Mapping:
            _set_field(workflow, v.CFG_Mapping, v.Default_CFG, "cfg")
        if v.Denoise_Mapping:
            _set_field(workflow, v.Denoise_Mapping, v.Default_Denoise, "denoise")
        if v.Model_Mapping and v.Default_Model:
            _set_field(workflow, v.Model_Mapping, v.Default_Model, "model")
        if v.Sampler_Mapping and v.Default_Sampler:
            _set_field(workflow, v.Sampler_Mapping, v.Default_Sampler, "sampler")
        if v.Scheduler_Mapping and v.Default_Scheduler:
            _set_field(workflow, v.Scheduler_Mapping, v.Default_Scheduler, "scheduler")

        return workflow

    async def execute_comfyui_workflow(
        self, workflow: dict, event_emitter=None
    ) -> Optional[str]:
        """Execute workflow on ComfyUI server with WebSocket + polling fallback"""
        try:
            base_url = self.valves.ComfyUI_Address.rstrip("/")
            headers = {}
            if self.valves.ComfyUI_API_Key:
                headers["Authorization"] = f"Bearer {self.valves.ComfyUI_API_Key}"

            async with aiohttp.ClientSession(headers=headers) as session:
                # Queue workflow
                async with session.post(
                    f"{base_url}/prompt",
                    json={"prompt": workflow, "client_id": self.client_id},
                    timeout=60,
                ) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        logger.error(
                            f"Failed to queue: {response.status} - {error_text}"
                        )
                        return None

                    result = await response.json()
                    prompt_id = result.get("prompt_id")

                if not prompt_id:
                    logger.error("No prompt_id returned")
                    return None

                logger.info(f"Workflow queued with ID: {prompt_id}")

                # Try WebSocket first, fall back to polling
                ws_url = f"{'ws' if not base_url.startswith('https') else 'wss'}://{base_url.split('://', 1)[-1]}/ws?clientId={self.client_id}"
                completed = await self.wait_for_completion_ws(
                    ws_url, prompt_id, event_emitter
                )

                if not completed:
                    logger.warning("WebSocket failed, falling back to polling")
                    completed = await self.wait_for_completion_polling(
                        session, base_url, prompt_id
                    )

                if not completed:
                    logger.error("Workflow did not complete successfully")
                    return None

                # Get result
                return await self.get_result_image_url(session, base_url, prompt_id)

        except Exception as e:
            logger.error(f"Workflow execution failed: {e}", exc_info=True)
            return None

    async def wait_for_completion_ws(
        self, ws_url: str, prompt_id: str, event_emitter=None
    ) -> bool:
        """Wait for workflow completion via WebSocket"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.ws_connect(ws_url, timeout=30) as ws:
                    start_time = asyncio.get_event_loop().time()

                    async for msg in ws:
                        if (
                            asyncio.get_event_loop().time() - start_time
                            > self.valves.max_wait_time
                        ):
                            logger.error("WebSocket timeout")
                            return False

                        if msg.type != aiohttp.WSMsgType.TEXT:
                            continue

                        try:
                            data = json.loads(msg.data)
                        except json.JSONDecodeError:
                            continue

                        msg_type = data.get("type")
                        msg_data = data.get("data", {})

                        if msg_type == "progress" and event_emitter:
                            value = msg_data.get("value", 0)
                            max_val = msg_data.get("max", 1)
                            if self.valves.show_detailed_progress:
                                await self.emit_status(
                                    event_emitter,
                                    f"Processing: {value}/{max_val} steps",
                                )
                            # For condensed mode, only show every 5th step or major milestones
                            elif value % 5 == 0 or value == max_val:
                                await self.emit_status(
                                    event_emitter,
                                    f"Processing: {value}/{max_val} steps",
                                )

                        elif (
                            msg_type == "executed"
                            and msg_data.get("prompt_id") == prompt_id
                        ):
                            logger.info("Workflow completed successfully")
                            return True
                        elif (
                            msg_type == "execution_error"
                            and msg_data.get("prompt_id") == prompt_id
                        ):
                            logger.error(
                                f"Workflow failed: {msg_data.get('exception_message')}"
                            )
                            return False

        except Exception as e:
            logger.warning(f"WebSocket error: {e}")
            return False

        return False

    async def wait_for_completion_polling(
        self, session: aiohttp.ClientSession, base_url: str, prompt_id: str
    ) -> bool:
        """Wait for completion using polling as fallback"""
        start_time = asyncio.get_event_loop().time()
        while True:
            if asyncio.get_event_loop().time() - start_time > self.valves.max_wait_time:
                logger.error("Polling timeout")
                return False

            try:
                async with session.get(
                    f"{base_url}/history/{prompt_id}", timeout=30
                ) as response:
                    if response.status != 200:
                        await asyncio.sleep(self.valves.poll_interval)
                        continue

                    history = await response.json()
                    if prompt_id in history:
                        outputs = history[prompt_id].get("outputs", {})
                        if outputs:  # Has outputs, consider complete
                            logger.info("Workflow completed (polling)")
                            return True
            except Exception as e:
                logger.warning(f"Polling error: {e}")

            await asyncio.sleep(self.valves.poll_interval)

    async def get_result_image_url(
        self, session: aiohttp.ClientSession, base_url: str, prompt_id: str
    ) -> Optional[str]:
        """Get the result image URL from ComfyUI history"""
        try:
            await asyncio.sleep(1)  # Brief wait for history to be available

            async with session.get(
                f"{base_url}/history/{prompt_id}", timeout=30
            ) as response:
                if response.status != 200:
                    logger.error(f"Failed to get history: {response.status}")
                    return None

                history = await response.json()
                if prompt_id not in history:
                    logger.error("Prompt ID not found in history")
                    return None

                # Find output images
                outputs = history[prompt_id].get("outputs", {})
                for node_id, node_output in outputs.items():
                    if "images" in node_output and node_output["images"]:
                        image_info = node_output["images"][0]  # Take first image
                        filename = image_info["filename"]
                        subfolder = image_info.get("subfolder", "")

                        # Build view URL
                        params = f"filename={filename}&type=output"
                        if subfolder:
                            params += f"&subfolder={subfolder}"

                        image_url = f"{base_url}/view?{params}"

                        # Convert host.docker.internal to localhost for browser accessibility
                        if "host.docker.internal" in image_url:
                            image_url = image_url.replace(
                                "host.docker.internal", "localhost"
                            )

                        logger.info(f"Found result image: {image_url}")
                        return image_url

                logger.error("No images found in outputs")
                return None

        except Exception as e:
            logger.error(f"Failed to get result image: {e}", exc_info=True)
            return None

    # --- Event Emitter Helpers ---
    async def emit_status(
        self, event_emitter: Optional[Callable], message: str, done: bool = False
    ):
        """Emit status updates to the client"""
        if event_emitter:
            await event_emitter(
                {
                    "type": "status",
                    "data": {
                        "status": "complete" if done else "in_progress",
                        "description": message,
                        "done": done,
                    },
                }
            )

    async def emit_message(self, event_emitter: Optional[Callable], message: str):
        """Emit a message directly to the chat interface"""
        if event_emitter:
            await event_emitter({"type": "message", "data": {"content": message}})

    def _prepare_minimal_request(self, body: dict) -> dict:
        """Prepare a minimal request that will be intercepted by outlet"""
        # Send minimal request to model that will be replaced by outlet
        minimal_body = {
            "model": body.get("model", "gpt-3.5-turbo"),
            "messages": [{"role": "user", "content": "status"}],
            "max_tokens": 5,
            "stream": False,
            "temperature": 0,
        }
        logger.info("Prepared minimal request for outlet interception")
        return minimal_body

    # --- Filter Interface ---
    async def inlet(
        self,
        body: dict,
        __user__: dict = None,
        __event_emitter__: Optional[Callable[[dict], Awaitable[None]]] = None,
    ) -> dict:
        """Process incoming requests when filter is toggled ON"""

        # Reset tracking variables
        self.last_result_url = None
        self.last_processing_failed = False
        self.processed_image_url = None

        messages = body.get("messages", [])

        # Check for images
        base64_image = self.get_image_from_messages(messages)
        if not base64_image:
            return body  # Pass through if no images

        # Get prompt
        prompt = self.get_prompt_from_messages(messages)
        if not prompt:
            prompt = "enhance this image"

        logger.info(
            f"ComfyUI Filter activated! Processing image with prompt: '{prompt[:100]}...'"
        )

        try:
            await self.emit_status(
                __event_emitter__, "Starting ComfyUI image processing..."
            )

            # Unload Ollama if needed
            if self.valves.unload_ollama_models:
                await self.emit_status(__event_emitter__, "Unloading Ollama models...")
                unload_all_models(self.valves.ollama_url)

            # Parse workflow
            try:
                workflow = json.loads(self.valves.ComfyUI_Workflow_JSON)
                logger.debug(f"Workflow loaded: {len(workflow)} nodes")
            except json.JSONDecodeError as e:
                logger.error(f"Invalid workflow JSON: {e}")
                await self.emit_status(
                    __event_emitter__, "Invalid workflow JSON configuration", done=True
                )
                return body

            # Determine if we need inline base64 or upload
            inline_b64 = False
            try:
                img_node_id, _ = _parse_mapping(self.valves.Image_Mapping)
                if img_node_id and img_node_id in workflow:
                    node_type = workflow[img_node_id].get("class_type", "")
                    inline_b64 = node_type in ("ETN_LoadImageBase64", "LoadImageBase64")
                    logger.info(
                        f"Image node {img_node_id} type={node_type}; inline_b64={inline_b64}"
                    )
            except Exception as e:
                logger.warning(f"Could not inspect image node: {e}")

            # Prepare image reference
            if inline_b64:
                image_ref = _normalize_base64_png(base64_image)
                await self.emit_status(
                    __event_emitter__, "Using inline base64 image..."
                )
            else:
                await self.emit_status(
                    __event_emitter__, "Uploading image to ComfyUI..."
                )
                image_ref = await self.upload_image_to_comfyui(base64_image)
                if not image_ref:
                    await self.emit_status(
                        __event_emitter__, "Image upload failed", done=True
                    )
                    return body

            # Configure workflow
            workflow = self.configure_workflow(workflow, prompt, image_ref)

            # Execute workflow
            await self.emit_status(__event_emitter__, "Processing with ComfyUI...")
            result_url = await self.execute_comfyui_workflow(
                workflow, __event_emitter__
            )

            if not result_url:
                await self.emit_status(
                    __event_emitter__, "ComfyUI processing failed", done=True
                )
                return body

            await self.emit_status(
                __event_emitter__, "Image processing complete!", done=True
            )

            # Handle response based on mode
            if self.valves.Response_Mode == "direct_injection":
                # Store result for outlet method
                self.processed_image_url = result_url

                # Add assistant response directly to messages
                image_response = (
                    f"Here is your edited image:\n\n![Generated Image]({result_url})"
                )
                body["messages"].append(
                    {"role": "assistant", "content": image_response}
                )

                # Prevent additional model response
                body["max_tokens"] = 1
                body["stream"] = False
                return body

            else:  # llm_instruction mode
                # Modify the user message to get the LLM to output exactly what we want
                # This is the key insight from the games hub filter!
                for msg in reversed(messages):
                    if msg.get("role") == "user":
                        msg["content"] = (
                            f"Respond with exactly this message and nothing else: '![Generated Image]({result_url})'"
                        )
                        break

                logger.info(
                    f"Modified user message to request exact image output: {result_url}"
                )
                return body

        except Exception as e:
            logger.error(f"ComfyUI Filter error: {e}", exc_info=True)
            await self.emit_status(
                __event_emitter__, f"Processing failed: {str(e)}", done=True
            )
            return body

    async def outlet(self, body: dict, __user__: dict = None) -> dict:
        """Process responses after they come back from the model (only for direct_injection mode)"""

        # Handle direct_injection mode only
        if self.valves.Response_Mode == "direct_injection" and self.processed_image_url:
            logger.info(f"Outlet intercepting model response to preserve image")

            success_message = f"![Generated Image]({self.processed_image_url})"

            if isinstance(body, dict) and "choices" in body:
                for choice in body["choices"]:
                    if "message" in choice:
                        choice["message"]["content"] = success_message
                        logger.info("Replaced message content in direct_injection mode")
                        break
                    elif "delta" in choice:
                        choice["delta"]["content"] = success_message
                        logger.info("Replaced delta content in direct_injection mode")
                        break

            self.processed_image_url = None

        return body
