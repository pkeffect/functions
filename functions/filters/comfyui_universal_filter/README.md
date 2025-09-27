# 🎨 ComfyUI Universal Filter - Combined

[![Open WebUI](https://img.shields.io/badge/Open%20WebUI-Compatible-blue)](https://openwebui.com/)
[![ComfyUI](https://img.shields.io/badge/ComfyUI-Integration-green)](https://github.com/comfyanonymous/ComfyUI)
[![Python](https://img.shields.io/badge/Python-3.8+-yellow)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-red)](LICENSE)
[![Version](https://img.shields.io/badge/Version-4.1.0-purple)](https://github.com/open-webui/open-webui)

> **Advanced ComfyUI filter with node:field mapping, robust base64 handling, VRAM management, and dual response modes**

## 🚀 Quick Start

### 📥 Installation
1. Copy the entire filter code from the source file
2. In Open WebUI, go to **Admin Panel** → **Settings** → **Functions**
3. Click **"+"** to add a new function
4. Paste the code and click **Save**
5. The filter will appear in your function list with a magic wand icon

### ⚡ Initial Setup
1. Configure your **ComfyUI_Address** (default: `http://host.docker.internal:8188`)
2. Add your **ComfyUI_Workflow_JSON** (export from ComfyUI using "Save (API Format)")
3. Set your **node:field mappings** based on your workflow
4. Test with an image + text prompt

---

## 📋 Overview

An advanced filter for Open WebUI that enables seamless integration with ComfyUI for AI image processing. This filter automatically detects images in conversations and processes them through ComfyUI workflows with intelligent VRAM management and dual response modes.

**Version:** 4.1.0  
**Authors:** pkeffect & therezz (reptar)  
**Required Open WebUI Version:** 0.5.0+

---

## 🔧 Core Features

### 🎯 Intelligent Image Detection
- **Multi-format Support**: Handles data URLs, raw base64, and bytes
- **Robust Base64 Decoding**: Fixes padding, supports URL-safe alphabet
- **PNG Normalization**: Converts all images to PNG format for ComfyUI

### 🗺️ Advanced Node:Field Mapping System
- **Dynamic Workflow Configuration**: Map any workflow parameter using `nodeID:fieldName` syntax
- **Required Mappings**: Prompt, Image, Seed
- **Optional Mappings**: Negative prompt, Steps, CFG, Denoise, Model, Sampler, Scheduler

### 💾 VRAM Management
- **Ollama Integration**: Automatically unloads Ollama models before ComfyUI processing
- **Memory Optimization**: Prevents VRAM conflicts between different AI systems

### 📡 Dual Response Modes
- **Direct Injection**: Bypasses LLM entirely, injects image directly into conversation
- **LLM Instruction**: Instructs the LLM to output specific markdown

---

## 🏗️ Technical Architecture

### 📥 Message Processing Pipeline

```python
# 1. Image Extraction
base64_image = get_image_from_messages(messages)
# Scans messages for base64 images in content or images fields

# 2. Prompt Extraction  
prompt = get_prompt_from_messages(messages)
# Extracts text from latest user message

# 3. Image Scrubbing (for LLM modes)
scrubbed_messages = strip_images_for_llm(messages)
# Removes image payloads to prevent LLM processing
```

### 🔄 Base64 Handling System

The filter implements a robust 3-layer base64 processing system:

#### Layer 1: Padding & Format Fixing
```python
def _fix_base64_padding(s: str) -> str:
    s = "".join(s.split())  # Remove whitespace
    if "-" in s or "_" in s:
        s = s.replace("-", "+").replace("_", "/")  # URL-safe conversion
    missing = len(s) % 4
    if missing:
        s += "=" * (4 - missing)  # Add padding
    return s
```

#### Layer 2: Robust Decoding
```python
def _decode_base64_robust(s: str) -> bytes:
    if s.startswith("data:image"):
        s = s.split(",", 1)[1]  # Strip data URL prefix
    try:
        return base64.b64decode(s, validate=False)
    except Exception:
        fixed = _fix_base64_padding(s)
        return base64.b64decode(fixed, validate=False)
```

#### Layer 3: PNG Normalization
```python
def _normalize_base64_png(image_input) -> str:
    raw = _decode_base64_robust(image_input)
    img = Image.open(BytesIO(raw))
    buf = BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")
```

### 🗺️ Node:Field Mapping System

The mapping system allows dynamic workflow configuration without hardcoded node IDs:

#### Mapping Parser
```python
def _parse_mapping(mapping: str) -> Tuple[Optional[str], Optional[str]]:
    # Input: "76:prompt" -> Output: ("76", "prompt")
    if not mapping or ":" not in mapping:
        return None, None
    node_id, field = mapping.split(":", 1)
    return node_id.strip(), field.strip()
```

#### Field Setter
```python
def _set_field(workflow: dict, mapping: str, value, name: str) -> bool:
    node_id, field = _parse_mapping(mapping)
    if not node_id or not field:
        return False
    
    node = workflow.get(node_id)
    if not node or "inputs" not in node:
        return False
        
    node["inputs"][field] = value  # Set the actual value
    return True
```

#### Workflow Configuration
```python
def configure_workflow(self, workflow: dict, prompt: str, image_ref: str) -> dict:
    # Required mappings
    _set_field(workflow, self.valves.Prompt_Mapping, prompt, "prompt")
    _set_field(workflow, self.valves.Image_Mapping, image_ref, "image") 
    _set_field(workflow, self.valves.Seed_Mapping, random.randint(0, 2**32 - 1), "seed")
    
    # Optional mappings with defaults
    if self.valves.Steps_Mapping:
        _set_field(workflow, self.valves.Steps_Mapping, self.valves.Default_Steps, "steps")
    # ... additional optional mappings
    
    return workflow
```

### 🚀 ComfyUI Execution Engine

#### Dual Execution Modes

**1. Inline Base64 Mode**
- For nodes like `ETN_LoadImageBase64` or `LoadImageBase64`
- Passes base64 data directly to workflow
- No upload required

**2. Upload Mode**  
- For standard `LoadImage` nodes
- Uploads image to ComfyUI's input directory
- Returns filename for workflow reference

#### WebSocket + Polling Hybrid
```python
async def execute_comfyui_workflow(self, workflow: dict, event_emitter=None):
    # 1. Queue workflow
    response = await session.post(f"{base_url}/prompt", 
                                json={"prompt": workflow, "client_id": self.client_id})
    prompt_id = result.get("prompt_id")
    
    # 2. Try WebSocket first
    completed = await self.wait_for_completion_ws(ws_url, prompt_id, event_emitter)
    
    # 3. Fallback to polling if WebSocket fails
    if not completed:
        completed = await self.wait_for_completion_polling(session, base_url, prompt_id)
    
    # 4. Extract result image URL
    return await self.get_result_image_url(session, base_url, prompt_id)
```

#### Progress Tracking
- **WebSocket Events**: Real-time progress updates via ComfyUI WebSocket
- **Detailed Mode**: Shows every step
- **Condensed Mode**: Shows every 5th step or milestones
- **Polling Fallback**: Uses `/history` endpoint when WebSocket unavailable

### 📡 Response Mode System

#### Mode 1: Direct Injection
```python
# In inlet():
self.processed_image_url = result_url
body["messages"].append({
    "role": "assistant", 
    "content": f"![Generated Image]({result_url})"
})
body["max_tokens"] = 1  # Prevent additional model response

# In outlet():
if self.processed_image_url:
    choice["message"]["content"] = f"![Generated Image]({self.processed_image_url})"
```

#### Mode 2: LLM Instruction
```python
# Modify user message to instruct LLM exactly what to output
for msg in reversed(messages):
    if msg.get("role") == "user":
        msg["content"] = f"Respond with exactly this message and nothing else: '![Generated Image]({result_url})'"
        break
```

---

## ⚙️ Complete Valve Configuration

### 🔧 System Settings

#### `priority: int = 0`
**Description:** Priority level for the filter function  
**Options:** Any integer (higher = higher priority)  
**Default:** `0`

---

### 🔗 ComfyUI Connection

#### `ComfyUI_Address: str`
**Description:** Address of the running ComfyUI server  
**Options:** Any valid URL  
**Default:** `"http://host.docker.internal:8188"`  
**Docker Note:** Use `host.docker.internal` if Open WebUI runs in Docker

#### `ComfyUI_API_Key: str`
**Description:** Optional Bearer token for ComfyUI authentication  
**Options:** Any string or empty  
**Default:** `""`

---

### 📋 Workflow Configuration

#### `ComfyUI_Workflow_JSON: str`
**Description:** The entire ComfyUI workflow in JSON format (API format)  
**Options:** Valid JSON workflow from ComfyUI "Save (API Format)"  
**Default:** `"{}"`  
**Type:** Textarea input

---

### 🗺️ Node:Field Mappings (Advanced)

#### `Prompt_Mapping: str`
**Description:** node:field for the text prompt  
**Format:** `"nodeID:fieldName"`  
**Example:** `"76:prompt"`  
**Default:** `"76:prompt"`

#### `Image_Mapping: str`
**Description:** node:field for the input image  
**Format:** `"nodeID:fieldName"`  
**Example:** `"78:image"`  
**Default:** `"78:image"`

#### `Seed_Mapping: str`
**Description:** node:field for the seed  
**Format:** `"nodeID:fieldName"`  
**Example:** `"3:seed"`  
**Default:** `"3:seed"`

#### `Negative_Mapping: str`
**Description:** node:field for negative prompt (optional)  
**Format:** `"nodeID:fieldName"` or empty  
**Example:** `"77:prompt"`  
**Default:** `""`

#### `Steps_Mapping: str`
**Description:** node:field for steps (optional)  
**Format:** `"nodeID:fieldName"` or empty  
**Example:** `"3:steps"`  
**Default:** `""`

#### `CFG_Mapping: str`
**Description:** node:field for cfg (optional)  
**Format:** `"nodeID:fieldName"` or empty  
**Example:** `"3:cfg"`  
**Default:** `""`

#### `Denoise_Mapping: str`
**Description:** node:field for denoise (optional)  
**Format:** `"nodeID:fieldName"` or empty  
**Example:** `"3:denoise"`  
**Default:** `""`

#### `Model_Mapping: str`
**Description:** node:field for model name (optional)  
**Format:** `"nodeID:fieldName"` or empty  
**Example:** `"4:ckpt_name"`  
**Default:** `""`

#### `Sampler_Mapping: str`
**Description:** node:field for sampler name (optional)  
**Format:** `"nodeID:fieldName"` or empty  
**Example:** `"3:sampler_name"`  
**Default:** `""`

#### `Scheduler_Mapping: str`
**Description:** node:field for scheduler (optional)  
**Format:** `"nodeID:fieldName"` or empty  
**Example:** `"3:scheduler"`  
**Default:** `""`

---

### 🎛️ Default Values

#### `Default_Steps: int`
**Description:** Default steps value  
**Options:** Any positive integer  
**Default:** `20`

#### `Default_CFG: float`
**Description:** Default cfg value  
**Options:** Any positive float  
**Default:** `7.0`

#### `Default_Denoise: float`
**Description:** Default denoise value  
**Options:** Float between 0.0 and 1.0  
**Default:** `0.75`

#### `Default_Negative: str`
**Description:** Default negative prompt  
**Options:** Any string  
**Default:** `""`

#### `Default_Model: str`
**Description:** Default model name  
**Options:** Any valid ComfyUI model name or empty  
**Default:** `""`

#### `Default_Sampler: str`
**Description:** Default sampler  
**Options:** ComfyUI sampler names: `"euler"`, `"euler_ancestral"`, `"heun"`, `"dpm_2"`, `"dpm_2_ancestral"`, `"lms"`, `"dpm_fast"`, `"dpm_adaptive"`, `"dpmpp_2s_ancestral"`, `"dpmpp_sde"`, `"dpmpp_2m"`, etc.  
**Default:** `"euler"`

#### `Default_Scheduler: str`
**Description:** Default scheduler  
**Options:** ComfyUI scheduler names: `"normal"`, `"karras"`, `"exponential"`, `"sgm_uniform"`, `"simple"`, `"ddim_uniform"`  
**Default:** `"normal"`

---

### 📡 Response Mode

#### `Response_Mode: str`
**Description:** Response mode for handling generated images  
**Options:**
- `"direct_injection"` - Add image directly to messages, bypass LLM
- `"llm_instruction"` - Tell LLM to output markdown image

**Default:** `"llm_instruction"`

---

### 💾 VRAM Management

#### `unload_ollama_models: bool`
**Description:** Unload all Ollama models from VRAM before running ComfyUI  
**Options:** `true` / `false`  
**Default:** `false`

#### `ollama_url: str`
**Description:** Ollama API URL for unloading models  
**Options:** Any valid URL  
**Default:** `"http://host.docker.internal:11434"`

---

### ⏱️ Timing Configuration

#### `max_wait_time: int`
**Description:** Max wait time for generation (seconds)  
**Options:** Any positive integer  
**Default:** `300`

#### `poll_interval: float`
**Description:** Polling interval for /history (seconds)  
**Options:** Any positive float  
**Default:** `1.0`

---

### 🖥️ UI Settings

#### `show_detailed_progress: bool`
**Description:** Show detailed step-by-step progress updates  
**Options:**
- `true` - Show every step update
- `false` - Show condensed updates (every 5th step)

**Default:** `false`

---

## 🔧 Setup Instructions

### 1️⃣ ComfyUI Workflow Preparation
1. Create your workflow in ComfyUI
2. Use **"Save (API Format)"** to export JSON
3. Note the node IDs for key components:
   - Text prompt input node
   - Image input node  
   - Seed/sampling parameter nodes

### 2️⃣ Filter Configuration in Open WebUI
1. Navigate to **Admin Panel** → **Settings** → **Functions**
2. Find "ComfyUI Universal Filter" in your function list
3. Click the **settings gear icon** to configure valves
4. Configure the `ComfyUI_Workflow_JSON` field with your workflow
5. Set up node:field mappings based on your workflow
6. Test with a simple image + prompt

### 3️⃣ Node ID Discovery
Use ComfyUI's developer tools or examine the API format JSON:
```json
{
  "76": {
    "class_type": "CLIPTextEncode", 
    "inputs": {
      "prompt": "your text here"  // Map as "76:prompt"
    }
  }
}
```

---

## 🐛 Troubleshooting

### ❌ Common Issues

**Workflow Not Executing**
- ✅ Verify ComfyUI server is running and accessible
- ✅ Check node:field mappings match your workflow
- ✅ Ensure workflow JSON is valid API format

**Images Not Processing** 
- ✅ Confirm image is in base64 format in message
- ✅ Check image upload/inline mode compatibility
- ✅ Verify image input node type and mapping

**VRAM Issues**
- ✅ Enable Ollama model unloading
- ✅ Adjust ComfyUI model settings
- ✅ Monitor system memory usage

### 🔍 Debug Logging
The filter provides comprehensive logging:
```python
logger.info(f"Found base64 image: {len(base64_data)} chars")
logger.debug(f"[map] prompt: node 76 field 'prompt' set -> {prompt}")
logger.warning(f"WebSocket failed, falling back to polling")
```

---

## 🔄 Workflow Examples

### 🎨 Image-to-Image Enhancement
```json
{
  "3": {
    "class_type": "KSampler",
    "inputs": {
      "seed": 0,          // Map: "3:seed"
      "steps": 20,        // Map: "3:steps" 
      "cfg": 7.0,         // Map: "3:cfg"
      "denoise": 0.75     // Map: "3:denoise"
    }
  },
  "76": {
    "class_type": "CLIPTextEncode",
    "inputs": {
      "prompt": ""        // Map: "76:prompt"
    }
  },
  "78": {
    "class_type": "LoadImage", 
    "inputs": {
      "image": ""         // Map: "78:image"
    }
  }
}
```

### 🔧 ControlNet Processing
```json
{
  "15": {
    "class_type": "ControlNetApply",
    "inputs": {
      "image": "",        // Map: "15:image"
      "strength": 1.0     // Map: "15:strength"
    }
  }
}
```

---

## 🚀 Performance Optimization

### ⚡ Speed Improvements
- Use WebSocket for real-time progress
- Enable Ollama unloading for VRAM efficiency
- Optimize workflow complexity
- Use appropriate polling intervals

### 💡 Memory Management
- PNG normalization reduces file sizes
- Base64 caching prevents re-processing
- Automatic cleanup of uploaded images
- Smart VRAM allocation

---

## 🔮 Advanced Features

### 🎛️ Custom Node Support
The filter supports any ComfyUI node through the mapping system:
```python
# Custom nodes examples
"CustomNode_1:parameter": "value"
"AnotherNode_5:setting": "config"
```

### 🔄 Multi-Image Workflows
Extend for multiple image inputs:
```python
Image_Mapping_1: "78:image"
Image_Mapping_2: "79:image" 
```

### 🎨 Dynamic Parameter Adjustment
Implement parameter extraction from prompts:
```python
# Extract parameters like "steps:30 cfg:8.5" from user input
# Apply to workflow dynamically
```

---

## 📚 API Reference

### 🔧 Core Methods

#### `get_image_from_messages(messages: List[Dict]) -> Optional[str]`
Extracts base64 image data from Open WebUI message format.

#### `configure_workflow(workflow: dict, prompt: str, image_ref: str) -> dict`
Applies node:field mappings to configure ComfyUI workflow.

#### `execute_comfyui_workflow(workflow: dict, event_emitter=None) -> Optional[str]`
Executes workflow on ComfyUI server and returns result image URL.

### 📡 Event System

#### Status Events
```python
await self.emit_status(event_emitter, "Processing with ComfyUI...", done=False)
```

#### Message Events  
```python
await self.emit_message(event_emitter, "Custom message to user")
```

---

## 🤝 Contributing

### 📋 Development Guidelines
- Follow existing code structure and naming conventions
- Add comprehensive logging for debugging
- Test with multiple workflow types
- Document new mapping options

### 🧪 Testing Checklist
- [ ] Base64 image detection and normalization
- [ ] Node:field mapping accuracy  
- [ ] WebSocket and polling execution
- [ ] Both response modes (direct/LLM instruction)
- [ ] VRAM management functionality
- [ ] Error handling and recovery

---

## 📄 License & Credits

**Authors:** pkeffect & therezz (reptar)  
**Open WebUI Integration:** Advanced filter system  
**ComfyUI Compatibility:** Full API format support

---

## 🔗 Related Documentation

- [Open WebUI Documentation](https://docs.openwebui.com/)
- [ComfyUI API Documentation](https://github.com/comfyanonymous/ComfyUI)
- [Open WebUI Filter Development](https://deepwiki.com/open-webui/open-webui/)