# 🎨 OpenRouter Universal Image Filter

[![Open WebUI](https://img.shields.io/badge/Open%20WebUI-Compatible-blue)](https://openwebui.com/)
[![OpenRouter](https://img.shields.io/badge/OpenRouter-API-green)](https://openrouter.ai/)
[![Python](https://img.shields.io/badge/Python-3.8+-yellow)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-red)](LICENSE)

> **Production-ready filter for OpenRouter vision and image generation models with comprehensive error handling, retry logic, and circuit breaker pattern**

---

## 🚀 Quick Start

### 📥 Installation

1. Copy the entire filter code
2. In Open WebUI: **Admin Panel** → **Settings** → **Functions**
3. Click **"+"** to add new function
4. Paste code and click **Save**
5. **Admin:** Configure OpenRouter API key in filter settings
6. **Users:** Configure personal preferences in your user settings

### ⚡ Initial Setup

**For Admins (Optional):**
1. Get API key from [OpenRouter](https://openrouter.ai/keys) (optional - only if providing fallback)
2. Go to Admin Panel → Functions → OpenRouter Image Filter → Settings (Valves)
3. Paste API key in **OPENROUTER_API_KEY** field (or leave empty)
4. Adjust system-wide limits if needed (timeout, max retries, etc.)
5. Save settings

**For Users (Choose ONE option):**

**Option 1: Use your existing OpenRouter connection (EASIEST)**
- If you already have OpenRouter configured in Settings → Models
- The filter automatically detects and uses your API key
- No additional configuration needed!

**Option 2: Add personal API key to filter**
1. Get your API key from [OpenRouter](https://openrouter.ai/keys)
2. Go to your Profile/Settings → Functions
3. Find "OpenRouter Image Filter"
4. Set your **OPENROUTER_API_KEY** in UserValves
5. Configure other preferences (models, quality, etc.)
6. Save and start using!

**Option 3: Use system fallback key**
- If admin has configured a fallback key
- Just configure your preferences (models, quality settings)
- Leave **OPENROUTER_API_KEY** empty in UserValves
- Uses admin's shared key

---

### 🎯 First Test

**Vision Test (upload an image):**
1. Upload an image in chat
2. Ask: "What's in this image?"
3. Filter automatically detects image and sends to your configured vision model
4. Get analysis back

**Generation Test (text only):**
1. Type: "Generate an image of a sunset over mountains"
2. Filter detects generation request
3. Switches to your configured generation model
4. Returns generated image

---

### 📥 Installation

1. Copy the entire filter code
2. In Open WebUI: **Admin Panel** → **Settings** → **Functions**
3. Click **"+"** to add new function
4. Paste code and click **Save**
5. Configure your OpenRouter API key in settings

### ⚡ Initial Setup

1. Get API key from [OpenRouter](https://openrouter.ai/keys)
2. Open filter settings and paste your key in **OPENROUTER_API_KEY**
3. Choose your preferred models:
   - **Vision**: `qwen/qwen-2-vl-7b-instruct` (default)
   - **Generation**: `google/gemini-2.5-flash-image-preview` (default)
4. Test by uploading an image and asking a question

---

## 📋 Overview

A **complete production filter** for Open WebUI that integrates OpenRouter's vision and image generation models with enterprise-grade reliability.

### ✨ Key Features

- 🔍 **Automatic Image Detection** - Detects images in conversations automatically
- 🎯 **Smart Mode Selection** - Auto-switches between vision analysis and image generation
- 🛡️ **Circuit Breaker Pattern** - Prevents cascading failures with automatic recovery
- 🔄 **Exponential Backoff** - Intelligent retry logic for transient failures
- 📐 **Image Optimization** - Automatic resize and compression to meet API limits
- ✅ **Comprehensive Validation** - Validates images before sending
- 📊 **Real-time Status Updates** - Progress feedback in Open WebUI interface
- 🔧 **Full Error Handling** - Graceful degradation with helpful error messages

---

## 🏗️ Architecture

### 🔄 Processing Flow

```
User Message → Inlet → Image Detection → Optimization → OpenRouter API → Outlet → Response
                ↓                                          ↓
         Format Images                            Handle Retries
         Validate Data                            Circuit Breaker
         Select Model                             Error Recovery
```

### 🧩 Core Components

#### **1. Circuit Breaker**
Prevents overwhelming the API with requests when failures occur:
- Tracks failure count
- Opens circuit after threshold (default: 5 failures)
- Automatically attempts to close after timeout (default: 60s)
- Resets on successful requests

#### **2. Retry Logic with Exponential Backoff**
Handles transient failures intelligently:
- Rate limits: 2^attempt + random jitter (0-0.5s)
- Server errors: 2^attempt fixed delay
- Timeouts: Progressive backoff
- Max retries configurable (default: 3)

#### **3. Image Optimization Pipeline**
Ensures images meet API requirements:
1. **Validation** - Verify format and integrity
2. **Resize** - Scale to optimal dimensions (≤1568px for Claude)
3. **Format Conversion** - Convert to RGB, remove alpha channel
4. **Compression** - JPEG with quality setting (default: 85)
5. **Size Check** - Verify under provider limits

#### **4. Multi-Format Support**
Handles various input formats:
- Open WebUI native format (`images` array)
- OpenAI multimodal format (`image_url` objects)
- Data URLs with/without prefix
- Raw base64 strings with automatic padding fix

---

## ⚙️ Complete Configuration Guide

### 🔑 API Configuration

#### `OPENROUTER_API_KEY` *(required)*
**Description:** Your OpenRouter API key  
**Get it from:** https://openrouter.ai/keys  
**Format:** String (e.g., `sk-or-v1-...`)  
**Security:** Stored securely, never logged

#### `OPENROUTER_BASE_URL`
**Description:** OpenRouter API base URL  
**Default:** `https://openrouter.ai/api/v1`  
**Note:** Only change if using a proxy

---

### 🤖 Model Selection

#### `vision_model`
**Description:** Model for image analysis/editing  
**Default:** `qwen/qwen-2-vl-7b-instruct`  
**Options:**
- `qwen/qwen-2-vl-7b-instruct` - Fast, cost-effective
- `qwen/qwen-2.5-vl-32b-instruct` - Higher quality
- `anthropic/claude-3-5-sonnet` - Best reasoning
- `openai/gpt-4o` - Strong multimodal
- `google/gemini-2.0-flash-exp:free` - Free option

#### `generation_model`
**Description:** Model for image generation  
**Default:** `google/gemini-2.5-flash-image-preview`  
**Options:**
- `google/gemini-2.5-flash-image-preview` - Fast generation
- Check OpenRouter for latest image generation models

#### `auto_model_selection`
**Description:** Auto-switch to vision model when images detected  
**Type:** Boolean  
**Default:** `true`  
**Use case:** Enable for seamless experience

---

### 🎛️ Operation Mode

#### `operation_mode`
**Description:** Control filter behavior  
**Type:** Dropdown  
**Options:**
- `auto` - Smart detection (recommended)
- `vision_only` - Only process uploaded images
- `generation_only` - Only generate new images

**Auto mode logic:**
```python
if "generate image" in prompt and no images uploaded:
    → generation_only
elif images uploaded:
    → vision_only
```

---

### 🖼️ Image Processing

#### `max_image_dimension`
**Description:** Maximum pixel dimension (width or height)  
**Type:** Integer  
**Default:** `1568`  
**Why 1568?** Optimal for Claude models (no server-side resize)  
**Range:** 512-2048 recommended

#### `auto_resize`
**Description:** Automatically resize large images  
**Type:** Boolean  
**Default:** `true`  
**Impact:** Reduces API costs and latency

#### `jpeg_quality`
**Description:** JPEG compression quality  
**Type:** Integer (1-100)  
**Default:** `85`  
**Balance:** 85 = good quality, smaller size  
**Notes:**
- 90-100: Minimal compression, large files
- 75-85: Balanced (recommended)
- 50-74: High compression, quality loss

#### `max_image_size_mb`
**Description:** Max image size before compression  
**Type:** Float  
**Default:** `5.0`  
**Provider limits:**
- OpenAI: 20MB
- Anthropic: 5MB (strict)
- Google: 20MB

---

### 🌐 API Settings

#### `request_timeout`
**Description:** API request timeout in seconds  
**Type:** Integer  
**Default:** `120`  
**Recommended:** 60-180 depending on network

#### `max_retries`
**Description:** Maximum retry attempts  
**Type:** Integer  
**Default:** `3`  
**Range:** 1-5 recommended  
**Note:** Total time = timeout × retries

#### `enable_circuit_breaker`
**Description:** Enable circuit breaker for resilience  
**Type:** Boolean  
**Default:** `true`  
**Benefits:**
- Prevents cascading failures
- Automatic recovery
- Protects API quota

---

### 🎨 Generation Parameters

#### `temperature`
**Description:** Sampling temperature for generation  
**Type:** Float (0.0-2.0)  
**Default:** `0.7`  
**Guide:**
- 0.0-0.3: Deterministic, consistent
- 0.4-0.8: Balanced creativity
- 0.9-2.0: Highly creative, varied

#### `max_tokens`
**Description:** Maximum tokens in response  
**Type:** Integer  
**Default:** `4096`  
**Note:** Doesn't affect image generation, only text

---

### 🔍 Debug & Logging

#### `enable_logging`
**Description:** Enable console logging  
**Type:** Boolean  
**Default:** `true`  
**Logs:**
- Image processing steps
- API calls and responses
- Error details
- Performance metrics

#### `log_requests`
**Description:** Log full request details  
**Type:** Boolean  
**Default:** `false`  
**Warning:** ⚠️ May expose sensitive data, use only for debugging

---

### 📡 Optional Headers

#### `app_name`
**Description:** Application name for OpenRouter tracking  
**Type:** String  
**Default:** `"Open WebUI"`  
**Purpose:** Helps OpenRouter provide better analytics

#### `site_url`
**Description:** Site URL for OpenRouter tracking  
**Type:** String  
**Default:** `"https://openwebui.com"`  
**Purpose:** Optional referrer tracking

---

## 🎯 Usage Examples

### 📸 Vision/Analysis Mode

**Upload an image and ask:**
```
What's in this image?
Describe the colors and composition
What brand is this logo?
Translate the text in this image
```

**The filter will:**
1. Detect the uploaded image
2. Optimize it (resize if needed)
3. Send to vision model
4. Return analysis

### 🎨 Generation Mode

**Text-only prompts:**
```
Generate an image of a sunset over mountains
Create a logo for a coffee shop called "Brew Haven"
Draw a cyberpunk city at night
Make an abstract artwork with blue and gold
```

**The filter will:**
1. Detect generation keywords
2. Switch to generation model
3. Add `modalities: ["image", "text"]`
4. Return generated image(s)

### 🔄 Mixed Workflows

**Edit/Transform (with image uploaded):**
```
Make this image black and white
Add a vintage filter to this photo
Remove the background from this image
Enhance the colors in this picture
```

**The filter automatically:**
- Routes to vision model (has image)
- Processes the uploaded image
- Sends edit instruction
- Returns result

---

## 🛠️ Advanced Configuration

### 🔧 Custom Model Setup

For specific use cases, you can configure different models:

**High-quality vision analysis:**
```python
vision_model: "anthropic/claude-3-5-sonnet:beta"
max_tokens: 8192
temperature: 0.3
```

**Fast, cost-effective processing:**
```python
vision_model: "qwen/qwen-2-vl-7b-instruct:free"
auto_resize: true
max_image_dimension: 1024
```

**Premium image generation:**
```python
generation_model: "google/gemini-2.5-flash-image-preview"
temperature: 0.9
```

---

## 🐛 Troubleshooting

### ❌ Common Issues

#### **"OpenRouter API key not configured"**
**Cause:** No API key available from any source  
**Solution:**
- **Option 1:** Configure OpenRouter in Settings → Models (recommended)
- **Option 2:** Add your personal key in filter UserValves
- **Option 3:** Ask admin to set a system fallback key
- Check the error message for specific guidance

#### **"No OpenRouter API key available"**
**Cause:** Filter couldn't find a key from user, model connection, or admin  
**Solution:**
1. Check if you have OpenRouter in Settings → Models
2. If not, add your personal key to UserValves
3. If using admin key, ask admin to configure it
4. See error message for detailed next steps

#### **"Circuit breaker is OPEN"**
**Cause:** Too many recent failures (system-wide)  
**Solution:**
1. Wait 60 seconds for auto-recovery
2. Admin should check API key validity
3. Verify OpenRouter service status at https://status.openrouter.ai
4. Admin can temporarily disable circuit breaker if needed

#### **"Authentication failed"**
**Cause:** Invalid or missing API key (admin issue)  
**Solution:**
1. Admin: Verify key at https://openrouter.ai/keys
2. Ensure no extra spaces in key
3. Check key hasn't expired or been revoked

#### **"Rate limit exceeded"**
**Cause:** Too many requests to OpenRouter  
**Solution:**
1. Filter will automatically retry with backoff
2. Wait a few seconds between requests
3. Admin may need to upgrade OpenRouter plan for higher limits

#### **"Image too large"**
**Cause:** Image exceeds admin-configured size limit  
**Solution:**
1. Enable `auto_resize: true` in your user settings
2. Lower `max_image_dimension` to 1024 in your settings
3. Reduce `jpeg_quality` to 75 in your settings
4. Contact admin if limit is too restrictive

#### **Images not being processed**
**Cause:** Various possible issues  
**Solution:**
1. Enable `enable_logging: true` in YOUR user settings
2. Check browser console for error messages
3. Verify your selected vision model supports images
4. Test with a small PNG image first
5. Contact admin if API key issues

#### **Model doesn't see images**
**Cause:** Non-vision model selected in your settings  
**Solution:**
1. Enable `auto_model_selection: true` in your settings
2. Manually select a vision model (e.g., `qwen/qwen-2-vl-7b-instruct`)
3. Verify model supports vision on OpenRouter model page

#### **Different users getting different results**
**Cause:** Normal - users have different settings!  
**Explanation:**
- Each user can configure their own models
- Different quality/compression settings per user
- Different temperature values produce different outputs
- This is by design for flexibility

---

## 📊 Performance Optimization

### ⚡ Speed Improvements (User Settings)

**For fastest processing:**
```python
vision_model: "qwen/qwen-2-vl-7b-instruct"  # Fastest vision model
max_image_dimension: 1024  # Smaller = faster
jpeg_quality: 75  # More compression = faster upload
auto_resize: true  # Enable optimization
```

**For best quality (slower):**
```python
vision_model: "anthropic/claude-3-5-sonnet"  # Best reasoning
max_image_dimension: 2048  # Full quality
jpeg_quality: 95  # Minimal compression
auto_resize: false  # Keep original
```

### 💰 Cost Optimization (User Settings)

**Free models only:**
```python
vision_model: "qwen/qwen-2-vl-7b-instruct:free"
generation_model: "google/gemini-2.0-flash-exp:free"
```

**Optimize images aggressively:**
```python
auto_resize: true
max_image_dimension: 1024  # Smaller = cheaper
jpeg_quality: 75  # More compression = cheaper
```

### 🔧 Admin Optimizations

**Reduce timeouts for faster failures:**
```python
request_timeout: 60  # Instead of 120
max_retries: 2  # Instead of 3
```

**Enforce stricter limits:**
```python
max_image_size_mb: 3.0  # Instead of 5.0
```

---

## 🔐 Security Best Practices

### 🛡️ API Key Management (Admin)

✅ **DO:**
- Store API key only in admin Valves
- Use environment variables for production deployments
- Rotate keys periodically
- Monitor usage on OpenRouter dashboard
- Set up billing alerts on OpenRouter

❌ **DON'T:**
- Share the admin API key with users
- Hardcode keys in the filter code
- Commit keys to version control
- Allow users to override the API key

### 🔒 Data Privacy

**For Admins:**
- Images are sent to OpenRouter (third-party service)
- Processed by underlying model providers (OpenAI, Anthropic, etc.)
- Review provider privacy policies before deployment
- Consider self-hosted alternatives for sensitive data
- Inform users about data handling

**For Users:**
- Your images are sent to external AI providers
- Don't upload sensitive/confidential images
- Enable logging only when debugging (may log prompts)
- Your settings are private (not shared with other users)

### 👥 Multi-User Security

**User isolation:**
- Each user has independent settings
- User A's logging doesn't affect User B
- Personal preferences stored separately
- No cross-user data leakage

**Admin controls:**
- API key shared (cost pooling)
- Enforce system-wide limits
- Monitor total usage across all users
- Can disable filter globally if needed

---

## 📈 Monitoring & Analytics

### 📊 Admin Monitoring

**Track overall usage:**
1. Go to https://openrouter.ai/activity
2. View all generations under your API key
3. Monitor costs per model
4. Track success/failure rates
5. Identify heavy users (by generation IDs in logs)

**System health:**
- Check circuit breaker status in logs
- Monitor retry rates
- Track timeout frequency
- Review error patterns

### 🔍 User Debugging

**Enable your personal logging:**
```python
enable_logging: true  # In YOUR user settings
```

**Your logs will show:**
- Image optimization metrics for your images
- Model selection decisions for your requests
- API calls for your requests only
- Your error messages with context
- Your retry attempts

**Note:** Your logging doesn't affect other users or system logs.

---

## 💡 Best Practices for Multi-User Deployments

### 👨‍💼 For Admins

**1. Set reasonable defaults:**
```python
# Admin Valves
max_image_size_mb: 5.0  # Reasonable limit
request_timeout: 120  # Generous timeout
max_retries: 3  # Good balance
enable_circuit_breaker: true  # Protect the system
```

**2. Monitor costs:**
- Check OpenRouter dashboard weekly
- Set billing alerts
- Review model usage patterns
- Consider rate limiting if needed

**3. Communicate with users:**
- Document which models are available
- Share cost information if relevant
- Explain system limits
- Provide troubleshooting guide

**4. Regular maintenance:**
- Rotate API keys quarterly
- Review error logs monthly
- Update model list as new options available
- Test with different user configurations

### 👤 For Users

**1. Choose models wisely:**
- Free models for casual use
- Premium models for important work
- Balance cost vs quality based on your needs

**2. Optimize your images:**
- Enable auto_resize for daily use
- Use lower quality for drafts
- Only use high quality when needed

**3. Be considerate:**
- Don't spam generation requests
- Respect shared resources
- Report issues to admin
- Disable logging when not debugging

**4. Experiment with settings:**
- Try different models to find favorites
- Adjust temperature for creative control
- Fine-tune image quality preferences
- Share good configurations with team

---

## 🎓 Advanced Use Cases

### 🏢 Enterprise Deployment

**Admin Configuration:**
```python
# Valves (Admin)
OPENROUTER_API_KEY: "sk-or-v1-..." # Company account
max_image_size_mb: 10.0  # Higher limit for professional use
request_timeout: 180  # Longer for complex processing
enable_circuit_breaker: true  # Critical for stability
app_name: "CompanyName Internal Tool"
site_url: "https://internal.company.com"
```

**Department-specific templates:**

**Design Team:**
```python
# UserValves
vision_model: "anthropic/claude-3-5-sonnet"
max_image_dimension: 2048
jpeg_quality: 95
temperature: 0.9
```

**Development Team:**
```python
# UserValves
vision_model: "qwen/qwen-2-vl-7b-instruct"
max_image_dimension: 1024
jpeg_quality: 80
temperature: 0.3
```

**Marketing Team:**
```python
# UserValves
generation_model: "google/gemini-2.5-flash-image-preview"
temperature: 0.8
operation_mode: "auto"
```

### 🎨 Creative Studio Setup

**Admin provides premium access:**
```python
# Valves
max_image_size_mb: 20.0  # Large files for pros
request_timeout: 300  # Complex generations
```

**Artists configure for quality:**
```python
# UserValves
vision_model: "anthropic/claude-3-5-sonnet"
generation_model: "google/gemini-2.5-flash-image-preview"
max_image_dimension: 2048
jpeg_quality: 95
temperature: 1.0  # Maximum creativity
auto_resize: false  # Keep original quality
show_status_updates: true  # See progress
```

### 🏫 Educational Institution

**Admin protects budget:**
```python
# Valves
max_image_size_mb: 3.0  # Smaller limit
request_timeout: 90  # Faster failures
max_retries: 2  # Less retry overhead
```

**Students use free models:**
```python
# UserValves (suggested defaults for students)
vision_model: "qwen/qwen-2-vl-7b-instruct:free"
generation_model: "google/gemini-2.0-flash-exp:free"
max_image_dimension: 1024
jpeg_quality: 75
temperature: 0.7
enable_logging: false  # Reduce noise
```

**Professors use premium:**
```python
# UserValves (professors override)
vision_model: "anthropic/claude-3-5-sonnet"
max_image_dimension: 1568
jpeg_quality: 90
```

---

## 🔄 Migration Guide

### 📦 Coming from ComfyUI Filter

**Key differences:**

| Aspect | ComfyUI Filter | OpenRouter Filter |
|--------|----------------|-------------------|
| Setup | Workflow JSON, node mapping | Just API key |
| Models | Local, self-hosted | 200+ cloud models |
| Configuration | Workflow-specific | Model-specific |
| Users | Shared workflow | Personal preferences |
| Cost | Hardware | Pay-per-use |

**Migration steps:**

1. **Admin:** Get OpenRouter API key
2. **Admin:** Install OpenRouter filter
3. **Admin:** Configure API key in Valves
4. **Users:** Choose equivalent models:
   - ComfyUI SDXL → `stabilityai/stable-diffusion-xl`
   - ComfyUI Flux → Check OpenRouter for Flux models
   - Custom workflows → Find similar models on OpenRouter
5. **Test:** Compare outputs
6. **Transition:** Disable ComfyUI filter when satisfied

**What you'll gain:**
- ✅ No local setup/maintenance
- ✅ Access to latest models instantly
- ✅ Per-user customization
- ✅ Better reliability (circuit breaker, retries)

**What you'll miss:**
- ❌ Full workflow control
- ❌ Custom nodes
- ❌ Free local processing

---

## 🚀 Roadmap & Future Features

### 🔮 Planned Enhancements

**System-level:**
- [ ] Usage quotas per user (admin setting)
- [ ] Model whitelist/blacklist (admin control)
- [ ] Cost tracking per user
- [ ] Custom model fallbacks
- [ ] Response caching

**User-level:**
- [ ] Multi-image generation support
- [ ] Image editing with masks
- [ ] Batch processing
- [ ] Favorite model presets
- [ ] Personal prompt templates
- [ ] Generation history

**Integration:**
- [ ] Webhook notifications
- [ ] External storage for images
- [ ] API usage reports
- [ ] Slack/Discord notifications
- [ ] Custom model providers

---

## 📞 Support & Community

### 🆘 Getting Help

**As a User:**
1. Check your personal settings (UserValves)
2. Enable logging in your settings
3. Try with default settings
4. Contact your admin if API key issues
5. Report bugs with logs

**As an Admin:**
1. Check admin settings (Valves)
2. Verify API key validity
3. Review OpenRouter dashboard
4. Check system logs
5. Review this documentation

**Community Resources:**
- [Open WebUI Discord](https://discord.gg/openwebui)
- [OpenRouter Discord](https://discord.gg/openrouter)
- [GitHub Discussions](https://github.com/open-webui/open-webui/discussions)
- [OpenRouter Docs](https://openrouter.ai/docs)

### 📋 Bug Reports

**Include in your report:**
1. Filter version
2. Open WebUI version
3. Your role (admin/user)
4. Error messages from logs (enable logging)
5. Steps to reproduce
6. Model being used
7. User settings (UserValves) if relevant
8. Admin settings (Valves) if relevant (admins only)

**For privacy:**
- Don't share API keys
- Redact sensitive prompts
- Mask user identifiable information

---

## 📄 License & Credits

**License:** MIT  
**Author:** Community  
**Platform:** [Open WebUI](https://openwebui.com)  
**API Provider:** [OpenRouter](https://openrouter.ai)

### 🙏 Acknowledgments

- Open WebUI team for the filter architecture and multi-user support
- OpenRouter for unified API access to multiple providers
- Community contributors and testers
- Users providing feedback for improvements

---

## ⚡ Quick Reference Card

### 🎯 Recommended Configurations

**Default (Balanced):**
```python
# UserValves
vision_model: "qwen/qwen-2-vl-7b-instruct"
generation_model: "google/gemini-2.5-flash-image-preview"
operation_mode: "auto"
auto_model_selection: true
max_image_dimension: 1568
jpeg_quality: 85
temperature: 0.7
auto_resize: true
show_status_updates: true
```

**Budget (Free/Cheap):**
```python
# UserValves
vision_model: "qwen/qwen-2-vl-7b-instruct:free"
generation_model: "google/gemini-2.0-flash-exp:free"
max_image_dimension: 1024
jpeg_quality: 75
auto_resize: true
```

**Quality (Premium):**
```python
# UserValves
vision_model: "anthropic/claude-3-5-sonnet"
max_image_dimension: 2048
jpeg_quality: 95
temperature: 0.9
auto_resize: false
```

**Speed (Fast):**
```python
# UserValves
vision_model: "qwen/qwen-2-vl-7b-instruct"
max_image_dimension: 1024
jpeg_quality: 75
temperature: 0.5
auto_resize: true
```

---

## 📋 FAQ

**Q: How do I configure my personal settings?**  
A: Go to your Profile/Settings → Functions → OpenRouter Image Filter → Configure UserValves

**Q: Can I use different models than other users?**  
A: Yes! Each user configures their own vision_model and generation_model in UserValves.

**Q: Why can't I change the API key?**  
A: API key is admin-only (in Valves) for security and cost management. Contact your admin.

**Q: My colleague gets better quality images. Why?**  
A: They probably have different UserValves settings (higher jpeg_quality, larger max_image_dimension, or different model).

**Q: Does my logging affect others?**  
A: No! Your enable_logging setting only affects your own requests.

**Q: Can admin see my prompts?**  
A: Only if they have access to system logs. Your UserValves are private, but API requests go through shared infrastructure.

**Q: Which models support vision?**  
A: Check OpenRouter's model list with "multimodal" tag. Common ones: GPT-4o, Claude 3.5, Qwen-VL series, Gemini 2.0

**Q: Can I use this with qwen-image or qwen-image-edit specifically?**  
A: Those specific models aren't on OpenRouter. Use Qwen-VL models for similar vision functionality.

**Q: How much does this cost?**  
A: Depends on models used and volume. Check [OpenRouter pricing](https://openrouter.ai/models). Admins see consolidated billing.

**Q: What if I want to use my own API key?**  
A: Current design uses shared admin key. You could fork the filter and modify Valves → UserValves for OPENROUTER_API_KEY, but this complicates cost tracking.

**Q: Can I generate multiple images at once?**  
A: Depends on the model. Some support it, some don't. Check model documentation on OpenRouter.

**Q: Why do status updates not show?**  
A: Check your UserValves setting: `show_status_updates` might be false.

---

**Version:** 1.0.0  
**Last Updated:** 2025  
**Compatibility:** Open WebUI 0.5.0+  
**Architecture:** Multi-user with Valves (Admin) + UserValves (Per-user)

## 📊 Performance Optimization

### ⚡ Speed Improvements

**1. Reduce image dimensions:**
```python
max_image_dimension: 1024  # Instead of 1568
```

**2. Increase compression:**
```python
jpeg_quality: 75  # Instead of 85
```

**3. Use faster models:**
```python
vision_model: "qwen/qwen-2-vl-7b-instruct"  # Fast
# vs
vision_model: "anthropic/claude-3-5-sonnet"  # Slower but better
```

### 💰 Cost Optimization

**1. Use free models when possible:**
```python
vision_model: "qwen/qwen-2-vl-7b-instruct:free"
generation_model: "google/gemini-2.0-flash-exp:free"
```

**2. Optimize images aggressively:**
```python
auto_resize: true
max_image_dimension: 1024
jpeg_quality: 75
```

**3. Enable circuit breaker:**
```python
enable_circuit_breaker: true  # Prevents wasteful retry storms
```

---

## 🔐 Security Best Practices

### 🛡️ API Key Management

✅ **DO:**
- Store API key in Valves (encrypted by Open WebUI)
- Use environment variables in production
- Rotate keys periodically
- Monitor usage on OpenRouter dashboard

❌ **DON'T:**
- Hardcode keys in the filter code
- Share keys publicly
- Commit keys to version control
- Enable `log_requests` in production (may log keys)

### 🔒 Data Privacy

**Image handling:**
- Images are sent to OpenRouter (third-party)
- Processed by underlying model providers
- Review provider privacy policies
- Use self-hosted models for sensitive data

**Logging:**
- Disable `log_requests` in production
- Logs may contain user prompts and metadata
- Enable only for debugging

---

## 📈 Monitoring & Analytics

### 📊 Track Usage

OpenRouter provides detailed analytics:
1. Go to https://openrouter.ai/activity
2. View generation IDs (logged by filter)
3. Monitor costs per model
4. Track success/failure rates

### 🔍 Debug Logging

Enable comprehensive logging:
```python
enable_logging: true
log_requests: true  # Only when debugging
```

**Logged information:**
- Image optimization metrics (size reduction)
- Model selection decisions
- API request/response details
- Error messages with context
- Retry attempts and backoff times

---

## 🚀 Roadmap & Future Features

### 🔮 Planned Enhancements

- [ ] Multi-image generation support
- [ ] Image editing with masks
- [ ] Batch processing for multiple images
- [ ] Advanced prompt engineering helpers
- [ ] Model cost comparison
- [ ] Automatic model fallback
- [ ] Response caching
- [ ] Streaming support for generation

### 💡 Feature Requests

Submit ideas at [Open WebUI Discussions](https://github.com/open-webui/open-webui/discussions)

---

## 🤝 Contributing

### 📝 Development Guidelines

1. Follow existing code structure
2. Add comprehensive error handling
3. Update documentation for changes
4. Test with multiple models
5. Consider backward compatibility

### 🧪 Testing Checklist

Before submitting changes:
- [ ] Test with vision models (uploaded images)
- [ ] Test with generation models (text prompts)
- [ ] Test error handling (invalid API key, rate limits)
- [ ] Test image optimization (various sizes)
- [ ] Test retry logic (simulate failures)
- [ ] Test circuit breaker (force multiple failures)
- [ ] Verify logging output
- [ ] Check performance impact

---

## 📄 License & Credits

**License:** MIT  
**Author:** Community  
**Platform:** [Open WebUI](https://openwebui.com)  
**API Provider:** [OpenRouter](https://openrouter.ai)

### 🙏 Acknowledgments

- Open WebUI team for the filter architecture
- OpenRouter for unified API access
- Community contributors and testers

---

## 🔗 Related Resources

- [Open WebUI Documentation](https://docs.openwebui.com/)
- [OpenRouter Documentation](https://openrouter.ai/docs)
- [OpenRouter Models](https://openrouter.ai/models)
- [Filter Development Guide](https://docs.openwebui.com/features/plugin/functions/filter/)

---

## 💬 Support

**Need help?**
- Check [Troubleshooting](#-troubleshooting) section
- Review logs with `enable_logging: true`
- Ask in [Open WebUI Discord](https://discord.gg/openwebui)
- Report bugs on [GitHub Issues](https://github.com/open-webui/open-webui/issues)

**Before asking for help, provide:**
1. Filter version
2. Open WebUI version
3. Error messages from logs
4. Steps to reproduce issue
5. Model being used

---

## ⚡ Quick Reference

### 🎯 Recommended Settings

**For most users:**
```python
vision_model: "qwen/qwen-2-vl-7b-instruct"
generation_model: "google/gemini-2.5-flash-image-preview"
operation_mode: "auto"
auto_model_selection: true
auto_resize: true
max_image_dimension: 1568
jpeg_quality: 85
enable_circuit_breaker: true
enable_logging: true
```

**For cost-sensitive:**
```python
vision_model: "qwen/qwen-2-vl-7b-instruct:free"
generation_model: "google/gemini-2.0-flash-exp:free"
max_image_dimension: 1024
jpeg_quality: 75
```

**For high-quality:**
```python
vision_model: "anthropic/claude-3-5-sonnet"
max_image_dimension: 1568
jpeg_quality: 95
request_timeout: 180
```

---

## 📋 FAQ

**Q: Which models support vision?**  
A: Check OpenRouter's model list with "multimodal" tag. Common ones: GPT-4o, Claude 3.5, Qwen-VL series, Gemini 2.0

**Q: Can I use this with local models?**  
A: No, this filter is specifically for OpenRouter's API. For local models, use the ComfyUI filter.

**Q: Why are my images being compressed?**  
A: Auto-optimization ensures images meet API size limits and reduces costs. Disable with `auto_resize: false` if needed.

**Q: Does this work with qwen-image or qwen-image-edit?**  
A: Those specific models aren't on OpenRouter. Use Qwen-VL models instead for similar functionality.

**Q: How much does this cost?**  
A: Depends on model and usage. Check [OpenRouter pricing](https://openrouter.ai/models). Some models are free.

**Q: Can I generate multiple images at once?**  
A: Currently single image per request. Multiple images may be supported by specific models - check model docs.

**Q: What's the difference from ComfyUI filter?**  
A: ComfyUI = self-hosted, workflow-based. OpenRouter = cloud API, model-agnostic, no setup needed.

---

**Version:** 1.0.0  
**Last Updated:** 2025  
**Compatibility:** Open WebUI 0.5.0+