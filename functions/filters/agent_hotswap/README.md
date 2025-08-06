# 🎭 Agent Hotswap

> **⚠️ IMPORTANT WARNING**: Do not change the name of this plugin after installing. It is best to just reinstall as a new name if you need to rename it.

[![Version](https://img.shields.io/badge/version-3.0.1-blue.svg)](https://github.com/open-webui/functions)
[![Open WebUI](https://img.shields.io/badge/Open%20WebUI-v0.5.0+-green.svg)](https://github.com/open-webui/open-webui)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)


Universal AI persona switching for Open WebUI with enhanced multi-persona support, streaming capabilities, and robust backup systems.

## ✨ Features

### 🔥 **Core Functionality**
- **🎭 Instant Persona Switching** - Switch between AI personalities with simple commands
- **🌊 Multi-Persona Conversations** - Use multiple personas in a single message
- **🎛️ Per-Model Assignments** - Assign different personas to specific models
- **💾 Persistent Personas** - Maintains persona state across conversations
- **🖥️ Interactive Persona Browser** - Popup window interface for easy browsing

### 🚀 **Enhanced Features**
- **📊 Structured Logging** - Advanced logging with metadata and performance metrics
- **🔄 Persona Versioning** - Track changes and manage persona versions
- **⚡ Stream Switching** - Real-time persona transitions during streaming (experimental)
- **🛡️ Automatic Backups** - Intelligent backup system with change detection
- **🔌 Plugin Integration** - Seamless integration with other Open WebUI plugins

### 🎯 **Smart Capabilities**
- **🧠 Content-aware Transitions** - Intelligent switching based on natural completion points
- **🔍 Change Detection** - Automatic backups when personas are modified
- **⚙️ Compatibility Mode** - Works with both modern and legacy Open WebUI versions
- **🏃‍♂️ Async Operations** - High-performance asynchronous processing

## 📦 Installation

### **Requirements**
- Open WebUI v0.6.0+
- Python packages: `pydantic>=2.0.0`, `aiofiles>=23.0.0`, `aiohttp>=3.8.0`

### **Install via Open WebUI**
1. Go to **Admin Panel** → **Functions**
2. Click **➕ Add Function**
3. Copy and paste the Agent Hotswap code
4. Save and enable the function

### **Install via Community Hub**
1. Visit [Open WebUI Community Functions](https://openwebui.com/f/pkeffect/agent_hotswap)
2. Click **Get** → **Import to WebUI**

## 🎮 Usage

### **Basic Commands**

| Command | Description |
|---------|-------------|
| `!agent` | Show help and system information |
| `!agent list` | Open interactive persona browser |
| `!{persona}` | Switch to specific persona (e.g., `!coder`) |
| `!reset` | Return to default assistant |

### **Multi-Persona Examples**

**📝 Sequential Personas:**
```
!teacher explain quantum physics !writer create a poem about it !scientist analyze both
```

**🔬 Per-Model Assignments:**
```
!persona1 teacher !persona2 scientist !multi debate the theory of relativity
```

**💡 Creative Combinations:**
```
!coder write a Python function !reviewer critique the code !teacher explain it simply
```

### **Available Personas**

The plugin comes with default personas including:
- **💻 Coder** - Programming and development expert
- **✍️ Writer** - Creative writing specialist
- **👨‍🏫 Teacher** - Educational content creator
- **🔬 Scientist** - Research and analysis expert
- **📊 Analyst** - Data analysis and insights
- **🎨 Artist** - Creative and design focused

*Use `!agent list` to browse all available personas with descriptions.*

## ⚙️ Configuration

### **Core Settings**
- **`keyword_prefix`** (default: `!`) - Command prefix for persona switching
- **`persistent_persona`** (default: `true`) - Remember personas across conversations
- **`show_persona_info`** (default: `true`) - Display active persona information

### **Enhanced Features**
- **`enable_stream_switching`** (default: `false`) - Experimental real-time switching
- **`enable_persona_versioning`** (default: `true`) - Track persona versions
- **`enable_structured_logging`** (default: `true`) - Advanced logging system
- **`enable_automatic_backups`** (default: `true`) - Automatic backup creation

### **Backup & Safety**
- **`max_backup_files`** (default: `10`) - Maximum backup files to retain
- **`auto_download_personas`** (default: `true`) - Auto-download persona updates
- **`protect_custom_personas`** (default: `true`) - Protect custom personas from overwrites

## 🔧 Advanced Features

### **📊 Structured Logging**
Enhanced logging provides detailed insights:
```
[AGENT_HOTSWAP:INFO] Persona switched to coder [user_id=alice123, chat_id=chat_456, persona_name=💻 Code Assistant]
[AGENT_HOTSWAP:DEBUG] Multi-persona sequence detected with 3 transitions [user_id=alice123, chat_id=chat_456]
```

### **🔄 Persona Versioning**
- **Automatic versioning** when personas are updated
- **Change detection** using content hashing
- **Version history** with timestamps and descriptions
- **Rollback capability** to previous versions

### **🛡️ Intelligent Backups**
- **Content-based** - Creates backup when personas change
- **Time-based** - Periodic backups (every 7 days by default)
- **Event-based** - Pre-modification backups
- **Smart cleanup** - Automatic old backup removal

### **⚡ Stream Switching (Experimental)**
When enabled, allows real-time persona transitions during streaming responses:
```
!teacher start explaining !writer then create a story
```
The response smoothly transitions from teacher to writer mid-stream.

## 🔌 Plugin Integration

Agent Hotswap integrates seamlessly with other Open WebUI plugins:

### **Integration Context**
Provides rich context to other plugins:
```json
{
  "agent_hotswap_active": true,
  "active_persona": "coder",
  "active_persona_name": "💻 Code Assistant",
  "per_model_active": false,
  "enhanced_features": {
    "streaming": true,
    "versioning": true,
    "structured_logging": true
  }
}
```

### **Multi-Model Support**
Perfect companion for multi-model plugins:
- **Per-model personas** - `!persona1 teacher !persona2 scientist`
- **Cross-model context** - Shared persona information
- **Conversation continuity** - Maintained across model switches

## 📁 File Structure

```
/app/backend/data/cache/functions/agent_hotswap/
├── personas.json           # Main persona definitions
├── persona_versions.json   # Version history
├── index.html             # Persona browser UI
├── personas.json.backup.*  # Automatic backups
└── ...
```

## 🐛 Troubleshooting

### **Common Issues**

**Personas not switching:**
- Check that `keyword_prefix` matches your commands
- Verify persona exists with `!agent list`
- Enable debug logging: `enable_debug = true`

**Multi-persona not working:**
- Ensure `enable_stream_switching` is set correctly
- Try classic mode: `enable_stream_switching = false`
- Check for proper command syntax

**Backups not creating:**
- Verify `enable_automatic_backups = true`
- Check Open WebUI logs for error messages
- Ensure sufficient disk space

### **Debug Information**
Enable detailed logging:
```python
enable_debug = true
enable_structured_logging = true
integration_debug = true
```

## 🔄 Updates & Maintenance

### **Automatic Updates**
- **Persona downloads** - Automatically fetches new personas
- **UI updates** - Downloads latest browser interface
- **Version management** - Tracks all changes

### **Manual Maintenance**
- **Backup cleanup** - Automatic old backup removal
- **Cache management** - Global persona caching for performance
- **Version pruning** - Configurable version history limits

## 🤝 Contributing

### **Custom Personas**
Create custom personas by editing `personas.json`:
```json
{
  "my_expert": {
    "name": "🎯 My Expert",
    "prompt": "You are My Expert, specializing in...",
    "description": "Expert in specific domain",
    "capabilities": ["expertise1", "expertise2"]
  }
}
```

### **Plugin Development**
- **Integration hooks** - Use provided context for plugin communication
- **Event system** - Subscribe to persona switching events
- **API compatibility** - Standard Open WebUI function interface

## 📄 License

MIT License - See project repository for details.

## 🔗 Links

- **🏠 Project Repository:** [GitHub](https://github.com/pkeffect/functions/tree/main/functions/filters/agent_hotswap)
- **🌐 Open WebUI Community:** [Functions Hub](https://openwebui.com/functions)
- **💬 Support:** [Open WebUI Discord](https://discord.gg/5rJgQTnV4s)
- **📖 Documentation:** [Open WebUI Docs](https://docs.openwebui.com/)

## 🙏 Acknowledgments

- **Open WebUI Team** - For the amazing platform
- **Community Contributors** - For feedback and persona suggestions
- **Beta Testers** - For extensive testing and bug reports

---

**Made with ❤️ for the Open WebUI community**

*Enhance your AI conversations with dynamic persona switching!*
