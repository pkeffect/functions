# 🎭 Agent Hotswap

> **Transform your OpenWebUI experience with intelligent AI persona switching**

[![Version](https://img.shields.io/badge/version-0.1.2-blue.svg)](https://github.com/open-webui/functions)
[![OpenWebUI](https://img.shields.io/badge/OpenWebUI-Compatible-green.svg)](https://github.com/open-webui/open-webui)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

---

## 🌟 Overview

**Agent Hotswap** is a powerful OpenWebUI filter that enables seamless switching between 50+ specialized AI personas with a simple command system. Each persona comes with unique capabilities, expertise, and communication styles, all built on a **Master Controller** foundation that provides universal OpenWebUI-native features.

### ✨ Key Features

- 🎛️ **Master Controller System** - Transparent foundation providing OpenWebUI capabilities to all personas
- 🚀 **Instant Persona Switching** - Simple `!command` syntax for immediate role changes  
- 📦 **Auto-Download Collection** - Automatically fetches the complete 50+ persona collection on first run
- 🔄 **Auto-Updates** - Keeps persona collection current with weekly checks
- ⚡ **Performance Optimized** - Smart caching, pre-compiled patterns, and efficient loading
- 🎨 **Rich Rendering Support** - LaTeX math, Mermaid diagrams, HTML artifacts built-in
- 💾 **Automatic Backups** - Safe persona management with rollback capabilities
- 🔧 **Cross-Platform** - Works with both Docker and native OpenWebUI installations

---

## 🚀 Quick Start

### 1️⃣ Install the Filter
**Easy Install:** 
Use this link to install natively: https://openwebui.com/f/pkeffect/agent_hotswap

**Manual Install:**
1. Copy the complete filter code (main.py)
2. Add as a new filter in OpenWebUI → Admin Panel → Filters
3. Enable the filter

### 2️⃣ Automatic Setup
The plugin automatically:
- Downloads the complete 50+ persona collection
- Creates necessary configuration files
- Sets up proper paths for your installation type

### 3️⃣ Start Using Personas
```bash
!list      # See all available personas
!coder     # Become a programming expert
!writer    # Transform into a creative writer  
!analyst   # Switch to data analysis mode
!reset     # Return to default assistant
```

---

## 💡 Usage Guide

### Core Commands

| Command | Purpose |
|---------|---------|
| `!list` | Display all available personas in a formatted table |
| `!reset`, `!default`, `!normal` | Return to standard assistant mode |
| `!{persona_name}` | Switch to any specific persona |

### Available Personas

The plugin includes 50+ specialized personas covering:

**🔧 Development & Tech**
- `!coder` - 💻 Code Assistant
- `!debug` - 🐛 Debug Specialist  
- `!cybersecurityexpert` - 🛡️ Cyber Guardian
- `!devopsengineer` - ⚙️ System Smoother
- `!blockchaindev` - 🔗 Chain Architect

**📝 Creative & Content**
- `!writer` - ✍️ Creative Writer
- `!novelist` - 📚 Story Weaver
- `!poet` - ✒️ Verse Virtuoso
- `!filmmaker` - 🎥 Movie Director
- `!artist` - 🎨 Creative Visionary

**📊 Business & Analysis**
- `!analyst` - 📊 Data Analyst
- `!consultant` - 💼 Business Consultant
- `!economist` - 📈 Market Analyst Pro
- `!projectmanager` - 📋 Task Mastermind
- `!marketingguru` - 📢 Brand Booster

**🎓 Education & Research**
- `!teacher` - 🎓 Educator
- `!researcher` - 🔬 Researcher
- `!philosopher` - 🤔 Deep Thinker
- `!historian` - 📜 History Buff
- `!linguist` - 🗣️ Language Expert

**🔬 Science & Health**
- `!physicist` - ⚛️ Quantum Physicist
- `!biologist` - 🧬 Life Scientist
- `!chemist` - 🧪 Molecule Master
- `!doctor` - 🩺 Medical Informant
- `!nutritionist` - 🥗 Dietitian Pro

*And 25+ more! Use `!list` to see the complete collection.*

### Persona Features

- **Automatic Introduction** - Each persona introduces itself on activation
- **Persistent Context** - Persona remains active across messages until changed
- **Specialized Knowledge** - Tailored expertise and communication style
- **OpenWebUI Integration** - Full access to LaTeX, Mermaid, artifacts, and more

---

## 🎯 Core Concepts

### Master Controller System

The **Master Controller** is the invisible foundation that powers every persona:

- **Always Active** - Automatically loads with every persona
- **OpenWebUI Native** - Provides LaTeX math, Mermaid diagrams, HTML artifacts, file processing
- **Transparent** - Users never see or interact with it directly
- **Smart Persistence** - Only removed on reset/default commands

### Persona Architecture

Each persona includes:
- **Name** - Display name with emoji for visual identification
- **Prompt** - Comprehensive system prompt defining personality and expertise  
- **Description** - User-facing explanation of capabilities
- **Rules** - Structured guidelines for behavior and responses

---

## 🛠️ Configuration

### Basic Settings

| Setting | Default | Description |
|---------|---------|-------------|
| `keyword_prefix` | `!` | Command prefix for persona switching |
| `case_sensitive` | `false` | Whether commands are case-sensitive |
| `persistent_persona` | `true` | Keep persona active across messages |
| `show_persona_info` | `true` | Display status messages for switches |

### Advanced Settings

| Setting | Default | Description |
|---------|---------|-------------|
| `status_message_auto_close_delay_ms` | `5000` | Auto-close delay for status messages |
| `debug_performance` | `false` | Enable performance debugging logs |

---

## 🏗️ System Architecture

### Automatic Path Detection

The plugin automatically detects your installation type:

**Docker Installation:**
```
/app/backend/data/cache/functions/agent_hotswap/
├── personas.json                          # Main configuration
├── backups/                              # Automatic backups
│   ├── personas_backup_2024-01-15_14-30-22.json
│   └── personas_backup_2024-01-15_14-25-18.json
```

**Native Installation:**
```
~/.local/share/open-webui/cache/functions/agent_hotswap/
├── personas.json                          # Main configuration  
├── backups/                              # Automatic backups
```

### Auto-Update System

- **Weekly Checks** - Automatically checks for updates to persona collection
- **Smart Detection** - Updates configs with fewer than 20 personas
- **Background Downloads** - Non-blocking updates in separate thread
- **Automatic Backups** - Creates timestamped backups before updates

---

## 🔧 Troubleshooting

### Common Issues

**❌ Persona Not Loading**
```
Solution:
1. Use !list to verify persona exists
2. Check that filter is enabled
3. Restart OpenWebUI if needed
```

**❌ Commands Not Recognized**
```
Solution:
- Check keyword_prefix setting (default: "!")
- Ensure case_sensitive matches your usage
- Verify filter is enabled and active
```

**❌ Auto-Download Failed**
```
Solution:
- Check internet connectivity
- Plugin will create minimal config with basic personas
- Auto-retry will happen on next restart
```

### Recovery

**Reset Configuration:**
1. Disable and re-enable the filter in OpenWebUI
2. Plugin will auto-download fresh persona collection
3. Use `!list` to verify restoration

---

## 🚀 Advanced Features

### Custom Persona Creation

Add custom personas by editing the `personas.json` file:

```json
{
  "custom_key": {
    "name": "🎯 Your Custom Persona",
    "prompt": "You are a specialized assistant for...",
    "description": "Brief description of capabilities",
    "rules": [
      "1. First behavioral rule",
      "2. Second behavioral rule"
    ]
  }
}
```

### Performance Optimization

- **Smart Caching** - Only reloads when files change
- **Pattern Pre-compilation** - Regex patterns compiled once  
- **Change Detection** - File modification time tracking
- **Lazy Loading** - Personas loaded on-demand

---

## 🤝 Contributing

### Bug Reports

Include the following information:
- **OpenWebUI Version** - Your OpenWebUI version
- **Filter Configuration** - Relevant valve settings  
- **Error Messages** - Full error text and logs
- **Reproduction Steps** - How to recreate the issue

### Persona Contributions

Guidelines for new personas:
- **Clear Purpose** - Well-defined role and expertise
- **Comprehensive Prompt** - Detailed behavioral instructions
- **User-Friendly Description** - Clear capability explanation

---

## 📄 License

This project is licensed under the MIT License.

---

## 🙏 Acknowledgments

- **OpenWebUI Team** - For the amazing platform
- **Community Contributors** - For persona collections and feedback

---

<div align="center">

**🎭 Transform your AI interactions with Agent Hotswap!**

*Seamless persona switching • Rich OpenWebUI integration • Automatic setup*

</div>
