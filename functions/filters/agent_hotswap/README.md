# 🎭 Agent Hotswap

> **Transform your OpenWebUI experience with intelligent AI persona switching**

[![Version](https://img.shields.io/badge/version-0.1.0-blue.svg)](https://github.com/pkeffect/agent_hotswap)
[![OpenWebUI](https://img.shields.io/badge/OpenWebUI-Compatible-green.svg)](https://github.com/open-webui/open-webui)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

---

## 🌟 Overview

**Agent Hotswap** is a powerful OpenWebUI filter that enables seamless switching between specialized AI personas with a simple command system. Each persona comes with unique capabilities, expertise, and communication styles, all built on a **Master Controller** foundation that provides universal OpenWebUI-native features.

### ✨ Key Features

- 🎛️ **Master Controller System** - Transparent foundation providing OpenWebUI capabilities to all personas
- 🚀 **Instant Persona Switching** - Simple `!command` syntax for immediate role changes  
- 📦 **Remote Persona Downloads** - Automatically fetch and apply persona collections from repositories
- 🔒 **Security-First Design** - Trusted domain whitelist and validation system
- ⚡ **Performance Optimized** - Smart caching, pre-compiled patterns, and efficient loading
- 🎨 **Rich Rendering Support** - LaTeX math, Mermaid diagrams, HTML artifacts built-in
- 💾 **Automatic Backups** - Safe persona management with rollback capabilities

---

## 🚨 Important: Getting Started

> **⚠️ FIRST STEP:** After installation, use the `!download_personas` command to get the complete persona collection. The system starts with basic defaults but the full experience requires downloading the official persona repository.

---

## 📋 Table of Contents

- [🚀 Quick Start](#-quick-start)
- [🏗️ Installation](#️-installation)
- [🎯 Core Concepts](#-core-concepts)
  - [Master Controller System](#master-controller-system)
  - [Persona Architecture](#persona-architecture)
  - [Command System](#command-system)
- [📥 Persona Management](#-persona-management)
  - [Downloading Personas](#downloading-personas)
  - [Security & Trust](#security--trust)
  - [Backup System](#backup-system)
- [🛠️ Configuration](#️-configuration)
  - [Basic Settings](#basic-settings)
  - [Advanced Options](#advanced-options)
  - [Performance Tuning](#performance-tuning)
- [💡 Usage Guide](#-usage-guide)
  - [Basic Commands](#basic-commands)
  - [Persona Switching](#persona-switching)
  - [System Commands](#system-commands)
- [🏗️ System Architecture](#️-system-architecture)
  - [File Structure](#file-structure)
  - [Caching System](#caching-system)
  - [Pattern Matching](#pattern-matching)
- [🔧 Troubleshooting](#-troubleshooting)
- [🚀 Advanced Features](#-advanced-features)
- [🤝 Contributing](#-contributing)

---

## 🚀 Quick Start

### 1️⃣ Install the Filter
1. Copy the complete filter code
2. Add as a new filter in OpenWebUI
3. Enable the filter and configure basic settings

### 2️⃣ Download Persona Collection
```
!download_personas
```
This downloads the official persona repository with 50+ specialized AI assistants.

### 3️⃣ Explore Available Personas
```
!list
```

### 4️⃣ Switch to a Persona
```
!coder    # Become a programming expert
!writer   # Transform into a creative writer  
!analyst  # Switch to data analysis mode
```

### 5️⃣ Reset When Needed
```
!reset    # Return to default assistant
```

---

## 🏗️ Installation

### Prerequisites
- OpenWebUI instance with filter support
- Administrator access to add filters
- Internet connectivity for persona downloads

### Step-by-Step Installation

1. **Access Filter Management**
   - Navigate to OpenWebUI Settings
   - Go to Admin Panel → Filters
   - Click "Add Filter"

2. **Install Agent Hotswap**
   - Copy the complete filter code
   - Paste into the filter editor
   - Set filter name: "Agent Hotswap"
   - Save and enable the filter

3. **Initial Configuration**
   - Review default settings in the Valves section
   - Adjust `keyword_prefix` if desired (default: `!`)
   - Configure `trusted_domains` for security
   - Enable `create_default_config` for automatic setup

4. **First-Time Setup**
   ```
   !download_personas
   ```
   This command will:
   - Download the official persona collection
   - Create backup of existing configuration
   - Apply new personas with merge strategy
   - Clear caches for immediate availability

---

## 🎯 Core Concepts

### Master Controller System

The **Master Controller** is the invisible foundation that powers every persona interaction:

#### 🎛️ What It Provides
- **LaTeX Mathematics**: `$$E=mc^2$$` rendering support
- **Mermaid Diagrams**: Automatic flowchart and diagram generation
- **HTML Artifacts**: Interactive content creation capabilities
- **File Processing**: CSV, PDF, image upload handling
- **Status Messages**: Real-time feedback and progress indicators

#### 🔄 How It Works
- **Always Active**: Automatically loads with every persona
- **Transparent**: Users never see or interact with it directly
- **Foundation Layer**: Provides OpenWebUI-native capabilities to all personas
- **Smart Persistence**: Only removed on reset/default commands

### Persona Architecture

Each persona consists of structured components:

```json
{
  "persona_key": {
    "name": "🎭 Display Name",
    "prompt": "Detailed system prompt defining behavior",
    "description": "User-facing description of capabilities",
    "rules": ["Rule 1", "Rule 2", "..."]
  }
}
```

#### 🧩 Persona Components
- **Name**: Display name with emoji for visual identification
- **Prompt**: Comprehensive system prompt defining personality and expertise
- **Description**: User-facing explanation of capabilities
- **Rules**: Structured guidelines for behavior and responses

### Command System

Agent Hotswap uses a prefix-based command system:

| Command Type | Syntax | Purpose |
|-------------|--------|---------|
| **Persona Switch** | `!persona_key` | Activate specific persona |
| **List Personas** | `!list` | Show available personas in table format |
| **Reset System** | `!reset`, `!default`, `!normal` | Return to standard assistant |
| **Download Personas** | `!download_personas [url] [--replace]` | Fetch remote persona collections |

---

**Available Personas**
| Command | Name | Command | Name |
| ---|--- | ---|--- |
| `!airesearcher` | 🤖 AI Pioneer | `!analyst` | 📊 Data Analyst |
| `!archaeologist` | 🏺 Relic Hunter | `!architect` | 🏗️ Master Builder |
| `!artist` | 🎨 Creative Visionary | `!astronomer` | 🔭 Star Gazer |
| `!biologist` | 🧬 Life Scientist | `!blockchaindev` | 🔗 Chain Architect |
| `!careercounselor` | 🧑‍💼 Career Navigator | `!chef` | 🧑‍🍳 Culinary Genius |
| `!chemist` | 🧪 Molecule Master | `!coder` | 💻 Code Assistant |
| `!consultant` | 💼 Business Consultant | `!cybersecurityexpert` | 🛡️ Cyber Guardian |
| `!debug` | 🐛 Debug Specialist | `!devopsengineer` | ⚙️ System Smoother |
| `!doctor` | 🩺 Medical Informant | `!economist` | 📈 Market Analyst Pro |
| `!environmentalist` | 🌳 Nature's Advocate | `!ethicist` | 🧭 Moral Compass |
| `!fashiondesigner` | 👗 Style Icon | `!filmmaker` | 🎥 Movie Director |
| `!financialadvisor` | 💰 Wealth Sage | `!fitnesstrainer` | 💪 Health Coach |
| `!gamedesigner` | 🎮 Game Dev Guru | `!gardener` | 🌻 Green Thumb |
| `!geologist` | 🌍 Earth Explorer | `!historian` | 📜 History Buff |
| `!hrspecialist` | 🧑‍🤝‍🧑 People Partner Pro | `!interiordesigner` | 🛋️ Space Shaper |
| `!journalist` | 📰 News Hound | `!lawyer` | ⚖️ Legal Eagle |
| `!lifecoach` | 🌟 Goal Getter Guide | `!linguist` | 🗣️ Language Expert |
| `!marketingguru` | 📢 Brand Booster | `!mathematician` | ➕ Math Whiz |
| `!mechanic` | 🔧 Auto Ace | `!musician` | 🎶 Melody Maker |
| `!negotiator` | 🤝 Deal Maker Pro | `!novelist` | 📚 Story Weaver |
| `!nutritionist` | 🥗 Dietitian Pro | `!philosopher` | 🤔 Deep Thinker |
| `!photographer` | 📸 Image Capturer | `!physicist` | ⚛️ Quantum Physicist |
| `!poet` | ✒️ Verse Virtuoso | `!projectmanager` | 📋 Task Mastermind |
| `!psychologist` | 🧠 Mind Mender | `!publicspeaker` | 🎤 Oratory Coach |
| `!researcher` | 🔬 Researcher | `!roboticsengineer` | 🦾 Robot Builder |
| `!salesexpert` | 🤝 Deal Closer Pro | `!scriptwriter` | 🎬 Screen Scribe |
| `!sociologist` | 👥 Society Scholar | `!sommelier` | 🍷 Wine Connoisseur |
| `!teacher` | 🎓 Educator | `!travelguide` | ✈️ World Wanderer |
| `!writer` | ✍️ Creative Writer |   |   |

To revert to the default assistant, use one of these commands: `!reset`, `!default`, `!normal`

---

## 📥 Persona Management

### Downloading Personas

The download system enables automatic persona collection management:

#### 🌐 Basic Download
```bash
!download_personas
```
Downloads from the default repository with merge strategy.

#### 🔗 Custom Repository
```bash
!download_personas https://your-domain.com/personas.json
```
Download from a specific URL (must be in trusted domains).

#### 🔄 Replace Mode  
```bash
!download_personas --replace
```
Completely replaces local personas with remote collection.

#### 📊 Download Process
1. **URL Validation** - Verifies domain is trusted
2. **Content Retrieval** - Downloads JSON configuration  
3. **Structure Validation** - Ensures proper persona format
4. **Backup Creation** - Saves current configuration
5. **Merge/Replace** - Applies new personas based on strategy
6. **Cache Invalidation** - Refreshes system for immediate use

### Security & Trust

#### 🔒 Domain Whitelist
```
trusted_domains: "github.com,raw.githubusercontent.com,gitlab.com"
```
Only domains in this list can serve persona downloads.

#### 🛡️ Validation System
- **JSON Structure** - Validates persona configuration format
- **Required Fields** - Ensures name, prompt, description are present
- **Content Limits** - 1MB maximum download size
- **Timeout Protection** - 30-second download timeout

#### 🔍 Security Features
- HTTPS-only downloads
- Content-type validation
- Malicious URL detection
- Safe fallback on errors

### Backup System

#### 💾 Automatic Backups
- Created before every download/apply operation
- Timestamped for easy identification
- Stored in `backups/` subdirectory
- Automatic cleanup (keeps 5 most recent)

#### 📁 Backup Location
```
/app/backend/data/cache/functions/agent_hotswap/backups/
├── personas_backup_2024-01-15_14-30-22.json
├── personas_backup_2024-01-15_14-25-18.json
└── ...
```

---

## 🛠️ Configuration

### Basic Settings

#### 🎛️ Core Configuration
| Setting | Default | Description |
|---------|---------|-------------|
| `keyword_prefix` | `!` | Command prefix for persona switching |
| `case_sensitive` | `false` | Whether commands are case-sensitive |
| `persistent_persona` | `true` | Keep persona active across messages |
| `show_persona_info` | `true` | Display status messages for switches |

#### 🗂️ File Management
| Setting | Default | Description |
|---------|---------|-------------|
| `cache_directory_name` | `agent_hotswap` | Directory name for config storage |
| `config_filename` | `personas.json` | Filename for persona configuration |
| `create_default_config` | `true` | Auto-create default personas |

### Advanced Options

#### 📡 Download System
```python
default_personas_repo = "https://raw.githubusercontent.com/pkeffect/agent_hotswap/refs/heads/main/personas/personas.json"
trusted_domains = "github.com,raw.githubusercontent.com,gitlab.com"
download_timeout = 30
backup_count = 5
```

#### ⚡ Performance Settings
```python
debug_performance = false
status_message_auto_close_delay_ms = 5000
```

### Performance Tuning

#### 🚀 Optimization Features
- **Smart Caching** - Only reloads when files change
- **Pattern Pre-compilation** - Regex patterns compiled once
- **Lazy Loading** - Personas loaded on-demand
- **Change Detection** - File modification time tracking

---

## 💡 Usage Guide

### Basic Commands

#### 📋 List Available Personas
```
!list
```
Displays a formatted table showing:
- Command syntax for each persona
- Display names with emojis
- Reset command options

#### 🔄 Reset to Default
```
!reset      # Primary reset command
!default    # Alternative reset
!normal     # Another reset option
```

### Persona Switching

#### 🎭 Activate Persona
```
!coder      # Switch to Code Assistant
!writer     # Switch to Creative Writer
!analyst    # Switch to Data Analyst
!teacher    # Switch to Educator
!researcher # Switch to Researcher
```

#### 💬 Persona Behaviors
- **Automatic Introduction** - Each persona introduces itself on activation
- **Persistent Context** - Persona remains active until changed
- **Specialized Responses** - Tailored expertise and communication style
- **Master Controller Foundation** - OpenWebUI capabilities always available

### System Commands

#### 📥 Download Management
```bash
# Download default collection
!download_personas

# Download from specific URL  
!download_personas https://example.com/personas.json

# Replace all personas
!download_personas --replace

# Custom URL with replace
!download_personas https://example.com/custom.json --replace
```

---

## 🏗️ System Architecture

### File Structure

```
/app/backend/data/cache/functions/agent_hotswap/
├── personas.json                          # Main configuration
├── backups/                              # Automatic backups
│   ├── personas_backup_2024-01-15_14-30-22.json
│   └── personas_backup_2024-01-15_14-25-18.json
└── logs/                                 # Debug logs (if enabled)
```

#### 📄 personas.json Structure
```json
{
  "_master_controller": {
    "name": "🎛️ Master Controller",
    "hidden": true,
    "always_active": true,
    "priority": 0,
    "prompt": "=== OPENWEBUI MASTER CONTROLLER ===\n...",
    "description": "Universal OpenWebUI environment context"
  },
  "coder": {
    "name": "💻 Code Assistant", 
    "prompt": "You are the 💻 Code Assistant...",
    "description": "Expert programming assistance",
    "rules": [...]
  }
}
```

### Caching System

#### 🗄️ Smart Cache Features
- **File Modification Detection** - Only reloads when JSON changes
- **Validation Caching** - Remembers successful validations
- **Pattern Compilation Cache** - Stores compiled regex patterns
- **Invalidation Triggers** - Manual cache clearing on downloads

#### ⚡ Performance Benefits
- **Reduced I/O** - Minimizes file system access
- **Faster Switching** - Pre-compiled patterns for instant detection
- **Memory Efficiency** - Lazy loading of persona data
- **Change Tracking** - Timestamp-based modification detection

### Pattern Matching

#### 🔍 Regex Compilation
```python
# Compiled patterns for efficiency
prefix_pattern = re.compile(rf"{escaped_prefix}coder\b", flags)
reset_pattern = re.compile(rf"{escaped_prefix}(?:reset|default|normal)\b", flags)
list_pattern = re.compile(rf"{escaped_prefix}list\b", flags)
```

#### 🎯 Detection Strategy
1. **Command Preprocessing** - Normalize case if needed
2. **Pattern Matching** - Use pre-compiled regex for speed
3. **Priority Ordering** - System commands checked first
4. **Fallback Handling** - Graceful degradation on errors

---

## 🔧 Troubleshooting

### Common Issues

#### ❌ Download Failures
**Problem**: `!download_personas` fails with domain error
```
Solution: Check trusted_domains configuration
- Ensure domain is in whitelist: "github.com,raw.githubusercontent.com"
- Verify HTTPS protocol is used
- Check network connectivity
```

#### ❌ Persona Not Loading
**Problem**: Persona doesn't activate after switching
```
Solution: Check configuration and cache
1. Use !list to verify persona exists
2. Check personas.json syntax
3. Clear cache: restart OpenWebUI or modify config file
4. Review logs for validation errors
```

#### ❌ Commands Not Recognized
**Problem**: `!coder` or other commands don't work
```
Solution: Verify configuration
- Check keyword_prefix setting (default: "!")
- Ensure case_sensitive matches your usage
- Verify filter is enabled and active
- Test with !list first
```

### Debug Mode

#### 🐛 Enable Debugging
```python
debug_performance = true
```
Provides detailed timing and operation logs.

#### 📊 Performance Monitoring  
- Pattern compilation timing
- File loading performance
- Cache hit/miss ratios
- Command detection speed

### Recovery Procedures

#### 🔄 Reset Configuration
1. **Delete config file**: Remove `personas.json`
2. **Restart filter**: Toggle off/on in OpenWebUI
3. **Reload defaults**: System creates fresh configuration
4. **Re-download**: Use `!download_personas` to restore collection

#### 💾 Restore from Backup
1. **Locate backup**: Check `backups/` directory  
2. **Copy desired backup**: Rename to `personas.json`
3. **Clear cache**: Restart OpenWebUI or modify timestamp
4. **Verify**: Use `!list` to confirm restoration

---

## 🚀 Advanced Features

### Custom Persona Creation

#### 🎨 Manual Persona Addition
Edit `personas.json` directly to add custom personas:

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

### Repository Management

#### 🌐 Creating Persona Repositories
Structure for shareable persona collections:

```json
{
  "meta": {
    "version": "1.0.0",
    "author": "Your Name",
    "description": "Collection description"
  },
  "personas": {
    "specialist": {
      "name": "🎯 Specialist",
      "prompt": "...",
      "description": "..."
    }
  }
}
```

### Integration Patterns

#### 🔗 Workflow Integration
- **Development Teams**: Code review personas for different languages
- **Content Creation**: Writing personas for different styles/audiences  
- **Education**: Teaching personas for different subjects/levels
- **Analysis**: Specialized personas for different data types

---

## 🤝 Contributing

### Development Setup

#### 🛠️ Local Development
1. **Fork Repository** - Create your own copy
2. **Clone Locally** - Set up development environment
3. **Test Changes** - Use OpenWebUI test instance
4. **Submit PR** - Follow contribution guidelines

### Persona Contributions

#### 📝 Persona Guidelines
- **Clear Purpose** - Well-defined role and expertise
- **Comprehensive Prompt** - Detailed behavioral instructions
- **User-Friendly Description** - Clear capability explanation
- **Appropriate Rules** - Structured behavioral guidelines

#### 🧪 Testing Requirements
- **Validation** - Passes JSON schema validation
- **Functionality** - Commands work as expected  
- **Performance** - No significant slowdown
- **Compatibility** - Works with Master Controller system

### Bug Reports

#### 🐛 Reporting Issues
Include the following information:
- **OpenWebUI Version** - Your OpenWebUI version
- **Filter Configuration** - Relevant valve settings
- **Error Messages** - Full error text and logs
- **Reproduction Steps** - How to recreate the issue
- **Expected Behavior** - What should happen instead

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **OpenWebUI Team** - For the amazing platform
- **Community Contributors** - For persona collections and feedback
- **Beta Testers** - For early feedback and bug reports

---

## 📞 Support

- **GitHub Issues** - [Report bugs and request features](https://github.com/open-webui/functions/issues)
- **Discussions** - [Community support and questions](https://github.com/open-webui/functions/discussions)
- **Documentation** - This README and inline code documentation

---

<div align="center">

**🎭 Transform your AI interactions with Agent Hotswap!**

*Seamless persona switching • Rich OpenWebUI integration • Secure & performant*

</div>
