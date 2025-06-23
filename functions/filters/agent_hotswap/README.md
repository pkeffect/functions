# 🎭 Agent Hotswap

> **Revolutionary AI persona switching with dynamic multi-persona capabilities**

[![Version](https://img.shields.io/badge/version-0.2.0-blue.svg)](https://github.com/open-webui/functions)
[![OpenWebUI](https://img.shields.io/badge/OpenWebUI-Compatible-green.svg)](https://github.com/open-webui/open-webui)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

---

## 🌟 Overview

**Agent Hotswap** is the most advanced OpenWebUI filter for AI persona management, enabling seamless switching between 50+ specialized AI personas with breakthrough **dynamic multi-persona capabilities**. Execute complex workflows involving multiple experts in a single conversation, with automatic persona discovery and just-in-time loading.

### ✨ Revolutionary Features

- 🎛️ **Master Controller System** - Universal OpenWebUI capabilities foundation for all personas
- 🔄 **Dynamic Multi-Persona Sequences** - Multiple persona switches within a single prompt
- 🔍 **Universal Persona Detection** - Automatically works with any current or future personas
- ⚡ **Just-In-Time Loading** - Only loads personas actually requested for optimal performance
- 🚀 **Instant Persona Switching** - Simple `!command` syntax for immediate role changes  
- 📦 **Auto-Download Collection** - Automatically fetches the complete 50+ persona collection
- 🔄 **Auto-Updates** - Keeps persona collection current with weekly checks
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
2. Add as a new Function in OpenWebUI → Admin Panel → Functions
3. Enable the Function (also be sure to enable to Agent Swapper Icon in chat)

### 2️⃣ Automatic Setup
The plugin automatically:
- Downloads the complete 50+ persona collection
- Creates necessary configuration files
- Sets up proper paths for your installation type

### 3️⃣ Start Using Personas
```bash
# Single persona switching
!list      # See all available personas
!coder     # Become a programming expert
!writer    # Transform into a creative writer  
!analyst   # Switch to data analysis mode

# Revolutionary multi-persona sequences
!writer create a story about AI !physicist explain the science !teacher create study questions !artist design cover art

# Reset to default
!reset     # Return to standard assistant
```

---

## 🔥 **NEW: Dynamic Multi-Persona System**

### **Multi-Persona Sequences**
Execute complex workflows with multiple experts in a single prompt:

```bash
# Creative collaboration
!writer start a sci-fi story !physicist verify the science !historian add historical context !artist describe the visuals !writer conclude the story

# Educational deep-dive
!teacher introduce quantum mechanics !physicist explain the theory !engineer show applications !philosopher discuss implications

# Business analysis
!analyst present market data !economist add economic context !consultant recommend strategy !projectmanager create implementation timeline
```

### **How Multi-Persona Hotswapping Works**

#### **1. Universal Discovery Phase**
```
User Input: "!writer create story !teacher explain techniques !physicist add science"

↓ Universal Pattern Detection ↓

Discovered Commands: ['writer', 'teacher', 'physicist']
```

#### **2. Just-In-Time Loading**
```
Available Personas: 50+ in collection
↓ Smart Loading ↓
Loaded: Only 3 requested personas + Master Controller
Memory Usage: Minimal (only what's needed)
```

#### **3. Dynamic System Construction**
```
System Message Built:
├── Master Controller (OpenWebUI capabilities)
├── Writer Persona Definition  
├── Teacher Persona Definition
├── Physicist Persona Definition
└── Multi-Persona Execution Framework
```

#### **4. Sequence Parsing & Instruction Building**
```
Original: "!writer create story !teacher explain techniques !physicist add science"

↓ Parsed Into Structured Sequence ↓

Step 1 - Creative Writer: create story
Step 2 - Educator: explain techniques  
Step 3 - Quantum Physicist: add science
```

#### **5. Intelligent Execution**
The LLM receives comprehensive instructions and executes each persona switch seamlessly, maintaining context and flow throughout the entire sequence.

### **Universal Compatibility**
Works with **ANY** persona combination:
- **Current 50+ personas**: `!coder !analyst !economist`
- **Future personas**: Automatically detects new additions
- **Mixed combinations**: `!existing_persona !future_persona !another_new_one`
- **Unlimited sequences**: `!a !b !c !d !e !f !g !h !i !j...`

---

## 💡 Usage Guide

### Core Commands

| Command | Purpose |
|---------|---------|
| `!list` | Display all available personas in a formatted table |
| `!reset`, `!default`, `!normal` | Return to standard assistant mode |
| `!{persona_name}` | Switch to any specific persona |
| `!{persona1} task1 !{persona2} task2` | **NEW:** Multi-persona sequences |

### Single Persona Switching

```bash
!coder     # Switch to programming expert
!writer    # Become creative writer
!analyst   # Transform into data analyst
```

### Multi-Persona Workflows

```bash
# Content creation pipeline
!writer draft blog post !researcher fact-check claims !editor polish prose !marketer add compelling headlines

# Technical analysis
!analyst examine data !statistician run tests !consultant interpret results !presenter create executive summary

# Creative projects  
!novelist create plot !historian verify period details !scientist explain technology !artist design concepts
```

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
- **Persistent Context** - Single personas remain active across messages until changed
- **Specialized Knowledge** - Tailored expertise and communication style
- **OpenWebUI Integration** - Full access to LaTeX, Mermaid, artifacts, and more
- ****NEW:** Multi-Persona Transitions** - Smooth handoffs between experts in sequences

---

## 🎯 Core Concepts

### Master Controller System

The **Master Controller** is the invisible foundation that powers every persona:

- **Always Active** - Automatically loads with every persona
- **OpenWebUI Native** - Provides LaTeX math, Mermaid diagrams, HTML artifacts, file processing
- **Transparent** - Users never see or interact with it directly
- **Smart Persistence** - Only removed on reset/default commands

### Dynamic Multi-Persona Architecture

#### **Universal Detection Engine**
- **Pattern Recognition**: Automatically detects any `!{word}` pattern
- **Future-Proof**: Works with personas that don't exist yet
- **Smart Filtering**: Distinguishes between personas and special commands
- **Error Handling**: Gracefully handles unknown commands

#### **Just-In-Time Loading System**
```
Available: 50+ personas in collection
Requested: !writer !physicist !teacher
↓ Smart Loading ↓
Loaded: 3 personas + Master Controller
Memory: ~75% reduction vs loading all personas
Performance: Optimal regardless of collection size
```

#### **Dynamic System Message Construction**
Each multi-persona session gets a custom system message containing:
1. **Master Controller** - OpenWebUI capabilities foundation
2. **Requested Personas** - Only the personas actually needed
3. **Execution Framework** - Instructions for seamless switching
4. **Available Commands** - List of active personas for the session

### Persona Transition Control

Control how persona switches are displayed:

#### **Visible Transitions** (Default)
```
🎭 **Creative Writer**
Once upon a time, in a world where artificial intelligence...

🎭 **Quantum Physicist** 
The quantum mechanics underlying this scenario involve...

🎭 **Educator**
Let me explain these concepts in simpler terms...
```

#### **Silent Transitions** 
```
Once upon a time, in a world where artificial intelligence...

The quantum mechanics underlying this scenario involve...

Let me explain these concepts in simpler terms...
```

---

## 🛠️ Configuration

### Basic Settings

| Setting | Default | Description |
|---------|---------|-------------|
| `keyword_prefix` | `!` | Command prefix for persona switching |
| `case_sensitive` | `false` | Whether commands are case-sensitive |
| `persistent_persona` | `true` | Keep single personas active across messages |
| `show_persona_info` | `true` | Display status messages for switches |

### Advanced Settings

| Setting | Default | Description |
|---------|---------|-------------|
| `multi_persona_transitions` | `true` | **NEW:** Show transition announcements in multi-persona responses |
| `status_message_auto_close_delay_ms` | `5000` | Auto-close delay for status messages |
| `debug_performance` | `false` | Enable performance debugging logs |

### **NEW: Multi-Persona Transition Control**

The `multi_persona_transitions` valve controls how persona switches are displayed in multi-persona sequences:

**Enabled (Default):**
- Shows `🎭 **Persona Name**` announcements
- Clear visual indication of expert transitions
- Helpful for understanding which expert is responding

**Disabled:**
- Silent, seamless transitions
- Clean output without transition markers
- Personas switch invisibly behind the scenes

**When to disable transitions:**
- Creative writing where transitions would break immersion
- Professional reports requiring clean formatting
- When persona switches should be transparent to end users

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

### **NEW: Universal Detection Architecture**

#### **Pattern Compilation System**
```python
# Universal pattern matches any valid persona command
Pattern: !{word} where word = [a-zA-Z][a-zA-Z0-9_]*

Examples Matched:
✅ !coder, !writer, !analyst (existing)
✅ !quantum_engineer, !bioethicist (future)  
✅ !custom_persona_123 (user-defined)
❌ !123invalid (invalid format)
```

#### **Dynamic Loading Pipeline**
```
1. Parse Input → Discover Commands → [!writer, !physicist, !teacher]
2. Load Collection → Validate Existence → [✅writer, ✅physicist, ✅teacher]
3. Build System → Include Definitions → Master + 3 Personas
4. Create Instructions → Structure Sequence → Step-by-step execution
5. Execute → LLM follows sequence → Multi-expert response
```

### Auto-Update System

- **Weekly Checks** - Automatically checks for updates to persona collection
- **Smart Detection** - Updates configs with fewer than 20 personas
- **Background Downloads** - Non-blocking updates in separate thread
- **Automatic Backups** - Creates timestamped backups before updates

---

## 🚀 Advanced Use Cases

### Creative Collaboration
```bash
!writer start mystery novel !detective add investigative realism !psychologist develop character depth !editor polish prose !marketer create book blurb
```

### Technical Documentation
```bash
!engineer explain system architecture !coder provide implementation examples !teacher create tutorials !technical_writer polish documentation
```

### Business Strategy
```bash
!analyst present market data !economist add macro trends !consultant recommend strategies !projectmanager create timelines !presenter format executive summary
```

### Educational Content
```bash
!teacher introduce topic !researcher provide latest findings !philosopher explore implications !artist create visual aids !writer craft engaging narrative
```

### Problem Solving
```bash
!analyst define problem !researcher gather evidence !consultant brainstorm solutions !engineer evaluate feasibility !projectmanager plan implementation
```

## 🔧 Troubleshooting

### Common Issues

**❌ Multi-Persona Not Working**
```
Solution:
1. Ensure multiple !commands in single message
2. Check that filter is enabled
3. Verify personas exist with !list
4. Check transition settings in configuration
```

**❌ Unknown Persona Commands**
```
Behavior: System gracefully ignores unknown commands
Status: Shows "⚠️ Unknown: !invalid_persona"
Solution: Use !list to see available personas
```

**❌ Performance Issues with Large Sequences**
```
Optimization: System only loads requested personas
Memory: Scales efficiently regardless of sequence length
Tip: No performance penalty for complex workflows
```

**❌ Transitions Too Verbose/Invisible**
```
Solution: Adjust multi_persona_transitions valve
- Enable: Shows clear 🎭 **Persona** markers
- Disable: Silent, seamless transitions
```

### Recovery

**Reset Configuration:**
1. Disable and re-enable the filter in OpenWebUI
2. Plugin will auto-download fresh persona collection
3. Use `!list` to verify restoration

**Clear Persona State:**
```bash
!reset    # Clears all active personas
!default  # Alternative reset command
!normal   # Another reset option
```

---

## 🚀 Advanced Features

### Custom Persona Creation

Add custom personas by editing the `personas.json` file:

```json
{
  "custom_expert": {
    "name": "🎯 Your Custom Expert",
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

- **Universal Detection** - Works with unlimited personas
- **Smart Caching** - Only reloads when files change
- **Just-In-Time Loading** - Only loads requested personas
- **Pattern Pre-compilation** - Regex patterns compiled once  
- **Change Detection** - File modification time tracking

### **NEW: Multi-Persona Performance**

```
Traditional Approach: Load all 50+ personas
Memory Usage: High
Loading Time: Slow

Dynamic Approach: Load only requested personas
Memory Usage: Minimal
Loading Time: Instant
Scalability: Infinite
```

---

## 🤝 Contributing

### Bug Reports

Include the following information:
- **OpenWebUI Version** - Your OpenWebUI version
- **Filter Configuration** - Relevant valve settings  
- **Error Messages** - Full error text and logs
- **Reproduction Steps** - How to recreate the issue
- **Multi-Persona Details** - If issue involves persona sequences

### Persona Contributions

Guidelines for new personas:
- **Clear Purpose** - Well-defined role and expertise
- **Comprehensive Prompt** - Detailed behavioral instructions
- **User-Friendly Description** - Clear capability explanation
- **Multi-Persona Compatibility** - Works well with other experts

### Feature Requests

When requesting features:
- **Use Case** - Explain the specific workflow need
- **Multi-Persona Impact** - How it affects persona sequences
- **Performance Considerations** - Scalability requirements

---

## 📊 Performance Metrics

### **Traditional vs Dynamic Architecture**

| Metric | Traditional | **Dynamic Multi-Persona** |
|--------|-------------|---------------------------|
| **Memory Usage** | All personas loaded | Only requested personas |
| **Loading Time** | Fixed overhead | Scales with usage |
| **Flexibility** | Single persona | Unlimited combinations |
| **Future-Proofing** | Manual updates | Automatic discovery |
| **Performance** | Degrades with size | Constant performance |

### **Scalability Examples**

```bash
# 2 personas: ~95% memory savings
!writer !teacher

# 5 personas: ~90% memory savings  
!coder !analyst !economist !historian !artist

# 10 personas: ~80% memory savings
!writer !coder !teacher !physicist !artist !economist !historian !philosopher !consultant !researcher

# Performance remains optimal regardless of sequence complexity
```

---

## 📄 License

This project is licensed under the MIT License.

---

## 🙏 Acknowledgments

- **OpenWebUI Team** - For the amazing platform and architecture
- **Community Contributors** - For persona collections and feedback
- **Early Adopters** - For testing multi-persona workflows

---

## 🔮 Future Roadmap

- **Persona Chaining** - Automatic persona suggestions based on context
- **Workflow Templates** - Pre-built multi-persona sequences for common tasks
- **Performance Analytics** - Detailed metrics on persona usage patterns
- **Custom Transition Styles** - User-defined transition formatting
- **Persona Marketplace** - Community-contributed expert collections

---

<div align="center">

**🎭 Transform your AI interactions with Agent Hotswap!**

*Revolutionary multi-persona sequences • Universal compatibility • Infinite scalability*

### **Experience the future of AI interaction today**

</div>