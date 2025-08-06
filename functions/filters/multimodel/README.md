# 🎭 Multi-Model Filter for Open WebUI

> ⚠️ **Work in Progress** - This filter is actively being developed. Results may vary and features are subject to change. Please report issues and provide feedback!

An advanced multi-model orchestration filter for Open WebUI that enables seamless conversations between multiple local AI models with enhanced persona integration and reasoning capabilities.

## 🚀 Features

### 🤖 Multi-Model Orchestration
- Run conversations with 2-4 local Ollama models simultaneously
- Three conversation modes: General, Collaboration, and Debate
- Enhanced model display with icons and persona awareness
- Real-time streaming responses with formatted output

### 🎭 Advanced Persona Integration
- **Agent Hotswap Compatibility** - Seamlessly integrates with Agent Hotswap plugin
- **Per-Model Personas** - Assign different personas to each model
- **Single Persona Mode** - All models adopt the same persona
- **Visual Integration** - Persona names and icons in conversation headers

### 🧠 Enhanced Reasoning Capabilities
- **Native Reasoning Support** - Detects models with reasoning capabilities
- **Thinking Process Display** - Shows model reasoning in collapsible sections
- **Configurable Effort Levels** - Low/Medium/High reasoning token allocation
- **Thinking Tag Processing** - Handles various thinking tag formats

### 🎯 Conversation Modes
- **General Discussion** - Natural multi-perspective conversations
- **Collaboration Mode** - Models work together as specialized experts
- **Debate Mode** - Models argue different positions with domain expertise

## 📋 Requirements

- **Open WebUI** v0.6.0 or higher
- **Ollama** with at least 2 local models installed
- **Agent Hotswap Filter** (recommended for persona features)
- **Python** dependencies included in Open WebUI

## 🛠️ Installation

1. **Install Ollama Models**
   ```bash
   # Recommended models for best results
   ollama pull llama3.2:latest
   ollama pull qwen2.5:latest
   ollama pull mistral:latest
   ollama pull deepseek-r1:latest  # For advanced reasoning
   ```

2. **Install Filter**
   - Copy the filter code to your Open WebUI filters directory
   - Or install via the Open WebUI interface

3. **Install Agent Hotswap** (Optional but recommended)
   - Install the Agent Hotswap filter for full persona functionality
   - Set Agent Hotswap priority to 0, Multi-Model to 10

4. **Configure Models**
   - Go to filter settings and select your installed models
   - Ensure at least 2 models are selected

## 🎮 Quick Start

### Basic Commands
```bash
!multi                           # Show help
!multi test                      # Test configuration
!multi <topic>                   # Start general discussion
!multi collab <topic>            # Collaboration mode
!multi debate <topic>            # Debate mode
```

### With Persona Integration
```bash
# Single persona for all models
!teacher !multi collab Explain quantum physics

# Per-model persona assignments
!persona1 teacher !persona2 student !multi collab Machine learning basics
!persona1 ethicist !persona2 technologist !multi debate AI regulation
```

## 🔧 Configuration

### Filter Settings (Valves)

#### 🤖 Model Configuration
- **model_1-4**: Select your local Ollama models
- **max_turns_per_model**: Responses per model (1-5, default: 2)
- **response_temperature**: Creativity level (0.0-1.0, default: 0.7)
- **max_tokens**: Response length (50-2000, default: 500)

#### 🧠 Reasoning Settings
- **enable_reasoning**: Enable reasoning capabilities
- **reasoning_effort**: Token allocation (low/medium/high)
- **show_thinking_process**: Display reasoning steps

#### 🎭 Integration Settings
- **enable_agent_hotswap_integration**: Enable persona features
- **show_persona_names**: Display persona names in headers
- **integration_debug**: Debug persona assignments

#### ⚙️ Performance Settings
- **conversation_pace**: Delay between responses (0.5-3.0s)
- **response_timeout**: Max response time (30-120s)
- **enable_debug**: Debug logging

## 📊 Supported Models

### 🧠 Reasoning-Capable Models
- **DeepSeek-R1** 🧠 - Advanced chain-of-thought reasoning
- **Qwen-QwQ** 🤔 - Question-answering with reasoning
- **Claude Models** 🎭 - Anthropic's reasoning capabilities

### 🦙 General Purpose Models
- **Llama 3.2** 🦙 - Excellent general conversations
- **Qwen 2.5** 🧠 - Strong reasoning and analysis
- **Mistral** 🌪️ - Fast and efficient responses
- **Phi 3** 🔬 - Good for technical discussions
- **CodeLlama** 💻 - Programming and technical topics

## 🎭 Persona Examples

### Available Personas (via Agent Hotswap)
- 🎓 **Teacher** - Educational expert and clear communicator
- 🎒 **Student** - Curious learner asking thoughtful questions
- 🔬 **Scientist** - Evidence-based researcher and analyst
- 🧭 **Ethicist** - Moral philosopher examining ethical implications
- 💻 **Technologist** - Innovation-focused technical expert
- 📈 **Economist** - Market dynamics and economic analysis
- 🏛️ **Policymaker** - Governance and policy implementation
- ✍️ **Writer** - Creative content and communication specialist
- 📊 **Analyst** - Data analysis and insight generation

### Example Conversations
```bash
# Educational scenario
!persona1 teacher !persona2 student !multi collab Explain photosynthesis

# Policy analysis
!persona1 economist !persona2 policymaker !multi debate Universal basic income

# Technical discussion
!persona1 coder !persona2 analyst !multi collab Database optimization strategies

# Ethical debate
!persona1 ethicist !persona2 technologist !multi debate AI development ethics
```

## 🐛 Known Issues & Limitations

> ⚠️ **This is a work in progress** - expect inconsistencies and bugs!

### Current Limitations
- **Local Models Only** - No API model support yet
- **Variable Response Quality** - Results depend heavily on model selection
- **Persona Assignment** - May not work consistently with all models
- **Memory Usage** - Multiple simultaneous models can be resource-intensive
- **Response Timing** - Some models may timeout or respond slowly

### Known Issues
- Some models may not respond in certain scenarios
- Persona assignments might not be applied consistently
- Reasoning display may not work with all model outputs
- Stream processing can occasionally fail

### Troubleshooting
1. **Models not responding**: Check `ollama list` and model availability
2. **Personas not working**: Ensure Agent Hotswap is installed and running
3. **Poor quality responses**: Try different model combinations
4. **Timeouts**: Increase response timeout in settings
5. **Debug issues**: Enable debug logging in filter settings

## 🔍 Testing & Debugging

### Test Your Setup
```bash
!multi test  # Basic configuration test
!persona1 teacher !persona2 student !multi test  # Persona integration test
```

### Debug Information
Enable debug logging in filter settings to see:
- Model selection and assignment
- Persona integration status
- Response processing details
- Error messages and timeouts

## 🤝 Contributing

This project is actively developed and welcomes contributions:

### How to Help
- **Test different model combinations** and report results
- **Report bugs** with specific model configurations
- **Suggest improvements** for persona integration
- **Share successful conversation examples**
- **Improve documentation** and examples

### Feedback Needed
- Which model combinations work best?
- How can persona integration be improved?
- What conversation modes would be useful?
- Performance optimization suggestions

## 📈 Performance Tips

### Model Selection
- **Start with 2 models** for better reliability
- **Mix different model types** (reasoning + general purpose)
- **Test combinations** with `!multi test` first
- **Monitor resource usage** with multiple models

### Conversation Quality
- **Use clear, specific topics** for better responses
- **Leverage persona strengths** (teacher for education, scientist for research)
- **Adjust temperature** for creativity vs consistency
- **Set appropriate token limits** for response length

## 🎯 Roadmap

### Planned Features
- API model integration (OpenAI, Anthropic)
- Enhanced conversation memory
- Custom persona creation
- Conversation export/import
- Performance optimizations
- Better error handling

### Under Development
- Multi-turn conversation improvements
- Enhanced reasoning integration
- Better persona assignment reliability
- Stream processing enhancements

## 📝 License

MIT License - See filter header for details

## 🆘 Support

- **Issues**: Report via GitHub or Open WebUI community
- **Documentation**: Check `!multi` help command
- **Community**: Share experiences and tips
- **Updates**: Watch for filter updates and improvements

---

> 🎭 **Remember**: This is experimental software. Results will vary based on your models, hardware, and configuration. Start with simple tests and gradually explore advanced features!

**Happy multi-model conversations!** 🚀
