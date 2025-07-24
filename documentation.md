# ChatOS - Voice-Driven Operating System Assistant

## Overview

ChatOS is an intelligent voice-driven assistant that combines local AI models with cloud fallbacks to provide a seamless interaction experience. The system uses Model Context Protocol (MCP) to enable the AI to interact with your operating system, launch applications, manage files, and perform various system tasks through natural language commands.

## System Architecture

### Core Components

**Host Application** (`host/` directory)
- Voice recognition and speech processing
- Local-first AI model integration with cloud fallback
- Conversation management and state handling
- Audio recording and text-to-speech

**MCP Server** (`mcp_os/` directory)
- Tool execution engine
- Application launcher with OS-specific configurations
- File system operations
- Steam integration for gaming

### Key Features

- **Local-First AI**: Prioritizes local Ollama models for speed and privacy
- **Intelligent Fallback**: Automatically switches to OpenAI models when needed
- **Voice Interface**: Continuous listening with wake phrase detection
- **Cross-Platform**: Supports Windows, macOS, and Linux
- **Tool Integration**: Extensive application launching and system control
- **Sleep Mode**: Energy-efficient operation with wake commands

## Installation and Setup

### Prerequisites

- Python 3.8 or higher
- Two separate virtual environments (host and server)
- Ollama installed for local AI models
- OpenAI API key for fallback functionality

### Virtual Environment Setup

```bash
# Create server environment
python -m venv .venv-server
.venv-server\Scripts\activate  # Windows
source .venv-server/bin/activate  # Linux/macOS

# Install server dependencies
pip install fastmcp

# Create host environment  
python -m venv .venv-host
.venv-host\Scripts\activate  # Windows
source .venv-host/bin/activate  # Linux/macOS

# Install host dependencies
pip install openai python-dotenv webrtcvad pyaudio pyttsx3
```

### Environment Configuration

Create a `.env` file in the project root:

```env
# OpenAI Configuration
OPENAI_API_KEY=your_openai_api_key_here

# Model Configuration
USE_LOCAL_FIRST=true
LOCAL_CHAT_MODEL=llama3.1:8b-instruct-q4_0
FRONTIER_CHAT_MODEL=gpt-4o
OLLAMA_HOST=http://localhost:11434

# Audio Configuration
SAMPLE_RATE=16000
SILENCE_DURATION=1.5
PROCESSING_TIMEOUT=60.0

# Wake Phrase Configuration
STUCK_PHRASE="hello abraxas are you stuck"

# Mode Selection
CHATOS_CLI_MODE=false  # Set to true for CLI mode
```

## Launch System

### Windows Launch Script

The system uses a batch file (`run.bat`) to coordinate the startup:

```batch
@echo off
:: ────────────────────────────────────────────────
:: ChatOS launch script – Windows 10 / 11
:: place this file in ...\ChatOS\start_chat_os.bat
:: ────────────────────────────────────────────────

REM ── 1.  Define absolute paths (edit if you moved folders) ──────────────
set "CHATOS_DIR=%~dp0"
set "SERVER_VENV=%CHATOS_DIR%.venv-server"
set "HOST_VENV=%CHATOS_DIR%.venv-host"

REM ── 2.  Launch the MCP server in a new window ─────────────────────────
start "ChatOS-Server" ^
cmd /k ^
"^
    cd /d "%CHATOS_DIR%" ^& ^
    call "%SERVER_VENV%\Scripts\activate.bat" ^& ^
    python mcp_os\server.py ^
"

REM Wait for server to be ready
timeout /t 3 /nobreak > nul

REM ── 3.  Activate the host venv in *this* window and run the UI ────────
call "%HOST_VENV%\Scripts\activate.bat"
python host\enhanced_chat_host.py

REM ── 4.  Keep the window open after exit so you can read logs ──────────
echo.
echo Chat host has terminated.  Press any key to close this window.
pause > nul
```

### Launch Process

1. **Server Startup**: Launches MCP server in separate console window
2. **Initialization Delay**: 3-second wait for server readiness
3. **Host Activation**: Starts the main ChatOS interface
4. **Logging**: Keeps console open for debugging after exit

## Operating Modes

### Voice Mode (Default)

**Activation**: `CHATOS_CLI_MODE=false`

Features:
- Continuous audio monitoring
- Wake phrase detection when stuck
- Text-to-speech responses
- Sleep/wake functionality
- Automatic transcription and processing

**Voice Commands**:
- `"go to sleep"` - Enter sleep mode
- `"wake up"` / `"hello"` - Wake from sleep
- `"reset chat"` - Clear conversation history
- `"exit"` / `"quit"` - Shutdown system

### CLI Mode

**Activation**: `CHATOS_CLI_MODE=true`

Features:
- Text-based interaction
- Same tool capabilities as voice mode
- Faster development and debugging
- No audio processing overhead

**CLI Commands**:
- `exit`, `quit`, `goodbye` - Quit application
- `reset`, `clear`, `new chat` - Start new conversation
- `help` - Show available commands

## AI Model Configuration

### Local-First Architecture

**Primary Model**: Ollama-hosted local models
- Default: `llama3.1:8b-instruct-q4_0`
- Faster response times
- No API costs
- Privacy-focused

**Fallback Model**: OpenAI GPT models
- Default: `gpt-4o`
- Handles complex tool calling
- Automatic failover on local model issues
- Enhanced reasoning capabilities

### Failover Logic

```python
class FailoverChatProvider:
    def __init__(self, primary, backup, timeout=30):
        self.primary = primary      # Local Ollama model
        self.backup = backup        # OpenAI model
        self.timeout = timeout      # Local model timeout
```

**Failover Triggers**:
- Local model timeout
- Tool execution failures
- Network connectivity issues
- Model unavailability

## Tool System (MCP Integration)

### Application Launcher

The system can launch applications using natural language:

**Examples**:
- `"Open notepad"` → Launches Notepad
- `"Launch Word"` → Opens Microsoft Word
- `"Open calculator"` → Starts Calculator app

**Configuration**: `mcp_os/apps_config.json` contains OS-specific app mappings:

```json
{
  "windows": {
    "notepad": ["notepad.exe"],
    "word": [
      "C:\\Program Files\\Microsoft Office\\root\\Office16\\WINWORD.EXE",
      "winword.exe",
      "start winword"
    ]
  }
}
```

### Steam Integration

Gaming-specific tools for Steam users (host/voice_assistant/state.py):

- `list_steam_games` - Show installed games
- `launch_steam_game` - Start specific games
- `open_steam` - Launch Steam client

### File System Operations (host/voice_assistant/state.py)

File and folder management capabilities:

- `create_folder` - Create new directories
- `open_folder` - Open file explorer to location
- File system navigation and manipulation

### Cross-Platform Support

The tool system adapts to the current operating system:

- **Windows**: Uses `startfile()` and Registry integration
- **macOS**: Uses `open -a` command structure
- **Linux**: Uses standard executable launching

## State Management

### Assistant Modes

```python
class AssistantMode(Enum):
    LISTENING = "listening"      # Ready for commands
    RECORDING = "recording"      # Capturing audio
    PROCESSING = "processing"    # AI processing
    SPEAKING = "speaking"        # TTS playback
    STUCK_CHECK = "stuck_check"  # Wake phrase detection
    ERROR = "error"              # Error recovery
    SLEEPING = "sleeping"        # Sleep mode
```

### Conversation Management

- **Thread-safe state handling**: Prevents race conditions
- **Conversation history**: Maintains context across interactions
- **Interrupt handling**: Clean cancellation of operations
- **Stuck detection**: Monitors for unresponsive states

## Audio Processing

### Voice Activity Detection (VAD)

**WebRTC VAD Integration**:
- Configurable aggressiveness levels (0-3)
- Real-time speech detection
- Background noise filtering

**Audio Configuration**:
- **Sample Rate**: 16kHz (optimal for speech)
- **Silence Duration**: 1.5 seconds (end-of-speech detection)
- **Min Speech Duration**: 0.8 seconds (filter out noise)

### Speech Recognition

**Primary**: OpenAI Whisper API
- High accuracy transcription
- Robust noise handling
- Multiple language support

**Fallback**: Local Whisper models (configurable)

### Text-to-Speech

**Primary**: OpenAI TTS
- Natural voice synthesis
- Multiple voice options
- High quality output

**Fallback**: Pyttsx3 (local synthesis)

## Development and Debugging

### Logging System

**Configuration**: Adjustable log levels in `config.py`
```python
log_level: str = "INFO"          # DEBUG, INFO, WARNING, ERROR
log_file: str = "voice_assistant.log"
```

**Log Categories**:
- Mode transitions and state changes
- Tool execution results
- Model failover events
- Audio processing status
- Error handling and recovery

### CLI Development Mode

For rapid development and testing:

```bash
# Set CLI mode in environment
export CHATOS_CLI_MODE=true

# Launch directly
python host/enhanced_chat_host.py
```

### Debugging Tools

**Application Testing**:
```python
# Test app availability
test_app_availability()

# List all configured apps
list_apps()

# Search for specific apps
search_app("word")
```

## Security Considerations

### Path Safety

Application launcher includes safety restrictions:
- **Windows**: Only allows execution from trusted directories
- **macOS/Linux**: Restricted to standard application paths
- Path validation prevents arbitrary code execution

### Local Model Benefits

- **Privacy**: No data sent to external services for local operations
- **Security**: Reduced attack surface
- **Compliance**: Suitable for sensitive environments

## Troubleshooting

### Common Issues

**Local Model Not Starting**:
- Verify Ollama installation and model availability
- Check `OLLAMA_HOST` configuration
- Review server logs for connection errors

**Audio Issues**:
- Verify microphone permissions
- Check `webrtcvad` installation
- Adjust VAD aggressiveness settings

**Tool Execution Failures**:
- Review `apps_config.json` for correct paths
- Check application availability with `test_app_availability()`
- Verify MCP server connectivity

### Performance Optimization

**Local Model Performance**:
- Use quantized models (e.g., `q4_0`) for speed
- Adjust timeout settings based on hardware
- Monitor resource usage during operation

**Audio Processing**:
- Optimize VAD settings for environment
- Adjust silence thresholds for responsiveness
- Use appropriate sample rates

## Future Enhancements

### Planned Features

- **Plugin System**: Extensible tool architecture
- **Multi-Language Support**: International language packs
- **Enhanced Gaming Integration**: Broader game launcher support
- **Smart Home Integration**: IoT device control
- **Calendar and Email**: Productivity tool integration

### Customization Options

- **Voice Training**: Personal voice model adaptation
- **Custom Wake Phrases**: User-defined activation commands
- **Workflow Automation**: Multi-step command sequences
- **Context Awareness**: Location and time-based responses

## API Reference

### Key Classes

**ConversationManager**: Main interaction controller
**AssistantState**: Global state management
**FailoverChatProvider**: AI model coordination
**ContinuousAudioRecorder**: Audio capture system

### Configuration Parameters

**Audio Settings**:
- `sample_rate`: Audio sampling frequency
- `silence_duration`: End-of-speech detection
- `vad_aggressiveness`: Voice activity sensitivity

**Model Settings**:
- `local_chat_model`: Ollama model identifier
- `frontier_chat_model`: OpenAI model name
- `local_chat_timeout`: Local model timeout

**Processing Settings**:
- `processing_timeout`: Maximum processing time
- `tool_timeout`: Tool execution timeout
- `stuck_phrase`: Wake phrase for recovery

## License and Contributing

This project demonstrates advanced integration patterns for local-first AI systems with intelligent cloud fallbacks. The modular architecture supports easy extension and customization for specific use cases.

For development contributions, please maintain the separation between host and server components, and ensure all new tools follow the MCP protocol standards.


OPERATION FLOW:

# A. SERVER.PY (in mcp_os)

 server.py is the MCP (Model Context Protocol) server that:

Initializes FastMCP: Creates the MCP server instance
Loads Application Configuration: Reads apps_config.json for OS-specific app launching
Registers Tools: Sets up all the tools the AI can use:

launch_app() - Launch applications
list_apps() - Show available apps
search_app() - Find apps by name
test_app_availability() - Check which apps are actually available
launch_by_path() - Launch apps by full path
File system tools (from fs_tools.py)


Starts Listening: Waits for tool calls from the host application via stdio

Key Setup Actions:
python# Load OS-specific app configurations
load_apps_config("apps_config.json")

## Register file system tools
fs_tools.register_fs_tools(mcp)

## Start the MCP server
if __name__ == "__main__":
    mcp.run(transport="stdio")
The server runs in a separate console window and waits for the host application to connect and send tool execution requests. It's essentially the "backend" that gives the AI its ability to interact with your operating system.

After a 3-second delay, the batch file then launches the host application (enhanced_chat_host.py) which connects to this server.

