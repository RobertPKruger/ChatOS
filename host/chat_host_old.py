
# chat_host.py
#$Env:OPENAI_API_KEY = "sk-YOUR-NEW-KEY"
import asyncio
import json, subprocess, queue, threading, sys, io, time, os, uuid
import sounddevice as sd, soundfile as sf
#from pydub import AudioSegment
from openai import OpenAI
from pynput import keyboard   # simple hot-key to start/stop recording
from fastmcp import Client
from dotenv import load_dotenv
load_dotenv()  


client = OpenAI()

# Global MCP client
mcp_client = None

################################################################################
# 1.  Record microphone to an in-memory WAV buffer until you hit <space>.
################################################################################
def record_until_space(fs=16000):
    print("Hold space to talk…")
    q = queue.Queue()
    stop = threading.Event()            # will flip to True on key-up

    def kb_on_release(k):
        if k == keyboard.Key.space:     # finished speaking
            stop.set()
            return False                # stop this Listener

    kb = keyboard.Listener(on_release=kb_on_release)
    kb.start()

    def audio_cb(indata, frames, t, status):
        q.put(bytes(indata))

    rec = sd.RawInputStream(samplerate=fs, blocksize=0, dtype="int16",
                            channels=1, callback=audio_cb)
    rec.start()

    wav = io.BytesIO()
    with sf.SoundFile(wav, mode="w",
                      samplerate=fs,
                      channels=1,
                      subtype="PCM_16",
                      format="WAV") as f:
        while not stop.is_set():
            try:
                data = q.get(timeout=0.1)
                f.buffer_write(data, dtype="int16")
            except queue.Empty:
                pass

    rec.stop()
    kb.join()
    wav.seek(0)
    return wav        

################################################################################
# 2.  Chat loop – one request per utterance
################################################################################
SYSTEM = """You are my personal voice assistant. Use the tools provided to fulfil user requests.

When the user asks to open an application like "Open Notepad" or "Launch Calculator", use the launch_app tool and provide the appropriate app name as a parameter. For example:
- For "Open Notepad" → use launch_app with app="notepad"
- For "Open Calculator" → use launch_app with app="calc"
- For "Open File Explorer" → use launch_app with app="explorer"

Always provide the app parameter when using launch_app tool."""

async def converse_async():
    global mcp_client
    
    server_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "mcp_os", "server.py")
    )

    # Initialize MCP client with proper context management
    mcp_client = Client(server_path)
    
    async with mcp_client:
        # Build OpenAI-ready tools list
        tools = []
        for t in await mcp_client.list_tools():
            # Get the tool schema
            try:
                spec = t.openai_schema()              # new helper
            except AttributeError:
                # Fallback: build OpenAI schema from MCP tool
                spec = {
                    "name": t.name,
                    "description": getattr(t, "description", ""),
                    "parameters": getattr(t, "inputSchema", {"type": "object", "properties": {}})
                }
            
            # Ensure proper schema structure for OpenAI
            if "name" not in spec:
                spec["name"] = t.name
            if "description" not in spec:
                spec["description"] = getattr(t, "description", "")
            
            # Debug: print tool schema to see what's being sent
            print(f"Tool schema for {t.name}:", json.dumps(spec, indent=2))
            
            tools.append({"type": "function", "function": spec})

        history = [{"role": "system", "content": SYSTEM}]
        
        while True:
            # This runs in the main thread (synchronous)
            loop = asyncio.get_event_loop()
            wav = await loop.run_in_executor(None, record_until_space)
            print("…transcribing")

            # ----- sanity checks on recorded buffer ---------------------
            wav.seek(0)
            if len(wav.getbuffer()) < 10_000:
                print("No speech detected – hold space a bit longer.")
                continue

            wav.name = "speech.wav"           # hint for multipart encoder

            stt = client.audio.transcriptions.create(
                model="gpt-4o-transcribe",
                file=wav
            )

            user_text = stt.text.strip()
            print("You:", user_text)
            history.append({"role": "user", "content": user_text})

            completion = client.chat.completions.create(
                model="gpt-4o",
                messages=history,
                tools=tools,
                tool_choice="auto"
            )

            print("DEBUG raw reply\n", completion.model_dump_json(indent=2)[:800], "\n---")

            choice   = completion.choices[0]      # keep the whole choice
            message  = choice.message

            if choice.finish_reason == "tool_calls":
                # normalise both possible shapes into one list
                calls = []

                # new-style multi-tool list
                if getattr(message, "tool_calls", None):
                    calls = message.tool_calls

                # old-style single function_call
                elif getattr(message, "function_call", None):
                    calls = [message.function_call]

                # iterate only if we actually got something
                assistant_text = None
                for call in calls:
                    tool_out = await call_tool_async(call)
                    history.append({
                        "role": "tool", 
                        "tool_call_id": getattr(call, "id", str(uuid.uuid4())),
                        "content": tool_out
                    })
                    #assistant_text = tool_out
            else:
                assistant_text = message.content

            if user_text.strip().lower() in {"reset chat", "new chat"}:
                print("🔄  Starting a new chat thread.")
                history = [{"role": "system", "content": SYSTEM}]
                continue

            print("Assistant:", assistant_text)

            # TTS
            safe_text = assistant_text or "Okay."      # never send None / ""
            audio = client.audio.speech.create(
                model="tts-1",
                voice="nova",
                input=safe_text,
                response_format="wav"
            )

            buf = io.BytesIO(audio.read())
            data, fs = sf.read(buf, dtype="float32")  # soundfile decodes the WAV
            sd.play(data, fs)                         # sounddevice plays it
            sd.wait()

            history.append({"role": "assistant", "content": assistant_text})

async def call_tool_async(call):
    """
    Forward a tool-call (dict or ChatCompletionMessageToolCall) to the MCP server
    and return the text result asynchronously.
    """
    global mcp_client
    
    # ── 1. Normalise to simple values ───────────────────────────────
    if hasattr(call, "function"):                # new object shape
        name       = call.function.name
        args_json  = call.function.arguments or "{}"
    else:                                        # legacy dict shape
        name       = call.get("name") or call["function"]["name"]
        args_json  = (call.get("function", call)).get("arguments", "{}")

    args = json.loads(args_json)

    # ── 2. Call the MCP server ──────────────────────────────────────
    rsp = await mcp_client.call_tool(name, args)

    # Handle different response types
    if hasattr(rsp, "text"):
        return rsp.text
    elif hasattr(rsp, "content"):
        # Handle list of content objects
        if isinstance(rsp.content, list):
            text_parts = []
            for content in rsp.content:
                if hasattr(content, "text"):
                    text_parts.append(content.text)
                elif hasattr(content, "content"):
                    text_parts.append(str(content.content))
                else:
                    text_parts.append(str(content))
            return " ".join(text_parts)
        else:
            return str(rsp.content)
    else:
        return str(rsp)

def converse():
    """Synchronous wrapper to run the async conversation loop"""
    try:
        asyncio.run(converse_async())
    except KeyboardInterrupt:
        print("\nExiting...")
        sys.exit()

################################################################################
# 3.  Run until Ctrl-C
################################################################################
listener = keyboard.Listener(
    on_press=lambda k: None,
    on_release=lambda k: listener.stop() if k == keyboard.Key.space else None
)
listener.start()

if __name__ == "__main__":
    try:
        converse()
    except KeyboardInterrupt:
        sys.exit()