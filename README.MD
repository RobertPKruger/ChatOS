1. Create a .env file within the host directory, and add the following params:

# For very noisy environments
SILENCE_THRESHOLD=0.05
ENERGY_THRESHOLD_MULTIPLIER=2.5
MIN_SPEECH_DURATION=1.0
VAD_AGGRESSIVENESS=3
MIN_CONFIDENCE_LENGTH=4
CHATOS_CLI_MODE=false
USERNAME='<your name>'

# For quieter environments
'''
SILENCE_THRESHOLD=0.02
ENERGY_THRESHOLD_MULTIPLIER=1.5
MIN_SPEECH_DURATION=0.5
VAD_AGGRESSIVENESS=2
MIN_CONFIDENCE_LENGTH=3
USERNAME='<your name>'
'''

CHAT_PROVIDER=ollama
LOCAL_CHAT_MODEL=llama3.1:8b-instruct-q4_0
USE_LOCAL_FIRST=true
OLLAMA_HOST=http://localhost:11434
LOCAL_CHAT_TIMEOUT=30
LOCAL_MAX_TOKENS=512
LOCAL_CONFIDENCE_THRESHOLD=0.7

FRONTIER_CHAT_MODEL=gpt-4o

# Enhanced Processing Configuration  
PROCESSING_TIMEOUT=60.0
STUCK_CHECK_INTERVAL=5.0

# Api key
OPENAI_API_KEY= PASTE KEY HERE.

2. Install requirements to host and server environments.

# HOST side
python -m venv .venv-host              # only if it doesn’t exist yet
source .venv-host/bin/activate         # Windows: .venv-host\Scripts\activate
pip install --upgrade pip              # keeps pip itself current
pip install -r requirements/host.txt
deactivate

# SERVER side
python -m venv .venv-server
source .venv-server/bin/activate       # Windows: .venv-server\Scripts\activate
pip install --upgrade pip
pip install -r requirements/server.txt
deactivate

3. You can switch into the environments and launch the server and host with run.bat.

4. For the local model, you will want to install ollama:

# One-time per machine
brew install ollama            # or the installer/script
ollama serve &                 # or start the GUI app

# Per project
pip install ollama             # Python wrapper
ollama pull mistral            # download a model


Test file:

# my_code.py
import ollama

def main():
    # Make sure the Ollama server is running on localhost:11434
    reply = ollama.generate(
        model='mistral',           # any model you have pulled
        prompt='Summarize the Prisoner’s Dilemma in 100 words'
    )
    print(reply['response'])       # prints the text returned by the model

if __name__ == "__main__":
    main()



python my_code.py              # code calls localhost:11434