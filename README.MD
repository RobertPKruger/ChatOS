1. Create a .env file within the host directory, and add the following params:

# For very noisy environments
SILENCE_THRESHOLD=0.05
ENERGY_THRESHOLD_MULTIPLIER=2.5
MIN_SPEECH_DURATION=1.0
VAD_AGGRESSIVENESS=3
MIN_CONFIDENCE_LENGTH=4
USERNAME='<your name>'

# For quieter environments
SILENCE_THRESHOLD=0.02
ENERGY_THRESHOLD_MULTIPLIER=1.5
MIN_SPEECH_DURATION=0.5
VAD_AGGRESSIVENESS=2
MIN_CONFIDENCE_LENGTH=3


OPENAI_API_KEY= <PASTE KEY HERE.>

2. You can switch into the environments and launch the server and host with run.bat.