# voice_assistant/audio.py
"""
Audio recording and processing module
"""

import asyncio
import queue
import threading
import logging
import io
from typing import Optional
import numpy as np
import sounddevice as sd
import soundfile as sf

from .state import AssistantState, AssistantMode

logger = logging.getLogger(__name__)

class ContinuousAudioRecorder:
    """Manages continuous audio recording with mode awareness"""
    
    def __init__(self, sample_rate: int = 16000):
        self.sample_rate = sample_rate
        self.audio_queue = queue.Queue()
        self.recording = False
        self.stream = None
        self._lock = threading.Lock()
        
        # List available audio devices for debugging
        try:
            devices = sd.query_devices()
            logger.info("Available audio devices:")
            for i, device in enumerate(devices):
                logger.info(f"  {i}: {device['name']} - In:{device['max_input_channels']} Out:{device['max_output_channels']}")
        except Exception as e:
            logger.warning(f"Could not query audio devices: {e}")
        
    def start(self):
        """Start continuous recording"""
        with self._lock:
            if self.recording:
                return
                
            self.recording = True
            self.stream = sd.InputStream(
                samplerate=self.sample_rate,
                blocksize=int(self.sample_rate * 0.03),  # 30ms chunks for VAD
                dtype="int16",
                channels=1,
                callback=self._audio_callback
            )
            self.stream.start()
            logger.info("Started continuous audio recording")
    
    def stop(self):
        """Stop recording"""
        with self._lock:
            self.recording = False
            if self.stream:
                self.stream.stop()
                self.stream.close()
                self.stream = None
            logger.info("Stopped audio recording")
    
    def _audio_callback(self, indata, frames, time_info, status):
        """Audio stream callback"""
        if status:
            logger.warning(f"Audio callback status: {status}")
        if self.recording:
            self.audio_queue.put(bytes(indata))
    
    def get_audio_energy(self, audio_data: np.ndarray) -> float:
        """Calculate RMS energy of audio data"""
        return np.sqrt(np.mean(audio_data.astype(float)**2))
    
    async def record_until_silence(self, state: AssistantState, config, check_stuck_phrase: bool = False) -> Optional[io.BytesIO]:
        """Record audio until silence is detected, optionally checking for stuck phrase"""
        current_mode = state.get_mode()
        
        # Handle sleeping mode - always listen for wake commands
        if current_mode == AssistantMode.SLEEPING:
            logger.info("Listening for wake commands in sleep mode")
            return await self._record_in_sleep_mode(state, config)
        
        # Don't record in certain modes unless checking for stuck phrase
        if not check_stuck_phrase and current_mode not in [AssistantMode.LISTENING, AssistantMode.STUCK_CHECK]:
            return None
            
        logger.info(f"Listening for {'stuck phrase' if check_stuck_phrase else 'speech'} in mode: {current_mode.value}")
        
        return await self._record_normal_mode(state, config, check_stuck_phrase)
    
    async def _record_in_sleep_mode(self, state: AssistantState, config) -> Optional[io.BytesIO]:
        """Record audio specifically for sleep mode - listens for wake commands"""
        frames = []
        silence_frames = 0
        speech_frames = 0
        speech_started = False
        
        # Use shorter timeouts for sleep mode
        required_silence_frames = int(1.0 * self.sample_rate / (self.sample_rate * 0.03))  # 1 second
        min_speech_frames = int(0.5 * self.sample_rate / (self.sample_rate * 0.03))  # 0.5 seconds
        
        # Lower noise floor for sleep mode to catch wake words better
        noise_floor = config.silence_threshold * 0.5
        peak_energy = 0
        
        timeout_counter = 0
        max_timeout = 100  # About 10 seconds of waiting
        
        while self.recording and state.get_mode() == AssistantMode.SLEEPING:
            try:
                # Get audio chunk with timeout
                audio_chunk = self.audio_queue.get(timeout=0.1)
                timeout_counter = 0  # Reset timeout counter on successful audio
                
                # Convert to numpy array
                audio_data = np.frombuffer(audio_chunk, dtype=np.int16)
                energy = self.get_audio_energy(audio_data)
                
                # Update peak energy
                if energy > peak_energy:
                    peak_energy = energy
                
                # Use VAD if enabled and available
                if config.enable_vad and state.vad:
                    # VAD requires specific frame size
                    frame_duration = 30  # ms
                    required_samples = int(self.sample_rate * frame_duration / 1000)
                    
                    if len(audio_data) == required_samples:
                        is_speech = state.vad.is_speech(audio_chunk, self.sample_rate)
                        # Also require energy threshold for VAD
                        is_speech = is_speech and energy > noise_floor
                    else:
                        is_speech = energy > noise_floor
                else:
                    is_speech = energy > noise_floor
                
                if is_speech:
                    if not speech_started:
                        logger.debug(f"Potential wake word detected (energy: {energy:.4f}, threshold: {noise_floor:.4f})")
                        speech_started = True
                        frames.append(audio_chunk)
                        speech_frames += 1
                    else:
                        frames.append(audio_chunk)
                        speech_frames += 1
                    silence_frames = 0
                elif speech_started:
                    frames.append(audio_chunk)
                    silence_frames += 1
                    
                    if silence_frames >= required_silence_frames:
                        # Check if we have enough speech frames
                        if speech_frames >= min_speech_frames:
                            logger.debug(f"Wake command captured: {speech_frames} speech frames")
                            break
                        else:
                            # Reset for next potential wake word
                            frames = []
                            speech_frames = 0
                            speech_started = False
                            silence_frames = 0
                            peak_energy = 0
                        
            except queue.Empty:
                timeout_counter += 1
                if timeout_counter > max_timeout:
                    # Return None after timeout to prevent hanging
                    return None
                await asyncio.sleep(0.01)
                continue
            except Exception as e:
                logger.error(f"Error in sleep mode audio recording: {e}")
                break
        
        # Check if we captured something useful
        if not frames or speech_frames < min_speech_frames:
            return None
        
        # Convert frames to WAV
        return self._frames_to_wav(frames)
    
    async def _record_normal_mode(self, state: AssistantState, config, check_stuck_phrase: bool) -> Optional[io.BytesIO]:
        """Record audio in normal listening mode"""
        frames = []
        silence_frames = 0
        speech_frames = 0
        speech_started = False
        required_silence_frames = int(config.silence_duration * self.sample_rate / (self.sample_rate * 0.03))
        min_speech_frames = int(config.min_speech_duration * self.sample_rate / (self.sample_rate * 0.03))
        
        # Shorter requirements for stuck phrase detection
        if check_stuck_phrase:
            required_silence_frames = int(0.5 * self.sample_rate / (self.sample_rate * 0.03))
            min_speech_frames = int(0.3 * self.sample_rate / (self.sample_rate * 0.03))
        
        # Dynamic noise floor estimation
        noise_samples = []
        noise_floor = config.silence_threshold
        
        # Track peak energy to detect actual speech
        peak_energy = 0
        
        while self.recording and not state.interrupt_flag.is_set():
            try:
                # Check if mode changed (unless we're checking for stuck phrase)
                if not check_stuck_phrase:
                    current_mode = state.get_mode()
                    if current_mode not in [AssistantMode.LISTENING, AssistantMode.STUCK_CHECK]:
                        logger.debug(f"Mode changed to {current_mode.value}, stopping recording")
                        return None
                
                # Get audio chunk
                audio_chunk = self.audio_queue.get(timeout=0.1)
                
                # Convert to numpy array
                audio_data = np.frombuffer(audio_chunk, dtype=np.int16)
                energy = self.get_audio_energy(audio_data)
                
                # Update peak energy
                if energy > peak_energy:
                    peak_energy = energy
                
                # Update noise floor during initial silence
                if not speech_started and len(noise_samples) < 50:  # ~1.5 seconds of samples
                    noise_samples.append(energy)
                    if len(noise_samples) >= 20:
                        # Calculate noise floor with some headroom
                        noise_floor = np.percentile(noise_samples, 95) * config.energy_threshold_multiplier
                        noise_floor = max(noise_floor, config.silence_threshold)
                        noise_floor = min(noise_floor, config.max_energy_threshold)
                
                # Use VAD if enabled and available
                if config.enable_vad and state.vad:
                    # VAD requires specific frame size
                    frame_duration = 30  # ms
                    required_samples = int(self.sample_rate * frame_duration / 1000)
                    
                    if len(audio_data) == required_samples:
                        is_speech = state.vad.is_speech(audio_chunk, self.sample_rate)
                        # Also require energy threshold for VAD
                        is_speech = is_speech and energy > noise_floor
                    else:
                        is_speech = energy > noise_floor
                else:
                    is_speech = energy > noise_floor
                
                if is_speech:
                    if not speech_started:
                        # Require sustained energy above threshold
                        if energy > noise_floor * 1.5:  # Higher threshold for speech start
                            logger.debug(f"Speech detected (energy: {energy:.4f}, threshold: {noise_floor:.4f})")
                            speech_started = True
                            frames.append(audio_chunk)
                            speech_frames += 1
                    else:
                        frames.append(audio_chunk)
                        speech_frames += 1
                    silence_frames = 0
                elif speech_started:
                    frames.append(audio_chunk)
                    silence_frames += 1
                    
                    if silence_frames >= required_silence_frames:
                        # Check if we have enough speech frames
                        if speech_frames >= min_speech_frames:
                            # Also check if peak energy was significant
                            if peak_energy > noise_floor * 2:
                                logger.debug(f"Silence detected after {speech_frames} speech frames (peak energy: {peak_energy:.4f})")
                                break
                            else:
                                logger.debug(f"Ignoring low-energy speech (peak: {peak_energy:.4f})")
                                frames = []
                                speech_frames = 0
                                speech_started = False
                                silence_frames = 0
                                peak_energy = 0
                        else:
                            logger.debug(f"Ignoring short noise burst ({speech_frames} frames)")
                            frames = []
                            speech_frames = 0
                            speech_started = False
                            silence_frames = 0
                            peak_energy = 0
                        
            except queue.Empty:
                await asyncio.sleep(0.01)
                continue
            except Exception as e:
                logger.error(f"Error in audio recording: {e}")
                break
        
        if not frames or speech_frames < min_speech_frames:
            return None
        
        # Final peak energy check
        if peak_energy < noise_floor * 1.5:
            logger.debug(f"Rejecting low-energy audio (peak: {peak_energy:.4f})")
            return None
        
        return self._frames_to_wav(frames)
    
    def _frames_to_wav(self, frames) -> Optional[io.BytesIO]:
        """Convert audio frames to WAV format"""
        wav_buffer = io.BytesIO()
        try:
            with sf.SoundFile(
                wav_buffer,
                mode="w",
                samplerate=self.sample_rate,
                channels=1,
                subtype="PCM_16",
                format="WAV"
            ) as sound_file:
                for frame in frames:
                    sound_file.buffer_write(frame, dtype="int16")
        except Exception as e:
            logger.error(f"Error creating WAV file: {e}")
            return None
        
        wav_buffer.seek(0)
        wav_buffer.name = "speech.wav"
        return wav_buffer