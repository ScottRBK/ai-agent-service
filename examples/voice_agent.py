"""
Voice-enabled agent runner with STT/TTS integration.

Usage:
    python examples/voice_agent.py [agent_id] [provider_id]

Controls:
    SPACE + hold: Record voice input
    ENTER: Switch to text input mode
    Ctrl+C: Exit

Examples:
    python examples/voice_agent.py research_agent azure_openai_cc
    python examples/voice_agent.py cli_agent ollama
"""

import asyncio
import sys
import os
import io
import tempfile
import threading
import time
from pathlib import Path
import argparse
import wave
import select
import termios
import tty
from typing import Optional
from contextlib import contextmanager
import ctypes
import platform

try:
    import pyaudio
    import requests
    import subprocess
except ImportError as e:
    print(f"❌ Missing dependencies: {e}")
    print("Install with: pip install pyaudio requests")
    sys.exit(1)

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from app.core.agents.cli_agent import CLIAgent
from app.core.providers.manager import ProviderManager
from app.utils.logging import logger
from app.utils.chat_utils import clean_response_for_memory

# Voice service endpoints
STT_URL = "http://127.0.0.1:2022/v1/audio/transcriptions"
TTS_URL = "http://127.0.0.1:8880/v1/audio/speech"


# Audio settings
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000

@contextmanager
def suppress_stderr():
    """Context manager to temporarily suppress stderr output at file descriptor level"""
    # Save the original stderr file descriptor
    stderr_fd = sys.stderr.fileno()
    old_stderr_fd = os.dup(stderr_fd)
    
    # Open devnull
    devnull_fd = os.open(os.devnull, os.O_WRONLY)
    
    try:
        # Redirect stderr to devnull at the file descriptor level
        os.dup2(devnull_fd, stderr_fd)
        yield
    finally:
        # Restore original stderr
        os.dup2(old_stderr_fd, stderr_fd)
        os.close(old_stderr_fd)
        os.close(devnull_fd)

# Suppress ALSA warnings globally
if os.getenv('SUPPRESS_AUDIO_WARNINGS', 'true').lower() == 'true':
    # Set environment variables to minimize ALSA probing
    os.environ['AUDIODEV'] = 'null'
    os.environ['SDL_AUDIODRIVER'] = 'dummy'
    
    # For Linux systems, use ALSA lib error handler
    if platform.system() == 'Linux':
        try:
            # Try to load libasound and set error handler to null
            libasound = ctypes.cdll.LoadLibrary('libasound.so.2')
            # Set ALSA error handler to null function
            ERROR_HANDLER_FUNC = ctypes.CFUNCTYPE(None, ctypes.c_char_p, ctypes.c_int, 
                                                  ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p)
            def null_error_handler(filename, line, function, err, fmt):
                pass
            
            c_error_handler = ERROR_HANDLER_FUNC(null_error_handler)
            libasound.snd_lib_error_set_handler(c_error_handler)
        except:
            pass  # If we can't load libasound, continue anyway

class VoiceServices:
    def __init__(self):
        self.stt_available = self.check_service_health(STT_URL.replace("/v1/audio/transcriptions", "/health"))
        self.tts_available = self.check_service_health(TTS_URL.replace("/v1/audio/speech", "/health"))
        
        # Check for audio playback capability
        if self.tts_available:
            self.playback_method = self._detect_playback_method()
            if not self.playback_method:
                print(f"⚠️  Warning: No audio playback method available")
                self.tts_available = False
    
    def check_service_health(self, health_url: str) -> bool:
        try:
            response = requests.get(health_url, timeout=2)
            return response.status_code == 200
        except:
            return False
    
    def _detect_playback_method(self):
        """Detect the best available audio playback method"""
        # Go back to mpv as primary since it was working better
        methods = [
            ('mpv', ['mpv', '--version']),
            ('paplay', ['paplay', '--version']),
            ('aplay', ['aplay', '--version']),
            ('ffplay', ['ffplay', '-version'])
        ]
        
        for method, test_cmd in methods:
            try:
                with suppress_stderr():
                    subprocess.run(test_cmd, capture_output=True, check=True, timeout=2)
                sys.stdout.write(f"✅ Audio playback: Using {method}\n")
                sys.stdout.flush()
                return method
            except:
                continue
        return None
    
    def transcribe_audio(self, audio_file_path: str) -> Optional[str]:
        if not self.stt_available:
            return None
        
        try:
            with open(audio_file_path, 'rb') as audio_file:
                files = {'file': audio_file}
                data = {'model': 'whisper-1'}
                
                response = requests.post(STT_URL, files=files, data=data, timeout=30)
                response.raise_for_status()
                
                result = response.json()
                return result.get('text', '').strip()
        except Exception as e:
            print(f"❌ STT Error: {e}")
            return None
    
    def text_to_speech(self, text: str) -> bool:
        if not self.tts_available:
            return False
        
        try:
            # Try Kokoro format first
            data = {
                'text': text,
                'voice': 'af'  # Kokoro voice parameter
            }
            
            response = requests.post(TTS_URL, json=data, timeout=30)
            
            # If that fails, try OpenAI format
            if response.status_code != 200:
                data = {
                    'model': 'tts-1',
                    'input': text,
                    'voice': 'af_alloy'
                }
                response = requests.post(TTS_URL, json=data, timeout=30)
            
            response.raise_for_status()
            
            # Debug: Check response headers and content type
            content_type = response.headers.get('content-type', 'unknown')
            content_length = len(response.content)
            sys.stdout.write(f"   Audio response: {content_type}, {content_length} bytes\n")
            sys.stdout.flush()
            
            # For now, let's go back to the file method but with optimizations
            sys.stdout.write("   Using optimized file playback method\n")
            sys.stdout.flush()
            return self._play_audio_file_fallback(response.content, content_type)
        except Exception as e:
            sys.stdout.write(f"\n❌ TTS Error: {e}\n")
            sys.stdout.flush()
            # Try to get more details from response
            try:
                if 'response' in locals():
                    sys.stdout.write(f"   Response status: {response.status_code}\n")
                    sys.stdout.write(f"   Response text: {response.text[:200]}\n")
                    sys.stdout.flush()
            except:
                pass
            return False
    
    def _play_audio_file_fallback(self, audio_content: bytes, content_type: str):
        """Fallback method using temporary files"""
        # Determine file extension based on content type AND actual content
        if 'mp3' in content_type or 'mpeg' in content_type:
            suffix = '.mp3'
        elif 'wav' in content_type:
            suffix = '.wav'
        elif 'ogg' in content_type:
            suffix = '.ogg'
        else:
            # Check file magic bytes to determine format
            if audio_content.startswith(b'ID3') or audio_content.startswith(b'\xff\xfb'):
                suffix = '.mp3'  # MP3 file
            elif audio_content.startswith(b'RIFF'):
                suffix = '.wav'  # WAV file
            else:
                suffix = '.mp3'  # Default to MP3
        
        # Use RAM disk if available, otherwise /tmp
        temp_dirs = ['/dev/shm', '/tmp']
        temp_dir = None
        for td in temp_dirs:
            if os.path.exists(td) and os.access(td, os.W_OK):
                temp_dir = td
                break
        
        # Save audio to temporary file and play
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False, dir=temp_dir) as temp_audio:
            temp_audio.write(audio_content)
            temp_audio_path = temp_audio.name
        
        sys.stdout.write(f"   Saved audio to: {temp_audio_path} (size: {len(audio_content)} bytes)\n")
        sys.stdout.flush()
        
        try:
            self._play_audio_file(temp_audio_path)
            return True
        finally:
            os.unlink(temp_audio_path)
    
    def _play_audio_file(self, audio_path: str):
        """Play audio file using the best available method"""
        try:
            if self.playback_method == 'mpv':
                # Use mpv with simple, reliable settings
                with suppress_stderr():
                    result = subprocess.run([
                        'mpv', 
                        '--no-video',
                        '--really-quiet',  # Even less output
                        '--audio-device=pulse',
                        '--no-cache',  # Disable caching to see if that helps
                        audio_path
                    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            elif self.playback_method == 'paplay':
                with suppress_stderr():
                    result = subprocess.run([
                        'paplay', 
                        '--latency-msec=200',
                        '--process-time-msec=20',
                        audio_path
                    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            elif self.playback_method == 'aplay':
                with suppress_stderr():
                    result = subprocess.run(['aplay', audio_path], 
                                          stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            elif self.playback_method == 'ffplay':
                with suppress_stderr():
                    subprocess.run(['ffplay', '-nodisp', '-autoexit', audio_path], 
                                 stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            else:
                raise Exception("No audio playback method available")
        except Exception as e:
            sys.stdout.write(f"   Audio playback error: {e}\n")
            sys.stdout.flush()

class AudioRecorder:
    def __init__(self):
        with suppress_stderr():
            self.audio = pyaudio.PyAudio()
        self.recording = False
        self.frames = []
    
    def start_recording(self):
        self.recording = True
        self.frames = []
        
        with suppress_stderr():
            stream = self.audio.open(
                format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK
            )
        
        sys.stdout.write("\r🎤 Recording... (release SPACE to stop)\n")
        sys.stdout.flush()
        
        while self.recording:
            try:
                data = stream.read(CHUNK, exception_on_overflow=False)
                self.frames.append(data)
            except Exception as e:
                sys.stdout.write(f"\r\nRecording error: {e}\n")
                sys.stdout.flush()
                break
        
        stream.stop_stream()
        stream.close()
        sys.stdout.write(f"\r\n🔇 Recording stopped - captured {len(self.frames)} frames\n")
        sys.stdout.flush()
    
    def stop_recording(self):
        self.recording = False
    
    def save_recording(self, filename: str):
        if not self.frames:
            sys.stdout.write("\r\nNo audio frames to save\n")
            sys.stdout.flush()
            return False
        
        try:
            wf = wave.open(filename, 'wb')
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(self.audio.get_sample_size(FORMAT))
            wf.setframerate(RATE)
            wf.writeframes(b''.join(self.frames))
            wf.close()
            sys.stdout.write(f"\r\nSaved {len(self.frames)} frames to {filename}\n")
            sys.stdout.flush()
            return True
        except Exception as e:
            sys.stdout.write(f"\r\nError saving audio: {e}\n")
            sys.stdout.flush()
            return False
    
    def cleanup(self):
        self.audio.terminate()

class SimpleKeyListener:
    def __init__(self):
        self.space_pressed = False
        self.enter_pressed = False
        self.running = True
        
    def listen(self):
        old_settings = termios.tcgetattr(sys.stdin)
        try:
            tty.setraw(sys.stdin.fileno())
            while self.running:
                if select.select([sys.stdin], [], [], 0.1)[0]:
                    char = sys.stdin.read(1)
                    if char == ' ':
                        self.space_pressed = True
                    elif char in ['\r', '\n', '\x0d', '\x0a']:  # Various enter codes
                        self.enter_pressed = True
                    elif char == '\x03':  # Ctrl+C
                        self.running = False
                        break
        finally:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)

def print_usage():
    print("🎙️  Voice Agent - AI Agent Service with Voice")
    print("=" * 50)
    print()
    print("Usage:")
    print("  python examples/voice_agent.py [agent_id] [provider_id]")
    print()
    print("Controls:")
    print("  SPACE + hold: Record voice input")
    print("  ENTER: Switch to text input mode")
    print("  Ctrl+C: Exit")
    print()
    print("Available Agents:")
    print("  research_agent  - Web research with voice")
    print("  cli_agent      - Full featured with voice")
    print("  data_agent     - Data analysis with voice")
    print()

def validate_agent_config(agent_id: str) -> bool:
    try:
        from app.core.agents.agent_tool_manager import AgentToolManager
        agent_manager = AgentToolManager(agent_id)
        return agent_manager.config is not None
    except Exception as e:
        logger.error(f"Error validating agent {agent_id}: {e}")
        return False

def validate_provider(provider_id: str) -> bool:
    try:
        provider_manager = ProviderManager()
        provider_info = provider_manager.get_provider(provider_id)
        return provider_info is not None
    except Exception as e:
        logger.error(f"Error validating provider {provider_id}: {e}")
        return False

def get_provider_from_agent_config(agent_id: str) -> str:
    try:
        from app.core.agents.agent_tool_manager import AgentToolManager
        agent_manager = AgentToolManager(agent_id)
        provider = agent_manager.config.get("provider")
        return provider if provider else "azure_openai_cc"
    except Exception as e:
        logger.error(f"Error getting provider from agent config for {agent_id}: {e}")
        return "azure_openai_cc"

def parse_arguments():
    parser = argparse.ArgumentParser(description="Voice-enabled AI agent runner")
    parser.add_argument("agent_id", help="Agent ID to run")
    parser.add_argument("provider_id", nargs="?", help="Provider ID (optional)")
    parser.add_argument("--model", help="Model to use")
    parser.add_argument("--text-only", action="store_true", help="Disable voice, use text only")
    parser.add_argument("--setting", action="append", nargs=2, 
                       metavar=("KEY", "VALUE"), 
                       help="Model setting KEY VALUE")
    return parser.parse_args()

async def voice_interaction_loop(agent: CLIAgent, voice_services: VoiceServices, text_only: bool = False):
    recorder = AudioRecorder()
    listener = SimpleKeyListener()
    
    # Start keyboard listener in separate thread
    listener_thread = threading.Thread(target=listener.listen, daemon=True)
    listener_thread.start()
    
    sys.stdout.write("\r\n🎙️  Voice Agent Ready!\n")
    if not text_only and voice_services.stt_available and voice_services.tts_available:
        sys.stdout.write("💬 Controls: SPACE (toggle record), ENTER (text), Ctrl+C (exit)\n")
    else:
        sys.stdout.write("💬 Voice services unavailable - text mode only\n")
        sys.stdout.write("💬 Controls: ENTER (text), Ctrl+C (exit)\n")
        text_only = True
    sys.stdout.flush()
    
    recording_thread = None
    is_recording = False
    
    try:
        while listener.running:
            # Handle space press (toggle recording)
            if not text_only and listener.space_pressed and voice_services.stt_available:
                listener.space_pressed = False
                
                if not is_recording:
                    # Start recording
                    sys.stdout.write("\r\n🎤 Recording... (press SPACE again to stop)\n")
                    sys.stdout.flush()
                    recording_thread = threading.Thread(target=recorder.start_recording)
                    recording_thread.start()
                    is_recording = True
                else:
                    # Stop recording and process
                    sys.stdout.write("\r\n🔇 Stopping recording...\n")
                    sys.stdout.flush()
                    recorder.stop_recording()
                    if recording_thread:
                        recording_thread.join()
                    is_recording = False
                    
                    # Process audio
                    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
                        if recorder.save_recording(temp_file.name):
                            sys.stdout.write(f"\r\n🔄 Processing speech... (file size: {os.path.getsize(temp_file.name)} bytes)\n")
                            sys.stdout.flush()
                            text = voice_services.transcribe_audio(temp_file.name)
                            os.unlink(temp_file.name)
                            
                            if text:
                                sys.stdout.write(f"\r\n👤 You said: {text}\n")
                                sys.stdout.flush()
                                response = await agent.chat(text)
                                sys.stdout.write(f"🤖 Agent: {response}\n")
                                sys.stdout.flush()
                                
                                # Text-to-speech
                                if voice_services.tts_available:
                                    # For very long responses, offer summary option
                                    if len(response) > 1000:
                                        sys.stdout.write("🔊 Speaking response... (Press SPACE to interrupt)\n")
                                    else:
                                        sys.stdout.write("🔊 Speaking response...\n")
                                    sys.stdout.flush()
                                    voice_services.text_to_speech(clean_response_for_memory(response))
                            else:
                                sys.stdout.write("\r\n❌ Could not understand speech\n")
                                sys.stdout.flush()
                        else:
                            sys.stdout.write("\r\n❌ No audio recorded\n")
                            sys.stdout.flush()
                    
                    sys.stdout.write("\r\n💬 Ready for next input...\n")
                    sys.stdout.flush()
            
            # Handle enter press (text input)
            elif listener.enter_pressed:
                listener.enter_pressed = False
                
                # If currently recording, stop it first
                if is_recording:
                    recorder.stop_recording()
                    if recording_thread:
                        recording_thread.join()
                    is_recording = False
                    sys.stdout.write("\r\n🔇 Recording cancelled for text input\n")
                    sys.stdout.flush()
                
                sys.stdout.write("\r\n💬 Text mode - Enter your message:\n")
                sys.stdout.flush()
                
                # Stop the listener temporarily
                listener.running = False
                
                # Wait for listener thread to finish
                await asyncio.sleep(0.2)
                
                try:
                    text = input("👤 You: ").strip()
                    
                    if text:
                        response = await agent.chat(text)
                        sys.stdout.write(f"🤖 Agent: {response}\n")
                        sys.stdout.flush()
                        
                        # Optional TTS for text input too
                        if not text_only and voice_services.tts_available:
                            sys.stdout.write("🔊 Speaking response...\n")
                            sys.stdout.flush()
                            voice_services.text_to_speech(clean_response_for_memory(response))
                    
                    sys.stdout.write("\r\n💬 Ready for next input...\n")
                    sys.stdout.flush()
                except KeyboardInterrupt:
                    break
                
                # Restart the listener
                listener = SimpleKeyListener()
                listener_thread = threading.Thread(target=listener.listen, daemon=True)
                listener_thread.start()
            
            await asyncio.sleep(0.1)
    
    except KeyboardInterrupt:
        pass
    finally:
        if is_recording:
            recorder.stop_recording()
            if recording_thread:
                recording_thread.join()
        
        listener.running = False
        recorder.cleanup()
        sys.stdout.write("\r\n👋 Goodbye!\n")
        sys.stdout.flush()

async def main():
    args = parse_arguments()
    
    # Get provider
    provider_id = args.provider_id or get_provider_from_agent_config(args.agent_id)
    
    # Parse model settings
    model_settings = {}
    if args.setting:
        for key, value in args.setting:
            # Simple type conversion
            if value.lower() == 'true':
                model_settings[key] = True
            elif value.lower() == 'false':
                model_settings[key] = False
            else:
                try:
                    model_settings[key] = float(value) if '.' in value else int(value)
                except ValueError:
                    model_settings[key] = value
    
    sys.stdout.write(f"🎙️  Starting voice agent: {args.agent_id} with {provider_id}\n")
    sys.stdout.flush()
    
    # Validate configuration
    if not validate_agent_config(args.agent_id):
        sys.stdout.write(f"❌ Agent '{args.agent_id}' not found\n")
        sys.stdout.flush()
        print_usage()
        return
    
    if not validate_provider(provider_id):
        sys.stdout.write(f"❌ Provider '{provider_id}' not available\n")
        sys.stdout.flush()
        return
    
    # Initialize voice services
    voice_services = VoiceServices()
    sys.stdout.write(f"🎤 STT Service: {'✅' if voice_services.stt_available else '❌'}\n")
    sys.stdout.write(f"🔊 TTS Service: {'✅' if voice_services.tts_available else '❌'}\n")
    sys.stdout.flush()
    
    try:
        # Create agent
        agent = CLIAgent(args.agent_id, provider_id, model=args.model, model_settings=model_settings)
        
        # Start voice interaction
        await voice_interaction_loop(agent, voice_services, args.text_only)
        
    except Exception as e:
        print(f"❌ Error: {e}")
        logger.error(f"Voice agent error: {e}")

if __name__ == "__main__":
    if not os.path.exists("agent_config.json"):
        print("❌ agent_config.json not found")
        print("   Run from project root directory")
        sys.exit(1)
    
    asyncio.run(main())