import logging
import subprocess
import tempfile
import os
import queue
import numpy as np
import sounddevice as sd
import soundfile as sf
import noisereduce as nr
import yt_dlp
from contextlib import contextmanager
import time
import io

logger = logging.getLogger(__name__)

class AudioProcessor:
    """Handles audio streaming, processing, and playback."""
    
    def __init__(self, config):
        """Initialize the audio processor.
        
        Args:
            config (ConfigParser): Configuration object.
        """
        self.config = config
        self.ffmpeg_path = config['DEFAULT'].get('ffmpeg_path', 'ffmpeg')
        # Audio settings from [DEFAULT]
        self.sample_rate = int(config['DEFAULT'].get('sample_rate', '16000'))
        # Audio settings from [AUDIO]
        self.frame_duration_ms = float(config['AUDIO'].get('frame_duration_ms', '30'))
        self.frame_duration = self.frame_duration_ms / 1000.0  # Convert to seconds
        self.silence_duration_ms = float(config['AUDIO'].get('silence_duration_ms', '300'))
        self.silence_frames = int((self.silence_duration_ms / self.frame_duration_ms))
        self.noise_prop_decrease = float(config['AUDIO'].get('noise_prop_decrease', '0.8'))
        self.n_std_thresh_stationary = float(config['AUDIO'].get('n_std_thresh_stationary', '1.5'))
        self.n_fft = int(config['AUDIO'].get('n_fft', '256'))
        self.frame_size = int(self.sample_rate * self.frame_duration * 2)  # 16-bit mono
        self.vb_cable_device_index = int(config['DEFAULT'].get('vb_cable_device_index', '0'))
        self.volume_level = float(config['DEFAULT'].get('volume_level', '0.3'))
        self.ffmpeg_audio_process = None
        self.playback_queue = queue.Queue()


    @contextmanager
    def temp_audio_file(self, audio_data):
        """Create a temporary WAV file from audio data.
        
        Args:
            audio_data (bytearray): Raw audio data.
        
        Yields:
            str: Path to the temporary WAV file.
        
        Raises:
            IOError: If file creation or writing fails.
        """
        temp_file = None
        try:
            if not audio_data or len(audio_data) < self.frame_size:
                logger.error(f"Invalid audio data: length {len(audio_data)} bytes")
                raise IOError("Invalid audio data")
            
            temp_file = tempfile.NamedTemporaryFile(suffix='.wav', delete=False, dir=tempfile.gettempdir())
            temp_path = temp_file.name
            
            audio_array = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
            
            try:
                sf.write(temp_path, audio_array, self.sample_rate, format='WAV')

            except Exception as e:
                logger.error(f"Failed to write audio to {temp_path}: {e}")
                raise IOError(f"Failed to write audio: {e}")
            
            yield temp_path
            
        except Exception as e:
            logger.error(f"Error creating temporary audio file: {e}")
            raise
        finally:
            if temp_file:
                try:
                    temp_file.close()
                    if os.path.exists(temp_path):
                        os.remove(temp_path)
                except Exception as e:
                    logger.warning(f"Failed to delete temporary file {temp_path}: {e}")
    
    def get_audio_url(self, youtube_url: str) -> str:
        """Extract audio URL from YouTube video."""
        logger.info(f"Extracting audio URL from: {youtube_url}")
        try:
            ydl_opts = {
                'format': 'bestaudio/best',
                'quiet': True,
                'no_warnings': True,
            }
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(youtube_url, download=False)
                audio_url = info.get('url')
                if not audio_url:
                    logger.error("No audio URL found")
                    return ""

                return audio_url
        except Exception as e:
            logger.error(f"Failed to extract audio URL: {e}")
            return ""
    
    def start_ffmpeg_stream(self, audio_url: str) -> subprocess.Popen:
        """Start FFmpeg to stream audio."""
        logger.info("Starting FFmpeg audio stream")
        try:
            cmd = [
                self.ffmpeg_path,
                "-i", audio_url,
                "-f", "s16le",
                "-ar", str(self.sample_rate),
                "-ac", "1",
                "pipe:"
            ]
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                bufsize=self.frame_size * 10
            )
            time.sleep(1)
            if process.poll() is not None:
                stderr = process.stderr.read().decode('utf-8', errors='ignore')
                logger.error(f"FFmpeg audio stream failed: {stderr}")
                return None
            logger.info(f"FFmpeg audio stream started with PID: {process.pid}")
            self.ffmpeg_audio_process = process
            return process
        except Exception as e:
            logger.error(f"Failed to start FFmpeg audio stream: {e}")
            return None
    
    def process_frame_with_noisereduce(self, frame: bytes, noise_level=0.8) -> bytes:
        """Process audio frame with noise reduction."""

        try:
            if not frame or len(frame) != self.frame_size:
                logger.warning(f"Invalid frame size: {len(frame)}")
                return frame
            audio_array = np.frombuffer(frame, dtype=np.int16).astype(np.float32) / 32768.0
            reduced_audio = nr.reduce_noise(
                y=audio_array,
                sr=self.sample_rate,
                prop_decrease=noise_level,
                n_std_thresh_stationary=self.n_std_thresh_stationary,
                n_fft=self.n_fft
            )
            processed_audio = (reduced_audio * 32768.0).astype(np.int16).tobytes()
            return processed_audio
        except Exception as e:
            logger.error(f"Error processing audio frame: {e}")
            return frame
    
    def play_audio(self, audio_content: bytes):
        """Play audio content."""

        try:
            audio_array, samplerate = sf.read(io.BytesIO(audio_content), dtype='float32')
            sd.play(audio_array, samplerate, device=self.vb_cable_device_index)
            sd.wait()

        except Exception as e:
            logger.error(f"Error playing audio: {e}")