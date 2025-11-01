# Updated on 07/21/2025 (8:35 PM) to provide options for outputing the media: to OBS or to a file.
import threading
import time
import signal
import configparser
import logging
import re
from datetime import datetime
import queue
import subprocess
import tempfile
import webrtcvad
import soundfile as sf
import io
from audio_processing import AudioProcessor
from translation import Translator
from obs_control import OBSController
from gui import TranslationGUI
import sys
import numpy as np
import os

# Configure logging with DEBUG level for detailed diagnostics
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('translation.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Ensure console supports UTF-8 on Windows

if sys.platform == "win32":
    os.environ["PYTHONIOENCODING"] = "utf-8"
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')

class TranslationSystem:
    """Main class for the YouTube translation system."""
    
    def __init__(self, config):
        """Initialize the translation system.
        
        Args:
            config (ConfigParser): Configuration object.
        """
        logger.info("Initializing TranslationSystem")
        self.config = config

        # Update config with absolute paths for caption_file and logo_file
        cwd = os.getcwd()
        self.config['DEFAULT']['caption_file'] = os.path.join(cwd, 'captions.txt')
        self.config['DEFAULT']['logo_file'] = os.path.join(cwd, 'logo.png')
        self.output_mode = config['DEFAULT'].get('output_mode', 'File')  # Default to OBS
        self.audio_processor = AudioProcessor(config)
        self.translator = Translator(config)
        self.obs_controller = OBSController(config)
        self.stop_flag = threading.Event()
        self.first_translation_done = threading.Event()
        self.translation_done = threading.Event()  # Added: Signal translation completion
        self.video_start_time = None
        self.audio_stream_start_time = None
        self.audio_time_marker = 0
        self.audio_time_offset = 0
        self.caption_file = config['DEFAULT']['caption_file']
        #self.logo_file = config['DEFAULT']['logo_file']
        self.full_translation_file = None
        self.original_transcript_file = None
        self.video_delay = float(config['DEFAULT']['video_delay'])
        self.gui_update_queue = queue.Queue(maxsize=100)
        self.worker_threads = []
        #self.subtitle_file = os.path.join(cwd, f"subtitles_{datetime.now().strftime('%Y%m%d_%H%M%S')}.srt")
        self.subtitle_file = f"subtitles_{datetime.now().strftime('%Y%m%d_%H%M%S')}.srt"
        self.output_file = f"output_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
        self.audio_files = []  # Store temporary audio files
        self.subtitle_entries = []  # Store subtitle timings

    def translation_worker(self, process, youtube_url: str, translation_service: str, voice_value: str, noise_level: float):
        """Process audio stream, transcribe, translate, and queue playback.
        
        Args:
            process (subprocess.Popen): FFmpeg audio process.
            youtube_url (str): YouTube video URL.
            translation_service (str): Translation service ('Google' or 'Local').
            voice_value (str): TTS voice value (e.g., 'cmn-CN-Standard-B').
            noise_level (float): Noise reduction level.
        """
        logger.info("Starting translation worker.")
        
        if process is None or process.poll() is not None:
            stderr = process.stderr.read().decode('utf-8', errors='ignore') if process else ""
            logger.error(f"FFmpeg process not running: {stderr}")
            self.gui_update_queue.put(('status', f"Error: FFmpeg process not running: {stderr}"))
            self.stop_flag.set()
            return
        if self.stop_flag.is_set():
            logger.error("Stop flag already set before entering translation loop")
            self.gui_update_queue.put(('status', "Error: Translation stopped prematurely"))
            return
            
        vad = webrtcvad.Vad()
        vad.set_mode(2)
        ring_buffer = bytearray()
        buffer_data = bytearray()
        silence_counter = 0
        segment_start_time = None
        segment_id = 0

        READ_TIMEOUT = 30.0          # seconds without any new audio => consider EOF/stall
        POLL_INTERVAL = 1          # seconds to wait before checking process status again
        last_read_time = None
        print(f"last_read_time: {last_read_time}")

        # Watchdog thread to monitor process termination
        def watchdog():
            while last_read_time is None:
                time.sleep(1)  # Wait for initial read
            print(f"watchdog started, last_read_time: {last_read_time}")                
            while not self.stop_flag.is_set() and process.poll() is None:
                if time.time() - last_read_time > READ_TIMEOUT:
                   break
                time.sleep(POLL_INTERVAL)
            if not self.stop_flag.is_set():
                logger.info(f"Watchdog detected end of translation, last read time: {last_read_time}")
                self.translation_done.set()

        watchdog_thread = threading.Thread(target=watchdog, daemon=True)
        watchdog_thread.start()


        while process.poll() is None and not self.stop_flag.is_set() and not self.translation_done.is_set():
            try:
                frame = process.stdout.read(self.audio_processor.frame_size)

                if not frame:
                    continue

                raw_frame = frame  # Preserve raw frame for VAD
                if noise_level > 0:
                    frame = self.audio_processor.process_frame_with_noisereduce(frame, noise_level)
                self.audio_time_offset += self.audio_processor.frame_duration
                
                if not frame or len(frame) != self.audio_processor.frame_size:
                    logger.warning(f"Incomplete frame received: {len(frame)} bytes")
                    continue
                    
                is_speech = vad.is_speech(raw_frame, self.audio_processor.sample_rate)  # Use raw frame for VAD
                ring_buffer.extend(frame)
                
                if is_speech:
                    if segment_start_time is None:
                        segment_start_time = time.time()
                    buffer_data.extend(ring_buffer)
                    ring_buffer.clear()
                    silence_counter = 0
                else:
                    silence_counter += 1
                    if silence_counter >= self.audio_processor.silence_frames and buffer_data:
                        if len(buffer_data) < self.audio_processor.sample_rate * 2 * 0.3:
                            logger.info(f"Skipping too short buffer: {len(buffer_data)} bytes")
                            buffer_data.clear()
                            ring_buffer.clear()
                            silence_counter = 0
                            segment_start_time = None
                            self.audio_time_marker += self.audio_time_offset
                            self.audio_time_offset = 0
                            continue
                        segment_id += 1
                        try:
                            with self.audio_processor.temp_audio_file(buffer_data) as temp_file:
                                duration = len(buffer_data) / (self.audio_processor.sample_rate * 2)
                                logger.debug(f"Processing segment with duration: {duration:.2f}s")
                                transcript = self.translator.transcribe_audio(temp_file)
                                if not transcript.strip():
                                    logger.warning("Empty transcript returned by Whisper")
                                    buffer_data.clear()
                                    ring_buffer.clear()
                                    silence_counter = 0
                                    segment_start_time = None
                                    self.audio_time_marker += self.audio_time_offset
                                    self.audio_time_offset = 0
                                    continue
                                
                                try:
                                    self.gui_update_queue.put(('transcript', transcript), timeout=1)
                                    self.update_transcript_file(transcript)
                                except queue.Full:
                                    logger.warning("GUI update queue full, dropping transcript")
                                
                                try:
                                    translated_text = self.translator.translate_text(transcript, translation_service)
                                    try:
                                        self.gui_update_queue.put(('translation', translated_text), timeout=1)
                                    except queue.Full:
                                        logger.warning("GUI update queue full, dropping translation")
                                except Exception as e:
                                    logger.error(f"Translation error: {e}")
                                    buffer_data.clear()
                                    ring_buffer.clear()
                                    silence_counter = 0
                                    segment_start_time = None
                                    self.audio_time_marker += self.audio_time_offset
                                    self.audio_time_offset = 0
                                    continue
                                
                                try:
                                    audio_content = self.translator.synthesize_speech(
                                        translated_text, duration, voice_value
                                    )
                                except RuntimeError:
                                    logger.warning(f"Falling back to default voice: {self.config['VOICE_OPTIONS']['Male_Standard_B']}")
                                    audio_content = self.translator.synthesize_speech(
                                        translated_text, duration, self.config['VOICE_OPTIONS']['Male_Standard_B']
                                    )
                                if not self.first_translation_done.is_set():
                                    logger.info("Setting first_translation_done event")
                                    self.first_translation_done.set()
                                
                                playback_time = (self.video_start_time + self.audio_time_marker
                                                if self.video_start_time else time.time())
                                
                                #logger.info(f"Playback time: {playback_time:.2f}, Audio marker: {self.audio_time_marker:.2f}")
                                try:
                                    # Save audio to temporary file for File mode
                                    if self.output_mode == 'File':
                                        temp_dir = tempfile.gettempdir()
                                        logger.debug(f"Saving audio segment to temporary file in {temp_dir}")
                                        audio_file = os.path.join(temp_dir, f"audio_{segment_id}.wav")
                                        audio_array, sr = sf.read(io.BytesIO(audio_content), dtype="float32")
                                        sf.write(audio_file, audio_array, sr)
                                        self.audio_files.append((audio_file, playback_time))
                                        logger.info(f"Saved audio segment to {audio_file}")
                                        #flag = True

                                    self.audio_processor.playback_queue.put({
                                        'audio': audio_content,
                                        'text': translated_text,
                                        'play_at': playback_time
                                    }, timeout=1)
                                    last_read_time = time.time()
                                    print(f"Updated last_read_time: {last_read_time} for segment {segment_id}")

                                    self.update_translation_file(translated_text)

                                except queue.Full:
                                    logger.warning("Playback queue full, dropping audio")
                                except:
                                    logger.error("Error adding audio to playback queue")
                                    self.gui_update_queue.put(('status', "Error adding audio to playback queue"))
                                
                                # Update subtitle entries for File mode
                                if self.output_mode == 'File':
                                    caption_lines = []
                                    current_line = ""
                                    for char in translated_text:
                                        if len(current_line) >= 25:
                                            caption_lines.append(current_line)
                                            current_line = char
                                        else:
                                            current_line += char
                                    if current_line:
                                        caption_lines.append(current_line)
                                    
                                    line_duration = duration / len(caption_lines) if caption_lines else duration
                                    subtitle_start = self.audio_time_marker
                                    for line in caption_lines:
                                        self.subtitle_entries.append({
                                            'start': subtitle_start,
                                            'end': subtitle_start + line_duration,
                                            'text': line
                                        })
                                        subtitle_start += line_duration

                                # Clear buffers and reset state
                                buffer_data.clear()
                                ring_buffer.clear()
                                silence_counter = 0
                                segment_start_time = None
                                self.audio_time_marker += self.audio_time_offset
                                self.audio_time_offset = 0
                                logger.info(f"Processed segment {segment_id} with duration {duration:.2f}s")
                        except Exception as e:
                            logger.error(f"Error processing audio segment: {e}")
                            self.gui_update_queue.put(('status', f"Error processing audio: {e}"))
                            continue
            except Exception as e:
                logger.error(f"Translation worker error: {e}")
                self.gui_update_queue.put(('status', f"Error: {e}"))
                self.stop_flag.set()
                break
       
                
    def playback_worker(self):
        """Play translated audio and update captions."""
        #logger.info("Starting playback worker")
        empty_count = 0
        
        while not self.stop_flag.is_set():
            if self.video_start_time is None:
                time.sleep(0.1)
                continue
            
            self.gui_update_queue.put(('status', "Playing translation..."))
            try:
                item = self.audio_processor.playback_queue.get(timeout=5)
                audio_content = item['audio']
                translated_text = item['text']
                playback_time = item['play_at']
                
                while True:
                    now = time.time()
                    wait_time = playback_time - now
                    if wait_time <= 0:
                        break
                    time.sleep(min(wait_time, 0.05))
                
                caption_lines = []
                current_line = ""
                for char in translated_text:
                    if len(current_line) >= 25:
                        caption_lines.append(current_line)
                        current_line = char
                    else:
                        current_line += char
                if current_line:
                    caption_lines.append(current_line)
                
                audio_file = io.BytesIO(audio_content)
                data, samplerate = sf.read(audio_file, dtype='float32')
                total_duration = len(data) / samplerate
                line_duration = total_duration / len(caption_lines) if caption_lines else total_duration
                audio_thread = threading.Thread(
                    target=self.audio_processor.play_audio, args=(audio_content,)
                )
                caption_thread = threading.Thread(
                    target=self.update_captions, args=(caption_lines, line_duration)
                )

                audio_thread.start()
                caption_thread.start()
                audio_thread.join()
                caption_thread.join()
                empty_count = 0
            except queue.Empty:
                empty_count += 1
                if empty_count >= 5:
                    logger.info("Playback queue empty for too long. Stopping...")
                    self.stop_all()
                    break
                logger.debug("No audio to play. Waiting...")
                continue
            except Exception as e:
                logger.error(f"Playback worker error: {e}")
                self.gui_update_queue.put(('status', f"Error: {e}"))
                self.stop_flag.set()
                break

    def video_worker(self, youtube_url: str):
        """Process video stream and save to file with translated audio and subtitles."""
        if self.output_mode != 'File':
            return  # Only run for File mode
        logger.info(f"Waiting for translation to complete before starting video worker for URL: {youtube_url}")
        self.translation_done.wait()  # Wait for translation_worker to complete
        logger.info(f"Starting video worker for URL: {youtube_url}")

        try:
            import yt_dlp
            ydl_opts = {
                'format': 'bestvideo+bestaudio/best',
                'quiet': True,
                'no_warnings': True,
            }
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(youtube_url, download=False)
                #logger.debug(f"yt_dlp info keys: {list(info.keys())}")
                # Extract direct URL from requested_formats
                requested_formats = info.get('requested_formats')
                if requested_formats and isinstance(requested_formats, list):
                    # Get the video or audio stream URL (usually first is video, second is audio)
                    video_url = requested_formats[0].get('url')
                    if not video_url:
                        raise ValueError("Missing 'url' in requested_formats[0]")
                else:
                    raise ValueError("No 'requested_formats' found or it's not a list")

            logger.info(f"Extracted video URL: {video_url}")                
            # Generate ASS subtitle file
            self.generate_srt_subtitles()
            logger.info('Subtitle file created...')

          # FFmpeg command to mux video, audio, subtitles, and logo
            # 1920x1080 canvas, subs bottom-left with a semi-transparent box
            style = (
                "FontName=Microsoft YaHei,FontSize=22,"
                "PrimaryColour=&H00FFFFFF,"
                "BorderStyle=3,BackColour=&H99000000,Outline=3,Shadow=0,"
                "Alignment=1,WrapStyle=0,MarginL=25,MarginR=25,MarginV=20"
            )

            filter_complex = (
                # Resize first: fit inside 1920x1080, then pad to exact size
                "[0:v]scale=1920:1080:force_original_aspect_ratio=decrease,format=yuv420p,"
                "pad=1920:1080:(ow-iw)/2:(oh-ih)/2[v0];"
                # Burn subtitles on the 1920x1080 canvas
                f"[v0]subtitles=filename='{self.subtitle_file}':original_size=1920x1080:"
                f"force_style='{style}'[sub];"
                # Scale and overlay logo
                "[2:v]scale=90:90[logo];"
                "[sub][logo]overlay=56:56[vout]"
            )

            cmd = [
                "ffmpeg",
                "-re",
                "-i", video_url,                         # [0] video
                "-f", "concat", "-safe", "0",
                "-i", self.create_audio_concat_file(),   # [1] audio
                "-i", "logo.png",                        # [2] logo
                "-filter_complex", filter_complex,
                "-map", "[vout]",                        # map processed video
                "-map", "1:a:0?",                        # map your concat audio if present
                "-c:v", "libx264", "-preset", "fast",
                "-c:a", "aac",
                "-shortest",                             # end when the shortest stream ends
                "-y", self.output_file,
            ]

            print(f"ffmpeg command: {cmd}")
            process = subprocess.Popen(
                cmd,
                stderr=subprocess.PIPE,
                universal_newlines=True
            )
            self.ffmpeg_video_process = process
            
            # Monitor FFmpeg output
            while process.poll() is None and not self.stop_flag.is_set():
                stderr_line = process.stderr.readline()
                if stderr_line:
                    logger.debug(f"FFmpeg video: {stderr_line.strip()}")
                time.sleep(0.1)
                
            stderr = process.stderr.read()
            if process.poll() != 0:
                logger.error(f"FFmpeg video processing failed: {stderr}")
                #self.gui_update_queue.put(('status', f"Error saving video: {stderr}"))
            else:
                logger.info(f"Video saved to {self.output_file}")
                self.gui_update_queue.put(('status', f"Saved video to {self.output_file}"))
                
        except Exception as e:
            logger.error(f"Video worker error: {e}")
            #self.gui_update_queue.put(('status', f"Error: {e}"))
            self.stop_flag.set()

    def create_audio_concat_file(self):
        """Create a concat file for audio segments."""
        concat_file = os.path.join(tempfile.gettempdir(), f"concat_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
        with open(concat_file, 'w', encoding='utf-8') as f:
            for audio_file, _ in self.audio_files:
                f.write(f"file '{audio_file}'\n")
        return concat_file
        
    def generate_srt_subtitles(self):
        """Generate ASS subtitle file from subtitle entries."""
        # with open(self.subtitle_file, 'w', encoding='utf-8') as f:
        #     f.write("[Script Info]\nScriptType: v4.00+\n\n[V4+ Styles]\nFormat: Name, Fontname, Fontsize, PrimaryColour, Alignment, MarginV\nStyle: Default,Arial,24,&H00FFFF&,2,50\n\n[Events]\nFormat: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n")
        #     for entry in self.subtitle_entries:
        #         start = entry['start']
        #         end = entry['end']
        #         text = entry['text'].replace('\n', '\\N')
        #         start_str = f"{int(start//3600):02d}:{int((start%3600)//60):02d}:{start%60:06.2f}"
        #         end_str = f"{int(end//3600):02d}:{int((end%3600)//60):02d}:{end%60:06.2f}"
        #         f.write(f"Dialogue: 0,{start_str},{end_str},Default,,0,0,0,,{text}\n")
        with open(self.subtitle_file, 'w', encoding='utf-8') as f:
            for i, entry in enumerate(self.subtitle_entries, 1):
                start = entry['start']
                end = entry['end']
                text = entry['text'].replace('\n', ' ')
                start_str = f"{int(start//3600):02d}:{int((start%3600)//60):02d}:{int(start%60):02d},{int((start%1)*1000):03d}"
                end_str = f"{int(end//3600):02d}:{int((end%3600)//60):02d}:{int(end%60):02d},{int((end%1)*1000):03d}"
                f.write(f"{i}\n{start_str} --> {end_str}\n{text}\n\n")
        logger.info(f"Generated SRT subtitles at {self.subtitle_file}")

    def update_captions(self, caption_lines: list, line_duration: float):
        """Update caption file with text lines over time.
        
        Args:
            caption_lines (list): List of caption lines.
            line_duration (float): Duration per line in seconds.
        """
        caption_lines = [re.sub(r'[，。]', '  ', line) for line in caption_lines if line.strip()]
        for line in caption_lines:
            if self.stop_flag.is_set():
                break
            self.update_caption_file(line)
            time.sleep(line_duration)
        self.update_caption_file("")
        
    def update_transcript_file(self, text: str):
        """Add text to the transcript file.
        
        Args:
            text (str): Text to write.
        """
        try:
            with open(self.original_transcript_file, "a", encoding="utf-8") as f:
                f.write(text + "\n")
        except IOError as e:
            logger.error(f"Failed to write to transcript file: {e}")
            
    def update_translation_file(self, text: str):
        """Add text to the translation file.
        
        Args:
            text (str): Text to write.
        """
        try:
            with open(self.full_translation_file, "a", encoding="utf-8") as f:
                f.write(text + "\n")
        except IOError as e:
            logger.error(f"Failed to write to translation file: {e}")
            
    def update_caption_file(self, text: str):
        """Write text to the caption file for OBS.
        
        Args:
            text (str): Caption text.
        """
        try:
            with open(self.caption_file, "w", encoding="utf-8") as f:
                f.write(text)
        except IOError as e:
            logger.error(f"Failed to update caption file: {e}")
        except UnicodeEncodeError as e:
            logger.error(f"Unicode encoding error in caption file: {e}")
            
    def clear_caption_file(self):
        """Clear the caption file."""
        try:
            with open(self.caption_file, "w", encoding="utf-8") as f:
                f.write("")
        except IOError as e:
            logger.error(f"Failed to clear caption file: {e}")
            
    def start(self, youtube_url: str, translation_service: str, voice_value: str, noise_level: float, output_mode: str = None):
        """Start the translation process.
        
        Args:
            youtube_url (str): YouTube video URL.
            translation_service (str): Translation service ('Google' or 'Local').
            voice_value (str): TTS voice value (e.g., 'cmn-CN-Standard-B').
            noise_level (float): Noise reduction level.
            output_mode (str, optional): Output mode ('OBS' or 'File').            
        """
        logger.info(f"Starting translation system with URL: {youtube_url}, service: {translation_service}, voice: {voice_value}, noise_level: {noise_level}, output_mode: {output_mode or self.output_mode}")
        self.stop_flag.clear()
        self.first_translation_done.clear()
        self.translation_done.clear()  # Reset translation_done event
        self.video_start_time = None
        self.audio_time_marker = 0
        self.audio_time_offset = 0
        self.worker_threads = []
        self.subtitle_entries = []
        self.worker_threads = []
        self.full_translation_file = f"{self.config['DEFAULT']['full_translation_file']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        self.original_transcript_file = f"{self.config['DEFAULT']['original_transcript_file']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        self.output_mode = output_mode or self.config['DEFAULT']['output_mode']
        
        self.translator.initialize_translation_service(translation_service)
        self.clear_caption_file()
        
        audio_url = self.audio_processor.get_audio_url(youtube_url)
        if not audio_url:
            logger.error("Failed to get audio URL")
            self.gui_update_queue.put(('status', "Failed to get audio URL."))
            return
            
        self.audio_stream_start_time = time.time()
        process = self.audio_processor.start_ffmpeg_stream(audio_url)
        if not process or process.poll() is not None:
            stderr = process.stderr.read().decode('utf-8', errors='ignore') if process else ""
            logger.error(f"FFmpeg stream failed to start: {stderr}")
            self.gui_update_queue.put(('status', f"Failed to start FFmpeg stream: {stderr}"))
            return
            
        logger.info("Starting worker threads...")
        translation_thread = threading.Thread(
            target=self.translation_worker,
            args=(process, youtube_url, translation_service, voice_value, noise_level),
            daemon=True
        )
        translation_thread.start()
        self.worker_threads.append(translation_thread)
        
        playback_thread = threading.Thread(
            target=self.playback_worker,
            daemon=True
        )
        playback_thread.start()
        self.worker_threads.append(playback_thread)
        
        if self.output_mode == 'OBS':
            obs_thread = threading.Thread(
                target=self.obs_controller.setup_video,
                args=(youtube_url, self, self.first_translation_done),
                daemon=True
            )
            obs_thread.start()
            self.worker_threads.append(obs_thread)
        else:
            video_thread = threading.Thread(
                target=self.video_worker,
                args=(youtube_url,),
                daemon=True
            )
            video_thread.start()
            self.worker_threads.append(video_thread)
        
        logger.info("All worker threads started")
        
    def stop_all(self):
        """Stop all processes and clean up."""
        logger.info("Stopping all processes and threads")
        self.stop_flag.set()
        self.gui_update_queue.put(('status', "Stopping..."))
        
        # Clear playback queue
        try:
            while True:
                self.audio_processor.playback_queue.get_nowait()
                self.audio_processor.playback_queue.task_done()
        except queue.Empty:
            logger.info("Playback queue cleared")
        except Exception as e:
            logger.error(f"Error clearing playback queue: {e}")
            self.gui_update_queue.put(('status', f"Error clearing playback queue: {e}"))
        
        if self.audio_processor.ffmpeg_audio_process and self.audio_processor.ffmpeg_audio_process.poll() is None:
            try:
                if sys.platform == "win32":
                    self.audio_processor.ffmpeg_audio_process.terminate()
                    self.audio_processor.ffmpeg_audio_process.wait(timeout=2)
                else:
                    self.audio_processor.ffmpeg_audio_process.send_signal(signal.SIGINT)
                    self.audio_processor.ffmpeg_audio_process.wait(timeout=2)
                logger.info("Audio FFmpeg process terminated.")
            except subprocess.TimeoutExpired:
                self.audio_processor.ffmpeg_audio_process.kill()
                logger.warning("Audio FFmpeg process killed.")
            except Exception as e:
                logger.error(f"Error stopping audio process: {e}")
                
        if self.output_mode == 'File' and hasattr(self, 'ffmpeg_video_process') and self.ffmpeg_video_process and self.ffmpeg_video_process.poll() is None:
            try:
                if sys.platform == "win32":
                    self.ffmpeg_video_process.terminate()
                    self.ffmpeg_video_process.wait(timeout=2)
                else:
                    self.ffmpeg_video_process.send_signal(signal.SIGINT)
                    self.ffmpeg_video_process.wait(timeout=2)
                logger.info("Video FFmpeg process terminated.")
            except subprocess.TimeoutExpired:
                self.ffmpeg_video_process.kill()
                logger.warning("Video FFmpeg process killed.")
            except Exception as e:
                logger.error(f"Error stopping video process: {e}")
                self.gui_update_queue.put(('status', f"Error stopping video: {e}"))
                
        if self.output_mode == 'OBS':
            try:
                self.obs_controller.cleanup()
            except Exception as e:
                logger.error(f"Error during OBS cleanup: {e}")
                self.gui_update_queue.put(('status', f"Error during OBS cleanup: {e}"))
                
        for thread in self.worker_threads:
            if thread.is_alive():
                thread.join(timeout=2)
        self.worker_threads = []
        
        # Clean up temporary files for File mode
        #if self.output_mode == 'File':
        #    for audio_file, _ in self.audio_files:
        #        try:
        #            os.remove(audio_file)
        #        except OSError:
        #            pass
        #    if os.path.exists(self.subtitle_file):
        #        try:
        #            os.remove(self.subtitle_file)
        #        except OSError:
        #            pass
                
        logger.info("All threads and processes stopped")

def main():
    """Main entry point for the translation system."""
    config = configparser.ConfigParser()
    config_file = 'config.ini'
    try:
        logger.info(f"Loading config file from: {config_file}")
        if not config.read(config_file, encoding='utf-8'):
            logger.error(f"Failed to read config file: {config_file}")
            raise FileNotFoundError(f"Config file {config_file} not found or empty")
        logger.info("Successfully loaded config.ini")
    except Exception as e:
        logger.error(f"Error loading config: {e}")
        raise
    
    system = TranslationSystem(config)
    gui = TranslationGUI(system.start, system.stop_all, config)
    logger.info("GUI and TranslationSystem initialized")
    gui.run()

if __name__ == "__main__":
    main()