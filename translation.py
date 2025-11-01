import logging
from google.cloud import translate_v3 as translate, texttospeech
import whisper
from transformers import pipeline
import os
import soundfile as sf
import io
import numpy as np

logger = logging.getLogger(__name__)

class Translator:
    """Handles transcription, translation, and text-to-speech synthesis."""
    
    def __init__(self, config):
        """Initialize the translator with configuration.
        
        Args:
            config (ConfigParser): Configuration object.
        """
        self.config = config
        self.project_id = config['DEFAULT']['project_id']
        self.location = config['DEFAULT']['location']
        self.glossary_id = config['DEFAULT']['glossary_id']
        self.target_language = config['DEFAULT'].get('target_language', 'zh-CN')  # Default to Chinese
        self.google_client = None
        self.local_model = None
        self.whisper_model = None
        self.tts_client = texttospeech.TextToSpeechClient()

        
    def initialize_translation_service(self, service: str):
        """Initialize the translation service.
        
        Args:
            service (str): 'Google' for Google Cloud Translate, 'Local' for local model.
        """
        logger.info(f"Initializing translation service: {service}")
        if service == 'Google':
            try:
                self.google_client = translate.TranslationServiceClient()
                logger.info("Google Cloud Translate client initialized")
            except Exception as e:
                logger.error(f"Failed to initialize Google Cloud Translate: {e}")
                raise
        elif service == 'Local':
            try:
                model_path = self.config['DEFAULT']['local_translation_model']
                self.local_model = pipeline("translation", model=model_path, tokenizer=model_path)
                logger.info(f"Local translation model loaded: {model_path}")
            except Exception as e:
                logger.error(f"Failed to load local translation model: {e}")
                raise
        else:
            logger.error(f"Unknown translation service: {service}")
            raise ValueError(f"Unknown translation service: {service}")
            
    def transcribe_audio(self, audio_file: str) -> str:
        """Transcribe audio file using Whisper.
        
        Args:
            audio_file (str): Path to the audio file.
            
        Returns:
            str: Transcribed text.
        """
        try:
            if not os.path.exists(audio_file):
                logger.error(f"Audio file does not exist: {audio_file}")
                return ""
            if os.path.getsize(audio_file) < 100:
                logger.warning(f"Audio file too small: {audio_file}, size={os.path.getsize(audio_file)} bytes")
                return ""
            if self.whisper_model is None:
                self.whisper_model = whisper.load_model("large")
                logger.info("Whisper model loaded")

            result = self.whisper_model.transcribe(audio_file, fp16=False)
            transcript = result["text"].strip()
            logger.info(f"[Whisper STT] {transcript}")
            return transcript
        except Exception as e:
            logger.error(f"Transcription error: {e}")
            raise
            
    def translate_text(self, text: str, service: str) -> str:
        """Translate text using the specified service.
        
        Args:
            text (str): Text to translate.
            service (str): 'Google' or 'Local'.
            
        Returns:
            str: Translated text.
        """
        logger.info(f"Translating text: {text}")
        try:
            if not text.strip():
                logger.warning("Empty text provided for translation")
                return ""

            if service == 'Google':
                if not self.google_client:
                    raise ValueError("Google Cloud Translate client not initialized")
                parent = f"projects/{self.project_id}/locations/{self.location}"
                if self.glossary_id:
                    glossary_path = f"projects/{self.project_id}/locations/{self.location}/glossaries/{self.glossary_id}"
                    response = self.google_client.translate_text(
                        request={
                            "parent": parent,
                            "contents": [text],
                            "mime_type": "text/plain",
                            "source_language_code": "en",
                            "target_language_code": self.target_language,
                            "glossary_config": {
                                "glossary": glossary_path
                            }
                        }
                    )
                    translations = [t.translated_text for t in response.glossary_translations]
                    translated_text = translations[0] if translations else None
                    if translated_text is None:
                        logger.warning("No glossary translation returned, falling back to standard translation")
                        response = self.google_client.translate_text(
                            contents=[text],
                            target_language_code=self.target_language,
                            parent=parent,
                            mime_type="text/plain"
                        )
                        translated_text = ''.join([t.translated_text for t in response.translations])
                else:
                    response = self.google_client.translate_text(
                        contents=[text],
                        target_language_code=self.target_language,
                        parent=parent,
                        mime_type="text/plain"
                    )
                    translated_text = ''.join([t.translated_text for t in response.translations])
                logger.info(f"[Google Translate] {translated_text}")
                return translated_text
            elif service == 'Local':
                if not self.local_model:
                    raise ValueError("Local translation model not initialized")
                result = self.local_model(text, max_length=400)
                translated_text = result[0]['translation_text']
                logger.info(f"[Local Translate] {translated_text}")
                return translated_text
            else:
                logger.error(f"Unknown translation service: {service}")
                raise ValueError(f"Unknown translation service: {service}")
        except Exception as e:
            logger.error(f"Translation error: {e}")
            raise
            
    def synthesize_speech(self, text: str, target_duration: float, voice_value: str) -> bytes:
        """Synthesize speech for the given text with duration adjustment.
        
        Args:
            text (str): Text to synthesize.
            target_duration (float): Desired duration in seconds.
            voice_value (str): Voice identifier (e.g., 'cmn-CN-Standard-B').
            
        Returns:
            bytes: Audio content in WAV format.
        """
        logger.info(f"Synthesizing speech for text: {text}, voice: {voice_value}, target_duration: {target_duration:.2f}s")
        try:
            if not text.strip():
                logger.warning("Empty text provided for speech synthesis")
                return b""
            synthesis_input = texttospeech.SynthesisInput(text=text)
            voice = texttospeech.VoiceSelectionParams(
                language_code=voice_value.split('-')[0] + '-' + voice_value.split('-')[1],
                name=voice_value
            )
            audio_config = texttospeech.AudioConfig(
                audio_encoding=texttospeech.AudioEncoding.LINEAR16,  # Match original
                speaking_rate=1.0
            )
            response = self.tts_client.synthesize_speech(
                input=synthesis_input,
                voice=voice,
                audio_config=audio_config
            )
            audio_file = io.BytesIO(response.audio_content)
            data, samplerate = sf.read(audio_file, dtype='float32')
            actual_duration = len(data) / samplerate
            
            speaking_rate = actual_duration / target_duration if target_duration > 0 else 1.0
            if speaking_rate < 1.0:  # Too slow
                speaking_rate = 1.0
                audio_config.speaking_rate = speaking_rate
                response = self.tts_client.synthesize_speech(
                    input=synthesis_input,
                    voice=voice,
                    audio_config=audio_config
                )
                audio_file = io.BytesIO(response.audio_content)
                data, samplerate = sf.read(audio_file, dtype='float32')
                actual_duration = len(data) / samplerate
                
                silence_duration = target_duration - actual_duration
                if silence_duration > 0:
                    silence_samples = int(silence_duration * samplerate)
                    silence = np.zeros((silence_samples, data.shape[1]) if data.ndim > 1 else silence_samples, dtype='float32')
                    data = np.concatenate((data, silence))
                
                output_buffer = io.BytesIO()
                sf.write(output_buffer, data, samplerate, format='WAV')
                output_buffer.seek(0)
                logger.info(f"[TTS] Target: {target_duration:.2f}s, Actual: {actual_duration:.2f}s, Rate: {speaking_rate:.2f}, Added silence: {silence_duration:.2f}s")
                return output_buffer.read()
            else:  # Too fast
                speaking_rate = min(speaking_rate, 2.0)
                audio_config.speaking_rate = speaking_rate
                response = self.tts_client.synthesize_speech(
                    input=synthesis_input,
                    voice=voice,
                    audio_config=audio_config
                )
                logger.info(f"[TTS] Target: {target_duration:.2f}s, Actual: {actual_duration:.2f}s, Rate: {speaking_rate:.2f}")
                return response.audio_content
        except Exception as e:
            logger.error(f"Speech synthesis error: {e}")
            raise