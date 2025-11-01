import subprocess
import time
import logging
from threading import Event
from contextlib import contextmanager
import yt_dlp
from PIL import Image
import sys
try:
    from obswebsocket import obsws, requests
except ImportError:
    import obswebsocket
    from obswebsocket import requests

logger = logging.getLogger(__name__)

@contextmanager
def obs_websocket(host, port, password):
    """Context manager for OBS WebSocket connection."""
    ws = obsws(host, port, password)
    ws.connect()
    try:
        yield ws
    finally:
        ws.disconnect()

class OBSController:
    """Handles OBS WebSocket interactions and video streaming."""
    
    def __init__(self, config):
        """Initialize OBS controller.
        
        Args:
            config (ConfigParser): Configuration object.
        """
        self.host = config['DEFAULT']['obs_host']
        self.port = int(config['DEFAULT']['obs_port'])
        self.password = config['DEFAULT']['obs_password']
        self.scene = config['DEFAULT']['obs_scene']
        self.logo_file = config['DEFAULT']['logo_file']
        self.caption_file = config['DEFAULT']['caption_file']
        self.logo_pos_x = int(config['DEFAULT']['logo_pos_x'])
        self.logo_pos_y = int(config['DEFAULT']['logo_pos_y'])
        self.logo_size_x = int(config['DEFAULT']['logo_size_x'])
        self.logo_size_y = int(config['DEFAULT']['logo_size_y'])
        self.ffmpeg_path = config['DEFAULT']['ffmpeg_path']
        self.ffmpeg_video_process = None
        
    def setup_video(self, youtube_url: str, system, first_translation_done: Event):
        """Set up video source in OBS and start streaming after first translation is done.

        Args:
            youtube_url (str): YouTube video URL.
            system (TranslationSystem): Object that will store video start time.
            first_translation_done (Event): Event to wait for first translation.
        """
        logger.info("Starting OBS video setup")
        stream_url = None
        video_input_name = "YouTube"
        text_input_name = "Subtitle_auto"
        image_input_name = "Logo"

        try:
            # Step 1: Get the best stream URL using yt-dlp
            logger.info("Fetching video stream URL using yt-dlp")
            process = subprocess.Popen(
                ["yt-dlp", "-g", "-f", "best", youtube_url],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            stream_url_raw, _ = process.communicate()

            if not stream_url_raw:
                raise RuntimeError("Failed to get stream URL from yt-dlp.")
            stream_url = stream_url_raw.decode().strip()

            # Step 2: Connect to OBS WebSocket
            with obs_websocket(self.host, self.port, self.password) as ws:
                logger.info("Connected to OBS WebSocket")

                # Step 3: Wait for translation
                logger.info("Waiting for first translation to complete...")
                first_translation_done.wait()
                logger.info("Translation complete, proceeding with playback")

                # Step 4: Get canvas size
                video_settings = ws.call(requests.GetVideoSettings())
                canvas_width = video_settings.getBaseWidth()
                canvas_height = video_settings.getBaseHeight()

                # Step 5: Create ffmpeg video input
                logger.info("Creating video input in OBS")
                ws.call(requests.CreateInput(
                    sceneName=self.scene,
                    inputName=video_input_name,
                    inputKind="ffmpeg_source",
                    inputSettings={
                        "is_local_file": False,
                        "input": stream_url,
                        "looping": False,
                        "clear_on_media_end": False,
                        "restart_on_activate": True,
                        "close_when_inactive": False
                    }
                ))

                # Set video volume
                logger.info("Setting video input volume")
                ws.call(requests.SetInputVolume(
                    inputName=video_input_name,
                    inputVolumeMul=0.3
                ))

                # Scale video to fit canvas
                response = ws.call(requests.GetSceneItemId(sceneName=self.scene, sourceName=video_input_name))
                scene_item_id = response.getSceneItemId()

                source_settings = ws.call(requests.GetSourceSettings(sourceName=video_input_name))
                source_width = source_settings.datain.get("width", 640)  # Default to 640 if not found
                source_height = source_settings.datain.get("height", 360)  # Default to 360 if not found
                # Apply the transformation directly
                ws.call(requests.SetSceneItemTransform(
                    sceneName=self.scene,
                    sceneItemId=scene_item_id,
                    sceneItemTransform={
                        "positionX": 0.0,
                        "positionY": 0.0,
                        "rotation": 0.0,
                        "scaleX": canvas_width / source_width,  # Adjust scale to canvas width
                        "scaleY": canvas_height / source_height,  # Adjust scale to canvas height
                        "cropTop": 0,
                        "cropBottom": 0,
                        "cropLeft": 0,
                        "cropRight": 0
                    }
                ))

                ws.call(requests.SetSceneItemBounds(
                    sceneName=self.scene,
                    sceneItemId=scene_item_id,
                    boundsType=3,
                    boundsAlignment=0,
                    boundsWidth=canvas_width,
                    boundsHeight=canvas_height
                ))

                # Step 6: Create subtitle input
                logger.info("Creating subtitle input")
                ws.call(requests.CreateInput(
                    sceneName=self.scene,
                    inputName=text_input_name,
                    inputKind="text_gdiplus_v3",
                    inputSettings={
                        "read_from_file": True,
                        "file": self.caption_file,
                        "font": {
                            "face": "Microsoft YaHei",
                            "size": 86
                        },
                        "color1": 0xFFFFFFFF,  # White text
                        "background": True,
                        "bk_color": 0xFF000000,  # Opaque black
                        "bk_opacity": 90,
                        "outline": False,
                        "alignment": 0,  # Default (top-left) to avoid alignment conflicts
                        "extents": True,
                        "use_custom_extents": True,
                        "extents_cx": 1680,  # Wrap long lines
                        "extents_cy": 100  # Wrap long lines
                    }
                ))

                response = ws.call(requests.GetSceneItemId(sceneName=self.scene, sourceName=text_input_name))
                subtitle_item_id = response.getSceneItemId()

                ws.call(requests.SetSceneItemTransform(
                    sceneName=self.scene,
                    sceneItemId=subtitle_item_id,
                    sceneItemTransform={
                        "positionX": 120,
                        "positionY": 950
                    }
                ))

                # Step 7: Create logo input
                logger.info("Creating logo input")
                ws.call(requests.CreateInput(
                    sceneName=self.scene,
                    inputName=image_input_name,
                    inputKind="image_source",
                    inputSettings={
                        "file": self.logo_file
                    }
                ))

                response = ws.call(requests.GetSceneItemId(sceneName=self.scene, sourceName=image_input_name))
                logo_item_id = response.getSceneItemId()

                img_width, img_height = Image.open(self.logo_file).size
                ws.call(requests.SetSceneItemTransform(
                    sceneName=self.scene,
                    sceneItemId=logo_item_id,
                    sceneItemTransform={
                        "positionX": self.logo_pos_x,
                        "positionY": self.logo_pos_y,
                        "scaleX": self.logo_size_x / img_width,
                        "scaleY": self.logo_size_y / img_height,
                        "rotation": 0.0
                    }
                ))

                # Step 8: Set start time and register end event
                system.video_start_time = time.time()
                logger.info(f"Video playback started at {system.video_start_time}")

                def on_media_ended(event):
                    if event["inputName"] == video_input_name:
                        logger.info("Media ended, removing video input")
                        ws.call(requests.RemoveInput(inputName=video_input_name))

                ws.register(on_media_ended, "MediaInputEnded")

        except Exception as e:
            logger.error(f"OBS video setup error: {e}")
            raise

            
    def cleanup(self):
        """
        Clean up OBS inputs and stop FFmpeg.
        
        """
        with obs_websocket(self.host, self.port, self.password) as ws:
            try:
                response = ws.call(requests.RemoveInput(inputName="YouTube"))
                if response.status:
                    logger.debug("Removed YouTube_Video input")
                else:
                    logger.warning(f"Failed to remove YouTube_Video input: {response.datain['requestStatus']}")
                    
                response = ws.call(requests.RemoveInput(inputName="Logo"))
                if response.status:
                    logger.debug("Removed Logo input")
                else:
                    logger.warning(f"Failed to remove Logo input: {response.datain['requestStatus']}")

                response = ws.call(requests.RemoveInput(inputName="Subtitle_auto"))
                if response.status:
                    logger.debug("Removed subtitle input")
                else:
                    logger.warning(f"Failed to remove subtitle input: {response.datain['requestStatus']}")

            except Exception as e:
                logger.error(f"Error removing inputs: {e}")
            
        if self.ffmpeg_video_process and self.ffmpeg_video_process.poll() is None:
            try:
                self.ffmpeg_video_process.terminate()
                self.ffmpeg_video_process.wait(timeout=2)
                logger.info("FFmpeg video process terminated")
            except subprocess.TimeoutExpired:
                self.ffmpeg_video_process.kill()
                logger.warning("FFmpeg video process killed")
            except Exception as e:
                logger.error(f"Error stopping FFmpeg video process: {e}")
            self.ffmpeg_video_process = None