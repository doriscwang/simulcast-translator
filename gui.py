# Updated on 07/21/2025 (8:35 PM) to provide options for outputing the media: to OBS or to a file.
import tkinter as tk
from tkinter import messagebox
import re
import logging
import queue

logger = logging.getLogger(__name__)

class TranslationGUI:
    """Tkinter GUI for the YouTube translation system."""
    
    def __init__(self, start_callback, stop_callback, config):
        """Initialize the GUI.
        
        Args:
            start_callback (callable): Callback for start button.
            stop_callback (callable): Callback for stop button.
            config (ConfigParser): Configuration object.
        """
        self.root = tk.Tk()
        self.start_callback = start_callback
        self.stop_callback = stop_callback
        self.config = config
        self.is_translating = False
        try:
            if 'VOICE_OPTIONS' not in config:
                logger.error("VOICE_OPTIONS section missing in config.ini")
                raise KeyError("VOICE_OPTIONS section not found in config.ini")
            # Explicitly create a dictionary with only VOICE_OPTIONS keys
            self.voice_options = {key: config['VOICE_OPTIONS'][key] for key in config['VOICE_OPTIONS'].keys() if key not in config['DEFAULT']}
            if not self.voice_options:
                logger.warning("VOICE_OPTIONS section is empty. Using default voice.")
                self.voice_options = {'Default_Voice': 'cmn-CN-Standard-B'}
        except KeyError as e:
            logger.error(f"Error loading voice options: {e}")
            self.voice_options = {'Default_Voice': 'cmn-CN-Standard-B'}
            self.start_callback = lambda *args: self.root.after(0, lambda: messagebox.showerror("Error", "VOICE_OPTIONS section missing or invalid in config.ini. Using default voice."))
        
        self.status_var = tk.StringVar(value="Idle")
        self.transcript_var = tk.StringVar(value="")
        self.translation_var = tk.StringVar(value="")
        self.voice_var = tk.StringVar(value=list(self.voice_options.keys())[0])  # Set first voice as default
        self.translation_service = tk.StringVar(value="Google")
        self.output_mode = tk.StringVar(value=config['DEFAULT'].get('output_mode', 'OBS'))
        self.setup_gui()
        self.process_queue()  # Start polling the update queue
        
    def setup_gui(self):
        """Set up the GUI components."""
        self.root.protocol("WM_DELETE_WINDOW", self.stop_callback)
        self.root.title("Live YouTube Translator")
        
        tk.Label(self.root, text="YouTube URL:").grid(row=0, column=0, sticky="e", padx=10, pady=5)
        self.youtube_url_entry = tk.Entry(self.root, width=50)
        self.youtube_url_entry.grid(row=0, column=1, padx=10, pady=5)
        
        tk.Label(self.root, text="Translation Service:").grid(row=1, column=0, sticky="w", padx=10)
        tk.Radiobutton(
            self.root, text="Google Cloud", variable=self.translation_service, value="Google"
        ).grid(row=2, column=0, sticky="w", padx=10)
        tk.Radiobutton(
            self.root, text="Local Model", variable=self.translation_service, value="Local"
        ).grid(row=2, column=1, sticky="w", padx=10)
        
        tk.Label(self.root, text="Voice:").grid(row=3, column=0, sticky="e", padx=10, pady=5)
        tk.OptionMenu(
            self.root, self.voice_var, *self.voice_options.keys(),
            command=lambda v: logger.info(f"Selected voice: {v}")
        ).grid(row=3, column=1, padx=10, pady=5, sticky="w")
        
        tk.Label(self.root, text="Output Mode:").grid(row=4, column=0, sticky="e", padx=10, pady=5)
        tk.OptionMenu(self.root, self.output_mode, "OBS", "File", command=lambda v: logger.info(f"Selected output mode: {v}")
        ).grid(row=4, column=1, padx=10, pady=5, sticky="w")

        tk.Button(self.root, text="Start Translation", command=self.start_button_action).grid(
            row=5, column=0, padx=10, pady=10
        )
        tk.Button(self.root, text="Stop", command=self.stop_callback).grid(
            row=5, column=1, padx=10, pady=10
        )
        
        tk.Button(self.root, text="Close", command=self.close_action).grid(
            row=5, column=2, padx=10, pady=10
        )

        tk.Label(self.root, textvariable=self.status_var).grid(row=6, column=0, columnspan=3, pady=5)
        tk.Label(self.root, text="Transcript:").grid(row=7, column=0, sticky="ne", padx=10)
        tk.Label(
            self.root, textvariable=self.transcript_var, wraplength=400, justify="left"
        ).grid(row=7, column=1, columnspan=2, padx=10, pady=5, sticky="w")
        tk.Label(self.root, text="Translation:").grid(row=8, column=0, sticky="ne", padx=10)
        tk.Label(
            self.root, textvariable=self.translation_var, wraplength=400, justify="left"
        ).grid(row=8, column=1, columnspan=2, padx=10, pady=5, sticky="w")
        
        self.noise_slider = tk.Scale(
            self.root, from_=0, to=1, resolution=0.05, orient="horizontal", label="Noise Level", length=300
        )
        self.noise_slider.grid(row=9, column=0, columnspan=3, padx=10, pady=10)
        
    def process_queue(self):
        """Process GUI update queue periodically."""
        updated = False
        try:
            for _ in range(10):  # Process up to 10 items to avoid blocking
                try:
                    update_type, value = self.start_callback.__self__.gui_update_queue.get_nowait()
                    logger.debug(f"Processing GUI update: {update_type} = {value}")
                    if update_type == 'status':
                        self.status_var.set(value)
                        if value == "Stopping...":
                            self.is_translating = False
                    elif update_type == 'transcript':
                        self.transcript_var.set(value)
                    elif update_type == 'translation':
                        self.translation_var.set(value)
                    elif update_type == 'error':
                        self.root.after(0, lambda: messagebox.showerror("Error", value))
                    updated = True
                except queue.Empty:
                    break
        except Exception as e:
            logger.error(f"GUI update error: {e}")
        if updated:
            logger.debug("GUI updated successfully")
        self.root.after(20, self.process_queue)  # Poll every 20ms
        
    def start_button_action(self):
        """Handle start button click with URL validation."""
        youtube_url = self.youtube_url_entry.get()
        if not re.match(r'^https?://(www\.)?(youtube\.com|youtu\.be)/', youtube_url):
            self.start_callback.__self__.gui_update_queue.put(('error', "Please enter a valid YouTube URL."))
            return
        output_mode = self.output_mode.get()
        status_message = "Streaming to OBS..." if output_mode == 'OBS' else f"Saving to {self.start_callback.__self__.output_file}..."
        self.start_callback.__self__.gui_update_queue.put(('status', status_message))
        self.is_translating = True
        self.start_callback(
            youtube_url, self.translation_service.get(), self.voice_options[self.voice_var.get()], self.noise_slider.get(), self.output_mode.get()
        )
        
    def update_status(self, status: str):
        """Update the status label.
        
        Args:
            status (str): Status message.
        """
        self.start_callback.__self__.gui_update_queue.put(('status', status))
        
    def update_transcript(self, transcript: str):
        """Update the transcript display.
        
        Args:
            transcript (str): Transcribed text.
        """
        self.start_callback.__self__.gui_update_queue.put(('transcript', transcript))
        
    def update_translation(self, translation: str):
        """Update the translation display.
        
        Args:
            translation (str): Translated text.
        """
        self.start_callback.__self__.gui_update_queue.put(('translation', translation))

    def close_action(self):
        """Handle close button click or window close event."""
        logger.info("Close action triggered")
        try:
            if self.is_translating:
                logger.info("Translation is running, calling stop_callback")
                self.stop_callback()
                # Wait briefly to ensure cleanup completes
                self.root.after(100, self.finalize_close)
            else:
                logger.info("No translation running, closing GUI")
                self.finalize_close()
        except Exception as e:
            logger.error(f"Error during close action: {e}")
            self.finalize_close()
            
    def finalize_close(self):
        """Finalize GUI closure."""
        logger.info("Finalizing GUI closure")
        try:
            self.root.quit()
            self.root.destroy()
        except Exception as e:
            logger.error(f"Error destroying GUI: {e}")        
        
    def run(self):
        """Start the GUI main loop."""
        self.root.mainloop()