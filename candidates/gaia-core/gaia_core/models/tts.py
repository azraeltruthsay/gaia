"""
Text-to-speech module for GAIA D&D Campaign Assistant.
Handles all speech synthesis functionality.
"""

import logging
import pyttsx3
from typing import Optional, List

# Get the logger
logger = logging.getLogger("GAIA")

class SpeechManager:
    """Manages text-to-speech functionality."""
    
    def __init__(self, config):
        """
        Initialize with configuration.
        
        Args:
            config: Configuration object
        """
        self.config = config
        self.tts_engine = None
        self.initialize()
    
    def initialize(self) -> bool:
        """
        Initialize the text-to-speech engine.
        
        Returns:
            True if initialization is successful, False otherwise
        """
        try:
            self.tts_engine = pyttsx3.init()
            voices = self.tts_engine.getProperty('voices')
            
            logger.info("Text-to-speech engine initialized")
            print("\nAvailable Text-to-Speech Voices:")
            for index, voice in enumerate(voices):
                print(f"{index + 1}. Name: {voice.name}, ID: {voice.id}")
            
            self._select_voice(voices)
            return True
        except Exception as e:
            logger.warning(f"Could not initialize text-to-speech engine: {e}")
            self.tts_engine = None
            return False
    
    def _select_voice(self, voices: List) -> None:
        """
        Allow user to select a voice for text-to-speech.
        
        Args:
            voices: List of available voices
        """
        if not voices:
            return
            
        # Check if we should skip interactive selection
        if self.config.skip_tts_selection:
            print("Using default voice (skipping selection).")
            # Optionally select English voice if available
            for voice in voices:
                if "english" in voice.id.lower():
                    self.tts_engine.setProperty('voice', voice.id)
                    print(f"Automatically selected voice: {voice.name}")
                    return
            return
            
        while True:
            voice_choice_str = input("\nEnter the number of the voice you want to use (or press Enter for default): ").strip()
            if not voice_choice_str:
                print("Using default voice.")
                break
                
            try:
                voice_choice_index = int(voice_choice_str) - 1
                if 0 <= voice_choice_index < len(voices):
                    selected_voice_id = voices[voice_choice_index].id
                    self.tts_engine.setProperty('voice', selected_voice_id)
                    print(f"Voice set to: {voices[voice_choice_index].name}")
                    break
                else:
                    print("Invalid voice number. Please try again.")
            except ValueError:
                print("Invalid input. Please enter a number.")
    
    def speak(self, text: str) -> None:
        """
        Speak text using text-to-speech engine.
        
        Args:
            text: Text to speak
        """
        if not self.tts_engine:
            return
            
        try:
            self.tts_engine.say(text)
            self.tts_engine.runAndWait()
        except Exception as e:
            logger.warning(f"Error during speech: {e}")
    
    def stop(self) -> None:
        """Stop the text-to-speech engine."""
        if self.tts_engine:
            try:
                self.tts_engine.stop()
            except Exception as e:
                logger.warning(f"Error stopping speech engine: {e}")
    
    def set_properties(self, rate: Optional[int] = None, volume: Optional[float] = None) -> None:
        """
        Set speech engine properties.
        
        Args:
            rate: Speech rate (words per minute)
            volume: Volume (0.0 to 1.0)
        """
        if not self.tts_engine:
            return
            
        try:
            if rate is not None:
                self.tts_engine.setProperty('rate', rate)
            if volume is not None:
                self.tts_engine.setProperty('volume', volume)
        except Exception as e:
            logger.warning(f"Error setting speech properties: {e}")