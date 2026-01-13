from gtts import gTTS
import pygame
import time, os, logging

def play_audio_file(filename):
    try:
        if not os.path.exists(filename):
            raise FileNotFoundError(f"Không tìm thấy {filename}")
        if not pygame.mixer.get_init():
            pygame.mixer.init()
        pygame.mixer.music.load(filename)
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            time.sleep(0.1)
    except Exception as e:
        logging.error(f"Lỗi khi phát file {filename}: {e}")
        raise e

def play_sample(word, filename="sample.mp3"):
    try:
        tts = gTTS(text=word, lang="en")
        tts.save(filename)
        play_audio_file(filename)
    except Exception as e:
        logging.error(f"Lỗi TTS: {e}")
        raise e
