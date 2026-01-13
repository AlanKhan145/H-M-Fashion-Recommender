# record_module.py
import pyaudio, wave, logging
from config import SAMPLE_RATE, CHUNK, RECORD_SECONDS

def record_audio(filename="user_input.wav"):
    p = pyaudio.PyAudio()
    stream = None
    frames = []
    try:
        stream = p.open(format=pyaudio.paInt16, channels=1, rate=SAMPLE_RATE,
                        frames_per_buffer=CHUNK, input=True)
        logging.info("Bắt đầu ghi âm...")
        for _ in range(0, int(SAMPLE_RATE / CHUNK * RECORD_SECONDS)):
            frames.append(stream.read(CHUNK))
        logging.info("Ghi âm xong!")
    except Exception as e:
        logging.error(f"Lỗi khi ghi âm: {e}")
        raise e
    finally:
        if stream:
            stream.stop_stream()
            stream.close()
        p.terminate()

    try:
        with wave.open(filename, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(p.get_sample_size(pyaudio.paInt16))
            wf.setframerate(SAMPLE_RATE)
            wf.writeframes(b"".join(frames))
        logging.info(f"File ghi âm lưu tại {filename}")
    except Exception as e:
        logging.error(f"Lỗi khi lưu file: {e}")
        raise e
