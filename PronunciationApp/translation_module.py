from googletrans import Translator
import logging

def translate_english_to_vietnamese(text: str) -> str:
    translator = Translator()
    try:
        result = translator.translate(text, src="en", dest="vi")
        logging.info("Dịch thành công!")
        return result.text
    except Exception as e:
        logging.error(f"Lỗi khi dịch: {e}")
        return "Lỗi khi dịch"
