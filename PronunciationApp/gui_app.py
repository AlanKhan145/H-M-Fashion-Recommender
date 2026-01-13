import sys, time, os, logging
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QPushButton, QLineEdit, QLabel, QVBoxLayout,
                             QHBoxLayout, QProgressBar, QMessageBox)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from record_module import record_audio
from tts_module import play_sample, play_audio_file
from translation_module import translate_english_to_vietnamese
from history_module import save_word_to_history, load_history

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


# --- Worker cho TTS ---
class PlaySampleWorker(QThread):
    finished = pyqtSignal()
    error = pyqtSignal(str)

    def __init__(self, word, parent=None):
        super().__init__(parent)
        self.word = word

    def run(self):
        try:
            play_sample(self.word, "sample.mp3")
            self.finished.emit()
        except Exception as e:
            logging.error(f"Lỗi TTS: {e}")
            self.error.emit(str(e))


# --- Worker cho ghi âm ---
class RecordWorker(QThread):
    finished = pyqtSignal()
    error = pyqtSignal(str)

    def run(self):
        try:
            record_audio("user_input.wav")
            self.finished.emit()
        except Exception as e:
            logging.error(f"Lỗi ghi âm: {e}")
            self.error.emit(str(e))


# --- Giao diện chính ---
class PronunciationUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.current_theme = "light"
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle("Hệ Thống Luyện Phát Âm & Dịch Tiếng Anh - Next-Level")
        self.setGeometry(300, 300, 700, 600)
        self.central_widget = QWidget(self)
        self.setCentralWidget(self.central_widget)
        self.main_layout = QVBoxLayout(self.central_widget)

        self.banner = QLabel("Phát Âm Chuẩn - Chinh Phục Tiếng Anh Ngay Hôm Nay!", self)
        self.banner.setAlignment(Qt.AlignCenter)
        self.banner.setStyleSheet("font-size: 24px; font-weight: bold;")
        self.main_layout.addWidget(self.banner)

        self.word_entry = QLineEdit(self)
        self.word_entry.setPlaceholderText("Nhập từ/cụm từ cần luyện...")
        self.word_entry.setFixedWidth(300)
        self.main_layout.addWidget(self.word_entry, alignment=Qt.AlignCenter)

        # Nút chức năng: Phát mẫu, Ghi âm, Replay
        self.btn_layout = QHBoxLayout()
        self.play_button = QPushButton("Phát mẫu", self)
        self.record_button = QPushButton("Ghi âm", self)
        self.replay_sample_button = QPushButton("Replay mẫu", self)
        self.replay_record_button = QPushButton("Replay ghi âm", self)
        for btn in [self.play_button, self.record_button, self.replay_sample_button, self.replay_record_button]:
            self.btn_layout.addWidget(btn)
        self.main_layout.addLayout(self.btn_layout)

        # TÍNH NĂNG CHẤM ĐIỂM ĐÃ BỊ LOẠI BỎ

        self.progress = QProgressBar(self)
        self.progress.setFixedHeight(25)
        self.progress.setVisible(False)
        self.main_layout.addWidget(self.progress)

        self.result_label = QLabel("", self)
        self.result_label.setAlignment(Qt.AlignCenter)
        self.main_layout.addWidget(self.result_label)

        self.translate_button = QPushButton("Dịch sang tiếng Việt", self)
        self.translation_label = QLabel("", self)
        self.translation_label.setAlignment(Qt.AlignCenter)
        self.main_layout.addWidget(self.translate_button)
        self.main_layout.addWidget(self.translation_label)

        # Nút lịch sử, Clear và chuyển theme
        self.history_button = QPushButton("Lịch sử", self)
        self.clear_button = QPushButton("Clear", self)
        self.theme_toggle_button = QPushButton("Dark Mode", self)
        self.main_layout.addWidget(self.history_button, alignment=Qt.AlignCenter)
        self.main_layout.addWidget(self.clear_button, alignment=Qt.AlignCenter)
        self.main_layout.addWidget(self.theme_toggle_button, alignment=Qt.AlignCenter)

        # Kết nối tín hiệu
        self.play_button.clicked.connect(self.handle_play)
        self.record_button.clicked.connect(self.handle_record)
        self.replay_sample_button.clicked.connect(self.handle_replay_sample)
        self.replay_record_button.clicked.connect(self.handle_replay_record)
        self.translate_button.clicked.connect(self.handle_translation)
        self.history_button.clicked.connect(self.handle_history)
        self.clear_button.clicked.connect(self.handle_clear)
        self.theme_toggle_button.clicked.connect(self.toggle_theme)

    def toggle_theme(self):
        if self.current_theme == "light":
            self.setStyleSheet("background-color: #333; color: #eee;")
            self.theme_toggle_button.setText("Light Mode")
            self.current_theme = "dark"
        else:
            self.setStyleSheet("")
            self.theme_toggle_button.setText("Dark Mode")
            self.current_theme = "light"

    def disable_buttons(self):
        for btn in [self.play_button, self.record_button, self.replay_sample_button,
                    self.replay_record_button, self.clear_button,
                    self.translate_button, self.history_button, self.theme_toggle_button]:
            btn.setEnabled(False)

    def enable_buttons(self):
        for btn in [self.play_button, self.record_button, self.replay_sample_button,
                    self.replay_record_button, self.clear_button,
                    self.translate_button, self.history_button, self.theme_toggle_button]:
            btn.setEnabled(True)

    def handle_play(self):
        word = self.word_entry.text().strip()
        if not word:
            QMessageBox.warning(self, "Lỗi", "Vui lòng nhập từ cần luyện!")
            return
        self.disable_buttons()
        self.progress.setVisible(True)
        self.progress.setRange(0, 0)
        self.play_worker = PlaySampleWorker(word)
        self.play_worker.finished.connect(self.on_play_finished)
        self.play_worker.error.connect(lambda err: self.on_error(err))
        self.play_worker.start()

    def on_play_finished(self):
        self.progress.setVisible(False)
        self.replay_sample_button.setEnabled(True)
        self.enable_buttons()

    def handle_record(self):
        self.disable_buttons()
        self.progress.setVisible(True)
        self.progress.setRange(0, 0)
        self.record_worker = RecordWorker()
        self.record_worker.finished.connect(self.on_record_finished)
        self.record_worker.error.connect(lambda err: self.on_error(err))
        self.record_worker.start()

    def on_record_finished(self):
        self.progress.setVisible(False)
        self.replay_record_button.setEnabled(True)
        self.result_label.setText("🎤 Ghi âm hoàn tất! Nghe thử nào!")
        self.enable_buttons()

    def handle_replay_sample(self):
        try:
            if os.path.exists("sample.mp3"):
                play_audio_file("sample.mp3")
            else:
                QMessageBox.warning(self, "Lỗi", "Không có file mẫu!")
        except Exception as e:
            QMessageBox.critical(self, "Lỗi", f"Lỗi khi replay mẫu: {e}")

    def handle_replay_record(self):
        try:
            if os.path.exists("user_input.wav"):
                play_audio_file("user_input.wav")
            else:
                QMessageBox.warning(self, "Lỗi", "Không có file ghi âm!")
        except Exception as e:
            QMessageBox.critical(self, "Lỗi", f"Lỗi khi replay ghi âm: {e}")

    def handle_translation(self):
        word = self.word_entry.text().strip()
        if not word:
            QMessageBox.warning(self, "Lỗi", "Vui lòng nhập từ để dịch!")
            return
        translated_text = translate_english_to_vietnamese(word)
        self.translation_label.setText(f"Dịch: {translated_text}")
        save_word_to_history(word)

    def handle_history(self):
        history = load_history()
        msg = "\n".join(history) if history else "Chưa có từ nào được lưu."
        QMessageBox.information(self, "Lịch sử", msg)

    def handle_clear(self):
        self.word_entry.clear()
        self.result_label.setText("")
        self.translation_label.setText("")
        for file in ["sample.mp3", "user_input.wav"]:
            if os.path.exists(file):
                os.remove(file)

    def on_error(self, error_message):
        self.progress.setVisible(False)
        QMessageBox.critical(self, "Lỗi", error_message)
        self.enable_buttons()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = PronunciationUI()
    window.show()
    sys.exit(app.exec_())
