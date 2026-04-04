from pathlib import Path

from PySide6.QtCore import QThread
from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QLabel,
    QFileDialog,
    QListWidget,
    QTextEdit,
    QComboBox,
    QMessageBox,
    QProgressBar,
)

from src.audio.file_scanner import list_audio_file_names
from src.config import APP_TITLE, WINDOW_WIDTH, WINDOW_HEIGHT, DEFAULT_MODEL
from src.workers.transcription_worker import TranscriptionWorker


class MainWindow(QWidget):
    def __init__(self) -> None:
        super().__init__()

        self.setWindowTitle(APP_TITLE)
        self.resize(WINDOW_WIDTH, WINDOW_HEIGHT)

        self.selected_folder: str = ""
        self.thread: QThread | None = None
        self.worker: TranscriptionWorker | None = None

        self.title_label = QLabel("Selecione uma pasta com áudios")
        self.folder_label = QLabel("Nenhuma pasta selecionada")

        self.select_button = QPushButton("Selecionar pasta")
        self.select_button.clicked.connect(self.select_folder)

        self.model_combo = QComboBox()
        self.model_combo.addItems(["tiny", "base", "small", "medium", "large"])
        self.model_combo.setCurrentText(DEFAULT_MODEL)

        self.file_list = QListWidget()

        self.start_button = QPushButton("Iniciar transcrição")
        self.start_button.clicked.connect(self.start_transcription)

        self.stop_button = QPushButton("Parar")
        self.stop_button.clicked.connect(self.stop_transcription)
        self.stop_button.setEnabled(False)

        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)

        self.log_output = QTextEdit()
        self.log_output.setReadOnly(True)

        self.setup_layout()

    def setup_layout(self) -> None:
        layout = QVBoxLayout()

        layout.addWidget(self.title_label)
        layout.addWidget(self.folder_label)

        top_layout = QHBoxLayout()
        top_layout.addWidget(self.select_button)
        top_layout.addWidget(QLabel("Modelo:"))
        top_layout.addWidget(self.model_combo)

        action_layout = QHBoxLayout()
        action_layout.addWidget(self.start_button)
        action_layout.addWidget(self.stop_button)

        layout.addLayout(top_layout)
        layout.addWidget(QLabel("Arquivos encontrados:"))
        layout.addWidget(self.file_list)
        layout.addLayout(action_layout)
        layout.addWidget(self.progress_bar)
        layout.addWidget(QLabel("Logs:"))
        layout.addWidget(self.log_output)

        self.setLayout(layout)

    def log(self, message: str) -> None:
        self.log_output.append(message)

    def select_folder(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Selecione a pasta com os áudios")
        if not folder:
            return

        self.selected_folder = folder
        self.folder_label.setText(folder)
        self.load_audio_files()

    def load_audio_files(self) -> None:
        self.file_list.clear()

        file_names = list_audio_file_names(self.selected_folder)

        if not file_names:
            self.log("Nenhum áudio encontrado nessa pasta.")
            return

        self.file_list.addItems(file_names)
        self.log(f"{len(file_names)} arquivo(s) encontrado(s).")

    def start_transcription(self) -> None:
        if not self.selected_folder:
            QMessageBox.warning(self, "Aviso", "Selecione uma pasta primeiro.")
            return

        if self.file_list.count() == 0:
            QMessageBox.warning(self, "Aviso", "Nenhum áudio encontrado na pasta.")
            return

        self.progress_bar.setValue(0)
        self.log("Iniciando transcrição...")

        self.thread = QThread()
        self.worker = TranscriptionWorker(
            folder_path=self.selected_folder,
            model_name=self.model_combo.currentText(),
        )

        self.worker.moveToThread(self.thread)

        self.thread.started.connect(self.worker.run)
        self.worker.progress.connect(self.progress_bar.setValue)
        self.worker.log.connect(self.log)
        self.worker.finished.connect(self.on_finished)
        self.worker.failed.connect(self.on_failed)

        self.worker.finished.connect(self.thread.quit)
        self.worker.failed.connect(self.thread.quit)
        self.thread.finished.connect(self.thread.deleteLater)

        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)

        self.thread.start()

    def stop_transcription(self) -> None:
        if self.worker:
            self.worker.stop()
            self.log("Solicitação de parada enviada...")

    def on_finished(self) -> None:
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.log("Processo finalizado.")
        QMessageBox.information(self, "Concluído", "Transcrição finalizada com sucesso.")

    def on_failed(self, message: str) -> None:
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.log("Erro durante a transcrição:")
        self.log(message)
        QMessageBox.critical(self, "Erro", message)

    def closeEvent(self, event) -> None:
        if self.worker:
            self.worker.stop()
        event.accept()