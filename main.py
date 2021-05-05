# -*- coding: utf-8 -*-
import gc
from PyQt5.QtWidgets import QMenu, QFileDialog, QLineEdit
from PyQt5.QtCore import *
from tts_utils import *
from template import Ui_mainWindow
from PyQt5 import QtWidgets
import os
import re
from pydub import AudioSegment
import shutil
import wavio
import threading
from PyQt5.QtCore import QUrl
from PyQt5.QtMultimedia import QMediaPlayer
from PyQt5.QtMultimedia import *
import sys
from nltk.tokenize import word_tokenize
import pandas as pd
import torch

language = 'ru'
device = torch.device('cpu')
tokenset = '_~абвгдеёжзийклмнопрстуфхцчшщъыьэюя +.,!?…:;–'

class MyWin(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.ui = Ui_mainWindow()  # Экземпляр класса Ui_MainWindow, в нем конструктор всего GUI.
        self.ui.setupUi(self)
        self.ui.treatment_btn.clicked.connect(self.word_pr_start) #word_processing)
        self.ui.voice_btn.clicked.connect(self.text_sc_start) #text_scoring)
        self.ui.fon_btn.clicked.connect(self.open_fon)
        self.ui.clean_btn.clicked.connect(self.clean_text)
        self.ui.fileopen_btn.clicked.connect(self.file_open)
        self.ui.file_treat_btn.clicked.connect(self.file_sc_start) #file_scoring)
        self.ui.file_voice_btn.clicked.connect(self.file_names_start) #file_names)
        self.ui.label_format.setText('mp3')
        self.ui.label_voice.setText('Ksenia8000')

        self.player = QMediaPlayer()  # Инициализируем плеер
        self.content = QMediaContent()
        self.playlist_view = QLineEdit()
        #self.duration = 0
        self.playlist = QMediaPlaylist(self.player)  # Инициализируем плейлист
        self.playlist_dock_state = None
        self.player.setPlaylist(self.playlist)  # Устанавливаем плейлист в плеер

        self.ui.pause_btn.clicked.connect(self.file_pause)
        self.ui.stop_btn.clicked.connect(self.file_stop)
        self.ui.play_btn.clicked.connect(self.file_play)

        menu = QMenu(self)
        menu.addAction("Aidar8000", self.choose_Aidar8000)
        menu.addAction("Aidar16000", self.choose_Aidar16000)
        menu.addAction("Ksenia8000", self.choose_Ksenia8000)
        menu.addAction("Ksenia16000", self.choose_Ksenia16000)
        menu.addAction("Baya8000", self.choose_Baya8000)
        menu.addAction("Baya16000", self.choose_Baya16000)
        menu.addAction("Ruslan8000", self.choose_Ruslan8000)
        menu.addAction("Ruslan16000", self.choose_Ruslan16000)
        menu.addAction("Irina8000", self.choose_Irina8000)
        menu.addAction("Irina16000", self.choose_Irina16000)
        self.ui.vote_btn.setMenu(menu)

        menu_f = QMenu(self)
        menu_f.addAction("wav", self.format_wav)
        menu_f.addAction("mp3", self.format_mp3)
        menu_f.addAction("flac", self.format_flac)
        self.ui.format_btn.setMenu(menu_f)

        menu_fon = QMenu(self)
        menu_fon.addAction("Открыть", self.open_fon)
        menu_fon.addAction("Наложить на текст", self.overlay_bg_text_start) #overlay_background_text)
        menu_fon.addAction("Наложить на файл", self.overlay_bg_file_start) #overlay_background_file)
        self.ui.fon_btn.setMenu(menu_fon)

        menu_del = QMenu(self)
        menu_del.addAction("Текстовый файл", self.del_text_file)
        menu_del.addAction("Аудио файл", self.del_audio_file)
        menu_del.addAction("Фоновый файл", self.del_fon_file)
        self.ui.del_btn.setMenu(menu_del)

    def file_play(self):
        gc.collect()
        self.player.play()

    def file_pause(self):
        self.paused = True
        if self.player.state() == QMediaPlayer.PlayingState:
            self.player.pause()
        elif self.player.state() == QMediaPlayer.PausedState:
            self.player.play()

    def file_stop(self):
        self.player.stop()
        gc.collect()

    def format_wav(self):
        self.ui.label_format.setText("wav")

    def format_mp3(self):
        self.ui.label_format.setText("mp3")

    def format_flac(self):
        self.ui.label_format.setText("flac")

    def clean_text(self):
        self.ui.textEdit.clear()

    def word_pr_start(self):
        self.ui.lineEdit.clear()
        self.ui.lineEdit_down.clear()
        self.word_processing()

    def word_processing(self):
        txt = self.ui.textEdit.toPlainText()
        try:
            df = pd.read_csv("stress.csv")
            tokens = word_tokenize(txt)
            res = []
            for i, j in enumerate(tokens):
                a = df.loc[df['word'] == f'{j.lower()}']['stress']
                if len(a) > 0:
                    res.append(a.to_string(index=False))
                else:
                    res.append(j.lower())
            data = ' '.join(res)  # получить из списка строку
            self.ui.textEdit.clear()
            self.ui.textEdit.setPlainText(data)
            del res
        except:
            pass

    def text_sc_start(self):
        self.ui.lineEdit.clear()
        self.ui.lineEdit_down.clear()
        self.ui.lineEdit_down.setText("Озвучивание началось!")
        self.text_sc_tread()

    def text_sc_tread(self):
        threading.Thread(target=self.text_scoring, daemon=False).start()

    def text_scoring(self):
        format = self.ui.label_format.text()
        rat = self.ui.spinBox.value()
        mdl = self.ui.label_voice.text()
        model = init_jit_model(f"voices/{mdl}.jit", device)
        model = model.to(device)  # gpu or cpu
        wws = self.ui.textEdit.toPlainText()
        try:
            text3 = re.split(';|,|:|!|\*', wws)
            if not os.path.isdir("output"):
                os.mkdir("output")
            out = 'output'
            if not os.path.isdir("output_txt"):
                os.mkdir("output_txt")
            results = []
            for i, segm in enumerate(text3):
                audio = apply_tts(texts=[segm],
                                  model=model,
                                  sample_rate=int(rat),  # sample_rate,
                                  symbols=tokenset,
                                  device=device)
                for j, _audio in enumerate(audio):
                    au = audio[0].numpy()
                    results.append(f'{out}/test_{str(i).zfill(2)}.{format}')
                    wavio.write(f'{out}/test_{str(i).zfill(2)}.{format}', au, int(rat), sampwidth=2)
            file_n = open("output_txt/out_text_names.txt", "w", encoding='utf-8')
            file_n.write('\n'.join(results))
            file_n.close()
            del results
            self.text_stitching()
        except TypeError:
            self.ui.lineEdit_down.setText("Вставьте текст!")
            pass

    def text_stitching(self):
        format = self.ui.label_format.text()
        file_txt = "output_txt/out_text_names.txt"
        with open(file_txt, 'r') as f:
            rows = f.read().splitlines()
        wav_file_1 = AudioSegment.from_file(f"output/test_00.{format}")
        for row in rows:
            if row == f'output/test_00.{format}':
                pass
            else:
                wav_file_2 = AudioSegment.from_file(f"{row}")
                wav_file_1 += wav_file_2
        if not os.path.isdir("output_final"):
            os.mkdir("output_final")
        self.playlist.clear()
        try:
            wav_file_1.export(f"output_final/final_text_file.{format}", format=f"{format}")
            self.ui.lineEdit_down.setText(f"Текст озвучен -- output_final/final_text_file.{format}")
            self.ui.lineEdit.setText(f"output_final/final_text_file.{format}")
            playlist_item = self.ui.lineEdit.text()
            fullpath = QDir.current().absoluteFilePath(playlist_item)
            self.playlist.addMedia(QMediaContent(QUrl().fromLocalFile(fullpath)))
            self.player.setPlaylist(self.playlist)
            shutil.rmtree('output')
            f.close()
        except PermissionError:
            self.ui.lineEdit_down.setText(f"Переименуйте файл! -- output_final/final_file.{format.strip()}")

    def choose_Aidar8000(self):
        self.ui.label_voice.setText('Aidar8000')

    def choose_Aidar16000(self):
        self.ui.label_voice.setText('Aidar16000')

    def choose_Ksenia8000(self):
        self.ui.label_voice.setText('Ksenia8000')

    def choose_Ksenia16000(self):
        self.ui.label_voice.setText('Ksenia16000')

    def choose_Baya8000(self):
        self.ui.label_voice.setText('Baya8000')

    def choose_Baya16000(self):
        self.ui.label_voice.setText('Baya16000')

    def choose_Ruslan8000(self):
        self.ui.label_voice.setText('Ruslan8000')

    def choose_Ruslan16000(self):
        self.ui.label_voice.setText('Ruslan16000')

    def choose_Irina8000(self):
        self.ui.label_voice.setText('Irina8000')

    def choose_Irina16000(self):
        self.ui.label_voice.setText('Irina16000')

    def overlay_bg_file_start(self):
        try:
            self.ui.lineEdit.clear()
            self.ui.lineEdit_down.clear()
            self.ui.lineEdit_down.setText("Наложение фона началось!")
            self.overlay_bg_file_tread()
        except FileNotFoundError:
            self.ui.lineEdit_down.setText("Загрузите файл фона!")

    def overlay_bg_file_tread(self):
        threading.Thread(target=self.overlay_background_file, daemon=False).start()

    def overlay_background_file(self):
        format = self.ui.label_format.text()
        mmin = self.ui.lineEdit_mmin.text()
        smin = self.ui.lineEdit_smin.text()
        mmax = self.ui.lineEdit_mmax.text()
        smax = self.ui.lineEdit_smax.text()
        back_file = self.ui.lineEdit_lv.text()
        try:
            if len(back_file) > 0:
                sound1 = AudioSegment.from_file(f"output_final/final_file.{format}")
                sound2 = AudioSegment.from_file(f"{back_file}")
                if len(mmin) == 0 or len(smin) == 0 or len(mmax) == 0 or len(mmax) == 0:
                    output = sound1.overlay(sound2, position=0)
                    output.export(f"output_final/mixed_sounds.{format}", format=f"{format}")
                elif int(mmin) > 60 or int(smin) > 60 or int(mmax) > 60 or int(mmax) > 0:
                    output = sound1.overlay(sound2, position=0)
                    output.export(f"output_final/mixed_sounds.{format}", format=f"{format}")
                else:
                    startTime = int(mmin) * 60 * 1000 + int(smin) * 1000
                    endTime = int(mmax) * 60 * 1000 + int(smax) * 1000
                    sound2 = sound2[startTime:endTime]
                    output = sound1.overlay(sound2, position=0)
                    output.export(f"output_final/mixed_sounds.{format}", format=f"{format}")
                try:
                    self.playlist.clear()
                    self.ui.lineEdit_down.setText("Наложение фона завершено!")
                    self.ui.lineEdit.setText(f"output_final/mixed_sounds.{format}")
                    playlist_item = self.ui.lineEdit.text()
                    fullpath = QDir.current().absoluteFilePath(playlist_item)
                    self.playlist.addMedia(QMediaContent(QUrl().fromLocalFile(fullpath)))
                    self.player.setPlaylist(self.playlist)
                except PermissionError:
                    self.ui.lineEdit_down.setText(f"Переименуйте файл! -- output_final/final_file.{format.strip()}")
        except FileNotFoundError:
            self.ui.lineEdit_down.setText("Загрузите файл фона!")
            pass

    def overlay_bg_text_start(self):
        try:
            self.ui.lineEdit.clear()
            self.ui.lineEdit_down.clear()
            self.ui.lineEdit_down.setText("Наложение фона началось!")
            self.overlay_bg_text_read()
        except FileNotFoundError:
            self.ui.lineEdit_down.setText("Загрузите файл фона!")

    def overlay_bg_text_read(self):
        threading.Thread(target=self.overlay_background_text, daemon=False).start()

    def overlay_background_text(self):
        format = self.ui.label_format.text()
        mmin = self.ui.lineEdit_mmin.text()
        smin = self.ui.lineEdit_smin.text()
        mmax = self.ui.lineEdit_mmax.text()
        smax = self.ui.lineEdit_smax.text()
        back_file = self.ui.lineEdit_lv.text()
        try:
            if len(back_file) > 0:
                sound1 = AudioSegment.from_file(f"output_final/final_text_file.{format}")
                sound2 = AudioSegment.from_file(f"{back_file}")
                if len(mmin) == 0 or len(smin) == 0 or len(mmax) == 0 or len(mmax) == 0:
                    output = sound1.overlay(sound2, position=0)
                    output.export(f"output_final/mixed_text_sounds.{format}", format=f"{format}")
                elif int(mmin) > 60 or int(smin) > 60 or int(mmax) > 60 or int(mmax) > 0:
                    output = sound1.overlay(sound2, position=0)
                    output.export(f"output_final/mixed_text_sounds.{format}", format=f"{format}")
                else:
                    startTime = int(mmin) * 60 * 1000 + int(smin) * 1000
                    endTime = int(mmax) * 60 * 1000 + int(smax) * 1000
                    sound2 = sound2[startTime:endTime]
                    output = sound1.overlay(sound2, position=0)
                    output.export(f"output_final/mixed_text_sounds.{format}", format=f"{format}")
                try:
                    self.playlist.clear()
                    self.ui.lineEdit_down.setText("Наложение фона завершено!")
                    self.ui.lineEdit.setText(f"output_final/mixed_text_sounds.{format}")
                    playlist_item = self.ui.lineEdit.text()
                    fullpath = QDir.current().absoluteFilePath(playlist_item)
                    self.playlist.addMedia(QMediaContent(QUrl().fromLocalFile(fullpath)))
                    self.player.setPlaylist(self.playlist)
                except PermissionError:
                    self.ui.lineEdit_down.setText(f"Переименуйте файл! -- output_final/final_file.{format.strip()}")
        except FileNotFoundError:
            self.ui.lineEdit_down.setText("Загрузите файл фона!")
            pass

    def file_open(self):
        filename = QFileDialog.getOpenFileName(self, 'Open File', '', "Text files (*.txt)")
        if filename[0]:
            self.ui.lineEdit_rv.setText(f"{filename[0]}")
        else:
            pass

    def file_sc_start(self):
        self.ui.lineEdit.clear()
        self.ui.lineEdit_down.clear()
        self.ui.lineEdit_down.setText('Файл обрабатывается!')
        self.file_scoring_tr()

    def file_scoring_tr(self):
        threading.Thread(target=self.file_scoring, daemon=False).start()

    def file_scoring(self):
        try:
            filename = self.ui.lineEdit_rv.text()
            f = open(filename, 'r', encoding="utf-8")
            with f:
                data = f.read()
                df = pd.read_csv("stress.csv")
                tokens = word_tokenize(data)
                result = []
                for i, j in enumerate(tokens):
                    a = df.loc[df['word'] == f'{j.lower()}']['stress']
                    if len(a) > 0:
                        result.append(a.to_string(index=False))
                    else:
                        result.append(j.lower())
                data_str = ' '.join(result)  # получить из списка строку
                if not os.path.isdir("output_txt"):
                    os.mkdir("output_txt")
                file_sts = open("output_txt/out_file.txt", "w", encoding='utf-8')
                file_sts.write(data_str)
                f.close()
                del result
                self.file_scoring_final()
        except FileNotFoundError:
            pass

    def file_scoring_final(self):
        self.ui.lineEdit_down.setText('Файл обработан -- output_txt/out_file.txt')

    def file_names_start(self):
        self.ui.lineEdit.clear()
        self.ui.lineEdit_down.clear()
        self.ui.lineEdit_down.setText('Началось озвучивание файла. Процесс может занять много времени!')
        self.file_names_tread()

    def file_names_tread(self):
        threading.Thread(target=self.file_names, daemon=False).start()

    def file_names(self):
        try:
            file_txt = "output_txt/out_file.txt"
            f = open(file_txt, 'r', encoding="utf-8")
            format = self.ui.label_format.text()
            with f:
                data_txt = f.read()
                md = self.ui.label_voice.text()
                sr = self.ui.spinBox.value()
                model = init_jit_model(f"voices/{md}.jit", device)
                model = model.to(device)  # gpu or cpu
                text5 = re.split(';|,|:|!|\*', data_txt)
                try:
                    shutil.rmtree('output')
                except FileNotFoundError:
                    pass
                if not os.path.isdir("output"):
                    os.mkdir("output")
                out = 'output'
                name_files = []
                for i, segm in enumerate(text5):
                    audio = apply_tts(texts=[segm],
                                      model=model,
                                      sample_rate=int(sr),
                                      symbols=tokenset,
                                      device=device)
                    for j, _audio in enumerate(audio):
                        au = audio[0].numpy()
                        name_files.append(f'{out}/test_{str(i).zfill(2)}.{format}')
                        wavio.write(f'{out}/test_{str(i).zfill(2)}.{format}', au, int(sr), sampwidth=2)
                file_n = open("output_txt/out_name.txt", "w", encoding='utf-8')
                file_n.write('\n'.join(name_files))
                file_n.close()
                del name_files
            self.file_voices()
        except (FileNotFoundError, TypeError):
            pass

    def file_voices(self):
        format = self.ui.label_format.text()
        file_t = "output_txt/out_name.txt"
        with open(file_t, 'r') as f:
            lines = f.read().splitlines()
        wav_file_1 = AudioSegment.from_file(f"output/test_00.{format.strip()}")
        for line in lines:
            if line == f'test_00.{format.strip()}':
                pass
            else:
                wav_file_2 = AudioSegment.from_file(f'{line.strip()}')
                wav_file_1 += wav_file_2
        if not os.path.isdir("output_final"):
            os.mkdir("output_final")
        self.playlist.clear()
        try:
            wav_file_1.export(f"output_final/final_file.{format.strip()}", format=f"{format.strip()}")
            self.ui.lineEdit_down.setText(f"Озвучивание завершено! -- output_final/final_file.{format.strip()}")
            self.ui.lineEdit.setText(f"output_final/final_file.{format.strip()}")
            playlist_item = self.ui.lineEdit.text()
            fullpath = QDir.current().absoluteFilePath(playlist_item)
            self.playlist.addMedia(QMediaContent(QUrl().fromLocalFile(fullpath)))
            self.player.setPlaylist(self.playlist)
            shutil.rmtree('output')
            f.close()
        except PermissionError:
            self.ui.lineEdit_down.setText(f"Переименуйте файл! -- output_final/final_file.{format.strip()}")

    def open_fon(self):
        filename = QFileDialog.getOpenFileName(self, 'Open File', '', 'Audio (*.wav *.mp3 *.flac)')
        if filename[0]:
            self.ui.lineEdit_lv.setText(f"{filename[0]}")
        else:
            pass

    def del_text_file(self):
        self.ui.lineEdit_rv.setText("Открытие текстового файла")

    def del_audio_file(self):
        self.ui.lineEdit.clear()

    def del_fon_file(self):
        self.ui.lineEdit_lv.setText("Обрезка фона: m.min s.min m.max s.max")
        self.ui.lineEdit_mmax.clear()
        self.ui.lineEdit_smax.clear()
        self.ui.lineEdit_mmin.clear()
        self.ui.lineEdit_smin.clear()

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    app.setStyle('Fusion')
    window = MyWin()
    window.show()
    sys.exit(app.exec_())
