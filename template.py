# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'govor.ui'
#
# Created by: PyQt5 UI code generator 5.15.4
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_mainWindow(object):
    def setupUi(self, mainWindow):
        mainWindow.setObjectName("mainWindow")
        mainWindow.resize(917, 550)
        mainWindow.setMinimumSize(QtCore.QSize(917, 550))
        mainWindow.setMaximumSize(QtCore.QSize(917, 550))
        font = QtGui.QFont()
        font.setFamily("MV Boli")
        font.setPointSize(9)
        font.setBold(False)
        font.setWeight(50)
        mainWindow.setFont(font)
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap("stt-32.ico"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        mainWindow.setWindowIcon(icon)
        mainWindow.setIconSize(QtCore.QSize(32, 32))
        self.centralwidget = QtWidgets.QWidget(mainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.layoutWidget = QtWidgets.QWidget(self.centralwidget)
        self.layoutWidget.setGeometry(QtCore.QRect(14, 15, 896, 531))
        self.layoutWidget.setObjectName("layoutWidget")
        self.verticalLayout_7 = QtWidgets.QVBoxLayout(self.layoutWidget)
        self.verticalLayout_7.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout_7.setObjectName("verticalLayout_7")
        self.horizontalLayout_4 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_4.setObjectName("horizontalLayout_4")
        self.verticalLayout_5 = QtWidgets.QVBoxLayout()
        self.verticalLayout_5.setObjectName("verticalLayout_5")
        self.horizontalLayout_3 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_3.setSpacing(6)
        self.horizontalLayout_3.setObjectName("horizontalLayout_3")
        self.treatment_btn = QtWidgets.QPushButton(self.layoutWidget)
        self.treatment_btn.setMinimumSize(QtCore.QSize(100, 23))
        self.treatment_btn.setMaximumSize(QtCore.QSize(100, 23))
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(10)
        font.setBold(False)
        font.setWeight(50)
        self.treatment_btn.setFont(font)
        self.treatment_btn.setObjectName("treatment_btn")
        self.horizontalLayout_3.addWidget(self.treatment_btn)
        self.voice_btn = QtWidgets.QPushButton(self.layoutWidget)
        self.voice_btn.setMinimumSize(QtCore.QSize(100, 23))
        self.voice_btn.setMaximumSize(QtCore.QSize(100, 23))
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(10)
        font.setBold(False)
        font.setWeight(50)
        self.voice_btn.setFont(font)
        self.voice_btn.setObjectName("voice_btn")
        self.horizontalLayout_3.addWidget(self.voice_btn)
        self.verticalLayout_5.addLayout(self.horizontalLayout_3)
        self.lineEdit_lv = QtWidgets.QLineEdit(self.layoutWidget)
        self.lineEdit_lv.setMinimumSize(QtCore.QSize(206, 20))
        self.lineEdit_lv.setMaximumSize(QtCore.QSize(206, 20))
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(8)
        font.setBold(False)
        font.setWeight(50)
        self.lineEdit_lv.setFont(font)
        self.lineEdit_lv.setReadOnly(True)
        self.lineEdit_lv.setObjectName("lineEdit_lv")
        self.verticalLayout_5.addWidget(self.lineEdit_lv)
        self.horizontalLayout_4.addLayout(self.verticalLayout_5)
        self.verticalLayout = QtWidgets.QVBoxLayout()
        self.verticalLayout.setObjectName("verticalLayout")
        self.fon_btn = QtWidgets.QPushButton(self.layoutWidget)
        self.fon_btn.setEnabled(True)
        self.fon_btn.setMinimumSize(QtCore.QSize(100, 23))
        self.fon_btn.setMaximumSize(QtCore.QSize(100, 23))
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(10)
        font.setBold(False)
        font.setWeight(50)
        self.fon_btn.setFont(font)
        self.fon_btn.setObjectName("fon_btn")
        self.verticalLayout.addWidget(self.fon_btn)
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setSpacing(1)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.lineEdit_mmin = QtWidgets.QLineEdit(self.layoutWidget)
        self.lineEdit_mmin.setMinimumSize(QtCore.QSize(24, 20))
        self.lineEdit_mmin.setMaximumSize(QtCore.QSize(24, 20))
        self.lineEdit_mmin.setMaxLength(2)
        self.lineEdit_mmin.setObjectName("lineEdit_mmin")
        self.horizontalLayout.addWidget(self.lineEdit_mmin)
        self.lineEdit_smin = QtWidgets.QLineEdit(self.layoutWidget)
        self.lineEdit_smin.setMinimumSize(QtCore.QSize(24, 20))
        self.lineEdit_smin.setMaximumSize(QtCore.QSize(24, 20))
        font = QtGui.QFont()
        font.setFamily("MV Boli")
        font.setPointSize(8)
        font.setBold(False)
        font.setWeight(50)
        self.lineEdit_smin.setFont(font)
        self.lineEdit_smin.setMaxLength(2)
        self.lineEdit_smin.setObjectName("lineEdit_smin")
        self.horizontalLayout.addWidget(self.lineEdit_smin)
        self.lineEdit_mmax = QtWidgets.QLineEdit(self.layoutWidget)
        self.lineEdit_mmax.setMinimumSize(QtCore.QSize(24, 20))
        self.lineEdit_mmax.setMaximumSize(QtCore.QSize(24, 20))
        font = QtGui.QFont()
        font.setFamily("MV Boli")
        font.setPointSize(8)
        font.setBold(False)
        font.setWeight(50)
        self.lineEdit_mmax.setFont(font)
        self.lineEdit_mmax.setMaxLength(2)
        self.lineEdit_mmax.setObjectName("lineEdit_mmax")
        self.horizontalLayout.addWidget(self.lineEdit_mmax)
        self.lineEdit_smax = QtWidgets.QLineEdit(self.layoutWidget)
        self.lineEdit_smax.setMinimumSize(QtCore.QSize(24, 20))
        self.lineEdit_smax.setMaximumSize(QtCore.QSize(24, 20))
        font = QtGui.QFont()
        font.setFamily("MV Boli")
        font.setPointSize(8)
        font.setBold(False)
        font.setWeight(50)
        self.lineEdit_smax.setFont(font)
        self.lineEdit_smax.setMaxLength(2)
        self.lineEdit_smax.setObjectName("lineEdit_smax")
        self.horizontalLayout.addWidget(self.lineEdit_smax)
        self.verticalLayout.addLayout(self.horizontalLayout)
        self.horizontalLayout_4.addLayout(self.verticalLayout)
        self.verticalLayout_2 = QtWidgets.QVBoxLayout()
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.vote_btn = QtWidgets.QPushButton(self.layoutWidget)
        self.vote_btn.setMinimumSize(QtCore.QSize(100, 23))
        self.vote_btn.setMaximumSize(QtCore.QSize(100, 23))
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(10)
        font.setBold(False)
        font.setWeight(50)
        self.vote_btn.setFont(font)
        self.vote_btn.setObjectName("vote_btn")
        self.verticalLayout_2.addWidget(self.vote_btn)
        self.label_voice = QtWidgets.QLabel(self.layoutWidget)
        self.label_voice.setMinimumSize(QtCore.QSize(100, 20))
        self.label_voice.setMaximumSize(QtCore.QSize(100, 20))
        self.label_voice.setAlignment(QtCore.Qt.AlignCenter)
        self.label_voice.setObjectName("label_voice")
        self.verticalLayout_2.addWidget(self.label_voice)
        self.horizontalLayout_4.addLayout(self.verticalLayout_2)
        self.verticalLayout_3 = QtWidgets.QVBoxLayout()
        self.verticalLayout_3.setObjectName("verticalLayout_3")
        self.format_btn = QtWidgets.QPushButton(self.layoutWidget)
        self.format_btn.setMinimumSize(QtCore.QSize(100, 23))
        self.format_btn.setMaximumSize(QtCore.QSize(100, 23))
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(10)
        font.setBold(False)
        font.setWeight(50)
        self.format_btn.setFont(font)
        self.format_btn.setObjectName("format_btn")
        self.verticalLayout_3.addWidget(self.format_btn)
        self.label_format = QtWidgets.QLabel(self.layoutWidget)
        self.label_format.setMinimumSize(QtCore.QSize(100, 20))
        self.label_format.setMaximumSize(QtCore.QSize(100, 20))
        self.label_format.setAlignment(QtCore.Qt.AlignCenter)
        self.label_format.setObjectName("label_format")
        self.verticalLayout_3.addWidget(self.label_format)
        self.horizontalLayout_4.addLayout(self.verticalLayout_3)
        self.verticalLayout_6 = QtWidgets.QVBoxLayout()
        self.verticalLayout_6.setObjectName("verticalLayout_6")
        self.spinBox = QtWidgets.QSpinBox(self.layoutWidget)
        self.spinBox.setMinimumSize(QtCore.QSize(61, 26))
        self.spinBox.setMaximumSize(QtCore.QSize(61, 26))
        self.spinBox.setMinimum(8000)
        self.spinBox.setMaximum(16000)
        self.spinBox.setObjectName("spinBox")
        self.verticalLayout_6.addWidget(self.spinBox)
        self.label_rate = QtWidgets.QLabel(self.layoutWidget)
        self.label_rate.setMinimumSize(QtCore.QSize(61, 20))
        self.label_rate.setMaximumSize(QtCore.QSize(61, 20))
        font = QtGui.QFont()
        font.setFamily("MV Boli")
        font.setPointSize(9)
        font.setBold(False)
        font.setWeight(50)
        self.label_rate.setFont(font)
        self.label_rate.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.label_rate.setAlignment(QtCore.Qt.AlignCenter)
        self.label_rate.setObjectName("label_rate")
        self.verticalLayout_6.addWidget(self.label_rate)
        self.horizontalLayout_4.addLayout(self.verticalLayout_6)
        self.verticalLayout_4 = QtWidgets.QVBoxLayout()
        self.verticalLayout_4.setObjectName("verticalLayout_4")
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.fileopen_btn = QtWidgets.QPushButton(self.layoutWidget)
        self.fileopen_btn.setMinimumSize(QtCore.QSize(60, 23))
        self.fileopen_btn.setMaximumSize(QtCore.QSize(60, 23))
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(10)
        font.setBold(False)
        font.setWeight(50)
        self.fileopen_btn.setFont(font)
        self.fileopen_btn.setObjectName("fileopen_btn")
        self.horizontalLayout_2.addWidget(self.fileopen_btn)
        self.file_treat_btn = QtWidgets.QPushButton(self.layoutWidget)
        self.file_treat_btn.setMinimumSize(QtCore.QSize(80, 23))
        self.file_treat_btn.setMaximumSize(QtCore.QSize(80, 23))
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(10)
        font.setBold(False)
        font.setWeight(50)
        self.file_treat_btn.setFont(font)
        self.file_treat_btn.setObjectName("file_treat_btn")
        self.horizontalLayout_2.addWidget(self.file_treat_btn)
        self.file_voice_btn = QtWidgets.QPushButton(self.layoutWidget)
        self.file_voice_btn.setMinimumSize(QtCore.QSize(60, 23))
        self.file_voice_btn.setMaximumSize(QtCore.QSize(60, 23))
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(10)
        font.setBold(False)
        font.setWeight(50)
        self.file_voice_btn.setFont(font)
        self.file_voice_btn.setObjectName("file_voice_btn")
        self.horizontalLayout_2.addWidget(self.file_voice_btn)
        self.del_btn = QtWidgets.QPushButton(self.layoutWidget)
        self.del_btn.setMinimumSize(QtCore.QSize(60, 23))
        self.del_btn.setMaximumSize(QtCore.QSize(60, 23))
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(10)
        font.setBold(False)
        font.setWeight(50)
        self.del_btn.setFont(font)
        self.del_btn.setObjectName("del_btn")
        self.horizontalLayout_2.addWidget(self.del_btn)
        self.verticalLayout_4.addLayout(self.horizontalLayout_2)
        self.lineEdit_rv = QtWidgets.QLineEdit(self.layoutWidget)
        self.lineEdit_rv.setMinimumSize(QtCore.QSize(280, 20))
        self.lineEdit_rv.setMaximumSize(QtCore.QSize(280, 20))
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(8)
        font.setBold(False)
        font.setWeight(50)
        self.lineEdit_rv.setFont(font)
        self.lineEdit_rv.setAlignment(QtCore.Qt.AlignCenter)
        self.lineEdit_rv.setReadOnly(True)
        self.lineEdit_rv.setObjectName("lineEdit_rv")
        self.verticalLayout_4.addWidget(self.lineEdit_rv)
        self.horizontalLayout_4.addLayout(self.verticalLayout_4)
        self.verticalLayout_7.addLayout(self.horizontalLayout_4)
        self.textEdit = QtWidgets.QTextEdit(self.layoutWidget)
        self.textEdit.setMinimumSize(QtCore.QSize(890, 431))
        self.textEdit.setMaximumSize(QtCore.QSize(890, 431))
        self.textEdit.setObjectName("textEdit")
        self.verticalLayout_7.addWidget(self.textEdit)
        self.horizontalLayout_5 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_5.setObjectName("horizontalLayout_5")
        self.lineEdit_down = QtWidgets.QLineEdit(self.layoutWidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.lineEdit_down.sizePolicy().hasHeightForWidth())
        self.lineEdit_down.setSizePolicy(sizePolicy)
        self.lineEdit_down.setMinimumSize(QtCore.QSize(365, 28))
        self.lineEdit_down.setMaximumSize(QtCore.QSize(365, 28))
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(9)
        font.setBold(False)
        font.setWeight(50)
        self.lineEdit_down.setFont(font)
        self.lineEdit_down.setReadOnly(True)
        self.lineEdit_down.setObjectName("lineEdit_down")
        self.horizontalLayout_5.addWidget(self.lineEdit_down)
        self.clean_btn = QtWidgets.QPushButton(self.layoutWidget)
        self.clean_btn.setMinimumSize(QtCore.QSize(75, 28))
        self.clean_btn.setMaximumSize(QtCore.QSize(75, 28))
        font = QtGui.QFont()
        font.setFamily("MV Boli")
        font.setPointSize(9)
        font.setBold(False)
        font.setItalic(False)
        font.setWeight(50)
        self.clean_btn.setFont(font)
        self.clean_btn.setObjectName("clean_btn")
        self.horizontalLayout_5.addWidget(self.clean_btn)
        self.lineEdit = QtWidgets.QLineEdit(self.layoutWidget)
        self.lineEdit.setMinimumSize(QtCore.QSize(191, 28))
        self.lineEdit.setMaximumSize(QtCore.QSize(191, 28))
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(10)
        font.setBold(False)
        font.setWeight(50)
        self.lineEdit.setFont(font)
        self.lineEdit.setReadOnly(True)
        self.lineEdit.setObjectName("lineEdit")
        self.horizontalLayout_5.addWidget(self.lineEdit)
        self.play_btn = QtWidgets.QPushButton(self.layoutWidget)
        self.play_btn.setMinimumSize(QtCore.QSize(74, 28))
        self.play_btn.setMaximumSize(QtCore.QSize(74, 28))
        self.play_btn.setObjectName("play_btn")
        self.horizontalLayout_5.addWidget(self.play_btn)
        self.pause_btn = QtWidgets.QPushButton(self.layoutWidget)
        self.pause_btn.setMinimumSize(QtCore.QSize(74, 28))
        self.pause_btn.setMaximumSize(QtCore.QSize(74, 28))
        self.pause_btn.setObjectName("pause_btn")
        self.horizontalLayout_5.addWidget(self.pause_btn)
        self.stop_btn = QtWidgets.QPushButton(self.layoutWidget)
        self.stop_btn.setMinimumSize(QtCore.QSize(74, 28))
        self.stop_btn.setMaximumSize(QtCore.QSize(74, 28))
        self.stop_btn.setObjectName("stop_btn")
        self.horizontalLayout_5.addWidget(self.stop_btn)
        self.verticalLayout_7.addLayout(self.horizontalLayout_5)
        mainWindow.setCentralWidget(self.centralwidget)

        self.retranslateUi(mainWindow)
        QtCore.QMetaObject.connectSlotsByName(mainWindow)

    def retranslateUi(self, mainWindow):
        _translate = QtCore.QCoreApplication.translate
        mainWindow.setWindowTitle(_translate("mainWindow", "speech"))
        self.treatment_btn.setText(_translate("mainWindow", "????????????????????"))
        self.voice_btn.setText(_translate("mainWindow", "????????????????"))
        self.lineEdit_lv.setText(_translate("mainWindow", "?????????????? ????????: m.min s.min m.max s.max"))
        self.fon_btn.setText(_translate("mainWindow", "??????"))
        self.vote_btn.setText(_translate("mainWindow", "??????????"))
        self.label_voice.setText(_translate("mainWindow", "voice"))
        self.format_btn.setText(_translate("mainWindow", "????????????"))
        self.label_format.setText(_translate("mainWindow", "format"))
        self.label_rate.setText(_translate("mainWindow", "rate"))
        self.fileopen_btn.setText(_translate("mainWindow", "??????????????"))
        self.file_treat_btn.setText(_translate("mainWindow", "????????????????????"))
        self.file_voice_btn.setText(_translate("mainWindow", "????????????????"))
        self.del_btn.setText(_translate("mainWindow", "??????????????"))
        self.lineEdit_rv.setText(_translate("mainWindow", "???????????????? ???????????????????? ??????????"))
        self.clean_btn.setText(_translate("mainWindow", "clean"))
        self.play_btn.setText(_translate("mainWindow", "play"))
        self.pause_btn.setText(_translate("mainWindow", "pause"))
        self.stop_btn.setText(_translate("mainWindow", "stop"))
