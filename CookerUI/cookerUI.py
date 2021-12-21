# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'mainwindow.ui'
#
# Created by: PyQt5 UI code generator 5.15.6
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets



class Ui_MainWindow(object):
    progressStatus = 0
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(488, 249)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.down = QtWidgets.QPushButton(self.centralwidget)
        self.down.setGeometry(QtCore.QRect(20, 30, 93, 28))
        self.down.setObjectName("down")
        self.up = QtWidgets.QPushButton(self.centralwidget)
        self.up.setGeometry(QtCore.QRect(360, 30, 93, 28))
        self.up.setObjectName("up")
        self.TestButton = QtWidgets.QPushButton(self.centralwidget)
        self.TestButton.setGeometry(QtCore.QRect(190, 160, 93, 28))
        self.TestButton.setDefault(False)
        self.TestButton.setObjectName("TestButton")
        self.progressBar = QtWidgets.QProgressBar(self.centralwidget)
        self.progressBar.setGeometry(QtCore.QRect(60, 100, 371, 31))
        self.progressBar.setProperty("value", 0)
        self.progressBar.setObjectName("progressBar")
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(180, 70, 161, 16))
        self.label.setObjectName("label")
        self.testOutput = QtWidgets.QLabel(self.centralwidget)
        self.testOutput.setGeometry(QtCore.QRect(390, 160, 71, 16))
        self.testOutput.setObjectName("testOutput")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 488, 26))
        self.menubar.setObjectName("menubar")
        self.menuCook3r = QtWidgets.QMenu(self.menubar)
        self.menuCook3r.setObjectName("menuCook3r")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        self.menubar.addAction(self.menuCook3r.menuAction())
        self.TestButton.setCheckable(True)
        self.TestButton.clicked.connect(self.the_button_was_clicked)
        self.up.clicked.connect(self.progressUp)
        self.down.clicked.connect(self.progressDown)
        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def progressUp(self):  
        if(self.progressStatus < 100):
            self.progressStatus = self.progressStatus + 1
            self.progressBar.setValue(self.progressStatus)
    
    def progressDown(self):  
        if(self.progressStatus > 0):
            self.progressStatus = self.progressStatus - 1
            self.progressBar.setValue(self.progressStatus)

    def the_button_was_clicked(self):  
        _translate = QtCore.QCoreApplication.translate
        self.testOutput.setText(_translate("MainWindow", "ButtonPressed"))    

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.down.setText(_translate("MainWindow", "ProgressDown"))
        self.up.setText(_translate("MainWindow", "ProgressUp"))
        self.TestButton.setText(_translate("MainWindow", "Testbutton"))
        self.label.setText(_translate("MainWindow", "Cooking Progress"))
        self.testOutput.setText(_translate("MainWindow", "Testoutput"))
        self.menuCook3r.setTitle(_translate("MainWindow", "Cook3r"))

if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
