import sys
from PyQt5 import QtGui, QtWidgets
import Ui_change2sketch
from change_style import get_model, get_predict
import numpy as np
from tensorflow import keras



def open_file():
    filePath, _ = QtWidgets.QFileDialog.getOpenFileName()
    if filePath == '':
        return
    global pic
    pic = get_predict(model, filePath)
    ui.label.setPixmap(QtGui.QPixmap(filePath))
    height, width, _= pic.shape
    bytesPerLine = 3 * width
    qImg = QtGui.QImage(pic.data, width, height, bytesPerLine, QtGui.QImage.Format_BGR888).rgbSwapped()
    ui.label_2.setPixmap(QtGui.QPixmap(QtGui.QPixmap.fromImage(qImg)))
    ui.pushButton_2.setEnabled(True)

def save():
    filePath, _= QtWidgets.QFileDialog.getSaveFileName(filter='JPG(*.jpg)')
    if filePath == '':
        return
    prediction = keras.utils.array_to_img(pic)
    prediction.save(filePath)
 

if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_change2sketch.Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    ##### Start
    model = get_model()
    pic = np.array([])
    ui.pushButton_2.setEnabled(False)
    ui.pushButton.clicked.connect(open_file)
    ui.pushButton_2.clicked.connect(save)


    sys.exit(app.exec_())