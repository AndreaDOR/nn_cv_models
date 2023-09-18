from myUi import Ui_Form
import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog
import sys
from PyQt5 import QtWidgets, QtCore, QtGui
from PyQt5.QtWidgets import *
from imgPred import predict


class myUi(QMainWindow, Ui_Form):
    def __init__(self, parent=None):
        super(myUi, self).__init__(parent)
        self.setupUi(self)
        self.selectBut.clicked.connect(self.loadImg)

    def loadImg(self):
        self.fname, _ = QFileDialog.getOpenFileName(
            self, "请选择图片", ".", "图像文件(*.jpg *.jpeg *.png)"
        )
        if self.fname:
            self.predText.setText("打开文件成功")
            jpg = QtGui.QPixmap(self.fname).scaled(
                self.imgL.width(), self.imgL.height()
            )
            self.imgL.setPixmap(jpg)
            result, acc = predict(self.fname, "resnet34_pretrain.pth")
            self.predText.setText(f"预测结果{result},准确率:{acc:>0.3f}")
        else:
            self.predText.setText("打开文件失败")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    ui = myUi()
    ui.show()
    sys.exit(app.exec_())
