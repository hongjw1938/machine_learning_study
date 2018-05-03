import sys
from PyQt5.QtWidgets import *

class MyWindow(QMainWindow): #상속
    def __init__(self):
        super().__init__()
        self.setupUI()
        
    def setupUI(self):
        self.setWindowTitle("Review")
        btn1 = QPushButton("Click me", self)
        btn1.move(20, 20) #최상위 객체에서 x, y로 20씩 이동
        btn1.clicked.connect(self.btn1_clicked)
    
    def btn1_clicked(self):
        QMessageBox.about(self, "message", "clicked")
    
if __name__ == "__main__":
    app = QApplication(sys.argv)
    mywindow = MyWindow()
    mywindow.show()
    app.exec_()
        