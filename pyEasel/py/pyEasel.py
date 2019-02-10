import sys
sys.path.append('./')
from PyQt5.QtWidgets import QApplication
import GLViewer as glv
import EaselModule

from PyQt5.QtWidgets import QMainWindow
class Easel(QMainWindow):
    def __init__(self, Modules=[]):
        super().__init__()
        self.Modules = Modules
        print(len(self.Modules))

    def init(self):
        # for(Mod in self.Modules):
        #     print(Mod)
        pass

    def draw(self):
        pass

if __name__ == '__main__':
    app = QApplication(sys.argv)

    mainWindow = glv.GLViewer()
    mainWindow.show()
    sys.exit(app.exec_())