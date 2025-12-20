import sys
import matplotlib
matplotlib.use("QtAgg")

import matplotlib.pyplot as plt
from PyQt6.QtWidgets import QApplication

print("Python:", sys.executable)
print("Matplotlib backend:", matplotlib.get_backend())

# Qt event loop garanti olsun diye (özellikle VS Code/terminal kombinasyonlarında iyi gelir)
app = QApplication.instance() or QApplication([])

plt.plot([1, 2, 3], [1, 4, 9])
plt.title("QtAgg test")
plt.show()
