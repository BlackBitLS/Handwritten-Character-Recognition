import keras
import cv2
import numpy as np
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
import PyQt5.uic
import sys  # sys нужен для передачи argv в QApplication
from PyQt5 import QtWidgets
from PyQt5.QtWidgets import *
from PyQt5.QtCore import Qt, QPoint
from PyQt5.QtGui import *
from itrf import Gui


# скачиваем данные и разделяем на набор для обучения и тестовый
(x_train, y_train), (x_test, y_test) = mnist.load_data()

print(x_train.shape, y_train.shape)

num_classes = 10
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
input_shape = (28, 28, 1)

# преобразование векторных классов в бинарные матрицы
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print('Размерность x_train:', x_train.shape)
print(x_train.shape[0], 'Размер train')
print(x_test.shape[0], 'Размер test')


# Создание модели CNN
batch_size = 128
epochs = 2

model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),activation='relu',input_shape=input_shape))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))
opt = keras.optimizers.Adam(learning_rate=0.001)
model.compile(loss=keras.losses.categorical_crossentropy,optimizer= opt,metrics=['accuracy'])

# Обучение модели
hist = model.fit(x_train, y_train, batch_size = batch_size, epochs=epochs, verbose=1, validation_data=(x_test, y_test))
print("Модель успешно обучена")

#model.save('mnist.h5')
print("Модель сохранена как mnist.h5")
score = model.evaluate(x_test, y_test, verbose=0)
print('Потери при обучении:', score[0])
print('Точность при обучении:', score[1])


class Window(QtWidgets.QMainWindow, Gui.Ui_MainWindow):
    def __init__(self):
        # Это здесь нужно для доступа к переменным, методам
        # и т.д. в файле design.py
        super().__init__()
        self.setWindowFlag(Qt.MSWindowsFixedSizeDialogHint)
        self.setupUi(self)  # Это нужно для инициализации нашего дизайна
        self.setWindowTitle("Распознавание рукописных символов")
        self.drawing = False
        self.brushSize = 5
        self.brushColor = Qt.black
        self.lastPoint = QPoint()
        self.image = QImage(250, 250, QImage.Format_RGB32)
        self.image.fill(Qt.white)
        self.lineEdit_2.setReadOnly(True)
        self.lineEdit_2.setAlignment(Qt.AlignCenter)
        self.label.setStyleSheet("border: 2px solid blue; color: green")
        self.pushButton.clicked.connect(self.predict_digit2)
        self.pushButton_2.clicked.connect(self.clear)
        self.pushButton_3.clicked.connect(self.download)
        self.pushButton_4.clicked.connect(self.save)
        self.pushButton_4.clicked.connect(self.predict_digit)

    def download(self):
        filename, filetype = QFileDialog.getOpenFileName(self,
                                                         "Выбрать файл",
                                                         ".",
                                                         "JPG Files(*.jpg);;JPEG Files(*.jpeg);;\
                                                         PNG Files(*.png);;All Files(*)")
        self.lineEdit.setText(filename)
        pixmap = QPixmap()
        pixmap.load(filename)
        self.label.setPixmap(pixmap)
        self.label.setScaledContents(True)


    def predict_digit2(self):
        path = self.lineEdit.text()
        img = cv2.imread(fr'{path}')
        img_copy = img.copy()
        #img = cv2.resize(img, (300, 300))
        #Преобразование изображения в оттенки серого
        gray = cv2.cvtColor(img_copy, cv2.COLOR_BGR2GRAY)
        #Преобразование изоражения в ч/б цвет
        res, thresh = cv2.threshold(gray, 125, 255, cv2.THRESH_BINARY_INV)
        #Расширяет изображение
        dil = cv2.dilate(thresh, np.ones((38, 38), np.uint8), iterations=1)
        #Эрозия изображения
        erod = cv2.erode(dil, np.ones((12, 12), np.uint8), iterations=1)
        # изменение размерности для поддержки модели ввода
        img_res = cv2.resize(erod, (28, 28))
        img_final = np.reshape(img_res, (1, 28, 28, 1))

        # предстказание цифры
        res = model.predict([img_final])[0]
        digit= np.argmax(res)
        sdigit = str(digit)
        self.lineEdit_2.setText(sdigit)



    def predict_digit(self):

            img = cv2.imread(r'E:\ProgrVKR\res.jpeg', cv2.IMREAD_COLOR)
            img_copy = img.copy()
            #Преобразование изображения в оттенки серого
            gray = cv2.cvtColor(img_copy, cv2.COLOR_BGR2GRAY)
            #Преобразование изоражения в ч/б цвет
            res, tresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            #изменение размерности изображения для поддержки модели ввода
            img_res = cv2.resize(tresh, (28, 28), interpolation=cv2.INTER_AREA)
            #cv2.imshow("Enlarged", img_res)
            img_final = np.reshape(img_res, (1, 28, 28, 1))

            # предстказание цифры
            res = model.predict([img_final])[0]
            digit = np.argmax(res)
            sdigit = str(digit)
            self.lineEdit_2.setText(sdigit)

    # Событие нажатие мыши
    def mousePressEvent(self, event):
            if event.button() == Qt.LeftButton:
                self.drawing = True
                self.lastPoint = event.pos()

    # Событие движения мыши
    def mouseMoveEvent(self, event):
            if (event.buttons() & Qt.LeftButton) & self.drawing:
                painter = QPainter(self.image)
                painter.setPen(QPen(self.brushColor, self.brushSize, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
                painter.drawLine(self.lastPoint, event.pos())
                self.lastPoint = event.pos()
                self.update()
     # Событие отпускание мыши
    def mouseReleaseEvent(self, event):

        if event.button() == Qt.LeftButton:
            self.drawing = False
    #Рисование
    def paintEvent(self, event):
            canvasPainter = QPainter(self)
            canvasPainter.drawImage(0, 0, self.image)


    def save(self):
            self.image.save('res.jpeg')

    def clear(self):
            self.image.fill(Qt.white)
            self.lineEdit_2.clear()
            self.update()



def main():
    app = QtWidgets.QApplication(sys.argv)  # Новый экземпляр QApplication
    ui = PyQt5.uic.loadUi('itrf/Gui.ui')
    ui = Window()  # Создаём объект класса ExampleApp
    ui.show()  # Показываем окно
    app.exec_()  # и запускаем приложение


if __name__ == '__main__':  # Если мы запускаем файл напрямую, а не импортируем
    model = keras.models.load_model('mnist.h5')
    main()  # то запускаем функцию main()





