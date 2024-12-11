from modules.utils import *
import modules.cell_recognition as cr
import modules.symbol_train as st

pytesseract.pytesseract.tesseract_cmd = 'C:\Program Files\Tesseract-OCR/tesseract.exe'

filename = 'symbols_model.sav'


def main():
    SVM_model = pickle.load(open(filename, 'rb'))

    cell = cv2.imread("./Data Set/Training Set/Vertical/4/010.jpg")
    cell = cv2.cvtColor(cell, cv2.COLOR_BGR2GRAY)

    print(str(cr.cell_recognition(cell, 2, SVM_model)))


if __name__ == "__main__":
    main()