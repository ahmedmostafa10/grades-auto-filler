from modules.utils import *
import modules.text_recognition as tr

pytesseract.pytesseract.tesseract_cmd = 'C:\Program Files\Tesseract-OCR/tesseract.exe'

def main():
    
    img = cv2.imread("./imgs/120.jpg")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    print("The number is " + tr.text_recognition(img))


if __name__ == "__main__":
    main()