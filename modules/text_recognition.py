from modules.utils import *

def text_recognition(cell):
    # out = pytesseract.image_to_string(cell,config="outputbase digits")

    return digits_recognition(cell)

def digits_recognition(image):
    sharpen_kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    sharpen = cv2.filter2D(image, -1, sharpen_kernel)
    thresh = cv2.threshold(sharpen, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    possible_values_for_0 = ['O', 'o']
    possible_values_for_1 = ['|', 'i', 'I', '!', '/', "\\", "T"]
    possible_values_for_2 = ['z', 'zz', 'Zz', 'Z', 'zZ', "ZZ"]
    possible_values_for_4 = ['q']
    possible_values_for_5 = ['So', 's', 'S']
    possible_values_for_7 = ['we']
    possible_values_for_9 = ['a','9)']

    digit = pytesseract.image_to_string(thresh, lang='eng', config='--psm 6')
    trimmed = digit.strip()
    if(trimmed in possible_values_for_0):
        trimmed = "0"
    if(trimmed in possible_values_for_1):
        trimmed = "1"
    if(trimmed in possible_values_for_2):
        trimmed = "2"
    if(trimmed in possible_values_for_4):
        trimmed = "4"
    if(trimmed in possible_values_for_5):
        trimmed = "5"
    if(trimmed in possible_values_for_7):
        trimmed = "7"
    if(trimmed in possible_values_for_9):
        trimmed = "9"
    return trimmed