from modules.utils import *

# index = 0 -> Code
# index = 1 -> Digit
# index = 2 -> Symbol

def cell_recognition(cell, index, SVM_model):
    out = ""

    if index == 0:
        # detect the code
        out = code_recognition(cell)
    elif index == 1:
        # detect the digit
        out = digits_recognition(cell)
    elif index == 2:
        # detect the symbol
        
        # crop the image 
        cropped=cell[20:np.shape(cell)[0]-20,20:np.shape(cell)[1]-20]

        if np.sum(cropped) == 0:
            out = 'empty'
        else:
            out = symbol_recognition(cell, SVM_model)

    return out


def code_recognition(cell):
    custom_config = r'--oem 3 --psm 6 outputbase digits'
    text = pytesseract.image_to_string(cell, config=custom_config)

    return text


def digits_recognition(cell):
    # TODO
    # Still not so accurate

    sharpen_kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    sharpen = cv2.filter2D(cell, -1, sharpen_kernel)
    thresh = cv2.threshold(sharpen, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    possible_values_for_0 = ['O', 'o']
    possible_values_for_1 = ['|', 'i', 'I', '!', '/', "\\", "T"]
    possible_values_for_2 = ['z', 'zz', 'Zz', 'Z', 'zZ', "ZZ"]
    possible_values_for_4 = ['q']
    possible_values_for_5 = ['So', 's', 'S']
    possible_values_for_7 = ['we']
    possible_values_for_9 = ['a','9)']

    digit = pytesseract.image_to_string(thresh, config='--psm 10 --oem 1')

    trimmed = digit.strip()

    print(trimmed)

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




def symbol_recognition(cell, SVM_model):
  #Getting HOG of the Cell
  hog_features, _ = hog_fun (cell)
  
  # Send cell to the Model
  symbol = SVM_model.predict([hog_features])
 
  return symbol




# def detect_vertical_lines(image):
#     height = image.shape[0] // 4

#     verticalSE = cv2.getStructuringElement(cv2.MORPH_RECT, (1, height))
#     morphResult = cv2.morphologyEx(image, cv2.MORPH_OPEN, verticalSE)
#     contours, hierarchy = cv2.findContours(morphResult, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
#     verticalLinesCnt = len(contours)

#     return verticalLinesCnt


# def detect_horizontal_lines(image):
#     height = image.shape[1] // 4

#     verticalSE = cv2.getStructuringElement(cv2.MORPH_RECT, (height, 1))
#     morphResult = cv2.morphologyEx(image, cv2.MORPH_OPEN, verticalSE)
#     contours, hierarchy = cv2.findContours(morphResult, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
#     horizontalLinesCnt = len(contours)

#     return horizontalLinesCnt


    # # sharpen_kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    # # sharpen = cv2.filter2D(image, -1, sharpen_kernel)
    # # thresh = cv2.threshold(sharpen, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    # edges = cv2.Canny(image, 50, 150, apertureSize=3)


    # lines = cv2.HoughLinesP(edges, 2, np.pi / 180, 110, minLineLength=100, maxLineGap=40)

    # print(lines)

    # ans = np.zeros_like(image)

    # if lines is not None:
    #     for x1,y1,x2,y2 in lines[: , 0]:
    #         # a = np.cos(theta)
    #         # b = np.sin(theta)
    #         # x0 = a * rho
    #         # y0 = b * rho
    #         # x1 = int(x0 + 1000 * (-b))
    #         # y1 = int(y0 + 1000 * (a))
    #         # x2 = int(x0 - 1000 * (-b))
    #         # y2 = int(y0 - 1000 * (a))
    #         cv2.line(ans, (x1,y1), (x2,y2), (255,255,255), 5)

    # Save or display the result

