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
            out = "-"
        else:
            out = symbol_recognition(cell, SVM_model)

    if out == "Square":
        out = "0"

    elif out == "1_ver":
        out = "1"
    elif out == "2_ver":
        out = "2"
    elif out == "3_ver":
        out = "3"
    elif out == "4_ver":
        out = "4"
    elif out == "5_ver":
        out = "5"

    elif out == "1_hor":
        out = "4"
    elif out == "2_hor":
        out = "3"
    elif out == "3_hor":
        out = "2"
    elif out == "4_hor":
        out = "1"

    elif out == "Question":
        out = '?'
    elif out == "Check":
        out = "5"

    return out


def code_recognition(cell):
    custom_config = r'--oem 3 --psm 6 outputbase digits'
    text = pytesseract.image_to_string(cell, config=custom_config)

    ans = text.rstrip()

    return ans if ans != "" else "0000000"


def digits_recognition(cell):
    # TODO
    # Still not so accurate
    sharpen_kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    cell = cv2.filter2D(cell, -1, sharpen_kernel)

    cell = cv2.threshold(cell, 128, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    cell = cv2.medianBlur(cell, 3)
    cell = cv2.resize(cell, (300, 300))


    digit = pytesseract.image_to_string(cell, config='--psm 13 --oem 3 -c tessedit_char_whitelist=0123456789')

    possible_values_for_0 = ['0', 'O', 'o']
    possible_values_for_1 = ['1', '|', 'i', 'I', '!', '/', "\\", "T"]
    possible_values_for_2 = ['2', 'z', 'zz', 'Zz', 'Z', 'zZ', "ZZ"]
    possible_values_for_3 = ['3']
    possible_values_for_4 = ['4', 'q']
    possible_values_for_5 = ['5', 'So', 's', 'S']
    possible_values_for_6 = ['6', 'b']
    possible_values_for_7 = ['7', 'we']
    possible_values_for_8 = ['8', 'g']
    possible_values_for_9 = ['9', 'a','9)']

    print(digit)
    digit = digit.rstrip()
    trimmed = digit.strip()

    # print(trimmed)

    if(trimmed in possible_values_for_0):
        trimmed = "0"
    if(trimmed in possible_values_for_1):
        trimmed = "1"
    if(trimmed in possible_values_for_2):
        trimmed = "2"
    if(trimmed in possible_values_for_3):
        trimmed = "3"
    if(trimmed in possible_values_for_4):
        trimmed = "4"
    if(trimmed in possible_values_for_5):
        trimmed = "5"
    if(trimmed in possible_values_for_6):
        trimmed = "6"
    if(trimmed in possible_values_for_7):
        trimmed = "7"
    if(trimmed in possible_values_for_8):
        trimmed = "8"
    if(trimmed in possible_values_for_9):
        trimmed = "9"

    return trimmed




def symbol_recognition(cell, SVM_model):
  #Getting HOG of the Cell
  hog_features, _ = hog_fun (cell)
  
  # Send cell to the Model
  symbol = SVM_model.predict([hog_features])
 
  return symbol[0]
def show_images(images, titles=None):
        n_ims = len(images)
        if titles is None:
            titles = ['(%d)' % i for i in range(1, n_ims + 1)]
        fig = plt.figure()
        n = 1
        for image, title in zip(images, titles):
            a = fig.add_subplot(1, n_ims, n)
            if image.ndim == 2:
                plt.gray()
            plt.imshow(image)
            a.set_title(title)
            n += 1
        fig.set_size_inches(np.array(fig.get_size_inches()) * n_ims)
        plt.show()
def bubble_selection(row):
    row_answers=[]
    maxx=0
    ans=0
    counts=[]
    for i in range(len(row)):
        # _, binary_image = cv2.threshold(row[i], 127, 255, cv2.THRESH_BINARY)
        binary_image = (row[i]>160)
        # kernel = np.ones((3, 3), np.uint8)
        # dilated_image = cv2.dilate(binary_image, kernel, iterations=1)
        # eroded_image = cv2.erode(dilated_image, kernel, iterations=1)
        black_pixel_count = np.sum(binary_image == 0)
        counts.append(black_pixel_count)
        if(black_pixel_count>maxx):
            maxx=black_pixel_count
            ans=i
    thersould= 0.97* maxx
    for i in range(len(row)):
        if(counts[i]>thersould):
            if(i==0):
                row_answers.append('A')
            elif(i==1):
                row_answers.append('B')
            elif(i==2):
                row_answers.append('C')
            elif(i==3):
                row_answers.append('D')
            else:
                row_answers.append('E')
    return row_answers

def bubble_answers(bubbles_matrix):
    columns=len(bubbles_matrix)
    rowsPerColoumn=len(bubbles_matrix[0])
    answers=[]
    for i in range(columns):
        for j in range(rowsPerColoumn):
            row_answers=bubble_selection(bubbles_matrix[i][j])
            answers.append(row_answers)

    return answers



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

