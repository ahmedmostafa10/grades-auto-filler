from modules.utils import *

def text_recognition(cell):
    # out = pytesseract.image_to_string(cell,config="outputbase digits")

    return digits_recognition(cell)



def digits_recognition(image):
    sharpen_kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    sharpen = cv2.filter2D(image, -1, sharpen_kernel)
    thresh = cv2.threshold(sharpen, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]


    edges = cv2.Canny(image, 50, 150, apertureSize=3)

    lines = cv2.HoughLines(edges, 1, np.pi / 180, 120)


    ans = np.zeros_like(image)

    if lines is not None:
        for rho, theta in lines[:, 0]:
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            x1 = int(x0 + 1000 * (-b))
            y1 = int(y0 + 1000 * (a))
            x2 = int(x0 - 1000 * (-b))
            y2 = int(y0 - 1000 * (a))
            cv2.line(ans, (x1, y1), (x2, y2), (255, 255, 255), 5)

    # Save or display the result
    cv2.imwrite('lines_detected.jpg', ans)
    cv2.imshow('Detected Lines', ans)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


    return

    # possible_values_for_0 = ['O', 'o']
    # possible_values_for_1 = ['|', 'i', 'I', '!', '/', "\\", "T"]
    # possible_values_for_2 = ['z', 'zz', 'Zz', 'Z', 'zZ', "ZZ"]
    # possible_values_for_4 = ['q']
    # possible_values_for_5 = ['So', 's', 'S']
    # possible_values_for_7 = ['we']
    # possible_values_for_9 = ['a','9)']

    # digit = pytesseract.image_to_string(thresh, config='--psm 10 --oem 1')

    # trimmed = digit.strip()

    # print(trimmed)

    # if(trimmed in possible_values_for_0):
    #     trimmed = "0"
    # if(trimmed in possible_values_for_1):
    #     trimmed = "1"
    # if(trimmed in possible_values_for_2):
    #     trimmed = "2"
    # if(trimmed in possible_values_for_4):
    #     trimmed = "4"
    # if(trimmed in possible_values_for_5):
    #     trimmed = "5"
    # if(trimmed in possible_values_for_7):
    #     trimmed = "7"
    # if(trimmed in possible_values_for_9):
    #     trimmed = "9"
    # return trimmed