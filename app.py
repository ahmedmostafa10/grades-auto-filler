import uuid
from modules.utils import *
import modules.cell_recognition as cr
import modules.preprocessing as pp
import modules.symbol_train as st
import modules.extract_table_with_cells as et
import modules.data_parsing as dp
import modules.excel_generation as eg

pytesseract.pytesseract.tesseract_cmd = 'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'

filename = 'symbols_model.sav'


def main():
    SVM_model = pickle.load(open(filename, 'rb'))    

    pp.preprocess_image("./Data set/grade sheet/15.jpg")

    cropped_matrix, warped_colored_matrix, c = et.ExtractTableWithCells("./saved_image.jpg")

    switcher = { 0:0, 1:1, 2:2, 3:2}

    file = open('example.txt', 'w')

    for i, row in enumerate(cropped_matrix):
        if i != 0:
            line = ""
            for j, ele in enumerate(row):
                # index = 0 -> Code
                # index = 1 -> Digit
                # index = 2 -> Symbol

                ele = cv2.cvtColor(ele, cv2.COLOR_BGR2GRAY)
                
                # print(ele.shape)
                # cut only if code

                # ele = cv2.resize(ele, (185, 220))

                if j == 0:
                    ele = ele[60 : 170, 0 : 200]
                else:
                    ele = ele[5 : 180, 15 : 220]

                # print(ele.shape)

                ele = cv2.bitwise_not(ele)
                # io.imshow(ele)
                # io.show()

                # io.imsave("./train/" + str(uuid.uuid4()) + ".jpg", ele)
                
                ans = str(cr.cell_recognition(ele, switcher.get(j), SVM_model))

                # print(ans)

                line = line + " " + ans

            print(line)
            file.write(line + "\n")

    file.close()

    df = dp.parse_exam_data('example.txt', 3)

    df.to_csv('./example.csv',index=False)
    
    eg.generate_excel_from_csv('./example.csv', './example.xlsx')

if __name__ == "__main__":
    main()