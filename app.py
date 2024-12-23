import uuid
from modules.utils import *
import modules.cell_recognition as cr
import modules.symbol_train as st
import modules.extract_table_with_cells as et
import modules.data_parsing as dp
import modules.excel_generation as eg
import modules.extract_bubbles as eb

pytesseract.pytesseract.tesseract_cmd = 'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'

filename = 'symbols_model.sav'


def main():
    
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

    image_path="./Data set/grade sheet/13.jpg"
    model=1
    if(model == 1):
        cropped_matrix, warped_colored_matrix, c = et.ExtractTableWithCells(image_path)
        #show_images([cropped_matrix[0][3]],[""])
        switcher = { 0:0, 1:1, 2:2, 3:2}
        SVM_model = pickle.load(open(filename, 'rb'))    
        file = open('example.txt', 'w')

        for i, row in enumerate(cropped_matrix):
            if i != 0:
                line = ""
                for j, ele in enumerate(row):
                    # index = 0 -> Code
                    # index = 1 -> Digit
                    # index = 2 -> Symbol
                    # ele = cv2.cvtColor(ele, cv2.COLOR_BGR2GRAY)
                    # print(ele.shape)
                    # cut only if code
                    ele = cv2.resize(ele, (185, 220))
                    if j == 0:
                        ele = ele[60 : 170, 0 : 200]
                    else:
                        ele = ele[5 : 180, 15 : 220]

                    # print(ele.shape)
                    # ele = cv2.bitwise_not(ele)
                    # print(ele)
                    # for x in ele: 
                    #     print(' '.join(map(str, x)))

                    # show_images([ele],[""])
                    io.imshow(ele)
                    io.show()
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
    else:
        studentId,bubbles_matrix = eb.extract_bubbles(image_path)
        studentId=np.array(studentId)
        gray_image=(rgb2gray(studentId)*255).astype(np.uint8)
        studentId = (gray_image>160).astype(np.uint8)
        studentId=cv2.bitwise_not(studentId)
        show_images([studentId],[" "])
        id= cr.cell_recognition(studentId,0,None)
        print(id)

        answers=cr.bubble_answers(bubbles_matrix)
        for i in range(len(answers)):
            print(f"answer {i+1}: {answers[i]}")
        with open('./answer.txt', 'r') as file:
            letters = [line.strip() for line in file]
        print(letters)
        markdown=[]
        markdown.append(id)  # u should remove these two lines after appending id from photo
        for i in range(len(answers)):
            if(answers[i][0]==letters[i]):
                markdown.append(1)
            else:
                markdown.append(0)
        with open('output.txt', 'w') as file:
        # Write elements separated by spaces
            file.write(' '.join(map(str, markdown)))
        
        (dp.parse_exam_data("output.txt",len(answers))).to_csv('./output.csv',index=False)
        eg.generate_excel_from_csv('./output.csv', './output.xlsx')
        
if __name__ == "__main__":
    main()