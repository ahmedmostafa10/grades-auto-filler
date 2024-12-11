from modules.utils import *

def symbol_train():

    check_mark, question_mark, square, horizontal, vertical = trained_data()
    x_train = np.concatenate((check_mark, question_mark, square, *horizontal, *vertical), axis= 0)

    y_train = np.concatenate((["Check"]*np.shape(check_mark)[0], ["Question"]*np.shape(question_mark)[0], ["Square"]*np.shape(square)[0], ["1_hor"]*np.shape(horizontal[0])[0], ["2_hor"]*np.shape(horizontal[1])[0], ["3_hor"]*np.shape(horizontal[2])[0], ["4_hor"]*np.shape(horizontal[3])[0], ["1_ver"]*np.shape(vertical[0])[0], ["2_ver"]*np.shape(vertical[1])[0], ["3_ver"]*np.shape(vertical[2])[0], ["4_ver"]*np.shape(vertical[3])[0], ["5_ver"]*np.shape(vertical[4])[0]), axis=0)
    
    random_seed = 42  
    random.seed(random_seed)
    np.random.seed(random_seed)

    train_features, test_features, train_labels, test_labels = train_test_split(x_train, y_train, test_size=0.2, random_state=random_seed)

    print('############## Training', " SVM ", "##############")
    model = train_model(train_features, train_labels)

    accuracy = model.score(test_features, test_labels)

    print(model, 'accuracy:', accuracy*100, '%')

    return model


def images_to_features(relative_path):
    absolute_path = os.path.dirname(__file__)
    
    directory = os.path.join(absolute_path, relative_path)

    features_list = []

    for filename in os.listdir(directory):
        f = os.path.join(directory, filename)

        if os.path.isfile(f):
            image = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
            feature_hog, _ = hog_fun(image)
            features_list.append(feature_hog)

    return features_list



def trained_data():

    print('Loading dataset. This will take time ...')

    # Check Mark
    relative_path = "../Data Set/Training Set/Check"
    check_mark = images_to_features(relative_path)

    # Question Mark
    relative_path = "../Data Set/Training Set/QuestionMark"
    question_mark = images_to_features(relative_path)

    # Square
    relative_path = "../Data Set/Training Set/Square"
    square = images_to_features(relative_path)


    # Horizontal Lines
    horizontal_lines = []
    for i in range(1, 5):
        relative_path = "../Data Set/Training Set/Horizontal/" + str(i) 
        horizontal_lines.append(images_to_features(relative_path))
        

    # Verical Lines
    vertical_lines = []
    for i in range(1, 6):
        relative_path = "../Data Set/Training Set/Vertical/" + str(i)
        vertical_lines.append(images_to_features(relative_path))
        
    print('Finished loading dataset.')

    return check_mark, question_mark, square, horizontal_lines, vertical_lines
    


def train_model(x_train, y_train):
    
    model = svm.SVC()
    model.fit(x_train, y_train)

    return model    
