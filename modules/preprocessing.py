from modules.utils import *


def show_images(images,titles=None):
    #This function is used to show image(s) with titles by sending an array of images and an array of associated titles.
    # images[0] will be drawn with the title titles[0] if exists
    # You aren't required to understand this function, use it as-is.
    n_ims = len(images)
    if titles is None: titles = ['(%d)' % i for i in range(1,n_ims + 1)]
    fig = plt.figure()
    n = 1
    for image,title in zip(images,titles):
        a = fig.add_subplot(1,n_ims,n)
        if image.ndim == 2: 
            plt.gray()
        plt.imshow(image)
        a.set_title(title)
        n += 1
    fig.set_size_inches(np.array(fig.get_size_inches()) * n_ims)
    plt.show()

def apply_threshold(image, threshold):
    h,w=image.shape
    for i in range (h):
        for j in range (w):
            image[i][j]=image[i][j]>threshold
    return image

def load_image_with_orientation(image_path):
    image = Image.open(image_path)
    for orientation in ExifTags.TAGS.keys():
        if ExifTags.TAGS[orientation] == 'Orientation':
            break
    exif = image._getexif()
    if exif is not None:
        orientation = exif.get(orientation)
        if orientation == 3:
            image = image.rotate(180, expand=True)
        elif orientation == 6:
            image = image.rotate(270, expand=True)
        elif orientation == 8:
            image = image.rotate(90, expand=True)
    return np.array(image)


def preprocess_gradeSheet(image_path):
    image = load_image_with_orientation(image_path)
    #show_images([image],["original image"])
    gray_image= (rgb2gray(image[:,:,0:3])*255).astype(np.uint8)
    #show_images([gray_image],["gray_image"])
    Thresholded_Img = cv2.adaptiveThreshold(gray_image, 255,cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 21, 10)

    contours, _ = cv2.findContours(Thresholded_Img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour = max(contours, key=cv2.contourArea)
    epsilon = 0.02 * cv2.arcLength(contour, True)
    paper_corners = cv2.approxPolyDP(contour, epsilon, True)
    
    if len(paper_corners) == 4:
        paper_corners = paper_corners.reshape(4, 2).astype("float32")
    else:
        raise ValueError("Couldn't detect a quadrilateral from the contour.")
    
    sum_coords = paper_corners.sum(axis=1)
    diff_coords = np.diff(paper_corners, axis=1)
    
    top_left = paper_corners[np.argmin(sum_coords)]
    bottom_right = paper_corners[np.argmax(sum_coords)]
    top_right = paper_corners[np.argmin(diff_coords)]
    bottom_left = paper_corners[np.argmax(diff_coords)]
    
    paper_corners = np.array([top_left, top_right, bottom_right, bottom_left], dtype="float32")
    
    width_top = np.linalg.norm(paper_corners[0] - paper_corners[1])  # Top edge
    width_bottom = np.linalg.norm(paper_corners[2] - paper_corners[3])  # Bottom edge
    height_left = np.linalg.norm(paper_corners[0] - paper_corners[3])  # Left edge
    height_right = np.linalg.norm(paper_corners[1] - paper_corners[2])  # Right edge
    
    width = int(max(width_top, width_bottom)) 
    height = int(max(height_left, height_right))

    print(width, height)
    
    dst_points = np.array([
        [0, 0],
        [width - 1, 0],
        [width - 1, height - 1],
        [0, height - 1]
    ], dtype="float32")

    M = cv2.getPerspectiveTransform(paper_corners, dst_points)

    # Apply the warp
    warped_image = cv2.warpPerspective(Thresholded_Img, M, (width, height))
    return warped_image



def preprocess_bubbleSheet(image_path):
    image = load_image_with_orientation(image_path)
    #show_images([image],["original image"])
    gray_image= (rgb2gray(image[:,:,0:3])*255).astype(np.uint8)
    #gray_image = cv2.GaussianBlur(gray_image, (5, 5), 0)
    _, Thresholded_Img = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Find contours
    contours, _ = cv2.findContours(Thresholded_Img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour = max(contours, key=cv2.contourArea)

    # Approximate contour to polygon
    epsilon = 0.01 * cv2.arcLength(contour, True)
    paper_corners = cv2.approxPolyDP(contour, epsilon, True)

    # Debug information
    #print(f"Number of corners detected: {len(paper_corners)}")

    if len(paper_corners) == 4:
        paper_corners = paper_corners.reshape(4, 2).astype("float32")
    else:
        raise ValueError("Couldn't detect a quadrilateral from the contour.")

    #print(paper_corners)
    width_top = np.linalg.norm(paper_corners[0] - paper_corners[1])  # Top edge
    width_bottom = np.linalg.norm(paper_corners[2] - paper_corners[3])  # Bottom edge
    height_left = np.linalg.norm(paper_corners[0] - paper_corners[3])  # Left edge
    height_right = np.linalg.norm(paper_corners[1] - paper_corners[2])  # Right edge

    # Average the dimensions to get a more accurate width and height
    width = int(max(width_top, width_bottom))  # Use max or mean if desired
    height = int(max(height_left, height_right))
    # if width > height:
    #     width, height = height, width
    #print(width,height)
    dst_points = np.array([
        [0, 0],
        [width - 1, 0],
        [width - 1, height - 1],
        [0, height - 1]
    ], dtype="float32")

    M = cv2.getPerspectiveTransform(paper_corners, dst_points)
    # Apply the warp
    warped_image = cv2.warpPerspective(Thresholded_Img, M, (width, height))
    final_img = cv2.flip(warped_image, 1)
    return final_img