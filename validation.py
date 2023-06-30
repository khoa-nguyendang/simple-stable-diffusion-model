import cv2
from PIL import Image

try:
    file = r"./test.jpeg"
    # img = cv2.imread(file)
    img2 = Image.open(file) # open the image file
    img2.verify() # verify that it is, in fact an image
    # if img is None:
    #     print("invalid img")
    if img2 is None:
        print("invalid img2")
    # print(img.shape)
    print(img2.shape)

        
except Image.UnidentifiedImageError as e:
    
    print("UnidentifiedImageError error {0}".format(e))