import argparse
import cv2

# ap = argparse.ArgumentParser()
# ap.add_argument("--i", "--image", required=True, help="path to input image")
# args= vars(ap.parse_args())


# image = cv2.imread(args["image"])
image = cv2.imread("lena.png")
(h, w)= image.shape[:2]
cv2.imshow("Original Image", image)
#height= no of rows, width= no of columns, channnel= no of channels

#images are simply numpy arrays- with the origin at (0,0) that is top left
#value of the image
(b, g, r) = image[0,0]
print("Pixel at (0,0) - Red: {}, Green: {}, Blue: {}".format(r,g,b))

#image[y, x] no of rows, no of columns
image[20, 50]=(0, 0, 255)
(b, g, r) = image[20, 50]
print("Pixel at (20,50) - Red: {}, Green: {}, Blue: {}".format(r,g,b))

#since we are using Numpy arrays, we can apply array slicing to grab large
#chunks/regions of interest from the image-- here we grab the top left
#corner of the image
(cX, cY) = (w//2, h//2)

tl= image[0:cY, 0:cX]
cv2.imshow("Top Left corner", tl)

#in a similar fashiom, we can crop the top-right, bottom-right, and bottom-left
#corner of the image and then display them to our screen
tr= image[0:cY, cX:w]
br= image[cY:h, cX:w]
bl= image[cY:h, 0:cX]
cv2.imshow("Top-Right Corner", tr)
cv2.imshow("Bottom-Right Corner", br)
cv2.imshow("Bottom-left Corner", bl)

#set the top-right corner of the original image ti be green
image[0:cY, cX:w]=(0, 255, 0)
cv2.imshow("Upadted Top Right Green Image", image)
cv2.waitKey(0)

cv2.imwrite("New Image.jpg", image)