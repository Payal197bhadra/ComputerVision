import argparse
import cv2

# ap = argparse.ArgumentParser()
# ap.add_argument("--i", "--image", required=True, help="path to input image")
# args= vars(ap.parse_args())


# image = cv2.imread(args["image"])
image = cv2.imread("lena.png")
(h, w, c)= image.shape[:3]

#height= no of rows, width= no of columns, channnel= no of channels

print("Width : {} pixels". format(w))
print("Height : {} pixels". format(h))
print("Channels : {} pixels". format(c))

cv2.imshow("Image", image)
cv2.waitKey(0)

cv2.imwrite("New Image.jpg", image)