#import the necessary packages
import numpy as np
import cv2


#draw a rectangle
#create a black image
rectangle = np.zeros((300,300),dtype="uint8")

#draw rectangle in the black image
#------------------------------------------------------------------------------------------------------
#1st- image
#2nd- top left corner of rectangle
#3rd- bottom right corner of rectangle
#4th- color of the shape (pass RGB value like(255,0,0),for grayscale pass scalar value like 255)
#5th- thickness (if -1 is passed, it will fill the shape and default value is 1)
cv2.rectangle(rectangle, (25,25), (275,275), 255, -1)
cv2.imshow("Rectangle", rectangle)

#draw a circle
#create  a black image
circle=np.zeros((300,300),dtype="uint8")
# draw a circle in the black image
#----------------------------------------------------------------------------------------------------------------------
# 1st-image
# 2nd- center coordinate
# 3rd- radius
# 4th- color of the shape (pass RGB value like(255,0,0),for grayscale pass scalar value like 255)
# 5th- thickness (if -1 is passed, it will fill the shape and default value is 1)
cv2.circle(circle, (150,150),150,255,-1)
cv2.imshow("Circle",circle)

# a bitwise 'AND' is only 'True' when both inputs have a value that
# is "ON" -- in this case, the cv2.bitwise_and function examines
# every pixel in the rectangle and circle; if *BOTH* pixels have a
# value greater than zero then the pixel is turned 'ON' (i.e, 255)
# in the output image; otherwise, the output value is set to
# 'OFF' (i.e, 0)
bitwiseAnd = cv2.bitwise_and(rectangle, circle)
cv2.imshow("AND", bitwiseAnd)


# a bitwise 'OR' examines every pixel in the two inputs, and if
# *EITHER* pixel in the rectangle or circle is greater than zero,
# then the output pixel has a value of 255, otherwise it is 0
bitwiseOr = cv2.bitwise_or(rectangle, circle)
cv2.imshow("OR", bitwiseOr)

# the bitwise 'XOR' is identical to the 'OR' function, with one
# exception: both the rectangle and circle are not allowed to *BOTH*
# have values greater than 0 (only one can be 0)
bitwiseXor = cv2.bitwise_xor(rectangle, circle)
cv2.imshow("XOR", bitwiseXor)

# finally, the bitwise 'NOT' inverts the values of the pixels; pixels
# with a value of 255 become 0, and pixels with a value of 0 become
# 255
bitwiseNot = cv2.bitwise_not(circle)
cv2.imshow("NOT", bitwiseNot)

cv2.waitKey(0)
cv2.destroyAllWindows()

