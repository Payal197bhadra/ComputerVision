import numpy as np
import matplotlib.pyplot as plt
import cv2
def imagehistogram(img, do_print):
    img=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_hist=np.zeros([256])
    print(img_hist.shape)
    for x in range(img.shape[0]):
        for y in range(img.shape[1]):
            intensity=img[x,y]
            img_hist[intensity]+=1
    '''normalizing the Histogram by dividing number of pixels of each intensity with noralizing factor
    which is multiplication of image width and height means total no of pixels in image which is called 
    normalized histogram'''
    img_hist /= (img.shape[0] * img.shape[1])
    if do_print:
        print_histogram(img_hist, name="n_h_img", title="Normalized Histogram")
    return img_hist, img

def print_histogram(img_hist, name, title):
    plt.figure()
    plt.title(title)
    plt.plot(img_hist, color='#ef476f')
    plt.bar(np.arange(len(img_hist)), img_hist, color='#b7b7a4')
    plt.ylabel('Number of Pixels')
    plt.xlabel('Pixel Value')
    plt.show()
    # plt.savefig("hist_" + name)
img=cv2.imread("obama.jpg")
gray=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
imagehistogram(img,do_print=True)

#============================================================================
#using opencv python
#=============================================================================
hist=cv2.calcHist([img],[0], None, [256], [0, 256])
# matplotlib expects RGB images so convert and then display the image
# with matplotlib
plt.figure()
plt.axis("off")
plt.imshow(cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB))
# plot the histogram
plt.figure()
plt.title("Grayscale Histogram")
plt.xlabel("Bins")
plt.ylabel("# of Pixels")
plt.plot(hist)
plt.xlim([0, 256])


# normalize the histogram
hist /= hist.sum()
# plot the normalized histogram
plt.figure()
plt.title("Grayscale Histogram (Normalized)")
plt.xlabel("Bins")
plt.ylabel("% of Pixels")
plt.plot(hist)
plt.xlim([0, 256])
plt.show()

