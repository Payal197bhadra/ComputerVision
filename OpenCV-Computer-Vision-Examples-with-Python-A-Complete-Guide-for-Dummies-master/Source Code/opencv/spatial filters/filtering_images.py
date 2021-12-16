import cv2
import numpy as np
def point_operation(img, K, L):
    #apply point operation to given grayscale image
    img= np.asarray(img, dtype=np.float32)
    img = img * K + L
    # clip pixel values
    img[img > 255] = 255
    img[img < 0] = 0
    return np.asarray(img, dtype=np.uint8)
def main():
    #creating noise
    #adding weighted noise to an grayscale image
    img=cv2.imread("obama.jpg")
    img=cv2.resize(img, (400, 600))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # gray=cv2.resize(gray,(400,600))
    #creating a noisy image
    noise = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
    cv2.randu(noise, 0, 256)
    # print(noise)
    noisy_img=gray + np.array(0.1*noise, dtype=np.uint8)

    # k = 0.5, l = 0
    out1 = point_operation(gray, 0.2, 0)
    # k = 1., l = 10
    out2 = point_operation(gray, 1., 10)
    # k = 0.8, l = 15
    out3 = point_operation(gray, 0.7, 25)
    res = np.hstack((gray, out1, out2, out3))
    # gaussian_noise = np.zeros((img.shape[0], img.shape[1]),dtype=np.uint8)
    # cv2.randn(gaussian_noise, 120, 256)
    # img=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # gaussian_noise_img= img + np.array(0.2*gaussian_noise, dtype=np.uint8)
    # cv2.imshow('All zero values',gaussian_noise_img)
    median_filter= cv2.medianBlur(noisy_img, 7)
    bilateral_filter = cv2.bilateralFilter(noisy_img, 2, 75,75)
    final_image= np.hstack((gray, noisy_img, median_filter, bilateral_filter))



    # cv2.imshow("Noise image", final_image)
    cv2.imshow("Point Operation", res)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
if __name__ == '__main__':
  main()