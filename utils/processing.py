import matplotlib.pyplot as plt
import numpy as np
import cv2
from string import digits




BINARY_THREHOLD = 180
def image_smoothening(img):
    ret1, th1 = cv2.threshold(img, BINARY_THREHOLD, 255, cv2.THRESH_BINARY)
    ret2, th2 = cv2.threshold(th1, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    blur = cv2.GaussianBlur(img, (3, 3), 0)
    ret3, th3 = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return th3

def remove_noise_and_smooth(img):
    #img = cv2.imread(file_name, 0)
	img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	#filtered = cv2.adaptiveThreshold(img.astype(np.uint8), 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 41, 3)
	kernel = np.ones((2, 2), np.uint8)
	opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
	closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)
	ocr_image = image_smoothening(img)
	#or_image = cv2.bitwise_or(img, closing)
	return ocr_image

def plot_image(input_image) :
	output_image = remove_noise_and_smooth(input_image)
		
	# plot images
	fig, ax = plt.subplots(nrows=1, ncols=2)
	ax[0].imshow(cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB))
	ax[0].set_title('Input Image')
	ax[0].axis('off')
	#ax[1].imshow(cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB))
	ax[1].imshow(output_image, cmap='gray')
	ax[1].set_title('Gaussian Blurred')
	ax[1].axis('off')
	plt.show()
	return output_image
