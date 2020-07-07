# USAGE
# python text_recognition.py --east frozen_east_text_detection.pb --image images/example_01.jpg
# python text_recognition.py --east frozen_east_text_detection.pb --image images/example_04.jpg --padding 0.05
# ------------------------------------------------------------------------------------------------------------

from utils.detection import recognize_text, only_tesseract
from utils.ner import ner_nltk, ner_spacy
from utils.cam import get_image
import cv2
import argparse

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", type=str,
	help="path to input image")
ap.add_argument("-east", "--east", type=str,
	help="path to input EAST text detector")
ap.add_argument("-d", "--detector", type=str,
	help="type of detector [ocr]  / [east]")
ap.add_argument("-c", "--min-confidence", type=float, default=0.7,
	help="minimum probability required to inspect a region")
ap.add_argument("-w", "--width", type=int, default=320,
	help="nearest multiple of 32 for resized width")
ap.add_argument("-e", "--height", type=int, default=320,
	help="nearest multiple of 32 for resized height")
ap.add_argument("-p", "--padding", type=float, default=0.0,
	help="amount of padding to add to each border of ROI")
args = vars(ap.parse_args())


print('Starting Libcam')
# load the input image and grab the image dimensions
if args["image"] == "cam" :
	image = get_image()
else :
	image = cv2.imread(args["image"])

# Fetch ocr results
if args["detector"] == "east" :
	text = recognize_text(image,  args["width"], args["height"], args["padding"], args["min_confidence"], args["east"])
	# form a sentence of extracted words
	text = ' '.join(c for c in text)
else :
	text = only_tesseract(image)

print('OCR text :', text)

# Get Named Entities using spacy
df = ner_spacy(text)
# Get only PERSONS from extracted ners
df = df[df['Labels'] == 'PERSON'] 
names = set(df['Entities'])
# output check : names
# print('NAMES ARE:',names)

# split names into words to subtract from sentence words later
sep_names = [str(n).split() for n in names]
sep_names = [item for items in sep_names for item in items]
# output check : sep_names
# print('SEP NAMES:', (sep_names))


# split sentence into words 
text = text.replace('\n',' ')
sentence = [text.split(' ')]
# print('SENTENCE :',sentence)
sentence = (list(sentence)[0])
# output check : sentence
# print('SENTENCE :',sentence)

# get book name (sentence words - names)
title = [x for x in sentence if  x not in sep_names]


print('\n\n')
print('BOOK NAME :\n', ' '.join(c for c in title))
print('AUTHOR :\n', names)
