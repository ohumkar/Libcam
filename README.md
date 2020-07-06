# Libcam
Book cover text recognition

```
Main Project Structure : </br>
images
Libcam
 !--utils
 !   !--detection.py
 !   !--ner.py
 !   !--processing.py
 !
 !--main.py
 !--frozen_east_text_detection.pb
```
_detection.py_ → extract text regions using pretrained east model </br>
_preprocessing.py_ → image processing on detected text regions </br>
_main.py_ → run tesseract on detected text regions </br>
_ner.py_ → nlp functions to perform named entity recognition </br>

### How to run</br>
Main.py --east --image --detector --width --height --padding
</br>
Why developed
Approach</br>
Basic pipeline :</br>
<div align = "center">
<img align = "center" src = "Desktop - 3.jpg" >
</div>
</br>
Text Detection + Text Recognition + Classification</br>
Detection : Bounding boxes are detected using the EAST text detector. As opposed to the simple ocr task in which the text is usually over a plain background this task is a bit different as it needs to needs to detect text in natural scene images which contains much more noise in the background. Hence a robust text detector was to used.
Recognition : </br>
</br>

### Challenges faced</br>
OCR is never 100 % accurate and given the task of performing ocr on natural scene images the output is even of a lower quality. To help the tesseract to detect text, detection using another robust algorithm needed to be done. TExt regions detected by the east were then sent to the tesseract for recognition
Even though used output was not of desired quality. The output was mainly influenced by the following:</br>
- Performance of the EAST → fails to capture words → boxes include some part of other words → degrading the performance of the latter ocr task</br>
- Output of the tesseract →most of the times the ocr outputs garbage text strings or correct strings along with random characters. Such kind of output further makes the task of NER difficult finally resulting in a poor output of the program
</br>
### Improvements :</br>
- Image processing</br>
- OCR cleaing</br>

