
import cv2 # importing the OpenCV
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml') # face cascade is used to filter faces in all the image 
# and haar cascade is a pre-trained classifier to filter out faces, eyes etc
img = cv2.imread('lo.jpg.jpg') # Here we read the image and convert it to grayscale. Many operations in OpenCV are done in grayscale.
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # It is converted to grayscale to make it simpler and reduce errors
faces = face_cascade.detectMultiScale(gray, 
    scaleFactor=1.1,
    minNeighbors=6,
    minSize=(30, 30),
    )
'''
This function detects the actual face and is the key part of our code, so let’s go over the options:

 #The detectMultiScale function is a general function that detects objects. Since we are calling it on the face cascade, that’s what it detects.

 #The first option is the grayscale image.

 #The second is the scaleFactor. Since some faces may be closer to the camera, they would appear bigger than the faces in the back. The scale factor compensates for this.

 #The detection algorithm uses a moving window to detect objects. minNeighbors defines how many objects are detected near the current one before it declares the face found. minSize, meanwhile, gives the size of each window.
'''
for (x, y, w, h) in faces:
    cv2.rectangle(img, (x,y), (x+w, y+h), (255, 0, 0), 2) # This function returns 4 values: the x and y location of the rectangle, and the rectangle’s width and height (w , h).
# We use these values to draw a rectangle using the built-in rectangle() function.
cv2.imshow('Faces Found', img)
cv2.waitKey() # In the end, we display the image and wait for the user to press a key.


