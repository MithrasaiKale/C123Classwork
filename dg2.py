import cv2
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
#Python Imaging Library (PIL) - external library adds support for image processing capabilities
from PIL import Image
import PIL.ImageOps
import os, ssl

X, y = fetch_openml('mnist_784', version=1, return_X_y=True)
print(pd.Series(y).value_counts())
classes = ['0', '1', '2','3', '4','5', '6', '7', '8','9']
nclasses = len(classes)

x_train, x_test, y_train, y_test=train_test_split(X,y,random_state=9, train_size=7500, test_size=2500)
x_trainScale=x_train/255.0
x_testScale=x_test/255.0
clf=LogisticRegression(solver="saga", multi_class="multinomial").fit(x_trainScale,y_train)
ypred=clf.predict(x_testScale)
acc=accuracy_score(y_test, ypred)
print(acc)

cap=cv2.VideoCapture(0)
while(True):
    try:
        ret, frame=cap.read()
        gray=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        height, width=gray.shape
        upperLeft=(int(width/2-56), int(height/2-56))
        bottomRight=(int(width/2+56), int(height/2+56))
        cv2.rectangle(gray,upperLeft, bottomRight, (0,255,0), 2)
        roi=gray[upperLeft[1]: bottomRight[1], upperLeft[0]: bottomRight[0]]
        im_pil=Image.fromarray(roi)
        # convert() to grayscale image - 'L' format means each pixel represented by a single value from 0 to 255
        img_bw=im_pil.convert('L')
        imgbwResized=img_bw.resize((28,28), Image.ANTIALIAS)
        imgInverted=PIL.ImageOps.invert(imgbwResized)
        pixel_filter=20
        min_pixel=np.percentile(imgInverted, pixel_filter)
        imgInverted_scaled=np.clip(imgInverted-min_pixel, 0, 255)
        max_pixel=np.max(imgInverted)
        imgInverted_scaled=np.asarray(imgInverted_scaled)/max_pixel
        test_sample=np.array(imgInverted_scaled).reshape(1,784)
        test_pred=clf.predict(test_sample)
        print(f"The predicted class for digit Class is {test_pred}")
        cv2.imshow("Frame", gray)
        if cv2.waitKey(1)&0xFF==ord("q"):
            break
    except Exception as e:
        pass
cap.release()
cv2.destroyAllWindows()