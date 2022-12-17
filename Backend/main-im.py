from fer import FER
import cv2
import os
import matplotlib.pyplot as plt
import csv
import pandas as pd
from os.path import exists
import os
import time

print("Start")
path = "C:/Users/Admin/Desktop/AI/Backend/input"
dir = os.listdir(path)
counter = 1
currentframe = 0
i = 0

while(True):
    print("waiting")
    print(len(dir))

    if(len(dir) == 0):
        time.sleep(15)
        path = "C:/Users/Admin/Desktop/AI/Backend/input"
        dir = os.listdir(path)

    if len(dir) != 0:
        print(print("input/test" + str(counter) +".webm"))
        if exists("input/test" + str(counter) +".webm") == True:
            cap = cv2.VideoCapture("input/test" + str(counter) +".webm")
            #i = 0

            while (cap.isOpened()):
                ret, frame = cap.read()
                if ret == False:
                    break
                cv2.imwrite('data/tester' + str(i) + '.jpg', frame)
                i += 1

            cap.release()
            cv2.destroyAllWindows()
            counter += 1
        if (len(dir) < counter):
            break

currentpicture = 0

thislist = []

while(True):
    if(exists('./data/tester' + str(currentpicture) + '.jpg') == False):
        break

    img = cv2.imread('./data/tester' + str(currentpicture) + '.jpg')
    detector = FER(mtcnn=True)
    print(detector.detect_emotions(img))
    thislist.extend(detector.detect_emotions(img))
    emotion, score = detector.top_emotion(img)
    print(str(emotion) + " " + str(score))
    currentpicture += 10

resultlist = []
indices = 0

with open("emotionvalues.csv",'w') as file:
        writer = csv.writer(file)
        writer.writerow(['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral'])
		
        for element in thislist:
            first = (list(thislist[indices].items())[1])[1].values()
            writer.writerow(list(first))
            indices += 1
			
df = pd.read_csv("emotionvalues.csv")

fig = df.plot(figsize=(20, 16), fontsize=26).get_figure()
#fig = df.plot()

plt.show()