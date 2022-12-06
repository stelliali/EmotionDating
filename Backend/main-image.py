from fer import FER
import cv2
import os
import matplotlib.pyplot as plt
import csv
import pandas as pd
from os.path import exists
import shutil

currentframe = 0

shutil.copy('Downloads/statement1.webm', 'input/statement1.webm')
shutil.copy('Downloads/statement2.webm', 'input/statement2.webm')
shutil.copy('Downloads/statement3.webm', 'input/statement3.webm')
shutil.copy('Downloads/statement4.webm', 'input/statement4.webm')
shutil.copy('Downloads/statement5.webm', 'input/statement5.webm')

while(True):
    if exists("input/statement5.webm"):
        cam = cv2.VideoCapture("input/statement5.webm")
        break
		
while(True):
	ret,frame = cam.read()

	if ret:
		name = './data/frame' + str(currentframe) + '.jpg'
		print ('Creating...' + name)

		cv2.imwrite(name, frame)

		currentframe += 3
		cam.set(1, currentframe)
	else:
		break

cam.release()
cv2.destroyAllWindows()

currentpicture = 0

thislist = []

while(currentpicture < 76):
	img = cv2.imread('./data/frame' + str(currentpicture) + '.jpg')
	detector = FER(mtcnn=True)
	print(detector.detect_emotions(img))
	thislist.extend(detector.detect_emotions(img))
	currentpicture += 3

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
fig.savefig('result-fe/my_figureLive.png')

plt.show()