from fer import Video
from fer import FER
from os.path import exists, isfile
import os
import time
import csv
from numpy import dot
from numpy.linalg import norm
import cosSimilaritylive
import time
import shutil


print("Start")
#path = "C:/Users/Admin/Desktop/AI/Backend/input"
#path = "D:/ProgramData/AC/EmotionDating/Backend/input"
path = "C:/Users/SofieLouise/IdeaProjects/EmotionDating/Backend/input"
dir = os.listdir(path)
counter = 1
length = 50
while(True):
    while not exists("input/statement.webm"):
        time.sleep(1)
    print("First Statement")
    while not exists("input/statement(1).webm"):
        time.sleep(1)
    print("Second")
    while not exists("input/statement(2).webm"):
        time.sleep(1)
    print("Third")
    os.rename("input/statement.webm", "input/statement_1.webm")
    os.rename("input/statement(1).webm", "input/statement_2.webm")
    os.rename("input/statement(2).webm", "input/statement_3.webm")

    print("copy ready!")
    print("waiting")
    print(len(dir))

    if(len(dir) == 0):
        time.sleep(15)
        #path = "C:/Users/Admin/Desktop/AI/Backend/input"
        path = "input"
        dir = os.listdir(path)

    if len(dir) != 0:
        print("input/statement_" + str(counter) +".webm")
        if exists("input/statement_" + str(counter) +".webm") == True:
            print("Start " + str(counter))
            video_filename = "input/statement_" + str(counter) +".webm"
            video = Video(video_filename)
		
		    # Analyze video, displaying the output
            detector = FER(mtcnn=True)
            raw_data = video.analyze(detector, display=False)
            df = video.to_pandas(raw_data)
            df = video.get_first_face(df)
            df = video.get_emotions(df)

            list = [(df['angry'].sum())/length, (df['disgust'].sum())/length, (df['fear'].sum())/length, (df['happy'].sum())/length, (df['sad'].sum())/length, (df['surprise'].sum())/length, (df['neutral'].sum())/length]
			
            with open('emotionvalues.csv','a') as file:
                writer = csv.writer(file, lineterminator='\n')
                #writer.writerow(['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral'])
                writer.writerow(list)
                
            print("End " + str(counter))
            break
            counter += 1
            if (len(dir) < counter):
                break
    # time.sleep(70)    
    

cosSimilaritylive.cosSim()