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
path = "D:/ProgramData/AC/EmotionDating/Backend/input"
dir = os.listdir(path)
counter = 1
length = 50 
while(True):
    while not exists("D:/ProgramData/AC/statement.webm"):
        time.sleep(1)
    if isfile("D:/ProgramData/AC/statement.webm"):
        shutil.copy('D:/ProgramData/AC/statement.webm', 'D:/ProgramData/AC/EmotionDating/Backend/input/statement_1.webm')

    while not exists("D:/ProgramData/AC/statement (1).webm"):
        time.sleep(1)
    if isfile("D:/ProgramData/AC/statement (1).webm"):
        shutil.copy('D:/ProgramData/AC/statement (1).webm', 'D:/ProgramData/AC/EmotionDating/Backend/input/statement_2.webm')

    while not exists("D:/ProgramData/AC/statement (2).webm"):
        time.sleep(1)
    if isfile("D:/ProgramData/AC/statement (2).webm"):
        shutil.copy('D:/ProgramData/AC/statement (2).webm', 'D:/ProgramData/AC/EmotionDating/Backend/input/statement_3.webm')

    print("copy ready!")
    print("waiting")
    print(len(dir))

    if(len(dir) == 0):
        time.sleep(15)
        #path = "C:/Users/Admin/Desktop/AI/Backend/input"
        path = "D:/ProgramData/AC/EmotionDating/Backend/input"
        dir = os.listdir(path)

    if len(dir) != 0:
        print("D:/ProgramData/AC/EmotionDating/Backend/input/statement_" + str(counter) +".webm")
        if exists("D:/ProgramData/AC/EmotionDating/Backend/input/statement_" + str(counter) +".webm") == True:
            print("Start " + str(counter))
            video_filename = "D:/ProgramData/AC/EmotionDating/Backend/input/statement_" + str(counter) +".webm"
            video = Video(video_filename)
		
		    # Analyze video, displaying the output
            detector = FER(mtcnn=True)
            raw_data = video.analyze(detector, display=False)
            df = video.to_pandas(raw_data)
            df = video.get_first_face(df)
            df = video.get_emotions(df)

            list = [(df['angry'].sum())/length, (df['disgust'].sum())/length, (df['fear'].sum())/length, (df['happy'].sum())/length, (df['sad'].sum())/length, (df['surprise'].sum())/length, (df['neutral'].sum())/length]
			
            with open('D:/ProgramData/AC/EmotionDating/Backend/emotionvalues.csv','a') as file:
                writer = csv.writer(file, lineterminator='\n')
                #writer.writerow(['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral'])
                writer.writerow(list)
                
            print("End " + str(counter))
            counter += 1
            if (len(dir) < counter):
                break
    # time.sleep(70)    
    

cosSimilaritylive.cosSim()