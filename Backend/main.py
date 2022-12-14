from fer import Video
from fer import FER
from os.path import exists
import os
import time

print("Start")
path = "C:/Users/Admin/Desktop/AI/Backend/input"
dir = os.listdir(path)
counter = 1
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
            print("Start " + str(counter))
            video_filename = "input/test" + str(counter) +".webm"
            video = Video(video_filename)
		
		    # Analyze video, displaying the output
            detector = FER(mtcnn=True)
            raw_data = video.analyze(detector, display=True)
            df = video.to_pandas(raw_data)
            df = video.get_first_face(df)
            df = video.get_emotions(df)

		    # Plot emotions
            fig = df.plot(figsize=(20, 16), fontsize=26).get_figure()
		    # Filename for plot
            fig.savefig('result-fe/my_figure' + str(counter) + '.png')
            print(video.get_emotions(df))
            print("End " + str(counter))
            counter += 1
            if (len(dir) < counter):
                break