from fer import Video
from fer import FER
from os.path import exists

print("Start")
while(True):
    print("waiting")
    if exists("input/test.mp4") == True:
        video_filename = "input/test.mp4"
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
        fig.savefig('result-fe/my_figure.png')
        print(video.get_emotions(df))
    break