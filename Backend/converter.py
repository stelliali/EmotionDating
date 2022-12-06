import ffmpeg

class MpConv:	
	def convert_mp4(self, input_file, output_file):
		print(input_file)
		print(output_file)
		try:
			stream = ffmpeg.input(input_file)
			stream = ffmpeg.output(stream, output_file)		
			ffmpeg.run(stream)
		except:
			print('b')

ffm = MpConv()
#ffm.convert("input/test.webm", "input/test.mp4")
ffm.convert_mp4("input/test.webm", "input/test.mp4")