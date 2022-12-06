import cv2

cap = cv2.VideoCapture('test.webm')

i=0
while(cap.isOpened()):
    ret, frame = cap.read()
    if ret == False:
        break
    if i%3 == 0:
        cv2.imwrite('C:\Users\SofieLouise\IdeaProjects\EmotionDating\Backend\output\tester'+str(i)+'.jpg',frame)
    i+=1

cap.release()
cv2.destroyAllWindows()