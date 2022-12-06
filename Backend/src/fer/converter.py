import cv2

cap = cv2.VideoCapture('statement5.webm')

i=0
while(cap.isOpened()):
    ret, frame = cap.read()
    if ret == False:
        break
    cv2.imwrite('tester'+str(i)+'.jpg',frame)
    i+=1

cap.release()
cv2.destroyAllWindows()