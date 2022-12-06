import cv2

def convert(x):
    cap = cv2.VideoCapture('input\statement'+x+'.webm')

    i=0
    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret == False:
            break
        if i%3 == 0:
            cv2.imwrite('input\demo'+str(i)+'.jpg',frame)
        i+=1

    cap.release()
    cv2.destroyAllWindows()