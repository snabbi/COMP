import cv2
import numpy as np
from time import time

########## Brightest spot methods
def show_brightest_spot(frame):
    maxLoc = _get_brightest_spot(frame)
    cv2.circle(frame, maxLoc, 10, (255,0,0), 2)

def show_brightest_spot_loop(frame):
    maxLoc = _get_brightest_spot_loop(frame)
    cv2.circle(frame, maxLoc, 10, (255,0,0), 2)


######### "Reddest" spot methods
def show_reddest_spot(frame):
    mask = _create_mask(frame,np.array([30,30,100],dtype="uint8"), np.array([70,70,250],dtype="uint8"))
    output = cv2.bitwise_and(frame,frame,mask=mask)
    (_, _, _, maxLoc) = cv2.minMaxLoc(output[:,:,2])
    cv2.circle(frame,maxLoc,10,(0,255,0),2)

def show_reddest_spot_loop(frame):
    mask = _create_mask(frame,np.array([30,30,100],dtype="uint8"), np.array([70,70,250],dtype="uint8"))
    output = cv2.bitwise_and(frame,frame,mask=mask)
    maxLoc = _get_maxLoc_loop(output[:,:,2])
    cv2.circle(frame,maxLoc,10,(0,255,0),2)

######### Help functions
def _create_mask(frame,lower,upper):
    return cv2.inRange(frame,lower,upper)

def _get_brightest_spot(frame):
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    (_, _, _, maxLoc) = cv2.minMaxLoc(gray)
    return maxLoc

def _get_brightest_spot_loop(frame) -> tuple:
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    return _get_maxLoc_loop(gray)

def _get_maxLoc_loop(frame) -> tuple:
    m_col, m_row = 0,0
    m_val = 0
    for i in range(frame.shape[0]):
        for j in range(frame.shape[1]):
            if frame[i,j] > m_val:
                m_val = frame[i,j]
                m_col,m_row = i,j
    return (m_row,m_col)

def main():
    frames = 0
    N = 20
    vid = cv2.VideoCapture(0)
    #vid = cv2.VideoCapture('http://10.128.110.35:8080/video')
    start = time()
    while True:
        if frames == N:
            print(N/(time()-start))
            start = time()
            frames = 0
        ret,frame = vid.read()
        
        #show_brightest_spot(frame)
        show_reddest_spot(frame)

        #show_brightest_spot_loop(frame)
        #show_reddest_spot_loop(frame)
        if not ret:
            print('failed to grab frame')
            break
        
        cv2.imshow("Capturing",frame)
        frames += 1
        if cv2.waitKey(1) & 0xff == ord('q'):
            break

    vid.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()