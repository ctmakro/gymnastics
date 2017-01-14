import cv2
import numpy as np
import time
imgw = 200
imgh = 300

lastshow = time.time()

def showbar(actions,idx):
    l = len(actions)
    actions*=.5
    global imgw
    imgw = max(imgw,l)

    while np.mean(actions) < - imgh/2:
        actions += imgh

    while np.mean(actions) > imgh/2+10:
        actions -= imgh

    if hasattr(showbar,'im'):
        showbar.im *= [0.93,0.99,0.995]
    else:
        showbar.im = np.zeros((imgh,imgw,3),dtype='float32')

    im = showbar.im

    segw = int(imgw/l)

    for i in range(l):
        hei = actions[i]
        hei = int(-hei + imgh/2)

        if i==idx:
            c = np.array([1.,1.,1.])
            im[hei-2:hei+2,(i)*segw:(i+1)*segw] = 1.
        else:
            c = np.array([.4,.5,.8])
            pass


        im[hei:hei+1,i*segw:(i+1)*segw,:] = c
        # im[hei:hei+2,i*segw:i*segw+segw-3,:] += 0.2 * 0.8

    im = np.clip(im,a_max=1.0,a_min=0.0)

    global lastshow
    if time.time()-lastshow>0.1:
        lastshow=time.time()
        cv2.imshow('stat',im)
        cv2.waitKey(1)

def test():
    showbar([0,10,20,-10,-20])
