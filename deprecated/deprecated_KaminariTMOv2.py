import cv2
import numpy as np
import math

# read image
img1 = cv2.imread(r"./sample_images/scene1-2.jpeg")
img2 = cv2.imread(r"./sample_images/scene1-1.jpeg")
img3 = cv2.imread(r"./sample_images/scene1+1.jpeg")
img4 = cv2.imread(r"./sample_images/scene1+2.jpeg")

imgs = [img1, img2, img3, img4]
res = []
for img in imgs:

    # METHOD 1: RGB

    # convert img to gray
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # compute gamma = log(mid*255)/log(mean)
    mid = 0.5
    mean = np.mean(gray)
    print("mean", mean)
    gamma = math.log(mid*255)/math.log(mean)
    print("gamma",gamma)

    # do gamma correction
    img_gamma1 = np.power(img, gamma).clip(0,255).astype(np.uint8)



    # METHOD 2: HSV (or other color spaces)

    # convert img to HSV
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hue, sat, val = cv2.split(hsv)

    # compute gamma = log(mid*255)/log(mean)
    mid = 0.5
    mean = np.mean(val)
    print("mean", mean)
    gamma = math.log(mid*255)/math.log(mean)
    print("gamma",gamma)
    if gamma>4: 
        gamma=4.0 #create 
        print("gamma capped to",gamma)

    # do gamma correction on value channel
    val_gamma = np.power(val, gamma).clip(0,255).astype(np.uint8)

    # combine new value channel with original hue and sat channels
    hsv_gamma = cv2.merge([hue, sat, val_gamma])
    img_gamma2 = cv2.cvtColor(hsv_gamma, cv2.COLOR_HSV2BGR)

    # METHOD 3: XYZ (or other color spaces)

    # convert img to XYZ
    print("XYZ")
    XYZ = cv2.cvtColor(img, cv2.COLOR_BGR2XYZ)
    # add constant to image so gamma works
    constant = 1
    XYZ += np.full(XYZ.shape, constant, dtype=np.uint8)
    X, Y, Z = cv2.split(XYZ)
    print(XYZ[0])
    # compute gamma = log(mid*255)/log(mean)
    mid = 0.5
    mean = np.mean(Y)
    print("mean", mean)
    gamma = math.log(mid*255)/math.log(mean)
    print("gamma",gamma)
    # do gamma correction on value channel
    Y_gamma = np.power(Y, gamma).clip(0,255).astype(np.uint8)
    print(Y_gamma[0])
    # combine new value channel with original hue and sat channels
    XYZ_gamma = cv2.merge([X, Y_gamma, Z]) #why is the middle channel duplicated here???
    print(XYZ_gamma[0])
    img_gamma3 = cv2.cvtColor(XYZ_gamma, cv2.COLOR_XYZ2BGR)
       

    # # show results
    # cv2.imshow('input', img)
    # cv2.imshow('result1', img_gamma1)
    # cv2.imshow('result2', img_gamma2)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # save results
    res.append(img_gamma1)
    res.append(img_gamma2)
    res.append(img_gamma3)
    # cv2.imwrite('lioncuddle1_gamma1.jpg', img_gamma1)
    # cv2.imwrite('lioncuddle1_gamma2.jpg', img_gamma2)
    # cv2.imwrite('lioncuddle1_gamma3.jpg', img_gamma3)
i=0
for ress in res:
    cv2.imwrite("test"+str(i)+".jpg", ress)
    i+=1
