import numpy as np
from PIL import Image
from matplotlib import pyplot as plt

img = Image.open(r"./sample_images/cat.jpg")
img = np.array(img)
print(img)

def sRGB_to_perLight(img):
    shape = img.shape
    greyshape = img.shape[:-1]
    #sRGB to RGB [0,1]
    img = np.divide(img, 255.0)
    #RGB to linear (un-gamma)
    limg = img.reshape(np.product(shape)) #squeeze for convinience
    print("RGB",limg)
    linearize = np.vectorize(lambda x: x/12.92 if x <= 0.04045 else ((x+0.055)/1.055)**2.4)
    limg = linearize(limg)
    print("linear RGB",limg)
    #linear to luminance
    limg = limg.reshape(shape) #unsqueeze for convinience #note from later: convinience my ass, I need to optimize this
    print(limg)
    greyscalize = lambda x: x[0]*0.2126 + x[1]*0.7152 + x[2]*0.0722
    lumimg = np.ndarray(greyshape)
    for i in range(len(lumimg)):
        for j in range(len(lumimg[i])):
            lumimg[i][j] = greyscalize(limg[i][j])
    print("luminance",lumimg.shape,lumimg[0])
    #luminance to perceived lightness [0,100]
    perLightze = np.vectorize(lambda x : x * 903.3 if x<=0.008856 else x**(1/3)*116-16)
    perLight = perLightze(lumimg)
    print("perLight", perLight)
    return perLight

def decolorize(img):
    '''
    This http://cadik.posvete.cz/color_to_gray_evaluation/cadik08perceptualEvaluation.pdf claims that decolorize, in the paper
    https://www.cl.cam.ac.uk/techreports/UCAM-CL-TR-649.pdf, which uses the Y channel to represent luminance as perceived by
    the human eye.
    '''
    #constants (paper)
    r = 0.2989 
    g = 0.5870
    b = 0.1140
    shape = img.shape
    greyshape = img.shape[:-1]
    grayimg = np.ndarray(greyshape)
    decolorize_smol = lambda x: x[0]*r + x[1]*g + x[2]*b
    #sRGB to RGB [0,1]
    img = np.divide(img, 255.0)
    for i in range(len(img)):
        for j in range(len(img[0])):
            grayimg[i][j] = decolorize_smol(img[i][j])
    return grayimg

perLight = sRGB_to_perLight(img)
# plt.hist(perLight.flatten(),20)
# plt.show()
# img_adjust = np.multiply(perLight, 255/100)


grayimg = decolorize(img)
# plt.hist(grayimg.flatten(),20)
# plt.show()
# img_adjust_2 = np.multiply(grayimg, 255)
# img_adjust_2 = Image.fromarray(img_adjust_2)
# img_adjust_2.show()
#refs
#https://stackoverflow.com/questions/596216/formula-to-determine-perceived-brightness-of-rgb-color

plt.hist([perLight.flatten(),np.multiply(grayimg, 100).flatten()],20)
plt.axvline(x = np.mean(perLight.flatten()), color = 'orange' )
plt.axvline(x = np.mean(np.multiply(grayimg, 100).flatten()), color = 'blue')
plt.vlines(x = np.median(perLight.flatten()), ymin = 0, ymax = 18000, color = 'orange', linestyles = 'dotted')
plt.vlines(x = np.median(np.multiply(grayimg, 100).flatten()), ymin = 0, ymax = 18000, color = 'blue', linestyles = 'dotted')
plt.show()
