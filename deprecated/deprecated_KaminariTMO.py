'''
雷-TMO / Kaminari-TMO
Pau Blasco i Roca
26/10/2023
Descripció del bitxo
'''
#imports and dependencies
import numpy as np
import scipy.stats as stats
from PIL import Image
from matplotlib import pyplot as plt
import cv2

#hyperparams
PERTOL = 2
STOPS = 6

#Fairman Matrix and F-1 matrix
F = np.asmatrix([[0.49,0.31,0.2],[0.17697,0.81240,0.01063],[0,0.01,0.99]])
F_inv = np.linalg.inv(F) #double checked and seems correct

#helper functions
def RGB_to_sRGB(img):
    new_image = np.zeros(img.shape)
    for i in range(img.shape[0]):
        new_image[i] = img[i]/255
    return new_image

def sRGB_to_RGB(img):
    for pix in img:
        pix*=255
    return img.astype(int)

def green_correct(img):
    for pix in img:
        pix[1]-=12
        if pix[1]<=0:
            pix[1]=0
    return img.astype(int)

def getNorm(norm_y, pix):
    mark = 0
    val = 1/len(norm_y)
    for i in range(len(norm_y)):
        mark+=val
        if pix<mark:
            return norm_y[i]
    return norm_y[-1]

def sRGB_XYZ(img):
    for pix in img:
        pix = np.matmul(F, pix)
    return img

def XYZ_sRGB(img):
    for pix in img:
        pix = np.matmul(F_inv, pix)
    return img

def gamma(img):
    for pix in img:
        pix[1] = pix[1]**1
    return img

#input
#exposures = [img1, img2, img3, img4, img5, img6]
#test input and preproc
# img1 = Image.open(r"./sample_images/640px-StLouisArchMultExpEV-4.72.jpg")
# img2 = Image.open(r"./sample_images/640px-StLouisArchMultExpEV-1.82.jpg")
# img3 = Image.open(r"./sample_images/640px-StLouisArchMultExpEV+1.51.jpg")
# img4 = Image.open(r"./sample_images/640px-StLouisArchMultExpEV+4.09.jpg")
img1 = Image.open(r"./sample_images/scene1+2.jpeg")
img2 = Image.open(r"./sample_images/scene1+1.jpeg")
img3 = Image.open(r"./sample_images/scene1-1.jpeg")
img4 = Image.open(r"./sample_images/scene1-2.jpeg")
exposures = []
evs = [2, 1, -1, -2]
for img in [img1,img2,img3,img4]: # bruh
    img = np.array(img)
    shape_org = img.shape
    shape_new = [shape_org[0]*shape_org[1],shape_org[2]]
    img = np.reshape(img, shape_new)
    img = RGB_to_sRGB(img)
    exposures.append(img)

for image in exposures:
    image = sRGB_XYZ

def method_1(exposures):
    gray_levels = []
    gray_contri = []
    gray_modula = []
    for image in exposures:
        Ylevel = [pixel[1] for pixel in image]
        gray_levels.append(Ylevel)
        plt.hist(Ylevel,20) #deb
        plt.show() #deb
        lo_p = np.percentile(Ylevel,PERTOL)
        hi_p = np.percentile(Ylevel,100-PERTOL)
        median = np.median(Ylevel)
        sd = min(abs(lo_p-median),abs(hi_p-median))
        norm_x = np.linspace(0, 1, 32)
        norm_y = stats.norm.pdf(norm_x,median,sd)
        
        #Modulation (calculate the contribution matrix):
        Y_contribution = [getNorm(norm_y/max(norm_y),pix) for pix in Ylevel]
        gray_contri.append(Y_contribution)

        #Modulated matrix
        Y_modulated = np.multiply(Ylevel, Y_contribution)
        gray_modula.append(Y_modulated)
    return gray_modula

def method_2(exposures):
    gray_levels = []
    gray_contri = []
    gray_modula = []
    for image in exposures:
        #initial set-up
        Ylevel = [pixel[1] for pixel in image]
        median = np.median(Ylevel)

        #noise-reduction so gamma correction doesn't go boom
        convert_back = np.array(Ylevel).reshape([shape_org[0],shape_org[1]]) #very inefficient
        convert_back = (convert_back*255).astype(np.uint8) #very inefficient
        thing = cv2.fastNlMeansDenoising(convert_back, None, 5.0, 7, 21)
        deb = Image.fromarray(thing) #deb
        deb.show()
        Ylevel = np.array(thing).reshape(shape_new[0])/255

        #Apply gamma correction to Y-Level
        K = 0.10
        gamma = np.log(median)/np.log(K)
        print("gamma is",gamma) #deb
        deb = Image.fromarray(np.array(Ylevel).reshape([shape_org[0],shape_org[1]])*255) #deb
        deb.show() #deb
        Ylevel = [pixel**(1/gamma) for pixel in Ylevel]
        # plt.hist(Ylevel,20) #deb
        # plt.show() #deb
        deb = Image.fromarray(np.array(Ylevel).reshape([shape_org[0],shape_org[1]])*255) #deb
        deb.show() #deb
        input()
        gray_levels.append(Ylevel)
        lo_p = np.percentile(Ylevel,PERTOL)
        hi_p = np.percentile(Ylevel,100-PERTOL)
        median = np.median(Ylevel)
        sd = min(abs(lo_p-median),abs(hi_p-median))
        norm_x = np.linspace(0, 1, 32)
        norm_y = stats.norm.pdf(norm_x,median,sd)

        #Modulation (calculate the contribution matrix):
        Y_contribution = [getNorm(norm_y/max(norm_y),pix) for pix in Ylevel]
        gray_contri.append(Y_contribution)

        #Modulated matrix
        Y_modulated = np.multiply(Ylevel, Y_contribution)
        gray_modula.append(Y_modulated)
    return gray_modula

def de_extreme(pix):
    EXT_TOL = 0.05
    if pix<EXT_TOL:
        return EXT_TOL
    elif pix>1-EXT_TOL:
        return 1-EXT_TOL
    return pix

def method_3(exposures, evs):
    gray_levels = []
    gray_contri = []
    gray_modula = []
    gray_median = []
    for i in range(len(exposures)):
        image = exposures[i]
        #initial set-up
        Ylevel = [de_extreme(pixel[1]) for pixel in image]
        median = np.median(Ylevel)

        #noise-reduction so gamma correction doesn't go boom
        convert_back = np.array(Ylevel).reshape([shape_org[0],shape_org[1]]) #very inefficient
        convert_back = (convert_back*255).astype(np.uint8) #very inefficient
        thing = cv2.fastNlMeansDenoising(convert_back, None, 5.0, 7, 21)
        Ylevel = np.array(thing).reshape(shape_new[0])/255

        #Apply gamma correction to Y-Level
        K = 1/3
        B = 1 #this makes it look greener for some reason!
        gamma = (1/(K+np.exp((evs[i]-K)/B))+1)/2
        print("applying gamma",gamma) #deb
        Ylevel = [pixel**(1/gamma) for pixel in Ylevel]
        gray_levels.append(Ylevel)

        #Modulation (calculate the contribution matrix):
        Y_contribution = [0.25 for pix in Ylevel]
        gray_contri.append(Y_contribution)

        #Modulated matrix
        Y_modulated = np.multiply(Ylevel, Y_contribution)
        gray_modula.append(Y_modulated)
        gray_median.append(np.median(Ylevel))
    return gray_modula, gray_median

# gray_modula = method_1(exposures) #method 1
# gray_modula = method_2(exposures) #method2
gray_modula, gray_median = method_3(exposures, evs) #method 3
final_Ylevel = np.zeros(shape_new[0])
for i in range(4):
    final_Ylevel+=gray_modula[i]
    deb_im1 = Image.fromarray((final_Ylevel.reshape([shape_org[0],shape_org[1]])*255).astype(np.uint8))
    deb_im1.show() #deb

#color interpolation

def color_interpolation(exposures, final_Ylevel):
    #weights inverse to the square of the EV, but such that w_0 + ... + w_n = 1
    Y = final_Ylevel
    weights = np.array([1/ev**2 for ev in evs])
    weights /= sum(weights)
    print(weights, sum(weights))
    final_XYZlevels = np.zeros(shape = shape_new)
    for i in range(len(exposures)):
        img = exposures[i]
        w = weights[i]
        for j in range(len(img)):
            pix = img[j]
            Xj = pix[0]*w
            Zj = pix[2]*w
            final_XYZlevels[j] += [Xj, Y[j]/4, Zj] #constant to reduce greenness
    return final_XYZlevels

final_XYZ_levels = color_interpolation(exposures, final_Ylevel)
final_XYZ_levels_gamma_corrected = gamma(final_XYZ_levels)
final_sRGB_levels = XYZ_sRGB(final_XYZ_levels_gamma_corrected)
final_RGB_levels = sRGB_to_RGB(final_sRGB_levels)
final_RGB_levels = green_correct(final_RGB_levels)
final_image = np.reshape(final_RGB_levels, shape_org)


cv2.imwrite('Escena1_Kaminari7.jpg', final_image)
