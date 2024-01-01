from PIL import Image, ImageEnhance
from scipy.ndimage import gaussian_filter
import numpy as np
import matplotlib.pyplot as plt
import math
import imageio
import cv2

GAMMAS = [1.8,1,0.6]
SATCONST = 1.15

F = np.asmatrix([[0.5767309, 0.1855540, 0.1881852],
                [0.2973769, 0.6273491, 0.0752741],
                [0.0270343, 0.0706872, 0.9911085]])
F_inv = np.linalg.inv(F) #double checked and seems correct

def gammafun(x,gamm):
    return x**(1/gamm)
vgamm = np.vectorize(gammafun)

def rgb_to_XYZ(img): #increible, ho ignora completament. m'encanta python de merda
    #Fairman Matrix
    for v in img:
        for pix in v:
            pix = np.matmul(F,pix)
    return img

def XYZ_to_rgb(img):
    #Inv fairman matrix
    for v in img:
        for pix in v:
            pix = np.matmul(F_inv,pix)
    return img

#clustering on expositions:
from sklearn import cluster
import numpy as np
import matplotlib.pyplot as plt

def kmeans(data, k):
    data = data.reshape((-1,1))
    cluster_centers = np.array([0.15,0.4, 0.8],dtype=np.float32).reshape((-1,1))
    k_m = cluster.KMeans(n_clusters=k,init=cluster_centers,n_init=2,random_state=0)
    k_m.fit(data)
    values = k_m.cluster_centers_.squeeze()
    print(k_m.cluster_centers_)
    #labels = k_m.labels_
    labels = k_m.predict(data)
    return(values, labels)

def filter_overexposed(v, tol):
    if v>tol: return 1
    return 0
vfoexp = np.vectorize(filter_overexposed)

def gamma_clustered(pixel, label):
    return gammafun(pixel, GAMMAS[label])
vgamclus = np.vectorize(gamma_clustered)

### default gamma correction (base method)
def base(img):
    img = vgamm(img,2.2)
    print(np.max(img)/np.min(img[np.nonzero(img)]))
    imgtest = np.uint8(img*255)
    im = Image.fromarray(imgtest)
    im.show()

#####################################################################

imageio.plugins.freeimage.download()

for num in range(4):
    img = imageio.imread(r"./sample_images/scene"+str(num+1)+".hdr", format="HDR-FI")
    print(np.max(img)/np.min(img[np.nonzero(img)])) #dynamic range
    x,y,_ = img.shape

    fiY = np.ndarray(shape=[x,y])
    #extract color from initial image, X,Z values:
    ogR,ogG,ogB = cv2.split(img)
    ogXYZ = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    ogX,ogY,ogZ = cv2.split(ogXYZ)

    #getting the expositions (?)
    for i in range(4):
        expo = np.clip(img*(2**12), 2**i, 2**(12-4+i))/2**(8+i)
        XYZ = cv2.cvtColor(expo, cv2.COLOR_RGB2HLS)
        X,Y,Z = cv2.split(XYZ)
        print("image",i,np.min(Y),np.max(Y),np.average(Y))
        Y255 = np.uint8(Y*255)

        values, labels = kmeans(data=Y,k=3)
        img_segm = np.choose(labels, values)
        img_segm.shape = [x,y]
        # im = Image.fromarray(np.uint8(img_segm*255))
        # im.show()

        Y_partialgamma = []
        for i in range(len(Y)):
            Y_partialgamma.append(gamma_clustered(Y[i],labels[i]))
        Y_partialgamma = np.array(Y_partialgamma).reshape([x,y])
        #im = Image.fromarray(np.uint8(Y_partialgamma*255))
        fiY+=Y_partialgamma

    fiY/=4 #average it

    initial = img
    im = Image.fromarray(np.uint8(initial*255))
    #im.show()
    cv2.imwrite("results/finalinitial"+str(num)+".jpg", cv2.cvtColor(np.uint8(initial*255), cv2.COLOR_RGB2BGR))

    #add a bit of saturation to compensate, these operations seem to desaturate a bit the image
    fiZ = np.clip(ogZ*SATCONST,0,1)

    final = cv2.merge([ogX, np.float32(fiY), fiZ])
    final = cv2.cvtColor(final, cv2.COLOR_HLS2RGB)
    im = Image.fromarray(np.uint8(final*255))
    #im.show()
    # cv2.imwrite("results/finalfinal"+str(num)+".jpg", cv2.cvtColor(np.uint8(final*255), cv2.COLOR_RGB2BGR))

    #balancing to get darkest spots and recover some burnt features
    mix = (final*6.5+initial*3.5)/10
    im = Image.fromarray(np.uint8(mix*255))
    #im.show()s
    cv2.imwrite("results/finalmix"+str(num)+".jpg", cv2.cvtColor(np.uint8(mix*255), cv2.COLOR_RGB2BGR))