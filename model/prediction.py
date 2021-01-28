import cv2 as cv
from flask import  jsonify
import argparse
import numpy as np
import json
from os.path import dirname, join
from PIL import Image
import base64
import io
#protoPath = join(dirname(__file__), "deploy.prototxt")
#modelPath = join(dirname(__file__), "hed_pretrained_bsds.caffemodelor")


threshold_mask = 0.7 #85%
threshold_FD = 1.75
# parser = argparse.ArgumentParser()
# parser.add_argument('--prototxt', help='Path to deploy.prototxt',default='deploy.prototxt', required=False)
# parser.add_argument('--caffemodel', help='Path to hed_pretrained_bsds.caffemodel',default='hed_pretrained_bsds.caffemodel', required=False)

#args, unknown = parser.parse_known_args()
def binarizeImage(contour):
  ret, thresh = cv.threshold(contour, 127, 255, 0)
  return thresh

def reconstructImage(im,patch,c):
  w,h,_=im.shape
  reconstructed_im = np.ones((im.shape))
  a = 0
  
  for i in range(0,w,c):
    b = 0
    for j in range(0,h,c):
      lig=0
      col=0

      if i+c <= w :
        lig=i+c 
      else:
        lig=w

      if j+c <= h :
        col=j+c 
      else:
        col=h
      reconstructed_im[i:lig,j:col] *=patch[a,b] * 255.0
      b +=1
    a +=1
  return reconstructed_im

class CropLayer(object):
    def __init__(self, params, blobs):
        self.xstart = 0
        self.xend = 0
        self.ystart = 0
        self.yend = 0

    # Our layer receives two inputs. We need to crop the first input blob
    # to match a shape of the second one (keeping batch size and number of channels)
    def getMemoryShapes(self, inputs):
        inputShape, targetShape = inputs[0], inputs[1]
        batchSize, numChannels = inputShape[0], inputShape[1]
        height, width = targetShape[2], targetShape[3]

        self.ystart = (inputShape[2] - targetShape[2]) // 2
        self.xstart = (inputShape[3] - targetShape[3]) // 2
        self.yend = self.ystart + height
        self.xend = self.xstart + width

        return [[batchSize, numChannels, height, width]]

    def forward(self, inputs):
        return [inputs[0][:,:,self.ystart:self.yend,self.xstart:self.xend]]

def makeBlackBorders(im):
  result = im.copy()
  w,h,c = im.shape
  for i in range(w):
    for j in range(h):
      if i == 0 :
        result[i,j] = [0.0,0.0,0.0]
      elif i == w-1 :
        result[i,j] = [0.0,0.0,0.0]

  for i in range(h):
    for j in range(w):
      if i == 0 :
        result[j,i] = [0.0,0.0,0.0]
      elif i == h-1 :
        result[j,i] = [0.0,0.0,0.0]

  return result

def countWhitePixels(patch):
  w,h,c=patch.shape
  cp = 0
  for i in range(w):
    for j in range(h):
      if (patch[i,j] == [255.0,255.0,255.0]).all() :
        cp = cp + 1 
  
  return cp

def constructPatches(im,c):
  patches=[]

  w,h,_=im.shape
  for i in range(0,w,c):
    p = []
    for j in range(0,h,c):
      lig=0
      col=0

      if i+c <= w :
        lig=i+c 
      else:
        lig=w

      if j+c <= h :
        col=j+c 
      else:
        col=h

      patch=im[i:lig,j:col]
      nb_white_pix = countWhitePixels(patch)
      p.append(nb_white_pix)
    patches.append(p)
  
  return np.array(patches)

def fractal_dimension(Z,threshold=0.9):
    def boxcount(Z, k):
        S = np.add.reduceat(
            np.add.reduceat(Z, np.arange(0, Z.shape[0], k), axis=0),
                               np.arange(0, Z.shape[1], k), axis=1)
        return len(np.where((S > 0) & (S < k*k))[0])
    Z = (Z < threshold)
    p = min(Z.shape)
    n = 2**np.floor(np.log(p)/np.log(2))
    n = int(np.log(n)/np.log(2))
    sizes = 2**np.arange(n, 1, -1)
    counts = []
    for size in sizes:
        counts.append(boxcount(Z, size))
    coeffs = np.polyfit(np.log(sizes), np.log(counts), 1)
    return -coeffs[0]
    
def localize(net,im):
    i  = 1
    im = cv.bilateralFilter(im,9,75,75)
    resize_patch = int(np.max(im.shape)/5)

    inp = cv.dnn.blobFromImage(im, scalefactor=1.0, size=(500,500),
                                mean=(104.00698793, 116.66876762, 122.67891434),
                                swapRB=False, crop=False)
    net.setInput(inp)
    out = net.forward()
    out = out[0, 0]
    out = cv.resize(out, (im.shape[1], im.shape[0]))
    out = 255 * out
    out = out.astype(np.uint8)
    bf = cv.bilateralFilter(out,9,100,100)
    out = binarizeImage(bf)
    ret, thresh = cv.threshold(out,0,255,cv.THRESH_BINARY_INV+cv.THRESH_OTSU)
    thresh = thresh.astype(np.uint8)
    contour = np.zeros(im.shape)
    contours,_ = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
    cv.drawContours(contour, contours, -1, tuple([255]*im.shape[-1]), 1)
    contour = makeBlackBorders(contour)
    patches = constructPatches(contour,resize_patch )
    accepted_value_patch = threshold_mask * np.amax(patches)
    mask_patch = patches > accepted_value_patch
    reconstructed_im = reconstructImage(im,mask_patch,resize_patch)

    reconstructed_im  = reconstructed_im.astype(np.uint8)
    im_gray = cv.cvtColor(reconstructed_im, cv.COLOR_BGR2GRAY)
    
    contour = im.copy()
    
    contours,_ = cv.findContours(im_gray, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
    cv.drawContours(contour, contours, -1, (0,0,255), 3)
    #cv.imwrite("BB"+str(i)+".png",contour)
    
    return contour

def classify(net,im):
    i  = 1

    h,w,c = im.shape
    inp = cv.dnn.blobFromImage(im, scalefactor=1.0, size=(500,500),
                                mean=(104.00698793, 116.66876762, 122.67891434),
                                swapRB=False, crop=False)
    net.setInput(inp)
    out = net.forward()
    out = out[0, 0]
    out = cv.resize(out, (im.shape[1], im.shape[0]))
    out = 255 * out
    out = out.astype(np.uint8)
    thresh = cv.adaptiveThreshold(out,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C,\
                cv.THRESH_BINARY,11,2)
    thresh = thresh.astype(np.uint8)
    contour = np.zeros(im.shape)
    
    contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
    cv.drawContours(contour, contours, -1, tuple([255]*im.shape[-1]), 1)
    contour = makeBlackBorders(contour)
    contour = np.array(contour, dtype=np.uint8)
    I = cv.cvtColor(contour, cv.COLOR_BGR2GRAY)/255.0
    
    Z = 1.0 - I
    fd = fractal_dimension(Z,threshold=0.5)
    print("Image: ",i," fractal dimension",fd)
    
    prediction = 0
    if  fd >= threshold_FD :
        prediction = 1
    return prediction 

def predict(data,net):
    npimg = np.fromstring(data, np.uint8)
    im = cv.imdecode(npimg,cv.IMREAD_COLOR)
    #image = Image.open(data)
    #image = np.array(image)
    #im = image.astype('float32')
    y = classify(net,im)
    print("y",y)
    
    if y == 1:
        loc = localize(net,im)
        loc = np.array(loc)
        img = Image.fromarray(loc.astype("uint8"))
        rawBytes = io.BytesIO()
        img.save(rawBytes, "JPEG")
        rawBytes.seek(0)
        img_base64 = base64.b64encode(rawBytes.read())
        a = format(img_base64.decode('utf-8'))
        obj = { 'image': str(a), 'class': 'detected class: crowd' }
    else : 
        obj = { 'image': None, 'class': 'detected class: no crowd' }
    
    return json.dumps(obj)

