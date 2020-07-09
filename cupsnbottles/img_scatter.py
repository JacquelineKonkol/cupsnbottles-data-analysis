import seaborn
seaborn.set_style("whitegrid", {'axes.grid' : False})


import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import seaborn
seaborn.set_style("whitegrid", {'axes.grid' : False})

import numpy as np

import matplotlib
import PIL


def imageScatter(x,y,imgs,cls=None,probs=None,labels=None,ax=None,img_scale=(50,50),frame_width=3):
    '''scatter plot showing thumbnails as scatter symbols with frame and text for visualizing different things'''
    # from scipy.misc import imresize # deprecated, better pass PIL Image instead
    if ax is None:
        ax = plt.gca()
    if cls is not None:
        uniqueCls = np.unique(cls)
        classInd = [np.where(uniqueCls==cl)[0][0] for cl in cls]
        clsCols = np.multiply(matplotlib.cm.rainbow(np.linspace(0, 1, len(uniqueCls))),255)
        clsCols = np.delete(clsCols, 3, 1)

    for ind in range(len(x)):
        if type(imgs[ind]) == PIL.JpegImagePlugin.JpegImageFile:
            img = imgs[ind].resize(img_scale,resample=PIL.Image.BICUBIC)
        else:
            #img = imresize(imgs[ind],img_scale)
            img = np.array(PIL.Image.fromarray(imgs[ind][...,:3]).resize(img_scale))

        if cls is not None:
            img = frameImage(img,clsCols[classInd[ind]],frame_width,3,True)
        if probs is not None:
            img = frameImage(img,probs[ind],frame_width,12)

        im = OffsetImage(img, zoom=1)
        xa, ya = np.atleast_1d(x[ind], y[ind])
        artists = []
        for x0, y0 in zip(xa, ya):
           ab = AnnotationBbox(im, (x0, y0), xycoords='data', frameon=False,box_alignment=(0.5,0.5))
           artists.append(ax.add_artist(ab))
        ax.update_datalim(np.column_stack([x, y]))
        ax.autoscale()

        font = {'family': 'serif',
                'color': 'red',
                'weight': 'normal',
                'size': 30,
                }

        if labels != None:
            ax.text(xa, ya, str(labels[ind]), fontdict=font)
            ax.set_zorder(1)


    return artists


def frameImage(img,col,bw=2,side=15,image_scale_up=False):
    '''draws a frame into an image, bw = borderwidth, col=color, side is a bitmask were the frame should appear'''
    #if not isinstance(col,np.ndarray):
    #    if isinstance(col,float) and col < 1:
    #        col = col * 255
    #    col = np.array([col,col,col])

    nimg = img
    if image_scale_up:
        nimg = np.zeros((img.shape[0]+bw*2,img.shape[1]+bw*2,img.shape[2]),np.uint8)
        nimg[bw:-bw,bw:-bw,:] = img[:,:,:]

    img = np.array(PIL.Image.fromarray(img[..., :3]).resize((25, 25)))

    for i,c in enumerate(col):
        if side & 1: nimg[0:bw,:,i] = c*255
        if side & 2: nimg[:,0:bw,i] = c*255
        if side & 4: nimg[-bw:,:,i] = c*255
        if side & 8: nimg[:,-bw:,i] = c*255
    return nimg
