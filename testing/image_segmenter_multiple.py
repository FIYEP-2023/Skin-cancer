# Import some packages we have used so far + new ones for segmentation

import numpy as np
import matplotlib.pyplot as plt
from skimage.color import rgb2gray
from skimage import data

from skimage.filters import gaussian
from skimage.segmentation import active_contour
import os

# Now with skin lesions
# Read files from /testimages
files = os.listdir('testimages')

for file_im in files:
    im = plt.imread('testimages/' + file_im)
    # Remove fourth dimension - transparency
    im = im[:,:,:3]
    im = rgb2gray(im)
    # im = im[0:1500,1000:2500]

    #Resize for speed
    from skimage.transform import resize
    im = resize(im, (im.shape[0] // 4, im.shape[1] // 4), anti_aliasing=True)

    blur = int(im.shape[0]/50)
    points = int(0.52*im.shape[0]+16.92)
    radius = im.shape[0]/3

    im2 = gaussian(im, blur)
    plt.imshow(im2, cmap="gray")

    # Circle for this image
    s = np.linspace(0, 2*np.pi, points)   #Number of points on the circle
    r = im.shape[0]/2 + radius*np.sin(s)            #Row 
    c = im.shape[1]/2 + radius*np.cos(s)            #Column
    init2 = np.array([r, c]).T

    # Run active contour segmentation, the snake will be an array of the same shape as init
    snake = active_contour(im2, init2, w_line=0)


    # Show
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.imshow(im, cmap=plt.cm.gray)
    ax.plot(init2[:, 1], init2[:, 0], '--r', lw=3)
    ax.plot(snake[:, 1], snake[:, 0], '-b', lw=3)

    plt.show()
    print(file_im)

    # Export image to file
    plt.imsave('snakes/' + file_im, im)