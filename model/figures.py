import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import skimage
from scipy import ndimage

# Load csv
df = pd.read_csv('data/metadata.csv')

# Get segmented images
segmented = os.listdir('data/segmented')
segmented = [x for x in segmented if "mask" not in x]

# Get not segmented images
not_segmented = os.listdir('data/not_segmented')

# Get all
all_images = segmented + not_segmented

def clean_image(img: np.ndarray, to_binary: bool = False):
    # Check if image contains alpha channel
    if img.shape[2] == 4:
        # Remove alpha channel
        img = img[:, :, :3]
    
    # Convert to binary if specified
    if to_binary:
        # Merge all color channels
        img = img.mean(axis=2)
        # Convert to binary
        img = (img > 0).astype(np.uint8)

    return img

def create_figures():
    diagnosis_distribution()

    lesion_segment()
    
    asymmetry()

    filters()
    filter_histograms()

    optimisation_graphs()

def diagnosis_distribution():
    def all_distribution():
        categories = {
            'BCC': [],
            'MEL': [],
            'SEK': [],
            'SCC': [],
            'ACK': [],
            'NEV': [],
        }

        for file in all_images:
            category = df.loc[df['img_id'] == file]['diagnostic'].values[0]
            categories[category].append(file)

        return categories
    
    def has_cancer(cat):
        return cat in ['BCC', 'MEL', 'SCC']
    
    # Diagnoses distribution
    dist = all_distribution()
    dic = { k: len(v) for k, v in dist.items() }
    dic = {k: v for k, v in sorted(dic.items(), key=lambda item: item[1], reverse=True)}

    # Cancer distribution
    cancer = sum([ v for k, v in dic.items() if has_cancer(k) ])
    not_cancer = sum([ v for k, v in dic.items() if not has_cancer(k) ])

    # Plot side by side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    ax1.bar(dic.keys(), dic.values())
    ax1.set_title('Distribution of the diagnoses')
    ax2.bar(['Benign (ACK, NEV, SEK)', 'Malignant (BCC, MEL, SCC)'], [not_cancer, cancer])
    ax2.set_title('Distribution of cancerous vs non-cancerous diagnoses')
    
    # Save to file
    plt.savefig('figures/diagnosis_distribution.png')

def lesion_segment():
    num = 400
    og = segmented[num]
    mask = segmented[num].replace('.png', '_mask.png')

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    ax1.imshow(plt.imread(f'data/segmented/{og}'))
    ax1.set_title('Original image')
    ax2.imshow(plt.imread(f'data/segmented/{mask}'))
    ax2.set_title('Segmented image')

    plt.subplots_adjust(wspace=-0.35)

    plt.savefig('figures/lesion_segment.png', bbox_inches='tight')

def filters():
    img_name = "PAT_86_131_107.png"
    mask_name = img_name.replace('.png', '_mask.png')
    img = clean_image(plt.imread(f'data/segmented/{img_name}'))
    mask = clean_image(plt.imread(f'data/segmented/{mask_name}'), to_binary=True)

    filtered = skimage.feature.multiscale_basic_features(img, channel_axis=2)
    filtered = np.array([filtered[:,:,3], filtered[:,:,1], filtered[:,:,10]])
    for i in filtered:
        i[mask == 0] = 0

    fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(16, 5))
    ax1.imshow(img)
    ax1.set_title('Original image')
    ax2.imshow(filtered[0], cmap='gray')
    ax2.set_title('Filter 3')
    ax3.imshow(filtered[1], cmap='gray')
    ax3.set_title('Filter 1')
    ax4.imshow(filtered[2], cmap='gray')
    ax4.set_title('Filter 10')

    plt.savefig('figures/filters.png', bbox_inches='tight')

def filter_histograms():
    img_name = "PAT_86_131_107.png"
    mask_name = img_name.replace('.png', '_mask.png')
    img = clean_image(plt.imread(f'data/segmented/{img_name}'))
    mask = clean_image(plt.imread(f'data/segmented/{mask_name}'), to_binary=True)

    filtered = skimage.feature.multiscale_basic_features(img, channel_axis=2)
    n_features = filtered.shape[2]
    lesion_features = [filtered[:,:,i][mask==1] for i in range(n_features)]

    from skimage.color import rgb2gray
    fix, axs = plt.subplots(1, 4, figsize=(20, 5))
    og_gray = rgb2gray(img)
    og_gray[mask==0] = 0
    og_gray = og_gray.flatten()
    axs[0].hist(og_gray, bins=10)
    axs[0].set_title('Original image')
    axs[1].hist(lesion_features[3]*256, bins=10)
    axs[1].set_title('Filter 3')
    axs[2].hist(lesion_features[1]*256, bins=10)
    axs[2].set_title('Filter 1')
    axs[3].hist(lesion_features[10]*256, bins=10)
    axs[3].set_title('Filter 10')

    plt.savefig('figures/filter_histograms.png', bbox_inches='tight')

def optimisation_graphs():
    pass

# Copied from feature_extractor.py
def asymmetry_example(mask):
    com = ndimage.center_of_mass(mask) # find center of mass
    com = (int(com[0]), int(com[1])) # turn coords into integers
    
    # creat mask with lesion border
    brush = ndimage.generate_binary_structure(2, 1) # create 2 dim erosion brush
    eroded = ndimage.binary_erosion(mask, brush) # eroded mask
    border = mask - eroded # lesion border
    rows, cols = np.nonzero(border) # find coords of border
    coords = zip(rows, cols) # zip coords
    
    # find distance from center of mass to each border pixel
    dist_list = []
    for r, c in coords:
        '''find manhattan distance from center of mass - faster but bigger array'''
        # dist_list.append(abs(r - com[0]) + abs(c - com[1]))
        '''find euclidean distance from center of mass - slower but smaller array'''
        dist_list.append(int(np.sqrt((r - com[0])**2 + (c - com[1])**2)))
    
    # max distance from center of mass to edge of mask + 10 pixels
    max_dist = max(dist_list) + 10
    
    # slice the mask into a square of side length max_dist+10 with com at the center
    r1 = com[0] - max_dist  # lower bound for row
    r2 = com[0] + max_dist  # upper bound for row
    c1 = com[1] - max_dist  # lower bound for col
    c2 = com[1] + max_dist  # upper bound for col

    # if any of the bounds are outside the image
    # add empty rows/columns to mask until distance to com is max_dist+10
    # this keeps com in center and mask square
    if r1 < 0:
        mask = np.append(np.zeros([abs(r1), mask.shape[1]]), mask, 0)
        r2 += abs(r1)
        r1 = 0
    if c1 < 0:
        mask = np.append(np.zeros([mask.shape[0], abs(c1)]), mask, 1)
        c2 += abs(c1)
        c1 = 0
    if r2 > mask.shape[0]:
        mask = np.append(mask, np.zeros([r2-mask.shape[0], mask.shape[1]]), 0)
        r2 = mask.shape[0]
    if c2 > mask.shape[1]:
        mask = np.append(mask, np.zeros([mask.shape[0], c2-mask.shape[1]]), 1)
        c2 = mask.shape[1]

    # make the square around the lesion
    new_mask = mask[r1:r2,c1:c2]

    # check and correct if new_mask is uneven in either axis
    if new_mask.shape[0] %2 != 0:
        # add a row of zeros to the bottom
        new_mask = np.append(new_mask,np.zeros([1,new_mask.shape[1]]),0)
    if new_mask.shape[1] %2 != 0:
        # add a column of zeros to the right
        new_mask = np.append(new_mask,np.zeros([new_mask.shape[0],1]),1)
    ''' checks symmetry along the vertical axis (left-right) '''
    # split mask into two halves along the vertical axis 
    mask_left, mask_right = np.split(new_mask,2,axis=1)
    # convert to signed integers to prevent underflow
    mask_left = mask_left.astype(np.int8)
    mask_right = mask_right.astype(np.int8)
    
    # invert the left half of the mask
    reflect_mask_left = np.flip(mask_left, axis=1)
    # convert to signed integers to prevent underflow
    reflect_mask_left = reflect_mask_left.astype(np.int8)

    # find the absolute difference between halves
    sym = np.abs(mask_right-reflect_mask_left)

    fig, axs = plt.subplots(2, 2)
    axs[0, 0].imshow(mask_left,cmap='gray',)
    axs[0, 0].set_title('mask_left')
    axs[0, 1].imshow(reflect_mask_left,cmap='gray',)
    axs[0, 1].set_title('reflect_mask_left')
    axs[1, 0].imshow(mask_right,cmap='gray',)
    axs[1, 0].set_title('mask_right')
    axs[1, 1].imshow(sym,cmap='gray',)
    axs[1, 1].set_title('sym')
    # plt.show()
    plt.savefig('figures/asymmetry_example.png', bbox_inches='tight')

# Porbably needs "os" library added
def plot_asymmetry(img: np.ndarray, title: str = None):
    z = clean_image(plt.imread(f"data/resized/{img}"), to_binary=True)
    plt.imshow(z,cmap='gray',interpolation='none')
    plt.title(title)
    # plt.show()
    plt.savefig(f"figures/asymmetry_{img}")

def asymmetry():
    # List of image & title tuples
    imgs = [('PAT_1451_1560_616_mask.png','Compactness score: 0.842'),('PAT_270_417_728_mask.png','Compactness score: 3.556'),('PAT_82_125_907_mask.png','Asymmetry score: 0.113'),('PAT_97_151_587_mask.png','Asymmetry score: 0.518'),('PAT_270_417_728_mask.png','Asymmetry score: 0.824')]

    # Run plot_image() for each image
    for img in imgs:
        plot_asymmetry(img[0], img[1])

    # Run asymmetry example:
    y = clean_image(plt.imread(f"data/resized/PAT_779_1472_741_mask.png"), to_binary=True)
    asymmetry_example(y)