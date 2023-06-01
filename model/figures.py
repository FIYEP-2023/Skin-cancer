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