import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

# Load csv
df = pd.read_csv('data/metadata.csv')

# Get segmented images
segmented = os.listdir('data/segmented')
segmented = [x for x in segmented if "mask" not in x]

# Get not segmented images
not_segmented = os.listdir('data/not_segmented')

# Get all
all_images = segmented + not_segmented

def create_figures():
    diagnosis_distribution()

    lesion_segment()
    
    asymmetry_overlap()
    asymmetry_score()
    
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

def asymmetry_overlap():
    pass

def asymmetry_score():
    pass

def filters():
    pass

def filter_histograms():
    pass

def optimisation_graphs():
    pass