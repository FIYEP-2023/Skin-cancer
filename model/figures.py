import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import skimage
from scipy import ndimage
import pickle
from typing import Union, Tuple
from model.topk import TopK
import numpy.typing as npt
from model.classifiers import KNN, Logistic, train_splits, evaluate_splits


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

    color_figures()

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
    # COPIED FROM main.py
    # this is so cringe holy shit literally copied the entirety of main just to create two fucking figures
    def validate_file(filepath: str, prerequisite: Union[str, None] = None):
        if not os.path.exists(filepath) or os.stat(filepath).st_size == 0:
            raise FileNotFoundError(f"The file {filepath} does not exist or is empty. " + (f"Please run {prerequisite} first." if prerequisite is not None else ""))

    def topk(top_k_k: Union[int, None] = None):
        # Make sure our data exists
        validate_file("data/features/X.pkl", "--extract all --images all")
        validate_file("data/features/y.pkl", "--extract all --images all")
        validate_file("data/features/img_names.pkl", "--extract all --images all")
        validate_file("data/training/training_splits.pkl", "--split")

        # Load data
        with open("data/features/X.pkl", "rb") as f:
            X: np.ndarray = np.array(pickle.load(f))
        with open("data/features/y.pkl", "rb") as f:
            y: np.ndarray = np.array(pickle.load(f))
        with open("data/features/img_names.pkl", "rb") as f:
            img_names: list[str] = pickle.load(f)
        with open("data/training/training_splits.pkl", "rb") as f:
            training_splits: list[npt.NDArray[np.str_]] = pickle.load(f)

        # Set top_k_k to feature count if not specified
        if top_k_k is None:
            top_k_k = X.shape[1]

        # Train topk on each training split
        topks = []
        for i, train in enumerate(training_splits):
            # Get indices of images in this split
            indices = [img_names.index(img) for img in train]
            # Get features for this split
            X_split = X[indices]
            y_split = y[indices]
            # Fit topk
            topk = TopK(X_split, y_split, top_k_k)
            topk.DO_ABCD_ONLY = False
            topk.fit()
            topks.append(topk)
        
        # Train topk on all training data
        # Merge all training splits (remove duplicates)
        train = np.concatenate(training_splits)
        train = np.unique(train)
        # Get indices of images in this split
        indices = [img_names.index(img) for img in train]
        # Get features for this split
        X_split = X[indices]
        y_split = y[indices]
        # Fit topk
        topk = TopK(X_split, y_split, top_k_k)
        topk.fit()
        
        # Save to file
        with open("data/training/full_pca.pkl", "wb") as f:
            pickle.dump(topk, f)
        with open("data/training/cross_val_pcas.pkl", "wb") as f:
            pickle.dump(topks, f)

        # Close loaded files
        del X, y, img_names, training_splits

    def train(k_neighbors: int = 5):
        # Make sure our data exists
        validate_file("data/features/X.pkl", "--extract all --images all")
        validate_file("data/features/y.pkl", "--extract all --images all")
        validate_file("data/features/img_names.pkl", "--extract all --images all")
        validate_file("data/training/cross_val_pcas.pkl", "--pca")
        validate_file("data/training/full_pca.pkl", "--pca")
        validate_file("data/training/validation_splits.pkl", "--split")
        validate_file("data/training/test_data.pkl", "--split")

        # Load data
        with open("data/features/X.pkl", "rb") as f:
            X: np.ndarray = np.array(pickle.load(f))
        with open("data/features/y.pkl", "rb") as f:
            y: np.ndarray = np.array(pickle.load(f))
        with open("data/features/img_names.pkl", "rb") as f:
            img_names: list[str] = pickle.load(f)
        with open("data/training/cross_val_pcas.pkl", "rb") as f:
            topks: list[TopK] = pickle.load(f)
        with open("data/training/full_pca.pkl", "rb") as f:
            full_topk: TopK = pickle.load(f)
        with open("data/training/validation_splits.pkl", "rb") as f:
            validation_splits: list[npt.NDArray[np.str_]] = pickle.load(f)
        with open("data/training/test_data.pkl", "rb") as f:
            test_data: npt.NDArray[np.str_] = np.array(pickle.load(f))

        # Process validation data
        X_val_splits = []
        y_val_splits = []
        for i, split in enumerate(validation_splits):
            # Get indices of images in this split
            indices = [img_names.index(img) for img in split]
            topk = topks[i]
            # X
            X_val = topk.transform(X[indices])
            X_val_splits.append(X_val)
            # y
            y_val = y[indices]
            y_val_splits.append(y_val)


        # Train models on every split
        models = train_splits(topks, X_val_splits, y_val_splits, n_neighbors=k_neighbors)

        # Train full model on all training data
        # Get indices of images in test data
        indices = [img_names.index(img) for img in test_data]
        # Get features for this split
        X_test = full_topk.transform(X[indices])
        y_test = y[indices]
        # Fit KNN
        knn = KNN(full_topk, X_test, y_test, n_neighbors=k_neighbors)
        log = Logistic(full_topk, X_test, y_test)

        # Save to file
        with open("data/training/cross_val_models.pkl", "wb") as f:
            pickle.dump(models, f)
        with open("data/training/full_knn.pkl", "wb") as f:
            pickle.dump(knn, f)
        with open("data/training/full_log.pkl", "wb") as f:
            pickle.dump(log, f)
        
        # Close loaded files
        del X, y, img_names, topks, full_topk, validation_splits, test_data

    def test_top_k(max_k: int, k_neighbors: int = 5):
        """
        Test different values of K for top K and plots K vs. f1 score.
        """
        k_neighbors = 5 if k_neighbors is None else k_neighbors
        # Run tests
        x = []
        y = []
        for n in range(1, max_k+1):
            topk(n)
            train(k_neighbors=k_neighbors)

            # Load data
            with open("data/training/cross_val_models.pkl", "rb") as f:
                models: list[Tuple[KNN, Logistic]] = pickle.load(f)
            
            # Evaluate
            stats = evaluate_splits(models)
            f1 = stats["f1"][0] # [0] for knn
            x.append(n)
            y.append(f1)

            # Close loaded files
            del models
        
        # Plot results
        fig, ax = plt.subplots()
        ax.plot(x, y)
        ax.set_xlabel("K")
        ax.set_ylabel("F1 score")
        ax.set_title("K vs. F1 score")

        # Save x, y and plot to file
        with open("data/training/components_x.pkl", "wb") as f:
            pickle.dump(x, f)
        with open("data/training/components_y.pkl", "wb") as f:
            pickle.dump(y, f)
        plt.savefig(f"figures/components_plot_{max_k}.png")

    def test_neighbors(max_neighbors: int):
        """
        Test different numbers of neighbors for KNN and plots neighbors vs. errors.
        """
        # Run tests
        x = []
        y = []
        for k in range(1, max_neighbors+1, 2): # Only odd numbers
            train(k)

            # Load data
            with open("data/training/cross_val_models.pkl", "rb") as f:
                models: list[Tuple[KNN, Logistic]] = pickle.load(f)
            
            # Evaluate
            stats = evaluate_splits(models)
            cf_knn, cf_log = stats["confusion_matrix"]
            x.append(k)
            y.append(cf_knn[1,0] + cf_knn[0,1])

            # Close loaded files
            del models
        
        # Create new plots
        fig, ax = plt.subplots()
        ax.plot(x, y)
        ax.set_xlabel("Number of neighbors")
        ax.set_ylabel("Errors")
        ax.set_title("Number of neighbors vs. Errors")

        # Save x, y and plot to file
        with open("data/training/neighbors_x.pkl", "wb") as f:
            pickle.dump(x, f)
        with open("data/training/neighbors_y.pkl", "wb") as f:
            pickle.dump(y, f)
        plt.savefig(f"figures/neighbors_plot_{max_neighbors}.png")

    test_neighbors(100)
    test_top_k(50, k_neighbors=11)

def color_figures():
    def manhatten(true_color, pixel_color):
        return np.sum(np.abs(true_color - pixel_color))

    img_name = 'PAT_108_161_423'
    mask_name = img_name.replace('.png', '_mask.png')
    img = clean_image(plt.imread(f'data/segmented/{img_name}'))
    mask = clean_image(plt.imread(f'data/segmented/{mask_name}'), to_binary=True)

    mask = mask[:, :, 0].astype(np.uint8)

    color_dict = {
        'white':[(175, 172, 167),0],
        'light-brown':[(143, 100, 76),0],
        'dark-brown':[(82, 70, 67),0],
        'blue-grey':[(59, 63, 75),0],
        'red':[(146, 80, 86),0],
        'black':[(48, 51, 49),0] 
    }

    slic = skimage.segmentation.slic(img, n_segments=100, compactness=10, sigma=1, start_label=1, mask=mask)
    img_new = img.copy()
    img_res = img.copy()
    for i in np.unique(slic):
        if np.sum(mask[slic == i]) == 0:
            continue
        img_new[slic == i] = np.mean(img_new[slic == i], axis=0)
        norm = np.mean(img_new[slic == i], axis=0)
        rgb = np.array((norm * 255).astype(int))
        min_dist = 1000
        color = None
        for key, value in color_dict.items():
            dist = manhatten(rgb, value[0])
            if dist < min_dist:
                min_dist = dist
                color = key
        color_dict[color][1] += 1
        img_res[slic == i] = np.array(color_dict[color][0]) / 255   
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].imshow(img_new)
    ax[1].imshow(img_res)

    plt.savefig('figures/color.png')  