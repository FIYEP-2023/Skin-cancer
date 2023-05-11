import argparse
import os
import matplotlib.pyplot as plt
from PIL import Image
import pickle
# types
import numpy as np

# Custom
from model.feature_extractor import FeatureExtractor
from model.pca import PCA
from model.data_splitter import DataSplitter
from model.logger import LogTypes, Logger

def add_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--extract", "-e", help="extract a specific feature or all features")
    parser.add_argument("--resize", "-rs", help="resize all images to a specific size", action="store_true")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--random", "-r", help="extract features from a random image in the dataset", action="store_true")
    group.add_argument("--image", "-i", help="image to extract features from. Format like 'PAT_86_1082_41.png'. The .png is optional", default=None)
    group.add_argument("--images", "-is", help="number of images to extract features from, or 'all' for all images", default=None)
    parser.add_argument("--has_cancer", "-hc", help="check if the images have cancer", action="store_true")
    # pca
    parser.add_argument("--pca", "-p", help="perform pca on the extracted features", action="store_true")
    parser.add_argument("--min_variance", "-mv", help="the minimum variance to keep", default=0.8)
    # split
    parser.add_argument("--split", "-s", help="split the data into training and testing sets", action="store_true")
    parser.add_argument("--train_size", "-ts", help="the size of the training set", default=0.8)
    parser.add_argument("--folds", "-f", help="the number of folds to use for cross validation", default=5)
    args = parser.parse_args()
    return args

def validate_folder(path: str, prerequisite: str = None):
    if not os.path.exists(path) or os.listdir(path) == []:
        raise ValueError(f"Folder {path} does not exist or is empty." + (f"Please run {prerequisite} first." if prerequisite is not None else ""))

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

def extract_features(args):
    extractor = FeatureExtractor()
    # Prepare
    # Select resized images if they exist, else go with the original ones
    if os.path.exists("data/resized"):
        path = "data/resized"
        imgs = os.listdir("data/resized")
    else:
        Logger.log("Resized images not found, proceeding with original sizes.", LogTypes.WARNING)
        path = "data/segmented"
        imgs = os.listdir("data/segmented")

    # Choose random image
    if args.random:
        random_image = np.random.choice(imgs)
        # Remove _mask from name if present
        random_image = random_image.replace("_mask", "")
        args.image = random_image
        Logger.log(f"Using image {random_image}")
    # Choose user specified image
    if args.image is not None:
        imgs = [i for i in imgs if args.image in i]

    def extract(img, mask):
        # Check if "all" is specified
        if args.extract == "all":
            feats = extractor.do_all(img, mask)
            return feats
    
        feat = extractor.extract_feat(args.extract, img, mask)
        return feat

    if args.images is not None:
        non_masks = [i for i in imgs if "mask" not in i]
        np.random.shuffle(non_masks)
        if args.images.isdigit():
            Logger.log(f"Using {args.images} randomly selected images")
            non_masks = non_masks[:int(args.images)]

        feats = []
        for i, img_name in enumerate(non_masks):
            Logger.log(f"{i+1}/{len(non_masks)}", end="\r")
            img = plt.imread(f"{path}/{img_name}")
            mask = plt.imread(f"{path}/{img_name[:-4]}_mask.png")
            img = clean_image(img)
            mask = clean_image(mask, to_binary=True)
            result = extract(img, mask)
            feats.append(result)
            Logger.log(f"Feature count: {len(result)}")
        
        # Get labels
        labels = DataSplitter.has_cancer(non_masks)

        # Save features and labels to file
        np.save("data/features/X.npy", feats)
        np.save("data/features/y.npy", labels)

        return feats

    # Choose first image and corresponding mask
    img = [i for i in imgs if "mask" not in i][0]
    mask = img[:-4] + "_mask.png"

    # Load image and clean image
    img = plt.imread(f"{path}/{img}")
    mask = plt.imread(f"{path}/{mask}")
    img = clean_image(img)
    mask = clean_image(mask, to_binary=True)
    
    return extract(img, mask)

def resize_images(size=(1024, 1024)):
    Logger.log("Resizing images...")
    imgs = os.listdir("data/segmented")
    for i, img_name in enumerate(imgs):
        Logger.log(f"{i+1}/{len(imgs)}", end="\r")
        img = Image.open(f"data/segmented/{img_name}")
        img = img.resize(size)
        img.save(f"data/resized/{img_name}")
    Logger.log("Done!")

def split_training(train_size: None, folds: None):
    # Make sure our data exists
    validate_folder("data/features", "--extract")
    # Load features and labels from file
    X = np.load("data/features/X.npy")
    y = np.load("data/features/y.npy")
    Logger.log(f"Splitting data with {X.shape[0]} samples and {X.shape[1]} features", level = LogTypes.INFO)
    # Make sure X and y have the same amount of samples
    if X.shape[0] != y.shape[0]:
        raise ValueError(f"X and y have different amounts of samples ({X.shape[0]} and {y.shape[0]} respectively)")

    # Add y as column in X
    data = np.append(X, y.reshape(-1, 1), axis=1)
    # Split
    trains, validates, test = DataSplitter(data).split(train_size=train_size, folds=folds)

    # Save to file (cant use numpy because they are no arrays)
    with open("data/training/training_splits.npy", "wb") as f:
        pickle.dump(trains, f)
    with open("data/training/validation_splits.npy", "wb") as f:
        pickle.dump(validates, f)
    with open("data/training/test_data.npy", "wb") as f:
        pickle.dump(test, f)
    Logger.log(f"Saved {len(trains)} training splits, {len(validates)} validation splits and {len(test)} test data to file", level=LogTypes.INFO)

def pca(min_variance: float = None):
    # Make sure our data exists
    validate_folder("data/training", "--split")

    # Load data with pickle
    with open("data/training/training_splits.npy", "rb") as f:
        training_splits = pickle.load(f)
    with open("data/training/validation_splits.npy", "rb") as f:
        validation_splits = pickle.load(f)
    with open("data/training/test_data.npy", "rb") as f:
        test_data = pickle.load(f)

    def fit_pca(X: np.ndarray, y: np.ndarray):
        # Perform PCA
        pca = PCA(X, y)
        pca.fit(min_variance=min_variance)
        Logger.log(f"PCA result has {pca.pca_result_pruned.shape[1]} components", level=LogTypes.INFO)
        return pca

    # Train PCA
    pcas = []
    for i in range(len(training_splits)):
        X = training_splits[i][:, :-1]
        y = training_splits[i][:, -1]
        pca = fit_pca(X, y)
        pcas.append(pca)
    
    # Save PCA
    np.save("data/training/pcas.npy", pcas)

def main():
    args = add_args()
    
    if args.extract is not None:
        extract_features(args)
    
    if args.resize:
        resize_images()
    
    if args.has_cancer:
        if args.image is None:
            raise ValueError("Please specify an image to check for cancer using -i or --image")
        # add .png if not present
        if ".png" not in args.image:
            args.image += ".png"
        # throw error if using mask
        if "_mask" in args.image:
            raise ValueError("Please specify an image without the mask")
        result = DataSplitter.has_cancer(np.array([args.image]))
        Logger.log(f"Image {args.image} has cancer: {result[0]}", level=LogTypes.INFO)
    
    if args.pca:
        pca(args.min_variance)
    
    if args.split:
        split_training(args.train_size, args.folds)

if __name__ == '__main__':
    main()