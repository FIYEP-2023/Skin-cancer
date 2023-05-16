import argparse
import os
import matplotlib.pyplot as plt
from PIL import Image
import pickle
# types
import numpy as np
from typing import Tuple

# Custom
from model.feature_extractor import FeatureExtractor
from model.pca import PCA
from model.data_splitter import DataSplitter
from model.logger import LogTypes, Logger
from model.classifiers import KNN, LogisticRegression, train_splits, evaluate_splits

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
    parser.add_argument("--n_components", "-c", help="number of principal components in PCA", default=None)
    # Testing
    parser.add_argument("--test_components", "-tc", help="test different numbers of components from 1 to n_components for PCA", action="store_true")
    parser.add_argument("--test_neighbors", "-tn", help="test different numbers of neighbors from 1 to n_neighbors for KNN", action="store_true")
    # split
    parser.add_argument("--split", "-s", help="split the data into training and testing sets", action="store_true")
    parser.add_argument("--train_size", "-ts", help="the size of the training set", default=None)
    parser.add_argument("--folds", "-f", help="the number of folds to use for cross validation", default=5)
    # training
    parser.add_argument("--train", "-t", help="train all models on the data", action="store_true")
    parser.add_argument("--k_neighbors", "-k", help="number of neighbors to use for KNN", default=5)
    # evaluation
    parser.add_argument("--eval", "-ev", help="evaluate cross-validated model and get statistics", action="store_true")
    parser.add_argument("--probability_threshold", "-pt", help="the probability threshold to use for logistic regression", default=0.5)
    args = parser.parse_args()
    return args

def validate_file(filepath: str, prerequisite: str = None):
    if not os.path.exists(filepath) or os.stat(filepath).st_size == 0:
        raise FileNotFoundError(f"The file {filepath} does not exist or is empty. " + (f"Please run {prerequisite} first." if prerequisite is not None else ""))

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

        Logger.log(f"Extracting features from {len(non_masks)} images")
        feats = []
        for i, img_name in enumerate(non_masks):
            Logger.log(f"Extracting {i+1}/{len(non_masks)}")
            img = plt.imread(f"{path}/{img_name}")
            mask = plt.imread(f"{path}/{img_name[:-4]}_mask.png")
            img = clean_image(img)
            mask = clean_image(mask, to_binary=True)
            result = extract(img, mask)
            feats.append(result)
            Logger.log(f"Feature count: {len(result)}")
        
        # Get labels
        labels = FeatureExtractor.has_cancer(non_masks)

        # Save features and labels to file
        with open("data/features/X.pkl", "wb") as f:
            pickle.dump(feats, f)
        with open("data/features/y.pkl", "wb") as f:
            pickle.dump(labels, f)
        # Save image names
        with open("data/features/img_names.pkl", "wb") as f:
            pickle.dump(non_masks, f)

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

def split_data(train_size: float = 0.8, folds: int = 5):
    # Select resized images if they exist, else go with the original ones
    if os.path.exists("data/resized"):
        path = "data/resized"
    else:
        Logger.log("Resized images not found, proceeding with original sizes.", LogTypes.WARNING)
        path = "data/segmented"
    imgs = os.listdir(path)
    # Remove masks
    imgs = [i for i in imgs if "mask" not in i]

    # Send to splitter
    splitter = DataSplitter(imgs)
    # These are arrays of image names, not actual data
    trains, validates, test = splitter.split(train_size, folds)

    # Save to file (cant use numpy because their length does not match)
    with open("data/training/training_splits.pkl", "wb") as f:
        pickle.dump(trains, f)
    with open("data/training/validation_splits.pkl", "wb") as f:
        pickle.dump(validates, f)
    with open("data/training/test_data.pkl", "wb") as f:
        pickle.dump(test, f)
    Logger.log(f"Saved {len(trains)} training splits, {len(validates)} validation splits and {len(test)} test data to file", level=LogTypes.INFO)

def pca(n_components: int = None):
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
        training_splits: list[np.ndarray[str]] = pickle.load(f)

    # Set n_components to feature count if not specified
    if n_components is None:
        n_components = X.shape[1]

    # Train PCA on each training split
    pcas = []
    for i, train in enumerate(training_splits):
        Logger.log(f"Training PCA on split {i+1}/{len(training_splits)}")
        # Get indices of images in this split
        indices = [img_names.index(img) for img in train]
        # Get features for this split
        X_split = PCA.normalise(X[indices])
        y_split = y[indices]
        # Fit PCA
        pca = PCA(X_split, y_split, n_components)
        pca.fit()
        pcas.append(pca)
    
    # Train PCA on all training data
    Logger.log("Training PCA on all training data")
    # Merge all training splits (remove duplicates)
    train = np.concatenate(training_splits)
    train = np.unique(train)
    # Get indices of images in this split
    indices = [img_names.index(img) for img in train]
    # Get features for this split
    X_split = PCA.normalise(X[indices])
    y_split = y[indices]
    # Fit PCA
    pca = PCA(X_split, y_split, n_components)
    pca.fit()
    
    # Save to file
    with open("data/training/full_pca.pkl", "wb") as f:
        pickle.dump(pca, f)
    with open("data/training/cross_val_pcas.pkl", "wb") as f:
        pickle.dump(pcas, f)
    Logger.log("Saved PCAs to file", level=LogTypes.INFO)

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
        pcas: list[PCA] = pickle.load(f)
    with open("data/training/full_pca.pkl", "rb") as f:
        full_pca: PCA = pickle.load(f)
    with open("data/training/validation_splits.pkl", "rb") as f:
        validation_splits: list[np.ndarray[str]] = pickle.load(f)
    with open("data/training/test_data.pkl", "rb") as f:
        test_data: np.ndarray[str] = np.array(pickle.load(f))

    # Process validation data
    X_val_splits = []
    y_val_splits = []
    for i, split in enumerate(validation_splits):
        Logger.log(f"Processing validation split {i+1}/{len(validation_splits)}")
        # Get indices of images in this split
        indices = [img_names.index(img) for img in split]
        pca = pcas[i]
        # X
        X_val = pca.normalise(X[indices])
        X_val = pca.transform(X_val)
        X_val_splits.append(X_val)
        # y
        y_val = y[indices]
        y_val_splits.append(y_val)


    # Train models on every split
    models = train_splits(pcas, X_val_splits, y_val_splits, n_neighbors=k_neighbors)

    # Train full model on all training data
    Logger.log("Training full model on all training data")
    # Get indices of images in test data
    indices = [img_names.index(img) for img in test_data]
    # Get features for this split
    X_test = full_pca.normalise(X[indices])
    X_test = full_pca.transform(X_test)
    y_test = y[indices]
    # Fit KNN
    knn = KNN(full_pca, X_test, y_test, n_neighbors=k_neighbors)

    # Save to file
    with open("data/training/cross_val_models.pkl", "wb") as f:
        pickle.dump(models, f)
    with open("data/training/full_knn.pkl", "wb") as f:
        pickle.dump(knn, f)
    
    # Close loaded files
    del X, y, img_names, pcas, full_pca, validation_splits, test_data

def evaluate(probability_threshold):
    """
    Evaluate cross-trained models trained with train()
    """
    # Make sure our data exists
    validate_file("data/training/cross_val_models.pkl", "--train")

    # Load data
    with open("data/training/cross_val_models.pkl", "rb") as f:
        model: list[Tuple(KNN, LogisticRegression)] = pickle.load(f)
    
    # Evaluate
    Logger.log("Evaluating cross-trained models")
    for knn, log in model:
        knn.probability = True
        log.probability = True
        knn.probability_threshold = probability_threshold
        log.probability_threshold = probability_threshold
    confs = [(knn.get_confusion_matrix(), log.get_confusion_matrix()) for knn, log in model]
    knn_conf = np.mean([conf[0] for conf in confs], axis=0)
    log_conf = np.mean([conf[1] for conf in confs], axis=0)

    stats = evaluate_splits(model, True, probability_threshold=probability_threshold)
    acc_knn, acc_log = stats["accuracy"]
    prec_knn, prec_log = stats["precision"]
    rec_knn, rec_log = stats["recall"]
    f1_knn, f1_log = stats["f1"]
    roc_auc_knn, roc_auc_log = stats["roc_auc"]
    Logger.log("Cross-trained model stats (KNN):")
    Logger.log(f"Accuracy: {acc_knn:.4f}")
    Logger.log(f"Precision: {prec_knn:.4f}")
    Logger.log(f"Recall: {rec_knn:.4f}")
    Logger.log(f"F1: {f1_knn:.4f}")
    Logger.log(f"ROC AUC: {roc_auc_knn:.4f}")
    Logger.log(f"Confusion matrix:  Actual values")
    Logger.log(f"                     1       0    ")
    Logger.log(f"                 +-----------------+")
    Logger.log(f"    Predicted 1: |  {knn_conf[0][0]:.1f}   {knn_conf[0][1]:.1f}  |")
    Logger.log(f"              0: |  {knn_conf[1][0]:.1f}    {knn_conf[1][1]:.1f}    |")
    Logger.log(f"                 +-----------------+")

    Logger.log("Cross-trained model stats (Logistic Regression):")
    Logger.log(f"Accuracy: {acc_log:.4f}")
    Logger.log(f"Precision: {prec_log:.4f}")
    Logger.log(f"Recall: {rec_log:.4f}")
    Logger.log(f"F1: {f1_log:.4f}")
    Logger.log(f"ROC AUC: {roc_auc_log:.4f}")
    Logger.log(f"Confusion matrix:  Actual values")
    Logger.log(f"                     1        0    ")
    Logger.log(f"                 +-----------------+")
    Logger.log(f"    Predicted 1: |  {log_conf[0][0]:.1f}   {log_conf[0][1]:.1f}  |")
    Logger.log(f"              0: |  {log_conf[1][0]:.1f}     {log_conf[1][1]:.1f}    |")
    Logger.log(f"                 +-----------------+")


    # Close loaded files
    del model

def test_components(max_components: int, k_neighbors: int = 5):
    """
    Test different numbers of components for PCA and plots components vs. f1 score.
    """
    k_neighbors = 5 if k_neighbors is None else k_neighbors
    # Run tests
    Logger.log("Testing different numbers of components for PCA")
    x = []
    y = []
    for n in range(1, max_components+1):
        pca(n)
        train(k_neighbors=k_neighbors)

        # Load data
        with open("data/training/cross_val_models.pkl", "rb") as f:
            models: list[Tuple[KNN, LogisticRegression]] = pickle.load(f)
            knns: list[KNN] = [model[0] for model in models]
        
        # Evaluate
        stats = evaluate_splits(knns)
        f1 = stats["f1"]
        x.append(n)
        y.append(f1)

        # Close loaded files
        del knns
    
    # Plot results
    Logger.log("Plotting results")
    plt.plot(x, y)
    plt.xlabel("Number of components")
    plt.ylabel("F1 score")
    plt.title("Number of components vs. F1 score")

    # Save x, y and plot to file
    Logger.log("Saving results to file")
    with open("data/training/components_x.pkl", "wb") as f:
        pickle.dump(x, f)
    with open("data/training/components_y.pkl", "wb") as f:
        pickle.dump(y, f)
    plt.savefig(f"data/training/components_plot_{max_components}.png")

    plt.show()

def test_neighbors(max_neighbors: int):
    """
    Test different numbers of neighbors for KNN and plots neighbors vs. f1 score.
    """
    # Run tests
    Logger.log("Testing different numbers of neighbors for KNN")
    x = []
    y = []
    for k in range(1, max_neighbors+1):
        Logger.log(f"Testing {k} neighbors")
        train(k)

        # Load data
        with open("data/training/cross_val_models.pkl", "rb") as f:
            models: list[Tuple[KNN, LogisticRegression]] = pickle.load(f)
        
        # Evaluate
        stats = evaluate_splits(models)
        f1_knn, f1_log = stats["f1"]
        x.append(k)
        y.append(f1_knn)

        # Close loaded files
        del models
    
    # Plot results
    Logger.log("Plotting results")
    plt.plot(x, y)
    plt.xlabel("Number of neighbors")
    plt.ylabel("F1 score")
    plt.title("Number of neighbors vs. F1 score")

    # Save x, y and plot to file
    Logger.log("Saving results to file")
    with open("data/training/neighbors_x.pkl", "wb") as f:
        pickle.dump(x, f)
    with open("data/training/neighbors_y.pkl", "wb") as f:
        pickle.dump(y, f)
    plt.savefig(f"data/training/neighbors_plot_{max_neighbors}.png")

    plt.show()

def main():
    args = add_args()
    
    if args.has_cancer:
        if args.image is None:
            raise ValueError("Please specify an image to check for cancer using -i or --image")
        # add .png if not present
        if ".png" not in args.image:
            args.image += ".png"
        # throw error if using mask
        if "_mask" in args.image:
            raise ValueError("Please specify an image without the mask")
        result = FeatureExtractor.has_cancer(np.array([args.image]))
        Logger.log(f"Image {args.image} has cancer: {result[0]}", level=LogTypes.INFO)

    if args.resize:
        resize_images()

    if args.extract is not None:
        extract_features(args)

    if args.split:
        if args.train_size is not None and args.folds is not None:
            split_data(args.train_size, args.folds)
        elif args.train_size is not None:
            split_data(args.train_size)
        elif args.folds is not None:
            split_data(folds=args.folds)
        else:
            split_data()

    if args.test_components:
        if args.n_components is None:
            raise ValueError("Please specify the number of components to test using -n or --n_components")
        test_components(int(args.n_components), args.k_neighbors)
        return # If we're testing components, we don't need to do anything else
    if args.test_neighbors:
        if args.k_neighbors is None:
            raise ValueError("Please specify the number of neighbors to test using -k or --k_neighbors")
        test_neighbors(int(args.k_neighbors))
        return

    if args.pca:
        if args.n_components is not None:
            pca(int(args.n_components))
        else:
            pca()
    
    if args.train:
        if args.k_neighbors is not None:
            train(int(args.k_neighbors))
        else:
            train()

    if args.eval:
        evaluate(float(args.probability_threshold))

if __name__ == '__main__':
    main()