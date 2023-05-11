import argparse
import os
import matplotlib.pyplot as plt
# types
import numpy as np

from model.logger import LogTypes, Logger

def add_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--extract", "-e", help="extract a specific feature or all features")
    parser.add_argument("--resize", "-rs", help="resize all images to a specific size", action="store_true")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--random", "-r", help="extract features from a random image in the dataset", action="store_true")
    group.add_argument("--image", "-i", help="image to extract features from. Format like 'PAT_86_1082_41.png'. The .png is optional", default=None)
    group.add_argument("--images", "-is", help="number of images to extract features from, or 'all' for all images", default=None)
    # pca
    parser.add_argument("--pca", "-p", help="perform pca on the extracted features", action="store_true")
    parser.add_argument("--has_cancer", "-hc", help="check if the images have cancer", action="store_true")
    args = parser.parse_args()
    return args

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
    from feature_extraction.feature_extractor import FeatureExtractor
    extractor = FeatureExtractor()
    # Prepare
    # Select resized images if they exist, else go with the original ones
    if os.path.exists("data/resized"):
        path = "data/resized"
        imgs = os.listdir("data/resized")
    else:
        print("Warning: Resized images not found, proceeding with original sizes.")
        path = "data/segmented"
        imgs = os.listdir("data/segmented")

    # Choose random image
    if args.random:
        random_image = np.random.choice(imgs)
        # Remove _mask from name if present
        random_image = random_image.replace("_mask", "")
        args.image = random_image
        print(f"Using image {random_image}")
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
            print(f"Using {args.images} randomly selected images")
            non_masks = non_masks[:int(args.images)]

        feats = []
        for i, img_name in enumerate(non_masks):
            print(f"{i+1}/{len(non_masks)}", end="\r")
            img = plt.imread(f"{path}/{img_name}")
            mask = plt.imread(f"{path}/{img_name[:-4]}_mask.png")
            img = clean_image(img)
            mask = clean_image(mask, to_binary=True)
            result = extract(img, mask)
            feats.append(result)
        
        # Get labels
        labels = has_cancer(non_masks)

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
    print("Resizing images...")
    from PIL import Image
    imgs = os.listdir("data/segmented")
    for i, img_name in enumerate(imgs):
        print(f"{i+1}/{len(imgs)}", end="\r")
        img = Image.open(f"data/segmented/{img_name}")
        img = img.resize(size)
        img.save(f"data/resized/{img_name}")
    print("Done!")

def has_cancer(img_names: np.ndarray):
    # Load csv
    import pandas as pd
    df = pd.read_csv("data/metadata.csv")
    # Get labels
    cancerous = ["BCC", "SCC", "MEL"]
    labels = []
    for img_name in img_names:
        diagnosis = df.loc[df["img_id"] == img_name]["diagnostic"].values[0]
        cancer = diagnosis in cancerous
        labels.append(cancer)
    
    return np.array(labels)

def pca():
    # Load features and labels from file
    X = np.load("data/features/X.npy")
    y = np.load("data/features/y.npy")

    # Perform PCA
    from model.pca import PCA
    pca = PCA(X, y)
    pca_result = pca.fit(min_variance=0.8)
    print(f"PC count: {pca_result.shape[1]}")

    # Get dummy validation data
    imgs = os.listdir("data/resized")
    imgs = [i for i in imgs if "mask" not in i]
    # Get 10 random images
    np.random.shuffle(imgs)
    imgs = imgs[:10]
    # Get corresponding masks
    masks = [i[:-4] + "_mask.png" for i in imgs]
    # Load images and masks
    X_validate = []
    for img_name, img_mask_name in zip(imgs, masks):
        img = plt.imread(f"data/resized/{img_name}")
        mask = plt.imread(f"data/resized/{img_mask_name}")
        img = clean_image(img)
        mask = clean_image(mask, to_binary=True)
        print(img_name[:-4])
        feat = extract_features({
            "extract": "all",
            "image": img_name[:-4],
            "random": None,
            "images": None
        })
        X_validate.append(feat)
    # get labels
    y_validate = has_cancer(imgs)

    from sklearn.neighbors import KNeighborsClassifier
    k = 4
    # Train KNN
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(pca_result, y)

    # Transform test data because we used PCA on the training data so the axes are different
    X_test_transformed = pca.transform(X_validate)
    # Discard the same components as we did for the training data
    n_components = pca_result.shape[1]
    X_test_transformed_pruned = X_test_transformed[:, :n_components]
    # Predict
    y_pred = knn.predict(X_test_transformed_pruned)
    # Calculate accuracy
    accuracy = np.mean(y_pred == y_validate)
    print(f'k={k}, accuracy={accuracy:.3f}')

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
        result = has_cancer(np.array([args.image]))
        print(result)
    
    if args.pca:
        pca()

if __name__ == '__main__':
    main()