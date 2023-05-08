import argparse
import os
import matplotlib.pyplot as plt
# types
import numpy as np

def add_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--extract", "-e", help="extract a specific feature or all features")
    parser.add_argument("--resize", "-rs", help="resize all images to a specific size", action="store_true")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--random", "-r", help="select a random image from the dataset", action="store_true")
    group.add_argument("--image", "-i", help="image to extract features from. Format like 'PAT_86_1082_41.png'. The .png is optional", default=None)
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
    # Prepare
    # Select resized images if they exist, else go with the original ones
    if os.path.exists("data/resized"):
        imgs = os.listdir("data/resized")
    else:
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
    # Choose first image and corresponding mask
    img = [i for i in imgs if "mask" not in i][0]
    mask = img[:-4] + "_mask.png"

    # Load image
    img = plt.imread(f"data/segmented/{img}")
    mask = plt.imread(f"data/segmented/{mask}")

    # Clean image
    img = clean_image(img)
    mask = clean_image(mask, to_binary=True)
    
    from feature_extraction.feature_extractor import FeatureExtractor
    extractor = FeatureExtractor()

    # Check if "all" is specified
    if args.extract == "all":
        feats = extractor.do_all(img, mask)
        return feats
    
    feat = extractor.extract_feat(args.extract, img, mask)
    return feat

def resize_images(size=(1024, 1024)):
    print("Resizing images...")
    from PIL import Image
    imgs = os.listdir("data/segmented")
    for img_name in imgs:
        img = Image.open(f"data/segmented/{img_name}")
        img = img.resize(size)
        img.save(f"data/resized/{img_name}")

def main():
    args = add_args()
    
    if args.extract is not None:
        result = extract_features(args)
        print(result)
    
    if args.resize:
        resize_images()

if __name__ == '__main__':
    main()