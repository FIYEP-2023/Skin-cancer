import argparse
import os
import matplotlib.pyplot as plt
# types
import numpy as np

def add_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", "-t", help="run main on a single image file", action="store_true")
    parser.add_argument("--extract", "-e", help="extract a specific feature or all features")
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

def extract_features(feat_type: str, img: np.ndarray, mask: np.ndarray):
    from feature_extraction.feature_extractor import FeatureExtractor
    extractor = FeatureExtractor()

    # Check if "all" is specified
    if feat_type == "all":
        feats = extractor.do_all(img, mask)
        return feats
    
    feat = extractor.extract_feat(feat_type, img, mask)
    return feat

def main():
    args = add_args()

    if not args.test:
        print("Only test supported for now. Use -t.")
        return
    
    # Get testing image
    imgs = os.listdir("data/segmented")
    # Choose first image and corresponding mask
    img = [i for i in imgs if "mask" not in i][0]
    mask = img[:-4] + "_mask.png"

    # Load image
    img = plt.imread(f"data/segmented/{img}")
    mask = plt.imread(f"data/segmented/{mask}")

    # Clean image
    img = clean_image(img)
    mask = clean_image(mask, to_binary=True)
    
    # Feature extraction
    if args.extract is not None:
        result = extract_features(args.extract, img, mask)
        print(result)


if __name__ == '__main__':
    main()