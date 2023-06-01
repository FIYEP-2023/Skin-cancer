import pickle
import numpy as np
import matplotlib.pyplot as plt

from model.feature_extractor import FeatureExtractor
from model.classifiers import KNN
from model.logger import Logger
from main import clean_image

PROBABILITY_THRESHOLD = 0.5

with open("model/group3_classifier.sav", "rb") as f:
    knn: KNN = pickle.load(f)
knn.probability = True
knn.probability_threshold = PROBABILITY_THRESHOLD
knn.train() # this is dumb, but we forgot to call train() during training.
knn.topk.DO_ABCD_ONLY = True # We're not doing general features

def classify(img, mask):
    img = clean_image(img)
    mask = clean_image(mask, to_binary=True)

    extractor = FeatureExtractor()
    features = extractor.do_all(img, mask)
    feature_matrix = np.array([features])

    # Can't use our own topk.transform because we only get 1 image and it requires multiple to standardise them
    X = feature_matrix[:, :8]

    prediction = knn.predict(X)
    proba = knn.proba(X)
    pred_label = "Malignant" if prediction[0] == 1 else "Benign"

    Logger.log(f"Predicted label: {prediction[0]} ({pred_label})")
    Logger.log(f"Predicted probability: {proba[0][1]}")
    return prediction[0], proba[0][1]