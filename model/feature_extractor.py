from scipy import ndimage # updated morphology in newest scipy(?)
import numpy as np # needs full import for np.delete(?)
import math # for pi and sqrt
from statistics import mean # for mean
# from timeit import default_timer as timer
import matplotlib.pyplot as plt # for testing
import skimage
import json
import pandas as pd
from typing import Union, List, Tuple
from numpy.typing import NDArray

class FeatureExtractor:
    @staticmethod
    def has_cancer(img_names: Union[NDArray[np.str_], List[str], Tuple[str]]):
        """
        Given an array of image names (in the form PAT_45.66.822.png), returns an array of booleans indicating whether or not the image has cancer.
        """
        # Load csv
        df = pd.read_csv("data/metadata.csv")
        # Get labels
        cancerous = ["BCC", "SCC", "MEL"]
        labels = []
        for img_name in img_names:
            diagnosis = df.loc[df["img_id"] == img_name]["diagnostic"].values[0]
            cancer = diagnosis in cancerous
            labels.append(cancer)
        
        return np.array(labels)

    def asymmetry(self, img, mask):
        ''' uncomment to time, also lines 217,218 '''
        # start = timer()

        area = np.sum(mask) # lesion area
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


        def split_vertical():
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

            ratio = 2*np.sum(sym)/area # multiplied by 2 to normalize in [0,1]

            ''' uncomment following to plot '''
            # import matplotlib.pyplot as plt
            # fig, axs = plt.subplots(2, 2)
            # axs[0, 0].imshow(mask_left)
            # axs[0, 0].set_title('mask_left')
            # axs[0, 1].imshow(reflect_mask_left)
            # axs[0, 1].set_title('reflect_mask_left')
            # axs[1, 0].imshow(mask_right)
            # axs[1, 0].set_title('mask_right')
            # axs[1, 1].imshow(sym)
            # axs[1, 1].set_title('sym')
            # plt.show()

            return ratio


        def split_horizontal():
            ''' checks symmetry along the horizontal axis (up-down) '''
            # split the mask into two halves along the horizontal axis 
            mask_up, mask_down = np.split(new_mask,2, axis=0)
            # convert to signed integers to prevent underflow
            mask_up = mask_up.astype(np.int8)
            mask_down = mask_down.astype(np.int8)

            # invert the left half of the mask
            reflect_mask_up = np.flip(mask_up, axis=0)
            # convert to signed integers to prevent underflow
            reflect_mask_up = reflect_mask_up.astype(np.int8)

            # find the absolute difference between halves
            sym = np.abs(mask_down-reflect_mask_up)

            ratio = 2*np.sum(sym)/area # multiplied by 2 to normalize in [0,1]

            ''' uncomment following to plot '''
            import matplotlib.pyplot as plt
            # fig, axs = plt.subplots(2, 2)
            # axs[0, 0].imshow(mask_up)
            # axs[0, 0].set_title('mask_up')
            # axs[0, 1].imshow(reflect_mask_up)
            # axs[0, 1].set_title('reflect_mask_up')
            # axs[1, 0].imshow(mask_down)
            # axs[1, 0].set_title('mask_down')
            # axs[1, 1].imshow(sym)
            # axs[1, 1].set_title('sym')
            # plt.show()

            return ratio


        def split_downwards_diagonal():
            ''' checks symmetry along the downwards diagonal axis (\) '''
            # upper half of the mask
            mask_up = np.triu(new_mask)
            # convert to signed integers to prevent underflow
            mask_up = mask_up.astype(np.int8)

            # flip across the diagonal by transposing
            rotate_mask_up=np.transpose(mask_up)
            # convert to signed integers to prevent underflow
            rotate_mask_up = rotate_mask_up.astype(np.int8)
                        
            # lower half of the mask
            mask_down = np.tril(new_mask)
            # convert to signed integers to prevent underflow
            mask_down = mask_down.astype(np.int8)

            # find the absolute difference between halves
            sym = np.abs(mask_down-rotate_mask_up)

            ratio=2*np.sum(sym)/area # multiplied by 2 to normalize in [0,1]

            ''' uncomment following to plot '''
            # import matplotlib.pyplot as plt
            # fig, axs = plt.subplots(2, 2)
            # axs[0, 0].imshow(mask_up)
            # axs[0, 0].set_title('mask_up')
            # axs[0, 1].imshow(rotate_mask_up)
            # axs[0, 1].set_title('rotate_mask_up')
            # axs[1, 0].imshow(mask_down)
            # axs[1, 0].set_title('mask_down')
            # axs[1, 1].imshow(sym)
            # axs[1, 1].set_title('sym')
            # plt.show()

            return ratio
        

        def split_upwards_diagonal():
            ''' checks symmetry along the upwards diagonal axis (/) '''
            # flip original mask to then apply same method as for downwards diagonal
            new_new_mask = np.flip(new_mask, axis=1)
            # upper half of the mask
            mask_up = np.triu(new_new_mask)
            # convert to signed integers to prevent underflow
            mask_up = mask_up.astype(np.int8)

            # flip across the diagonal by transposing
            rotate_mask_up=np.transpose(mask_up)
            # convert to signed integers to prevent underflow
            rotate_mask_up = rotate_mask_up.astype(np.int8)
                        
            # lower half of the mask
            mask_down = np.tril(new_new_mask)
            # convert to signed integers to prevent underflow
            mask_down = mask_down.astype(np.int8)

            # find the absolute difference between halves
            sym = np.abs(mask_down-rotate_mask_up)

            ratio=2*np.sum(sym)/area # multiplied by 2 to normalize in [0,1]

            ''' uncomment following to plot '''
            # import matplotlib.pyplot as plt
            # fig, axs = plt.subplots(2, 2)
            # axs[0, 0].imshow(mask_up)
            # axs[0, 0].set_title('mask_up')
            # axs[0, 1].imshow(rotate_mask_up)
            # axs[0, 1].set_title('rotate_mask_up')
            # axs[1, 0].imshow(mask_down)
            # axs[1, 0].set_title('mask_down')
            # axs[1, 1].imshow(sym)
            # axs[1, 1].set_title('sym')
            # plt.show()

            return ratio
        
        # return how symmetrical the lesion is
        # (higher is more asymmetrical, lower is more symmetric)
        ratio = mean([split_vertical(), split_horizontal(), split_downwards_diagonal(), split_upwards_diagonal()])

        ''' uncomment to time, also line 10 '''
        # end = timer()
        # print("elapsed time",end - start)

        return ratio

    def compactness(self, img, mask):
        brush = ndimage.generate_binary_structure(2, 1) # create 2 dim erosion brush
        eroded = ndimage.binary_erosion(mask, brush) # eroded mask
        p = np.sum(mask - eroded) # find perimeter

        ''' uncomment following to plot '''
        # import matplotlib.pyplot as plt
        # fig, axs = plt.subplots(2, 2)
        # axs[0, 0].imshow(mask)
        # axs[0, 0].set_title('mask')
        # axs[0, 1].imshow(brush)
        # axs[0, 1].set_title('brush')
        # axs[1, 0].imshow(eroded)
        # axs[1, 0].set_title('eroded')
        # axs[1, 1].imshow(mask-eroded)
        # axs[1, 1].set_title('mask-eroded')
        # plt.show()

        area = np.sum(mask) # lesion area

        c = (p**2)/(4*math.pi*area) # calculate compactness
        return c
    
    def color(self, img, mask):
        def manhatten(true_color, pixel_color):
            return np.sum(np.abs(true_color - pixel_color))
        
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
        #img_res = img.copy()
        for i in np.unique(slic):
            if np.sum(mask[slic == i]) == 0:
                continue
            #img_new[slic == i] = np.mean(img_new[slic == i], axis=0)
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
            #img_res[slic == i] = np.array(color_dict[color][0]) / 255   
        #fig, ax = plt.subplots(1, 2, figsize=(10, 5))
        #ax[0].imshow(img_new)
        #ax[1].imshow(img_res)  
           
        return [v[1] for v in color_dict.values()]

    def filters(self, img, mask):
        # Apply multiscale filters
        filtered = skimage.feature.multiscale_basic_features(img, channel_axis=2)
        
        n_features = filtered.shape[2]
        if not n_features == 72:
            raise ValueError(f"Expected 72 features, got {n_features}")
        # Extract each feature and get the leasion part only
        lesion_features = [filtered[:,:,i][mask==1] for i in range(n_features)]
        
        # Plot histograms for feature 3, 1 and 10 and the histograms for grayscale version of original image
        # from skimage.color import rgb2gray
        # fig, axs = plt.subplots(1, 4, figsize=(20, 5))
        # og_gray = rgb2gray(img)
        # og_gray[mask==0] = 0
        # og_gray = og_gray.flatten()
        # axs[0].hist(og_gray, bins=10)
        # axs[0].set_title("Original image")
        # axs[1].hist(lesion_features[3]*256, bins=10)
        # axs[1].set_title("Filter 3")
        # axs[2].hist(lesion_features[1]*256, bins=10)
        # axs[2].set_title("Filter 1")
        # axs[3].hist(lesion_features[10]*256, bins=10)
        # axs[3].set_title("Filter 10")
        # plt.show()

        #region Bin size creator
        # Create n bins of different width, with the same number of elements in each bin
        # n_bins = 10
        # quantile_bins = np.array([np.quantile(lesion_features[i]*256, np.arange(0,1,1/n_bins)) for i in range(n_features)])
        #
        # Verify that each bin has the same number of elements
        # print([np.sum((lesion_features[feat_num]*256 > quantile_bins[i]) & (lesion_features[feat_num]*256 < quantile_bins[i+1])) for i in range(len(quantile_bins)-1)])
        #endregion

        # Load bin widths from model/filter_bins_default.json
        bins = json.load(open("model/filter_bins_default.json", "r")).get("filter_bins")
        if len(bins) != len(lesion_features):
            raise ValueError(f"Expected {len(lesion_features)} bins, got {len(bins)}")
        # bins is a n*m array, where n is the number of features and m is the number of bins for each feature
        # each value is the upper bound of the bin. It is built from PAT_637_1434_684.

        # Turn each feature into a histogram
        hist_features = [np.histogram(lesion_features[i]*256, bins=bins[i])[0] for i in range(n_features)]
        
        # Each bin in each histogram is a feature, represented as the difference between the base image (PAT_637_1434_684) and this one
        merged = np.concatenate(hist_features)
        return merged
    
    def do_all(self, img, mask) -> np.ndarray:
        # do all of them
        asymmetry = np.array([self.asymmetry(img, mask)]) # 1 feature
        compactness = np.array([self.compactness(img, mask)]) # 1 feature
        color = np.array(self.color(img, mask)) # 6 features
        filters = self.filters(img, mask) # 72 features
        merged = np.concatenate([asymmetry, compactness, color, filters])
        return merged

    def extract_feat(self, feat: str, img, mask):
        """
        Runs the method with the name of `feat`
        """
        try:
            feat = getattr(self, feat)(img, mask)
        except AttributeError as e:
            print(f"Invalid feature name: {feat}, or error in the feature extractor:")
            print(e)
        return feat
