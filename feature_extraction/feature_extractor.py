from scipy import ndimage # updated morphology in newest scipy(?)
import numpy as np # needs full import for np.delete(?)
import math # for pi and sqrt
from statistics import mean # for mean
# from timeit import default_timer as timer

class FeatureExtractor:

    def asymmetry(self, img, mask):
        # start = timer()
        area = np.sum(mask) # lesion area
        com = ndimage.center_of_mass(mask) # find center of mass
        com = (int(com[0]), int(com[1])) # turn coords into ints
        
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
        
        # max distance from center of mass to edge of mask
        max_distance = max(dist_list)
        
        # slice the mask into a square of side length max_distance + 10 with com at the center
        r1 = com[0] - (max_distance + 10)  # lower bound for row 
        r2 = com[0] + (max_distance + 10)  # upper bound for row 
        c1 = com[1] - (max_distance + 10)  # lower bound for col
        c2 = com[1] + (max_distance + 10)  # upper bound for col
        
        # for loop to make sure the bounds are not outside the image
        if r1 < 0 or c1 < 0:
            r1 = 0
            c1 = 0

        shortest = min(mask.shape[0], mask.shape[1])
        if r2 > mask.shape[0] or c2 > mask.shape[1]:
            r2 = shortest
            c2 = shortest

        # make the square around the lesion
        new_mask = mask[r1:r2,c1:c2]
        
        # if the image is uneven in any axis
        if new_mask.shape[0] %2 != 0 or new_mask.shape[1] %2 != 0:
            # add a row and column of zeros to make the mask even
            new_mask= np.append(new_mask,np.zeros([new_mask.shape[0],1]),1)
            new_mask= np.append(new_mask,np.zeros([1,new_mask.shape[1]]),0)

        def split_vertical():
            # split mask into two halves along the vertical axis 
            mask_left, mask_right = np.split(new_mask,2,axis=1)
            # convert to signed integers to prevent underflow
            mask_left = mask_left.astype(np.int8)
            mask_right = mask_right.astype(np.int8)
            
            # invert the left half of the mask
            reflect_mask_left = np.flip(mask_left, axis=1)
            # convert to signed integers to prevent underflow
            reflect_mask_left = reflect_mask_left.astype(np.int8)

            # find the absolute difference between the right half and the inverted left half
            sym = np.abs(mask_right-reflect_mask_left)

            # calculate the ratio of the sum of the absolute difference to the area of the mask
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
            # split the mask into two halves along the horizontal axis 
            mask_up, mask_down = np.split(new_mask,2, axis=0)
            # convert to signed integers to prevent underflow
            mask_up = mask_up.astype(np.int8)
            mask_down = mask_down.astype(np.int8)

            # invert the left half of the mask
            reflect_mask_up = np.flip(mask_up, axis=0)
            # convert to signed integers to prevent underflow
            reflect_mask_up = reflect_mask_up.astype(np.int8)

            # find the absolute difference between the right half and the inverted left half
            sym = np.abs(mask_down-reflect_mask_up)

            # calculate the ratio of the sum of the absolute difference to the area of the mask
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
        
        #return how symmetrical the lesion is
        # (higher is more asymmetrical, lower is more symmetric)
        ratio = mean([split_vertical(), split_horizontal(), split_downwards_diagonal(), split_upwards_diagonal()])
        # end = timer()
        # print("elapsed time",end - start)
        return ratio

    def compactness(self, img, mask):
        brush = ndimage.generate_binary_structure(2, 1) # create 2 dim erosion brush
        eroded = ndimage.binary_erosion(mask, brush) # eroded mask
        p = np.sum(mask - eroded) # find perimeter

        ''' use following to plot '''
        # import matplotlib.pyplot as plt
        ''' can use mask-eroded / mask / brush / eroded'''
        # plt.imshow(mask-eroded)
        # plt.show()

        area = np.sum(mask) # lesion area

        c = (p**2)/(4*math.pi*area) # calculate compactness
        return c
    
    def color(self, img, mask):
        print("owo")
        return 0

    def filters(self, img, mask):
        print("OwO")
        return 0
    
    def figure_something_out_you_are_original(self, img, mask):
        print(".w.")
        return 0
    
    def do_all(self, img, mask):
        # do all of them
        asymmetry = self.asymmetry(img, mask)
        compactness = self.compactness(img, mask)
        return (asymmetry, compactness)
    
    def extract_feat(self, feat: str, img, mask):
        """
        Runs the method with the name of `feat`
        """
        try:
            feat = getattr(self, feat)(img, mask)
        except AttributeError:
            print(f"Invalid feature name: {feat}")
        return feat
