class FeatureExtractor:
    def asymmetry(self, img, mask):
        print("uwu")
        return 0 # return the value of the feature

    def compactness(self, img, mask):
        ''' move these imports upwards if used in other functions too (?) '''
        from scipy import ndimage # updated morphology in newest scipy(?)
        import numpy as np # needs full import for np.delete(?)

        if img.shape[2] == 4: # if img has 4th dim (alpha)
            img = np.delete(img, 3, 2) # remove alpha

        brush = ndimage.generate_binary_structure(3, 1) # create 3 dim erosion brush
        eroded = ndimage.binary_erosion(mask, brush) # eroded mask
        p = np.sum(mask - eroded) # find perimeter

        area = np.sum(mask) # lesion area

        c = (p**2)/(4*area)
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
