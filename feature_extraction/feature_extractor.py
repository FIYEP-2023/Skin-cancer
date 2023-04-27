class FeatureExtractor:
    def asymmetry(self, img, mask):
        print("uwu")
        return 0 # return the value of the feature

    def compactness(self, img, mask):
        print(">w<")
        return 0 # return the value of the feature
    
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
