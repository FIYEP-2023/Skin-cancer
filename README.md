# Bachelor of Science in Data Science, ITU Copenhagen 2022/23 - First Year Project
## Second project: analysis of skin lesions  
  
# Running the code  
Install requirements with `pip install -r requirements.txt`. The code works with python 3.9.  
To generate figures, run `python main.py --figures`.  
To run the finished model on some data, put the images in "EVAL_IMGS" in the following format:
```bash
some_image_name.png
some_image_name_mask.png
```
Then run `python main.py --predict`  
Optionally include metadata: a .csv file with columns "img_id" and "diagnostic", a .npy two-dimensional numpy array with columns `["img_id.png", "diagnostic"]` or a .pkl numpy array with the same properties.  
Put this metadata files into "EVAL_IMGS".  
Predictions will be saved into a .npy and .csv file in "EVAL_RESULTS", as well as printed to the console and saved in "log.txt".  
If metadata is included, stats such as recall, f1-score and confusion matrix will also be printed to the console.

# Contributors  
[Alexander Thoren](https://github.com/TheColorman), [Josefine Nyeng](https://github.com/josefinenyeng), [Miranda Speyer-Larsen](https://github.com/mluonium) and [Pedro Prazeres](https://github.com/Pheadar).   
