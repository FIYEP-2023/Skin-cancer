# Task 0
Go through the data (images and meta-data) that you have available to understand what’s available to you and write a brief summary of your findings. For example:
-  What types of diagnoses are there, and how do they relate to each
other? You can use Google Scholar or other sources, to learn more
background about these diagnoses.
- Is there some missing data? Are there images of low quality? Etc.
- You can select a small (e.g. 100) subset images to focus on
at the start. The only requirement is that there are at least two
categories of images.

## Answer
- The images contain either skin cancer, or skin diseases. The cancer ones are BCC, SCC, MEL and BOD (merged within SCC). The rest, ACK, NEV, and SEK are skin diseases.   
- Some datapoints are missing, such as background_father and smoke.  
- Images are of varying quality and size, though all are square, though some are one pixel off e.g. 1189x1190.  
The sizes vary wildly, from 147x147 to 3474x3476. The focus of the images also varies.  
In some images, the lesion is obstructed, by things such as body hair.  
Some images do not have the lesions in center frame, and others contain multiple lesions.  
  
# Task 1A: segment images  
Create segmentations for some images. You can do this with image processing methods, or yourself with LabelStudio.  
  
# Task 1B: measure the features yourself  
Search for related work about the Asymmetry and Color features and how they are measured by dermatologists.  
Create an “annotation guide” for you and your group members, where you discuss at least 5 images together, and decide how to rate their Asymmetry and Color.  
Then split the images, such that each image is annotated by at least two people in your group. Save your annotations in a CSV file, such that there are as many columns as there are different annotators (+ one column for the image name), i.e. do not put annotations of diffferent people into the same column.  
Make sure your CSV file follows the guidelines outlined in [?].