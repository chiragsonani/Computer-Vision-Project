'''
@author: Prathmesh R Madhu.
For educational purposes only
'''
# -*- coding: utf-8 -*-
from __future__ import division

import skimage.io
import skimage.feature
import skimage.color
import skimage.transform
import skimage.util
import skimage.segmentation
import numpy as np
import time
import gc
from selective_search_implementation import SelectiveSearchProcessor

processor = SelectiveSearchProcessor()

def generate_segments(im_orig, scale, sigma, min_size):
    """
    Task 1: Segment smallest regions by the algorithm of Felzenswalb.
    1.1. Generate the initial image mask using felzenszwalb algorithm
    1.2. Merge the image mask to the image as a 4th channel
    """
    ### YOUR CODE HERE ###
    return processor.generate_segments(im_orig, scale, sigma, min_size)

def sim_colour(r1, r2):
    """
    2.1. calculate the sum of histogram intersection of colour
    """
    ### YOUR CODE HERE ###
    return processor.sim_colour(r1, r2)


def sim_texture(r1, r2):
    """
    2.2. calculate the sum of histogram intersection of texture
    """
    ### YOUR CODE HERE ###
    return processor.sim_texture(r1, r2)


def sim_size(r1, r2, imsize):
    """
    2.3. calculate the size similarity over the image
    """
    ### YOUR CODE HERE ###
    return processor.sim_size(r1, r2, imsize)


def sim_fill(r1, r2, imsize):
    """
    2.4. calculate the fill similarity over the image
    """
    ### YOUR CODE HERE ###
    return processor.sim_fill(r1, r2, imsize)

def calc_sim(r1, r2, imsize):
    return (sim_colour(r1, r2) + sim_texture(r1, r2)
            + sim_size(r1, r2, imsize) + sim_fill(r1, r2, imsize))

def calc_colour_hist(img):
    """
    Task 2.5.1
    calculate colour histogram for each region
    the size of output histogram will be BINS * COLOUR_CHANNELS(3)
    number of bins is 25 as same as [uijlings_ijcv2013_draft.pdf]
    extract HSV
    """
    BINS = 25
    hist = np.array([])
    ### YOUR CODE HERE ###
    hist = processor.calc_colour_hist(img)

    return hist

def calc_texture_gradient(img):
    """
    Task 2.5.2
    calculate texture gradient for entire image
    The original SelectiveSearch algorithm proposed Gaussian derivative
    for 8 orientations, but we will use LBP instead.
    output will be [height(*)][width(*)]
    Useful function: Refer to skimage.feature.local_binary_pattern documentation
    """
    ret = np.zeros((img.shape[0], img.shape[1], img.shape[2]))
    ### YOUR CODE HERE ###
    ret = processor.calc_texture_gradient(img)
    return ret

def calc_texture_hist(img):
    """
    Task 2.5.3
    calculate texture histogram for each region
    calculate the histogram of gradient for each colours
    the size of output histogram will be
        BINS * ORIENTATIONS * COLOUR_CHANNELS(3)
    Do not forget to L1 Normalize the histogram
    """
    BINS = 10
    hist = np.array([])
    ### YOUR CODE HERE ###
    hist = processor.calc_texture_hist(img)

    return hist

def extract_regions(img):
    '''
    Task 2.5: Generate regions denoted as datastructure R
    - Convert image to hsv color map
    - Count pixel positions
    - Calculate the texture gradient
    - calculate color and texture histograms
    - Store all the necessary values in R.
    '''
    R = {}
    ### YOUR CODE HERE ###
    R = processor.extract_regions(img)
    return R

def extract_neighbours(regions):

    def intersect(a, b):
        if (a["min_x"] < b["min_x"] < a["max_x"]
                and a["min_y"] < b["min_y"] < a["max_y"]) or (
            a["min_x"] < b["max_x"] < a["max_x"]
                and a["min_y"] < b["max_y"] < a["max_y"]) or (
            a["min_x"] < b["min_x"] < a["max_x"]
                and a["min_y"] < b["max_y"] < a["max_y"]) or (
            a["min_x"] < b["max_x"] < a["max_x"]
                and a["min_y"] < b["min_y"] < a["max_y"]):
            return True
        return False

    # Hint 1: List of neighbouring regions
    # Hint 2: The function intersect has been written for you and is required to check neighbours
    neighbours = []
    ### YOUR CODE HERE ###
    neighbours = processor.extract_neighbours(regions)

    return neighbours

def merge_regions(r1, r2):
    new_size = r1["size"] + r2["size"]
    rt = {}
    ### YOUR CODE HERE
    rt = processor.merge_regions(r1, r2)
    return rt


def selective_search(image_orig, scale=1.0, sigma=0.8, min_size=50):
    '''
    Selective Search for Object Recognition" by J.R.R. Uijlings et al.
    :arg:
        image_orig: np.ndarray, Input image
        scale: int, determines the cluster size in felzenszwalb segmentation
        sigma: float, width of Gaussian kernel for felzenszwalb segmentation
        min_size: int, minimum component size for felzenszwalb segmentation

    :return:
        image: np.ndarray,
            image with region label
            region label is stored in the 4th value of each pixel [r,g,b,(region)]
        regions: array of dict
            [
                {
                    'rect': (left, top, width, height),
                    'labels': [...],
                    'size': component_size
                },
                ...
            ]
    '''

    # Checking the 3 channel of input image
    assert image_orig.shape[2] == 3, "Please use image with three channels."
    imsize = image_orig.shape[0] * image_orig.shape[1]

    # Task 1: Load image and get smallest regions. Refer to `generate_segments` function.
    image = generate_segments(image_orig, scale, sigma, min_size)

    if image is None:
        return None, {}

    # Task 2: Extracting regions from image
    # Task 2.1-2.4: Refer to functions "sim_colour", "sim_texture", "sim_size", "sim_fill"
    # Task 2.5: Refer to function "extract_regions". You would also need to fill "calc_colour_hist",
    # "calc_texture_hist" and "calc_texture_gradient" in order to finish task 2.5.
    R = extract_regions(image)

    # Task 3: Extracting neighbouring information
    # Refer to function "extract_neighbours"
    neighbours = extract_neighbours(R)

    # Calculating initial similarities
    S = processor.compute_initial_similarities(neighbours, imsize)

    # Hierarchical search for merging similar regions
    print(f"[Stage 5/6] Hierarchical merging...")
    merge_start_time = time.time()
    initial_regions = len(R)
    iteration = 0
    
    while S != {}:

        # Get highest similarity - USE MAX INSTEAD OF SORTING!
        max_sim_item = max(S.items(), key=lambda x: x[1])
        i, j = max_sim_item[0]
        max_similarity = max_sim_item[1]
        
        # Stop if similarity is too low (early termination)
        if max_similarity < 0.1:
            print(f"[Stage 5/6] Early termination: max similarity {max_similarity:.3f} < 0.1")
            break

        # Task 4: Merge corresponding regions. Refer to function "merge_regions"
        t = max(R.keys()) + 1.0
        R[t] = merge_regions(R[i], R[j])

        # Task 5: Mark similarities for regions to be removed
        ### YOUR CODE HERE ###
        key_to_remove = []
        for key in S.keys():
            if i in key or j in key:
                key_to_remove.append(key)

        # Task 6: Remove old similarities of related regions
        ### YOUR CODE HERE ###
        for key in key_to_remove:
            del S[key]

        # Task 7: Calculate similarities with the new region
        ### YOUR CODE HERE ###
        # Only calculate similarities with neighbors of i and j
        neighbors_to_check = set()
        for key in key_to_remove:
            if i in key:
                other = key[1] if key[0] == i else key[0]
                if other != j:
                    neighbors_to_check.add(other)
            elif j in key:
                other = key[1] if key[0] == j else key[0]
                if other != i:
                    neighbors_to_check.add(other)
        
        for k in neighbors_to_check:
            if k in R:  # Make sure region still exists
                if (k < t):
                    key = (k, t)
                else:
                    key = (t, k)
                S[key] = calc_sim(R[k], R[t], imsize)
        
        iteration += 1
        if iteration % 100 == 0:
            print(f"[Stage 5/6] Merge iteration {iteration}: {len(R)} regions, {len(S)} similarities")
            gc.collect()

    # Complete merge tracking
    merge_duration = time.time() - merge_start_time
    print(f"[Stage 5/6] ✓ Hierarchical merging complete: {iteration} iterations in {merge_duration:.1f}s")

    # Task 8: Generating the final regions from R
    print(f"[Stage 6/6] Generating final proposals...")
    regions = []
    ### YOUR CODE HERE ###
    for k, r in R.items():
        regions.append({
            'rect': (r['min_x'], r['min_y'], r['max_x'] - r['min_x'], r['max_y'] - r['min_y']),
            'size': r['size'],
            'labels': r['labels']
        })
    
    print(f"[Stage 6/6] ✓ Generated {len(regions)} region proposals")
    return image, regions


