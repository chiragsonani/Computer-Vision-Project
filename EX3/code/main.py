'''
@author: Prathmesh R Madhu.
For educational purposes only
'''

# -*- coding: utf-8 -*-
from __future__ import (
    division,
    print_function,
)

import os
import time
import sys
import skimage.data
import skimage.io
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from selective_search import selective_search

class DualOutput:
    def __init__(self, file_path: str) -> None:
        self.terminal = sys.stdout
        self.log = open(file_path, 'w')
    
    def write(self, message: str) -> None:
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()
    
    def flush(self) -> None:
        self.terminal.flush()
        self.log.flush()
    
    def __del__(self) -> None:
        self.log.close()

def main():
    
    # Set up dual output to both terminal and file
    sys.stdout = DualOutput('../results/output.txt')
    
    # Define all test images from the three folders
    test_images = [
        '../data/arthist/adoration1.jpg',
        '../data/arthist/annunciation1.jpg',
        '../data/arthist/baptism1.jpg',
        '../data/chrisarch/ca-annun1.jpg',
        '../data/chrisarch/ca-annun2.jpg',
        '../data/chrisarch/ca-annun3.jpg',
        '../data/classarch/ajax3.jpg',
        '../data/classarch/leading1.jpg',
        '../data/classarch/pursuit2.jpg'
    ]
    
    # Process each image
    total_start_time = time.time()
    print(f"[INFO] Using {os.cpu_count()} CPU cores for parallel processing")
    
    for idx, image_path in enumerate(test_images, 1):
        print(f"\n{'='*80}")
        print(f"[INFO] Processing image {idx}/{len(test_images)}: {os.path.basename(image_path)}")
        print(f"[INFO] Full path: {image_path}")
        
        # loading a test image from '../data' folder
        image = skimage.io.imread(image_path)
        print(f"[INFO] Image shape: {image.shape}")
        
        # perform selective search
        image_start_time = time.time()
        image_label, regions = selective_search(
                                image,
                                scale=500,
                                min_size=20
                            )
        
        candidates = set()
        for r in regions:
            # excluding same rectangle (with different segments)
            if r['rect'] in candidates:
                continue
            
            # excluding regions smaller than 2000 pixels
            # you can experiment using different values for the same
            if r['size'] < 2000:
                continue
            
            # excluding distorted rects
            x, y, w, h = r['rect']
            if w/h > 1.2 or h/w > 1.2:
                continue
            
            candidates.add(r['rect'])
        
        image_duration = time.time() - image_start_time
        print(f"\n[SUCCESS] Found {len(candidates)} filtered region proposals in {image_duration:.1f}s")
        print(f"[SUCCESS] Total regions before filtering: {len(regions)}")
        
        # Draw rectangles on the original image
        fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(8, 8))
        ax.imshow(image)
        for x, y, w, h in candidates:
            rect = mpatches.Rectangle(
                (x, y), w, h, fill=False, edgecolor='red', linewidth=1
            )
            ax.add_patch(rect)
        plt.axis('off')
        
        # saving the image
        if not os.path.isdir('../results/'):
            os.makedirs('../results/')
        
        # Create output filename
        folder_name = image_path.split('/')[-2]
        file_name = image_path.split('/')[-1].replace('.jpg', '_proposals.jpg')
        output_path = f'../results/{folder_name}_{file_name}'
        
        fig.savefig(output_path, bbox_inches='tight', dpi=150)
        plt.close(fig)
        print(f"[SUCCESS] Saved result to: {output_path}")
    
    total_duration = time.time() - total_start_time
    print(f"\n{'='*80}")
    print(f"[COMPLETE] Processed all {len(test_images)} images in {total_duration:.1f}s")
    print(f"[COMPLETE] Average time per image: {total_duration/len(test_images):.1f}s")


if __name__ == '__main__':
    main()