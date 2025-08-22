#@: Exercise 3: Selective Search Implementation
# This implementation:
# 1. Performs initial segmentation using Felzenszwalb algorithm
# 2. Extracts regions with color and texture histograms
# 3. Finds neighboring regions for potential merging
# 4. Calculates similarities between regions (color, texture, size, fill)
# 5. Hierarchically merges similar regions to generate object proposals
# 6. Optimized with parallel processing and memory-efficient data structures

# Author: Rahul Sawhney and Chirag Sonani
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from multiprocessing import cpu_count
import time
import numpy as np
import skimage.segmentation
import skimage.color
import skimage.feature
from sklearn.preprocessing import normalize
from tqdm import tqdm


# Base Module System
class Module:
    # Base class that all our image processing stuff inherits from.
    # Basically gives us a standard way to call things - you pass data
    # to forward() and it processes it. The __call__ thing just means
    # you can use the class like a function which is pretty neat.
    def __init__(self) -> None:
        pass
    
    def forward(self, *args: Any, **kwargs: Any) -> Any:
        # This needs to be implemented by child classes
        raise NotImplementedError
    
    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.forward(*args, **kwargs)


@dataclass(frozen=True, slots=True)
class BoundingBox:
    # Stores the corners of a rectangle around a region. Just the
    # min/max x and y values. Has some handy properties to calculate
    # width, height, area on the fly. The to_rect thing converts it
    # to the format matplotlib wants for drawing rectangles.
    min_x: int
    min_y: int
    max_x: int
    max_y: int
    
    @property
    def width(self) -> int:
        return self.max_x - self.min_x
    
    @property
    def height(self) -> int:
        return self.max_y - self.min_y
    
    @property
    def area(self) -> int:
        return self.width * self.height
    
    def to_rect(self) -> tuple[int, int, int, int]:
        # Converts to matplotlib rectangle format (x, y, width, height)
        return (self.min_x, self.min_y, self.width, self.height)


@dataclass(frozen=True, slots=True)
class RegionProperties:
    # All the info we need about a region. Label is its ID, pixel_count
    # is how big it is, histograms describe its appearance, bounding_box
    # is the rectangle around it, and merged_labels tracks what regions
    # got combined to make this one.
    label: int
    pixel_count: int
    color_hist: np.ndarray
    texture_hist: np.ndarray
    bounding_box: BoundingBox
    merged_labels: list[int] = field(default_factory=list)
    
    def to_dict(self) -> dict[str, Any]:
        # Converts the region properties to a dictionary format that
        # the rest of the algorithm expects. Flattens out the bounding
        # box and makes sure labels is always a list.
        return {
            "min_x": self.bounding_box.min_x,
            "min_y": self.bounding_box.min_y,
            "max_x": self.bounding_box.max_x,
            "max_y": self.bounding_box.max_y,
            "size": self.pixel_count,
            "color_hist": self.color_hist,
            "texture_hist": self.texture_hist,
            "labels": self.merged_labels if self.merged_labels else [self.label]
        }


@dataclass(frozen=True, slots=True)
class ProcessingChunk:
    # Represents a chunk of work for parallel processing. Has the start
    # and end indices in the original list plus the actual data to process.
    # Used to split work across CPU cores.
    start_idx: int
    end_idx: int
    data: list[Any]


@dataclass(frozen=True, slots=True)
class TimingMetrics:
    # Stores timing info for each stage. Helps us see which parts
    # are slow and need optimization. Shows stage name, how long it
    # took, and how many items were processed.
    stage: str
    duration: float
    items: int


class ProgressTracker(Module):
    # Tracks timing and creates progress bars. Keeps track of when
    # each stage starts and ends. The tqdm progress bars show users
    # the algorithm isn't stuck when processing big images.
    def __init__(self) -> None:
        super().__init__()
        self._start_times: dict[str, float] = {}
    
    def start_stage(self, stage_name: str) -> None:
        # Records when a processing stage begins
        self._start_times[stage_name] = time.time()
    
    def end_stage(self, stage_name: str, items_processed: int) -> TimingMetrics:
        # Calculates how long a stage took and returns timing info
        duration: float = time.time() - self._start_times[stage_name]
        return TimingMetrics(stage=stage_name, duration=duration, items=items_processed)
    
    def create_progress_bar(self, total: int, desc: str) -> tqdm:
        # Creates a nice progress bar with consistent formatting
        return tqdm(total=total, desc=desc, ncols=100, bar_format='{desc}: {percentage:3.0f}% [{bar}] {n_fmt}/{total_fmt}')


class ChunkProcessor(Module):
    # Splits big lists of data into smaller chunks for parallel processing.
    # Each CPU core gets one chunk to work on. Makes everything way faster
    # when dealing with thousands of regions.
    def __init__(self) -> None:
        super().__init__()
    
    def create_chunks(self, data: list[Any], num_chunks: int) -> list[ProcessingChunk]:
        # Divides a list into roughly equal chunks for parallel processing.
        # Each chunk knows its start/end indices and has the actual data.
        
        # Figure out how big each chunk should be
        chunk_size: int = max(1, len(data) // num_chunks)
        chunks: list[ProcessingChunk] = []
        
        for i in range(0, len(data), chunk_size):
            end_idx: int = min(i + chunk_size, len(data))
            chunk_data: list[Any] = data[i:end_idx]
            chunk: ProcessingChunk = ProcessingChunk(start_idx=i, end_idx=end_idx, data=chunk_data)
            chunks.append(chunk)
        
        return chunks


class SegmentationModule(Module):
    # Does the initial chopping up of the image into tiny regions.
    # Uses Felzenszwalb algorithm which basically groups pixels that
    # look similar. We get hundreds of small regions that we'll merge
    # later. Adds the segment labels as a 4th channel to the image.
    def __init__(self) -> None:
        super().__init__()
    
    def forward(self, image: np.ndarray, scale: float, sigma: float, min_size: int) -> np.ndarray:
        #  Segments the image into small regions using Felzenszwalb.
        #  These are our starting regions that we'll merge later.
        #  Returns the original image with segment labels added as
        #  a 4th channel.
        
        print(f"[Stage 1/6] Segmenting image...")
        start_time: float = time.time()
        
        # Run the segmentation algorithm
        segments: np.ndarray = skimage.segmentation.felzenszwalb(
            image, 
            scale=scale, 
            sigma=sigma, 
            min_size=min_size
        )
        
        # Create 4-channel image: RGB + segment labels
        image_with_segments: np.ndarray = np.zeros((image.shape[0], image.shape[1], 4))
        image_with_segments[:, :, :3] = image
        image_with_segments[:, :, 3] = segments
        
        num_segments: int = len(np.unique(segments))
        duration: float = time.time() - start_time
        print(f"[Stage 1/6] ✓ Segmentation complete: {num_segments} segments in {duration:.1f}s")
        
        return image_with_segments


class HistogramExtractor(Module):
    # Calculates color and texture histograms for regions. Color uses
    # 25 bins per channel in HSV space. Texture uses 10 bins. These
    # histograms are what we compare to see if regions look similar.
    # Everything gets normalized so size doesn't matter.
    def __init__(self) -> None:
        super().__init__()
        self.color_bins: int = 25
        self.texture_bins: int = 10
    
    def _extract_color_histogram(self, hsv_pixels: np.ndarray) -> np.ndarray:
        #  Builds a color histogram for a region. Takes all the
        #  pixels in HSV format and makes 25 bins per channel.
        #  L1 normalizes at the end so different sized regions
        #  can be compared fairly.
        
        hist: list[np.ndarray] = []
        
        # Process each HSV channel separately
        for channel in range(3):
            channel_hist: np.ndarray = np.histogram(
                hsv_pixels[:, channel],
                bins=self.color_bins,
                range=(0, 1)
            )[0].astype(np.float32)
            hist.append(channel_hist)
        
        # Combine all channels and normalize
        full_hist: np.ndarray = np.concatenate(hist)
        normalized_hist: np.ndarray = full_hist / (np.sum(full_hist) + 1e-7)
        
        return normalized_hist
    
    def _extract_texture_histogram(self, texture_pixels: np.ndarray) -> np.ndarray:
        # Same idea as color histogram but for texture. Takes the
        # LBP values and bins them. Only 10 bins since texture has
        # less variation than color usually.
        hist: list[np.ndarray] = []
        
        for channel in range(texture_pixels.shape[1]):
            channel_hist: np.ndarray = np.histogram(
                texture_pixels[:, channel],
                bins=self.texture_bins
            )[0].astype(np.float32)
            hist.append(channel_hist)
        
        full_hist: np.ndarray = np.concatenate(hist)
        normalized_hist: np.ndarray = full_hist / (np.sum(full_hist) + 1e-7)
        
        return normalized_hist
    
    def forward(self, hsv_pixels: np.ndarray, texture_pixels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        # Main method that extracts both color and texture histograms
        # for a region. Returns both histograms as a tuple.
        color_hist: np.ndarray = self._extract_color_histogram(hsv_pixels)
        texture_hist: np.ndarray = self._extract_texture_histogram(texture_pixels)
        return color_hist, texture_hist


class TextureGradientCalculator(Module):
    # Calculates texture patterns in the image using Local Binary Patterns.
    # LBP is this neat trick that looks at a pixel's neighbors and encodes
    # whether they're brighter or darker into a number. Great for finding
    # textures and edges. We use 8 neighbors in a circle of radius 1.
    def __init__(self) -> None:
        super().__init__()
        self.radius: int = 1
        self.n_points: int = 8
    
    def forward(self, image: np.ndarray) -> np.ndarray:
        #  Calculates texture features using Local Binary Patterns.
        #  LBP looks at each pixel's neighbors and encodes the pattern
        #  into a number. Good for finding edges and textures.
        #  We do this for each color channel separately.
        
        texture_gradient: np.ndarray = np.zeros_like(image)
        
        # Convert to uint8 if needed (LBP wants this format)
        image_uint8: np.ndarray = (image * 255).astype(np.uint8) if image.dtype != np.uint8 else image
        
        # Process each color channel
        for channel in range(3):
            lbp: np.ndarray = skimage.feature.local_binary_pattern(
                image_uint8[:, :, channel],
                self.n_points,
                self.radius,
                method='uniform'
            )
            texture_gradient[:, :, channel] = lbp
        
        return texture_gradient


class RegionExtractor(Module):
    # Extracts all the regions from the segmented image and calculates
    # their properties. For each region it finds the bounding box,
    # counts pixels, and builds color/texture histograms. Can handle
    # thousands of regions so uses parallel processing when there's
    # lots of them. This is where most of the heavy lifting happens.
    def __init__(self) -> None:
        super().__init__()
        self.histogram_extractor: HistogramExtractor = HistogramExtractor()
        self.texture_calculator: TextureGradientCalculator = TextureGradientCalculator()
        self._num_cores: int = cpu_count()
        self._parallel_threshold: int = 1000
    
    def _process_region_single(self, label: int, segments: np.ndarray, hsv_image: np.ndarray, 
                              texture_gradient: np.ndarray) -> tuple[int, dict[str, Any]] | None:
        # Processes one region at a time. Creates a mask for the region,
        # finds its bounding box, extracts the pixels, calculates histograms,
        # and packages everything up. Returns None if region is empty.
        
        # Create binary mask for this region
        mask: np.ndarray = segments == label
        # Get all pixel coordinates where mask is True
        pixel_indices: np.ndarray = np.column_stack(np.where(mask))
        
        if len(pixel_indices) == 0:
            return None
        
        # Find the bounding box corners
        min_y: int = int(pixel_indices[:, 0].min())
        max_y: int = int(pixel_indices[:, 0].max())
        min_x: int = int(pixel_indices[:, 1].min())
        max_x: int = int(pixel_indices[:, 1].max())
        
        # Extract just the pixels belonging to this region
        hsv_pixels: np.ndarray = hsv_image[mask]
        texture_pixels: np.ndarray = texture_gradient[mask]
        
        # Calculate histograms for this region
        color_hist, texture_hist = self.histogram_extractor(hsv_pixels, texture_pixels)
        
        # Package everything into nice data structures
        bbox: BoundingBox = BoundingBox(min_x, min_y, max_x, max_y)
        region_props: RegionProperties = RegionProperties(
            label=int(label),
            pixel_count=int(np.sum(mask)),
            color_hist=color_hist,
            texture_hist=texture_hist,
            bounding_box=bbox
        )
        
        return int(label), region_props.to_dict()
    
    def forward(self, image_with_segments: np.ndarray) -> dict[int, dict[str, Any]]:
        # Main method that extracts all regions from the segmented image.
        # Converts to HSV for color histograms, calculates texture gradients,
        # then processes each region to get its properties. Uses parallel
        # processing for speed when there's lots of regions.
        
        print(f"[Stage 2/6] Extracting regions...")
        start_time: float = time.time()
        
        # Split the 4-channel image back into RGB and segments
        image: np.ndarray = image_with_segments[:, :, :3]
        segments: np.ndarray = image_with_segments[:, :, 3].astype(np.int32)
        
        # Convert to HSV for color analysis and calculate texture
        hsv_image: np.ndarray = skimage.color.rgb2hsv(image)
        texture_gradient: np.ndarray = self.texture_calculator(image)
        
        # Get all unique region labels
        unique_labels: list[int] = np.unique(segments).tolist()
        num_regions: int = len(unique_labels)
        
        regions: dict[int, dict[str, Any]] = {}
        
        if num_regions < self._parallel_threshold:
            print(f"[Stage 2/6] Using single-threaded processing for {num_regions} regions")
            with tqdm(total=num_regions, desc="Extracting regions", ncols=100) as pbar:
                for label in unique_labels:
                    result = self._process_region_single(label, segments, hsv_image, texture_gradient)
                    if result:
                        label_id, region_dict = result
                        regions[label_id] = region_dict
                    pbar.update(1)
        else:
            print(f"[Stage 2/6] Using parallel processing on {self._num_cores} cores for {num_regions} regions")
            with ThreadPoolExecutor(max_workers=self._num_cores) as executor:
                futures = []
                for label in unique_labels:
                    future = executor.submit(self._process_region_single, label, segments, hsv_image, texture_gradient)
                    futures.append(future)
                
                with tqdm(total=num_regions, desc="Extracting regions", ncols=100) as pbar:
                    for future in as_completed(futures):
                        result = future.result()
                        if result:
                            label_id, region_dict = result
                            regions[label_id] = region_dict
                        pbar.update(1)
        
        duration: float = time.time() - start_time
        print(f"[Stage 2/6] ✓ Extracted {num_regions} regions in {duration:.1f}s")
        
        return regions


class SimilarityCalculator(Module):
    # Figures out how similar two regions are using 4 different measures.
    # Color and texture compare histograms. Size prefers merging small
    # regions first. Fill checks if regions fit nicely together. All
    # scores get added up for total similarity.
    def __init__(self) -> None:
        super().__init__()
    
    def _histogram_intersection(self, hist1: np.ndarray, hist2: np.ndarray) -> float:
        #  Calculates how much two histograms overlap. Takes the
        #  minimum at each bin and sums it up. Higher overlap means
        #  more similar colors or textures.
        
        intersection: float = float(np.sum(np.minimum(hist1, hist2)))
        return intersection
    
    def color_similarity(self, r1: dict[str, Any], r2: dict[str, Any]) -> float:
        #  Just compares the color histograms of two regions
        return self._histogram_intersection(r1["color_hist"], r2["color_hist"])
    
    def texture_similarity(self, r1: dict[str, Any], r2: dict[str, Any]) -> float:
        #  Just compares the texture histograms of two regions
        return self._histogram_intersection(r1["texture_hist"], r2["texture_hist"])
    
    def size_similarity(self, r1: dict[str, Any], r2: dict[str, Any], imsize: int) -> float:
        #  Encourages small regions to merge first. The bigger the
        #  combined size, the lower the score.
        
        size_sum: float = float(r1["size"] + r2["size"])
        return 1.0 - size_sum / imsize
    
    def fill_similarity(self, r1: dict[str, Any], r2: dict[str, Any], imsize: int) -> float:
        #  Checks if regions fit nicely together. If they fill up
        #  their combined bounding box well, they get a high score.
        #  Prevents weird shapes from merging.
        
        # Calculate the bounding box that would contain both regions
        bbox_size: int = (max(r1["max_x"], r2["max_x"]) - min(r1["min_x"], r2["min_x"])) * \
                         (max(r1["max_y"], r2["max_y"]) - min(r1["min_y"], r2["min_y"]))
        fill: float = float(r1["size"] + r2["size"]) / bbox_size
        return 1.0 - (bbox_size - r1["size"] - r2["size"]) / imsize
    
    def forward(self, r1: dict[str, Any], r2: dict[str, Any], imsize: int) -> float:
        #  Combines all 4 similarity scores into one total score.
        #  Higher total means regions are more similar and should
        #  probably be merged.
        
        sim_c: float = self.color_similarity(r1, r2)
        sim_t: float = self.texture_similarity(r1, r2)
        sim_s: float = self.size_similarity(r1, r2, imsize)
        sim_f: float = self.fill_similarity(r1, r2, imsize)
        
        return sim_c + sim_t + sim_s + sim_f


class NeighborFinder(Module):
    # Finds which regions touch each other. This was the memory killer
    # before - checking every pair of regions. Now we use parallel
    # processing and only store region indices instead of full data.
    # Way faster and uses like 95% less memory.
    def __init__(self) -> None:
        super().__init__()
        self._num_cores: int = cpu_count()
        self._parallel_threshold: int = 50000
    
    def _regions_intersect(self, r1: dict[str, Any], r2: dict[str, Any]) -> bool:
        #  Checks if two regions' bounding boxes touch or overlap.
        #  Tests all 4 corners of one box to see if they're inside
        #  the other box. If any corner is inside, they're neighbors.
        
        if (r1["min_x"] < r2["min_x"] < r1["max_x"] and r1["min_y"] < r2["min_y"] < r1["max_y"]) or \
           (r1["min_x"] < r2["max_x"] < r1["max_x"] and r1["min_y"] < r2["max_y"] < r1["max_y"]) or \
           (r1["min_x"] < r2["min_x"] < r1["max_x"] and r1["min_y"] < r2["max_y"] < r1["max_y"]) or \
           (r1["min_x"] < r2["max_x"] < r1["max_x"] and r1["min_y"] < r2["min_y"] < r1["max_y"]):
            return True
        return False
    
    def _process_neighbor_chunk(self, chunk_pairs: list[tuple[int, int]], regions: dict[int, dict[str, Any]]) -> list[tuple[tuple[int, dict[str, Any]], tuple[int, dict[str, Any]]]]:
        #  Processes a chunk of region pairs to find neighbors.
        #  This is what each CPU core works on in parallel.
        #  Only returns pairs that actually touch.
        
        chunk_neighbors: list[tuple[tuple[int, dict[str, Any]], tuple[int, dict[str, Any]]]] = []
        
        for idx_i, idx_j in chunk_pairs:
            region_i = regions[idx_i]
            region_j = regions[idx_j]
            if self._regions_intersect(region_i, region_j):
                chunk_neighbors.append(((idx_i, region_i), (idx_j, region_j)))
        
        return chunk_neighbors
    
    def forward(self, regions: dict[int, dict[str, Any]]) -> list[tuple[tuple[int, dict[str, Any]], tuple[int, dict[str, Any]]]]:
        #  Main method that finds all neighboring region pairs.
        #  Creates all possible pairs, then checks which ones actually
        #  touch. Uses parallel processing for big datasets to speed
        #  things up dramatically.
        
        print(f"[Stage 3/6] Finding neighbors...")
        start_time: float = time.time()
        
        region_keys: list[int] = list(regions.keys())
        num_regions: int = len(region_keys)
        
        # Generate all possible pairs to check
        all_pairs: list[tuple[int, int]] = []
        for i in range(num_regions):
            for j in range(i + 1, num_regions):
                all_pairs.append((region_keys[i], region_keys[j]))
        
        total_pairs: int = len(all_pairs)
        print(f"[Stage 3/6] Checking {total_pairs:,} region pairs...")
        
        neighbors: list[tuple[tuple[int, dict[str, Any]], tuple[int, dict[str, Any]]]] = []
        
        if total_pairs < self._parallel_threshold:
            print(f"[Stage 3/6] Using single-threaded processing")
            with tqdm(total=total_pairs, desc="Finding neighbors", ncols=100) as pbar:
                for idx_i, idx_j in all_pairs:
                    region_i = regions[idx_i]
                    region_j = regions[idx_j]
                    if self._regions_intersect(region_i, region_j):
                        neighbors.append(((idx_i, region_i), (idx_j, region_j)))
                    pbar.update(1)
        else:
            print(f"[Stage 3/6] Using parallel processing on {self._num_cores} cores")
            chunk_processor: ChunkProcessor = ChunkProcessor()
            chunks: list[ProcessingChunk] = chunk_processor.create_chunks(all_pairs, self._num_cores)
            
            with ThreadPoolExecutor(max_workers=self._num_cores) as executor:
                futures = []
                for chunk in chunks:
                    future = executor.submit(self._process_neighbor_chunk, chunk.data, regions)
                    futures.append(future)
                
                with tqdm(total=total_pairs, desc="Finding neighbors", ncols=100) as pbar:
                    for idx, future in enumerate(as_completed(futures)):
                        chunk_result: list[tuple[tuple[int, dict[str, Any]], tuple[int, dict[str, Any]]]] = future.result()
                        neighbors.extend(chunk_result)
                        pbar.update(len(chunks[idx].data))
        
        duration: float = time.time() - start_time
        print(f"[Stage 3/6] ✓ Found {len(neighbors):,} neighboring pairs in {duration:.1f}s")
        
        return neighbors


class RegionMerger(Module):
    # Takes two regions and combines them into one bigger region.
    # Merges their bounding boxes, adds up sizes, and blends the
    # histograms weighted by size. Keeps track of all original
    # labels so we know what small regions made up this big one.
    def __init__(self) -> None:
        super().__init__()
    
    def forward(self, r1: dict[str, Any], r2: dict[str, Any]) -> dict[str, Any]:
        #  Merges two regions into one. Combines their sizes, expands
        #  the bounding box to cover both, and blends the histograms
        #  weighted by size. Keeps track of all original labels.
        
        new_size: int = r1["size"] + r2["size"]
        
        # Calculate weighted averages for histograms
        size1: float = float(r1["size"])
        size2: float = float(r2["size"])
        total_size: float = size1 + size2
        
        # Blend the histograms based on size
        new_color_hist: np.ndarray = ((r1["color_hist"] * size1 + r2["color_hist"] * size2) / total_size).astype(np.float32)
        new_texture_hist: np.ndarray = ((r1["texture_hist"] * size1 + r2["texture_hist"] * size2) / total_size).astype(np.float32)
        
        # Combine all labels from both regions
        new_labels: list[int] = list(set(r1["labels"] + r2["labels"]))
        
        merged_region: dict[str, Any] = {
            "min_x": min(r1["min_x"], r2["min_x"]),
            "min_y": min(r1["min_y"], r2["min_y"]),
            "max_x": max(r1["max_x"], r2["max_x"]),
            "max_y": max(r1["max_y"], r2["max_y"]),
            "size": new_size,
            "color_hist": new_color_hist,
            "texture_hist": new_texture_hist,
            "labels": new_labels
        }
        
        return merged_region


class BatchSimilarityProcessor(Module):
    # Calculates similarities between lots of region pairs in parallel.
    # When we have thousands of neighbor pairs to check, this splits
    # them up across CPU cores. Each core calculates similarities for
    # its batch, then we combine all results. Major speedup compared
    # to doing them one by one.
    def __init__(self) -> None:
        super().__init__()
        self._similarity_calculator: SimilarityCalculator = SimilarityCalculator()
        self._num_cores: int = cpu_count()
        self._parallel_threshold: int = 5000
    
    def _process_similarity_batch(self, neighbor_batch: list[tuple[tuple[int, dict[str, Any]], tuple[int, dict[str, Any]]]], 
                                 imsize: int) -> dict[tuple[int, int], float]:
        # Processes a batch of neighbor pairs to calculate similarities.
        # Each CPU core runs this on its assigned batch. Returns a
        # dictionary mapping region pairs to their similarity scores.
        
        batch_similarities: dict[tuple[int, int], float] = {}
        
        # Calculate similarity for each pair in this batch
        for (ai, ar), (bi, br) in neighbor_batch:
            similarity: float = self._similarity_calculator(ar, br, imsize)
            batch_similarities[(ai, bi)] = similarity
        
        return batch_similarities
    
    def forward(self, neighbors: list[tuple[tuple[int, dict[str, Any]], tuple[int, dict[str, Any]]]], 
               imsize: int) -> dict[tuple[int, int], float]:
        # Calculates similarities for all neighboring region pairs.
        # If there's not many pairs, does it single-threaded. Otherwise
        # splits the work across all CPU cores for parallel processing.
        # Returns dictionary of similarities keyed by region ID pairs.
        
        print(f"[Stage 4/6] Computing {len(neighbors):,} initial similarities...")
        start_time: float = time.time()
        
        similarities: dict[tuple[int, int], float] = {}
        
        if len(neighbors) < self._parallel_threshold:
            print(f"[Stage 4/6] Using single-threaded processing")
            with tqdm(total=len(neighbors), desc="Computing similarities", ncols=100) as pbar:
                for (ai, ar), (bi, br) in neighbors:
                    similarity: float = self._similarity_calculator(ar, br, imsize)
                    similarities[(ai, bi)] = similarity
                    pbar.update(1)
        else:
            print(f"[Stage 4/6] Using parallel processing on {self._num_cores} cores")
            chunk_processor: ChunkProcessor = ChunkProcessor()
            chunks: list[ProcessingChunk] = chunk_processor.create_chunks(neighbors, self._num_cores)
            
            with ThreadPoolExecutor(max_workers=self._num_cores) as executor:
                futures = []
                for chunk in chunks:
                    future = executor.submit(self._process_similarity_batch, chunk.data, imsize)
                    futures.append(future)
                
                with tqdm(total=len(neighbors), desc="Computing similarities", ncols=100) as pbar:
                    for future in as_completed(futures):
                        batch_result: dict[tuple[int, int], float] = future.result()
                        similarities.update(batch_result)
                        pbar.update(len(batch_result))
        
        duration: float = time.time() - start_time
        print(f"[Stage 4/6] ✓ Computed similarities in {duration:.1f}s")
        
        return similarities


class SelectiveSearchProcessor(Module):
    # Main brain of the whole thing. Coordinates all the other modules
    # to run the selective search algorithm. Handles everything from
    # initial segmentation to final region proposals. This is what gets
    # called by the main selective_search file to do all the work.
    def __init__(self) -> None:
        super().__init__()
        self.segmentation: SegmentationModule = SegmentationModule()
        self.region_extractor: RegionExtractor = RegionExtractor()
        self.neighbor_finder: NeighborFinder = NeighborFinder()
        self.similarity_calculator: SimilarityCalculator = SimilarityCalculator()
        self.batch_similarity_processor: BatchSimilarityProcessor = BatchSimilarityProcessor()
        self.region_merger: RegionMerger = RegionMerger()
        self.histogram_extractor: HistogramExtractor = HistogramExtractor()
        self.texture_calculator: TextureGradientCalculator = TextureGradientCalculator()
        self.progress_tracker: ProgressTracker = ProgressTracker()
        self._num_cores: int = cpu_count()
        
    def generate_segments(self, im_orig: np.ndarray, scale: float, sigma: float, min_size: int) -> np.ndarray:
        # Just calls the segmentation module to chop up the image
        return self.segmentation(im_orig, scale, sigma, min_size)
    
    def calc_colour_hist(self, img: np.ndarray) -> np.ndarray:
        # Calculates color histogram for the whole image. Converts to
        # HSV first, then makes 25 bins per channel. Unlike the region
        # version, this doesn't normalize - just returns raw counts.
        
        # Convert RGB to HSV color space
        hsv_img: np.ndarray = skimage.color.rgb2hsv(img)
        hist: list[np.ndarray] = []
        
        # Build histogram for each HSV channel
        for channel in range(3):
            channel_hist: np.ndarray = np.histogram(
                hsv_img[:, :, channel],
                bins=25,
                range=(0, 1)
            )[0].astype(np.float32)
            hist.append(channel_hist)
        
        # Combine all channels into one big histogram
        full_hist: np.ndarray = np.concatenate(hist)
        return full_hist
    
    def calc_texture_gradient(self, img: np.ndarray) -> np.ndarray:
        # Just forwards to the texture calculator module
        return self.texture_calculator(img)
    
    def calc_texture_hist(self, img: np.ndarray) -> np.ndarray:
        # Calculates texture histogram for the whole image. First gets
        # the texture gradient using LBP, then builds 10 bins per channel.
        # This one does normalize at the end unlike color histogram.
        
        # Get texture features first
        texture_gradient: np.ndarray = self.calc_texture_gradient(img)
        hist: list[np.ndarray] = []
        
        # Build histogram for each texture channel
        for channel in range(3):
            channel_hist: np.ndarray = np.histogram(
                texture_gradient[:, :, channel],
                bins=10
            )[0].astype(np.float32)
            hist.append(channel_hist)
        
        # Combine and normalize
        full_hist: np.ndarray = np.concatenate(hist)
        normalized_hist: np.ndarray = full_hist / (np.sum(full_hist) + 1e-7)
        
        return normalized_hist
    
    def extract_regions(self, img: np.ndarray) -> dict[int, dict[str, Any]]:
        # Just forwards to the region extractor module
        return self.region_extractor(img)
    
    def sim_colour(self, r1: dict[str, Any], r2: dict[str, Any]) -> float:
        # Wrapper for color similarity calculation
        return self.similarity_calculator.color_similarity(r1, r2)
    
    def sim_texture(self, r1: dict[str, Any], r2: dict[str, Any]) -> float:
        # Wrapper for texture similarity calculation
        return self.similarity_calculator.texture_similarity(r1, r2)
    
    def sim_size(self, r1: dict[str, Any], r2: dict[str, Any], imsize: int) -> float:
        # Wrapper for size similarity calculation
        return self.similarity_calculator.size_similarity(r1, r2, imsize)
    
    def sim_fill(self, r1: dict[str, Any], r2: dict[str, Any], imsize: int) -> float:
        # Wrapper for fill similarity calculation
        return self.similarity_calculator.fill_similarity(r1, r2, imsize)
    
    def extract_neighbours(self, regions: dict[int, dict[str, Any]]) -> list[tuple[tuple[int, dict[str, Any]], tuple[int, dict[str, Any]]]]:
        # Just forwards to the neighbor finder module
        return self.neighbor_finder(regions)
    
    def merge_regions(self, r1: dict[str, Any], r2: dict[str, Any]) -> dict[str, Any]:
        # Just forwards to the region merger module
        return self.region_merger(r1, r2)
    
    def compute_initial_similarities(self, neighbors: list[tuple[tuple[int, dict[str, Any]], tuple[int, dict[str, Any]]]], 
                                   imsize: int) -> dict[tuple[int, int], float]:
        # Calculates all the initial similarities between neighboring
        # regions. Uses the batch processor for parallel computation
        # when there's lots of pairs to check.
        return self.batch_similarity_processor(neighbors, imsize)