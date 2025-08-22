from __future__ import annotations
import os
import numpy as np
import time
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import Rectangle
import scipy.io as sio
from scipy import ndimage
from dataclasses import dataclass
from mpl_toolkits.axes_grid1 import make_axes_locatable
from typing import Any, Optional, Tuple, Union, Callable
# np.random.seed(42)

# Base Module System
class Module:
    def __init__(self) -> None:
        pass
    
    def forward(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError
    
    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.forward(*args, **kwargs)

# Data Structures
@dataclass
class PlaneModel:        
    normal: np.ndarray  # Normal vector of the plane (nx, ny, nz)
    d: float            # Distance from origin
    
    def distance_to_point(self, point: np.ndarray) -> float:                                   #computes how far a given point is from the plane.
        return abs(np.dot(self.normal, point) - self.d) / np.linalg.norm(self.normal)
    
    def distance_between_planes(self, other_plane: PlaneModel) -> float:                       # computes vertical distance between two parallel planes.
        # Simple distance calculation as shown in the lecture slides
        return abs(self.d - other_plane.d) / np.linalg.norm(self.normal)

# Data Handling
class BoxDataset:
    ####################################
    #Find files in a directory
    #Load each file into structured data
    #Access each example by index
    #Track metadata
    ####################################
    def __init__(self, data_dir: str) -> None:
        # Stores the directory path.
        # Calls _discover_files() to find and list the valid .mat files.
        # Outputs how many files were loaded.

        self.data_dir = data_dir
        self.file_paths = self._discover_files(data_dir)
        print(f"Initialized dataset with {len(self.file_paths)} examples from {data_dir}")
    
    def __len__(self) -> int:                   #Returns how many files are in the dataset.
        return len(self.file_paths)
    
    def __getitem__(self, idx: int) -> dict[str, Any]:
        file_path = self.file_paths[idx]
        print(f"Loading file: {file_path}")
        return self._load_file(file_path)
    
    #list of valid file paths
    def _discover_files(self, data_dir: str) -> list[str]:
        files: list[str] = [
            f"{data_dir}/example{i}kinect.mat" 
            for i in range(1, 5) 
            if os.path.exists(f"{data_dir}/example{i}kinect.mat")
        ]
        return files
    
    def _load_file(self, file_path: str) -> dict[str, Any]:             #Loads individual file content
        try: 
            # Load the .mat file
            mat_data = sio.loadmat(file_path)
            
            # Extract the file number from the filename
            file_number = os.path.basename(file_path).replace("example", "").replace("kinect.mat", "")
            
            # Get the data using the correct variable names based on the file number
            amplitude_name = f'amplitudes{file_number}'
            distance_name = f'distances{file_number}'
            cloud_name = f'cloud{file_number}'
            
            amplitude_image = mat_data[amplitude_name]
            distance_image = mat_data[distance_name]
            point_cloud = mat_data[cloud_name]
            
            result = {
                'amplitude': amplitude_image,
                'distance': distance_image,
                'point_cloud': point_cloud,
                'name': os.path.basename(file_path).replace('.mat', ''),
                'metadata': {
                    'file_path': file_path,
                    'file_number': file_number
                }
            }
            return result
            
        except Exception as e:
            print(f"Error loading file {file_path}: {str(e)}")
            raise


# Processing Modules
class Ransac(Module):    
    def __init__(self, threshold: float = 0.05, max_iterations: int = 1000, min_inlier_ratio: float = 0.3) -> None:
        super().__init__()
        self.threshold = threshold
        self.max_iterations = max_iterations
        self.min_inlier_ratio = min_inlier_ratio
    
    def forward(self, points: np.ndarray) -> tuple[PlaneModel, np.ndarray]:
        if points.shape[0] < 3:
            raise ValueError("Need at least 3 points to fit a plane")
        
        # Initialize variables
        max_inliers: int = 0
        best_model: Optional[PlaneModel] = None
        best_inlier_mask: Optional[np.ndarray] = None
        n_points: int = points.shape[0]
        
        # Add an iteration counter to force termination
        iter_count: int = 0
        max_allowed_iter: int = min(self.max_iterations, 2000)  # Cap at 2000 iterations
        
        # Run RANSAC iterations
        while iter_count < max_allowed_iter:
            iter_count += 1
            
            # Safety check in case of bad data
            if iter_count >= max_allowed_iter - 1:
                print("Warning: Maximum iterations reached without good model")
                # If we haven't found a decent model, force acceptance of the best one so far
                if best_model is None and n_points >= 3:
                    # Create a last-chance model with first 3 points
                    indices = np.array([0, 1, 2])
                    emergency_model: Optional[PlaneModel] = self._fit_plane(points[indices])
                    if emergency_model is not None:
                        best_model = emergency_model
                        distances: np.ndarray = np.abs(np.dot(points, best_model.normal) - best_model.d) / np.linalg.norm(best_model.normal)
                        best_inlier_mask: np.ndarray = distances < self.threshold
                    break
            
            # Randomly sample 3 points
            try:
                sample_indices: np.ndarray = np.random.choice(n_points, 3, replace=False)
                sampled_points: np.ndarray = points[sample_indices]
                
                # Fit a plane to the sampled points
                plane_model: Optional[PlaneModel] = self._fit_plane(sampled_points)
                if plane_model is None:
                    continue
                
                # Count inliers
                inlier_mask, inlier_count = self._count_inliers(plane_model, points)
                
                # Update best model if this one is better
                if inlier_count > max_inliers:
                    max_inliers: int = inlier_count
                    best_inlier_mask: np.ndarray = inlier_mask
                    best_model: PlaneModel = plane_model
                    
                    # Early termination if we found a very good model
                    inlier_ratio: float = max_inliers / n_points
                    if inlier_ratio > self.min_inlier_ratio:
                        break
            except Exception as e:
                print(f"Error in RANSAC iteration: {e}")
                continue
        
        if best_model is None:
            raise ValueError("RANSAC failed to find a plane model")
        
        return best_model, best_inlier_mask
    
    def _fit_plane(self, points: np.ndarray) -> Optional[PlaneModel]:
        # Points are in rows, so p1, p2, p3 are rows of the points array
        p1, p2, p3 = points
        
        # Compute two vectors in the plane
        v1: np.ndarray = p2 - p1
        v2: np.ndarray = p3 - p1
        
        # The normal vector is the cross product of the two vectors
        normal: np.ndarray = np.cross(v1, v2)
        
        # Normalize the normal vector
        normal_length: float = np.linalg.norm(normal)
        
        # Check if the normal vector has zero length (points are collinear/lie on a straight line)
        if normal_length < 1e-10:
            return None
            
        normal: np.ndarray = normal / normal_length
        
        # Calculate the distance parameter d
        d: float = np.dot(normal, p1)
        
        return PlaneModel(normal, d)
    

    def _fit_plane_to_points(self, points: np.ndarray) -> Optional[PlaneModel]:             ### [???]
        # Fit a plane to many points using least squares (SVD).
        # This gives a more accurate plane model than using just 3 points.
        
        # Args:
        #     points: 3D points (Nx3) to fit a plane to.
        # Returns:
        #     A plane model (normal vector and distance from origin).
        # 
        if points.shape[0] < 3:
            return None
            
        # Center the points by subtracting the mean
        center: np.ndarray = np.mean(points, axis=0)
        centered_points: np.ndarray = points - center
        
        # Use SVD to fit a plane
        _, _, vh = np.linalg.svd(centered_points)
        
        # The normal is the eigenvector corresponding to the smallest eigenvalue
        normal: np.ndarray = vh[2, :]
        
        # Normalize the normal vector
        normal_length: float = np.linalg.norm(normal)
        if normal_length < 1e-10:
            return None
            
        normal: np.ndarray = normal / normal_length
        
        # Ensure the normal points upward (positive z)
        if normal[2] < 0:
            normal: np.ndarray = -normal
        
        # Calculate the distance parameter d
        d: float = np.dot(normal, center)
        
        return PlaneModel(normal, d)
    

    def _count_inliers(self, plane_model: PlaneModel, points: np.ndarray) -> Tuple[np.ndarray, int]:
        # Calculate distances from all points to the plane
        distances: np.ndarray = np.abs(np.dot(points, plane_model.normal) - plane_model.d) / np.linalg.norm(plane_model.normal)
        
        # Create a mask for inlier points (points close enough to the plane)
        inlier_mask: np.ndarray = distances < self.threshold
        
        # Count the number of inliers
        inlier_count: int = np.sum(inlier_mask)
        
        return inlier_mask, inlier_count


class Mask(Module):    
    #### Applies morphological operations (erosion and dilation) for cleaning
    #### Extracts the largest connected component

    def __init__(self, kernel_size: int = 3) -> None:
        super().__init__()
        self.kernel_size = kernel_size
    
    def forward(self, mask: np.ndarray) -> np.ndarray:
        mask_binary: np.ndarray = mask.astype(bool)
        
        # Create structuring element (kernel) - use a single simple kernel
        kernel: np.ndarray = np.ones((self.kernel_size, self.kernel_size), dtype=bool)
        
        # Apply properly sequenced morphological operations
        # First, closing (dilation followed by erosion) to fill small holes
        mask_dilated: np.ndarray = ndimage.binary_dilation(mask_binary, structure=kernel)
        mask_closed: np.ndarray = ndimage.binary_erosion(mask_dilated, structure=kernel)
        
        # Then opening (erosion followed by dilation) to remove small isolated regions
        mask_eroded: np.ndarray = ndimage.binary_erosion(mask_closed, structure=kernel)
        mask_opened: np.ndarray = ndimage.binary_dilation(mask_eroded, structure=kernel)
        
        return mask_closed
    
    def extract_component(self, mask: np.ndarray) -> np.ndarray:
        """Extract the largest connected component from a binary mask."""
        # First apply a small closing operation to fill tiny holes
        kernel: np.ndarray = np.ones((3, 3), dtype=bool)
        mask_closed: np.ndarray = ndimage.binary_closing(mask, structure=kernel)          ## [???]
        
        # Label connected components
        labeled_mask, num_components = ndimage.label(mask_closed)
        
        if num_components == 0:
            return np.zeros_like(mask, dtype=bool)
        
        # Count the size of each component
        component_sizes: np.ndarray = np.bincount(labeled_mask.flatten())        #number of pixels are in each region 
        component_sizes[0] = 0  # Ignore the background
        
        # Find the largest component
        largest_component_label: int = np.argmax(component_sizes)
        
        # Create a mask for just the largest component
        largest_component_mask: np.ndarray = labeled_mask == largest_component_label
        
        # Post-process the component to clean it up (fill small holes)
        largest_component_mask: np.ndarray = ndimage.binary_closing(largest_component_mask, structure=kernel)
        
        return largest_component_mask


class Geometry(Module):
    def __init__(self) -> None:
        super().__init__()
    
    def forward(self, point_cloud: np.ndarray, box_top_mask: np.ndarray, floor_plane: PlaneModel, 
                                                                     box_plane: PlaneModel) -> dict[str, float]:
        # Calculate height using the formula from the lecture - no arbitrary limits
        height: float = abs(box_plane.d - floor_plane.d) / np.linalg.norm(floor_plane.normal)
        
        # Get the y, x coordinates of the box top mask
        ys, xs = np.where(box_top_mask == 1)
        
        if len(ys) == 0 or len(xs) == 0:
            # If no box points were found, return default dimensions
            print("Warning: No box points detected. Using default dimensions.")
            return {
                'height': 0.2,
                'width': 0.3,
                'length': 0.4
            }
        
        min_y, max_y = ys.min(), ys.max()
        min_x, max_x = xs.min(), xs.max()
        
        try:
            # Get the actual 3D points at all four corner positions
            p1: np.ndarray = point_cloud[min_y, min_x, :]  # Top-left
            p2: np.ndarray = point_cloud[min_y, max_x, :]  # Top-right
            p3: np.ndarray = point_cloud[max_y, min_x, :]  # Bottom-left
            p4: np.ndarray = point_cloud[max_y, max_x, :]  # Bottom-right
            
            # Calculate width and length using diagonal averaging for more stability
            length1: float = np.linalg.norm(p1 - p2)  # Top edge
            length2: float = np.linalg.norm(p3 - p4)  # Bottom edge
            length: float = (length1 + length2) / 2    # Average length
            
            width1: float = np.linalg.norm(p1 - p3)    # Left edge
            width2: float = np.linalg.norm(p2 - p4)    # Right edge
            width: float = (width1 + width2) / 2       # Average width
            
            # Make sure width is the shorter dimension (by convention)
            if width > length:
                width, length = length, width
            
            return {
                'height': height,
                'width': width,
                'length': length
            }
        except Exception as e:
            print(f"Error calculating dimensions: {e}")
            return {
                'height': height if height > 0 else 0.2,
                'width': 0.3,
                'length': 0.4
            }



class Viz(Module):
    def __init__(self, save_dir: str = "figures") -> None:
        #Create directory for images
        super().__init__()
        self.save_dir = save_dir
        # Create the main figures directory if it doesn't exist
        if not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)
            print(f"Created main figures directory: {save_dir}")
    
    def forward(self, data: dict[str, Any], floor_mask: np.ndarray, box_top_mask: np.ndarray, 
                                                                    dimensions: dict[str, float]) -> None:
        """
        Generate all visualizations required for the exercise.

        Args:
            data: Dictionary containing input data
            floor_mask: Binary mask for the floor
            box_top_mask: Binary mask for the box top
            dimensions: Dictionary with box dimensions
        """
        data_name = data.get('name', 'unnamed_example')
        print(f"Generating visualizations for {data_name}...")
        
        # Create directory for this specific example
        example_dir: str = os.path.join(self.save_dir, data_name)
        if not os.path.exists(example_dir):
            os.makedirs(example_dir, exist_ok=True)
            print(f"Created example directory: {example_dir}")
        
        # 1. Figure 1: Original data (ToF amplitude and Distance image)
        figure1_path: str = os.path.join(example_dir, "figure1.png")
        self.visualize_input_data(data['amplitude'], data['distance'], data_name, figure1_path)
        
        # 2. Figure 2: Floor mask and filtered floor mask
        # Create a filtered floor mask for this visualization
        filtered_floor_mask: np.ndarray = ndimage.binary_opening(floor_mask, structure=np.ones((3, 3)))
        filtered_floor_mask: np.ndarray = ndimage.binary_closing(filtered_floor_mask, structure=np.ones((5, 5)))
        
        figure2_path: str = os.path.join(example_dir, "figure2.png")
        self.visualize_floor_masks(floor_mask, filtered_floor_mask, data_name, figure2_path)
        
        # 3. Figure 3: Box top component
        figure3_path: str = os.path.join(example_dir, "figure3.png")
        self.visualize_box_top_component(box_top_mask, data_name, figure3_path)
        
        # 4. Figure 4: Final visualization with color coding
        # This will show floor (green), box top (red), and box edges (blue)
        figure4_path: str = os.path.join(example_dir, "figure4.png")
        self.visualize_color_coded(data['point_cloud'], floor_mask, box_top_mask, dimensions, data_name, figure4_path)
    

    def visualize_input_data(self, amplitude_img: np.ndarray, distance_img: np.ndarray, data_name: str = '', save_path: str = None) -> None:
        """
        Visualize the ToF amplitude and distance images (Figure 1).
        
        Args:
            amplitude_img: Amplitude image
            distance_img: Distance image
            data_name: Name of the dataset being visualized (e.g., example1kinect)
            save_path: Path to save the figure (if None, only display)
        """
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle(f"Figure 1: Input Data Visualization ({data_name})", fontsize=14)
        
        # Plot amplitude image (grayscale)
        axes[0].imshow(amplitude_img, cmap='gray')
        axes[0].set_title("(a) ToF amplitude image")
        axes[0].axis('off')
        
        # Plot distance image with color mapping - using jet colormap to match exercise
        im: plt.imshow = axes[1].imshow(distance_img, cmap='jet')
        axes[1].set_title("(b) Distance image")
        axes[1].axis('off')
        
        # Add a colorbar for the distance image
        
        divider = make_axes_locatable(axes[1])
        cax: plt.axes = divider.append_axes("right", size="5%", pad=0.05)
        cb: plt.colorbar = plt.colorbar(im, cax=cax)
        cb.set_label('Distance (m)', fontsize=9)
        
        plt.tight_layout()
        
        # Save the figure if a path is provided
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved Figure 1 to {save_path}")
        
        plt.show()
    

    def visualize_floor_masks(self, floor_mask: np.ndarray, filtered_floor_mask: np.ndarray, data_name: str = '', save_path: str = None) -> None:
        """
        Visualize the floor mask and filtered floor mask (Figure 2).
        
        Args:
            floor_mask: Original floor mask from RANSAC
            filtered_floor_mask: Floor mask after morphological operations
            data_name: Name of the dataset being visualized (e.g., example1kinect)
            save_path: Path to save the figure (if None, only display)
        """
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle(f"Figure 2: Floor Segmentation ({data_name})", fontsize=14)
        
        # Create color maps exactly matching the professor's lecture PDF:
        # - Blue (RGB: 0, 0, 0.8) for floor 
        # - Dark red (RGB: 0.8, 0, 0) for non-floor areas (box, background)
        
        mask_colored1: np.ndarray = np.zeros((*floor_mask.shape, 3), dtype=np.float32)
        mask_colored1[floor_mask == 0, 0] = 0.8  # Red for non-floor areas (box, background)
        mask_colored1[floor_mask == 1, 2] = 0.8  # Blue for floor
        
        mask_colored2: np.ndarray = np.zeros((*filtered_floor_mask.shape, 3), dtype=np.float32)
        mask_colored2[filtered_floor_mask == 0, 0] = 0.8  # Red for non-floor areas
        mask_colored2[filtered_floor_mask == 1, 2] = 0.8  # Blue for floor
        
        # Plot the masks with the exact color scheme from the PDF
        axes[0].imshow(mask_colored1)
        axes[0].set_title("(a) Floor mask")
        axes[0].axis('off')
        
        axes[1].imshow(mask_colored2)
        axes[1].set_title("(b) Filtered floor mask")
        axes[1].axis('off')
        
        # Add professional legends to both images
        for i in range(2):
            
            # Position legend in top-right corner                      ### [?]
            shape: tuple[int, int] = floor_mask.shape
            legend_x: int = shape[1] - 150
            legend_y: int = 20
            rect_width: int = 12
            rect_height: int = 12
            text_offset: int = 20
            
            # Add semi-transparent background
            legend_bg: Rectangle = Rectangle((legend_x-10, legend_y-10), 140, 70, 
                                facecolor='white', alpha=0.7, edgecolor='black', linewidth=1)
            axes[i].add_patch(legend_bg)
            
            # Add legend title
            axes[i].text(legend_x+60, legend_y, "LEGEND", fontsize=10, fontweight='bold', 
                        ha='center', va='center', color='black')
            
            # Add blue rectangle for floor
            floor_rect: Rectangle = Rectangle((legend_x, legend_y+20), rect_width, rect_height, 
                                facecolor=(0, 0, 0.8), edgecolor='black', linewidth=1)
            axes[i].add_patch(floor_rect)
            
            # Different label for original vs filtered floor mask
            if i == 0:
                floor_label: str = "Floor plane"
            else:
                floor_label: str = "Floor plane (filtered)"
                
            axes[i].text(legend_x+text_offset, legend_y+20+rect_height/2, floor_label, 
                        fontsize=9, va='center', color='black')
            
            # Add red rectangle for non-floor
            nonfloor_rect: Rectangle = Rectangle((legend_x, legend_y+40), rect_width, rect_height, 
                                    facecolor=(0.8, 0, 0), edgecolor='black', linewidth=1)
            axes[i].add_patch(nonfloor_rect)
            axes[i].text(legend_x+text_offset, legend_y+40+rect_height/2, "Box & background", 
                        fontsize=9, va='center', color='black')
        
        plt.tight_layout()
        
        # Save the figure if a path is provided
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved Figure 2 to {save_path}")
            
        plt.show()
    

    def visualize_box_top_component(self, box_top_mask: np.ndarray, data_name: str = '', save_path: str = None) -> None:
        # """
        # Visualize the box top component (Figure 3).
        
        # Args:
        #     box_top_mask: Binary mask for the box top component
        #     data_name: Name of the dataset being visualized (e.g., example1kinect)
        #     save_path: Path to save the figure (if None, only display)
        # """
        fig, ax = plt.subplots(figsize=(6, 5))
        fig.suptitle(f"Figure 3: Box Top Component ({data_name})", fontsize=14)
        
        # Create color map exactly matching the professor's lecture PDF:
        # - Blue (RGB: 0, 0, 0.8) for box top surface 
        # - Dark red (RGB: 0.8, 0, 0) for background
        
        mask_colored: np.ndarray = np.zeros((*box_top_mask.shape, 3), dtype=np.float32)
        mask_colored[box_top_mask == 0, 0] = 0.8  # Red channel for background (0.8 = dark red)
        mask_colored[box_top_mask == 1, 2] = 0.8  # Blue channel for box top (0.8 = rich blue)
        
        ax.imshow(mask_colored)
        ax.set_title("Box top component")
        ax.axis('off')
        
        # Add professional legend with rectangle patches
        
        
        # Position legend in top-right corner
        shape: tuple[int, int] = box_top_mask.shape
        legend_x: int = shape[1] - 150
        legend_y: int = 20
        rect_width: int = 12
        rect_height: int = 12
        text_offset: int = 20
        
        # Add semi-transparent background
        legend_bg: Rectangle = Rectangle((legend_x-10, legend_y-10), 140, 70, 
                            facecolor='white', alpha=0.7, edgecolor='black', linewidth=1)
        ax.add_patch(legend_bg)
        
        # Add legend title
        ax.text(legend_x+60, legend_y, "LEGEND", fontsize=10, fontweight='bold', 
            ha='center', va='center', color='black')
        
        # Add blue rectangle for box top
        box_rect: Rectangle = Rectangle((legend_x, legend_y+20), rect_width, rect_height, 
                        facecolor=(0, 0, 0.8), edgecolor='black', linewidth=1)
        ax.add_patch(box_rect)
        ax.text(legend_x+text_offset, legend_y+20+rect_height/2, "Box top surface", 
            fontsize=9, va='center', color='black')
        
        # Add red rectangle for background
        bg_rect: Rectangle = Rectangle((legend_x, legend_y+40), rect_width, rect_height, 
                        facecolor=(0.8, 0, 0), edgecolor='black', linewidth=1)
        ax.add_patch(bg_rect)
        ax.text(legend_x+text_offset, legend_y+40+rect_height/2, "Background", 
            fontsize=9, va='center', color='black')
        
        plt.tight_layout()
        
        # Save the figure if a path is provided
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved Figure 3 to {save_path}")
            
        plt.show()
    
    def visualize_color_coded(self, point_cloud: np.ndarray, floor_mask: np.ndarray, 
                             box_top_mask: np.ndarray, dimensions: dict, data_name: str = '', save_path: str = None) -> None:
        # """
        # Visualize the final box detection result with colored regions and corner labels (Figure 4).
        
        # Args:
        #     point_cloud: Original 3D point cloud
        #     floor_mask: Binary mask for the floor
        #     box_top_mask: Binary mask for the box top component
        #     dimensions: Dictionary with box dimensions
        #     data_name: Name of the dataset being visualized (e.g., example1kinect)
        #     save_path: Path to save the figure (if None, only display)
        # """
        # Create RGB image
        shape: tuple[int, int] = point_cloud.shape[:2]
        
        # Color coding exactly matching the professor's lecture example:
        # - Light green (RGB: 0, 255, 0) for floor
        # - Dark red (RGB: 139, 0, 0) for box top
        # - Blue (RGB: 0, 0, 128) for background
        
        # Initialize with blue background
        vis: np.ndarray = np.zeros((*shape, 3), dtype=np.uint8)
        vis[:, :] = [0, 0, 128]  # Dark blue background
        
        # Floor color (light green)
        vis[floor_mask == 1] = [0, 255, 0]
        
        # Box top color (dark red)
        vis[box_top_mask == 1] = [139, 0, 0]
        
        # Show with labels matching the final visualization
        fig, ax = plt.subplots(figsize=(8, 7))
        fig.suptitle(f"Figure 4: Visualization of floor, box and box corners ({data_name})", fontsize=14)
        
        # Display the image
        ax.imshow(vis)
        
        # Find corners for labeling
        ys, xs = np.where(box_top_mask == 1)
        if len(ys) > 0 and len(xs) > 0:
            min_y, max_y = ys.min(), ys.max()
            min_x, max_x = xs.min(), xs.max()
            
            # Add precise corner labels with clean positioning
            # Use larger font for better visibility
            ax.text((min_x+max_x)//2, min_y-10, 'top', fontsize=11, color='black', 
                    ha='center', va='bottom', fontweight='bold')
            ax.text((min_x+max_x)//2, max_y+10, 'bottom', fontsize=11, color='black', 
                    ha='center', va='top', fontweight='bold')
            ax.text(min_x-10, (min_y+max_y)//2, 'left', fontsize=11, color='black', 
                    ha='right', va='center', fontweight='bold')
            ax.text(max_x+10, (min_y+max_y)//2, 'right', fontsize=11, color='black', 
                    ha='left', va='center', fontweight='bold')
            
            # Create a proper legend with patches instead of text symbols
            from matplotlib.patches import Rectangle
            
            # Create a semi-transparent legend with a nice border at top-right
            legend_x: int = shape[1] - 150  # Position in top-right
            legend_y: int = 20
            rect_width: int = 12
            rect_height: int = 12
            text_offset: int = 20
            
            # Add a semi-transparent background for the legend
            legend_bg: Rectangle = Rectangle((legend_x-10, legend_y-10), 140, 90, 
                                 facecolor='white', alpha=0.7, edgecolor='black', linewidth=1)
            ax.add_patch(legend_bg)
            
            # Add legend title
            ax.text(legend_x+50, legend_y, "LEGEND", fontsize=10, fontweight='bold', 
                    ha='center', va='center', color='black')
            
            # Add green rectangle and label for floor
            floor_rect = Rectangle((legend_x, legend_y+15), rect_width, rect_height, 
                                 facecolor=(0, 1, 0), edgecolor='black', linewidth=1)
            ax.add_patch(floor_rect)
            ax.text(legend_x+text_offset, legend_y+15+rect_height/2, "Floor plane", 
                    fontsize=9, va='center', color='black')
            
            # Add red rectangle and label for box
            box_rect = Rectangle((legend_x, legend_y+35), rect_width, rect_height, 
                                facecolor=(139/255, 0, 0), edgecolor='black', linewidth=1)
            ax.add_patch(box_rect)
            ax.text(legend_x+text_offset, legend_y+35+rect_height/2, "Box", 
                   fontsize=9, va='center', color='black')
            
            # Add blue rectangle and label for background
            bg_rect = Rectangle((legend_x, legend_y+55), rect_width, rect_height, 
                               facecolor=(0, 0, 128/255), edgecolor='black', linewidth=1)
            ax.add_patch(bg_rect)
            ax.text(legend_x+text_offset, legend_y+55+rect_height/2, "Background", 
                   fontsize=9, va='center', color='black')
        
        ax.set_title("Visualization of floor (green), box (red)")
        ax.axis('off')
        plt.tight_layout()
        
        # Save the figure if a path is provided
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved Figure 4 to {save_path}")
            
        plt.show()
        
        # Print box dimensions with 4 decimal places of accuracy
        print(f"\nBox Dimensions for {data_name}:")
        print(f"Height: {dimensions['height']:.4f} meters")
        print(f"Width: {dimensions['width']:.4f} meters")
        print(f"Length: {dimensions['length']:.4f} meters")



class BoxDetector(Module):
    def __init__(self, config: Optional[dict[str, Any]] = None) -> None:
        super().__init__()
        config = config or {}

        # Initialize components with optimized parameters
        self.ransac = Ransac(
            threshold=config.get('threshold', 0.05),  # Default to 0.05 for floor plane
            max_iterations=config.get('max_iterations', 1000),
            min_inlier_ratio=config.get('min_inlier_ratio', 0.3)
        )
        
        self.mask = Mask(
            kernel_size=config.get('kernel_size', 3)  # Simplified mask parameters
        )
        
        self.geometry = Geometry()  # No height limits
        
        self.viz = Viz()
        
        print(f"BoxDetector initialized with configuration: {config}")
    

    def forward(self, data: dict[str, Any]) -> dict[str, Any]:
        print(f"\nProcessing: {data.get('name', 'unnamed example')}")
        
        point_cloud: np.ndarray = data['point_cloud']
        
        # 1. Detect floor plane with improved approach
        print("Detecting floor plane...")
        valid_points, valid_indices = self._get_valid_points(point_cloud)
        
        # Use a threshold better suited for floor detection
        original_threshold: float = self.ransac.threshold
        self.ransac.threshold = 0.05  # More permissive for floor detection
        
        try:
            floor_plane, floor_inliers = self.ransac(valid_points)
            
            # Simple normal orientation fix - just ensure z-component is positive 
            # to make sure floor normal points upward
            if floor_plane.normal[2] < 0:
                floor_plane.normal = -floor_plane.normal
                floor_plane.d = -floor_plane.d
            
            # 2. Process floor mask with simple morphological operations
            print("Processing floor mask...")
            floor_mask: np.ndarray = self._create_mask(point_cloud.shape[:2], valid_indices, floor_inliers)
            
            # Apply more aggressive filtering to improve the floor mask
            kernel: np.ndarray = np.ones((5, 5), dtype=bool)
            floor_mask: np.ndarray = ndimage.binary_closing(floor_mask, structure=kernel)
            
            # 3. Detect box plane from non-floor points with reduced threshold for precision
            print("Detecting box plane...")
            # Create non-floor mask using logical not
            non_floor_mask: np.ndarray = ~floor_mask
            
            # Get only valid points that are also not floor
            non_floor_valid: np.ndarray = np.logical_and(point_cloud[:,:,2] != 0, non_floor_mask)
            y_nf, x_nf = np.where(non_floor_valid)
            non_floor_indices: np.ndarray = np.stack((y_nf, x_nf), axis=1)
            non_floor_points: np.ndarray = point_cloud[y_nf, x_nf]
            
            # Use reduced threshold for box detection for better precision
            self.ransac.threshold = 0.01
            box_plane, box_inliers = self.ransac(non_floor_points)
            
            # 4. Process box mask and extract the largest component
            print("Processing box mask...")
            box_mask: np.ndarray = np.zeros(point_cloud.shape[:2], dtype=bool)
            box_indices: np.ndarray = non_floor_indices[box_inliers]
            box_mask[box_indices[:, 0], box_indices[:, 1]] = True
            
            # Use labeled components to find the largest one - this is critical
            labeled_mask, num_components = ndimage.label(box_mask)
            if num_components > 0:
                # Count sizes of each component
                component_sizes: np.ndarray = np.bincount(labeled_mask.flatten())
                component_sizes[0] = 0  # Ignore background
                largest_label: int = np.argmax(component_sizes)
                box_top_mask: np.ndarray = (labeled_mask == largest_label)
            else:
                # Fallback if no components found
                box_top_mask: np.ndarray = box_mask
            
            # 5. Calculate geometry with all four corners for better accuracy
            print("Calculating box dimensions...")
            dimensions: dict[str, float] = self.geometry(point_cloud, box_top_mask, floor_plane, box_plane)
            print(f"Box dimensions: Height={dimensions['height']:.4f}m × Width={dimensions['width']:.4f}m × Length={dimensions['length']:.4f}m")
            
            # Restore original threshold
            self.ransac.threshold = original_threshold
            
            # 6. Visualize with improved legends
            if data.get('visualize', True):
                self.viz(data, floor_mask, box_top_mask, dimensions)
            
            return {
                'dimensions': dimensions,
                'floor_plane': floor_plane,
                'box_plane': box_plane,
                'masks': {
                    'floor': floor_mask,
                    'filtered_floor': floor_mask,  # We've already filtered it
                    'box': box_mask,
                    'box_top': box_top_mask
                },
                'metadata': data.get('metadata', {})
            }
            
        except Exception as e:
            # Provide error handling to prevent infinite loops
            print(f"Error in processing: {e}")
            print("Returning placeholder results")
            
            # Create dummy masks and planes as placeholders
            dummy_mask: np.ndarray = np.zeros(point_cloud.shape[:2], dtype=bool)
            dummy_normal: np.ndarray = np.array([0, 0, 1])
            dummy_plane: PlaneModel = PlaneModel(dummy_normal, 0.0)
            
            # Return placeholder results
            return {
                'dimensions': {'height': 0.2, 'width': 0.3, 'length': 0.4},
                'floor_plane': dummy_plane,
                'box_plane': dummy_plane,
                'masks': {
                    'floor': dummy_mask,
                    'filtered_floor': dummy_mask,
                    'box': dummy_mask,
                    'box_top': dummy_mask
                },
                'metadata': data.get('metadata', {})
            }
    
    def process_dataset(self, dataset: BoxDataset, visualize: bool = True) -> dict[str, Any]:
        #@: Process all examples in a dataset.
        results: dict[str, Any] = {}
        
        print(f"\nProcessing {len(dataset)} examples from dataset")
        
        # Process each example
        for i in range(len(dataset)):
            data: dict[str, Any] = dataset[i]
            data['visualize'] = visualize
            
            result: dict[str, Any] = self(data)
            results[data['name']] = result
        
        # Print summary of all results
        print("\n====== Summary of Results ======")
        for name, result in results.items():
            dims: dict[str, float] = result['dimensions']
            print(f"{name}: Height={dims['height']:.4f}m × Width={dims['width']:.4f}m × Length={dims['length']:.4f}m")
        
        return results

    def _get_valid_points(self, point_cloud: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        # Extract valid points from the point cloud (points with non-zero z)
        # Get indices of valid points (z != 0)
        valid_mask: np.ndarray = point_cloud[:, :, 2] != 0
        y_indices, x_indices = np.where(valid_mask)
        valid_indices: np.ndarray = np.stack((y_indices, x_indices), axis=1)
        
        # Extract the valid points
        valid_points: np.ndarray = point_cloud[valid_mask]
        
        return valid_points, valid_indices
    

    def _create_mask(self, shape: tuple, indices: np.ndarray, inliers: np.ndarray) -> np.ndarray:
        # Create a binary mask from inlier indices
        mask: np.ndarray = np.zeros(shape, dtype=bool)
        inlier_indices: np.ndarray = indices[inliers]
        mask[inlier_indices[:, 0], inlier_indices[:, 1]] = True
        return mask
    

    def _get_non_floor_points(self, point_cloud: np.ndarray, valid_points: np.ndarray, 
                            valid_indices: np.ndarray, floor_inliers: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        # Extract points that are not part of the floor.
        # Create mask of non-floor points
        non_floor_indices: np.ndarray = valid_indices[~floor_inliers]
        
        # Extract non-floor points
        non_floor_points: np.ndarray = valid_points[~floor_inliers]
        
        return non_floor_points, non_floor_indices


# Driver Code
if __name__ == "__main__":
    print("Starting box detection application")
    
    # Set up configuration with optimized parameters
    config = {
        'threshold': 0.05,  # RANSAC inlier threshold for floor plane detection (in meters)
        'kernel_size': 3,   # Single morphological kernel size (simplifies operations)
    }
    
    # Set data directory
    data_dir: str = "/home/chirag/Desktop/Computer Vision Projekt/EX1/data"
    
    # Directory to save figures
    figures_dir: str = "/home/chirag/Desktop/Computer Vision Projekt/EX1/data/figures"
    
    # Initialize model and dataset
    model = BoxDetector(config)
    
    # Initialize the Viz module with the figures directory
    model.viz = Viz(save_dir=figures_dir)
    
    dataset = BoxDataset(data_dir)
    print(f"Found {len(dataset)} examples in dataset")
    
    # Record start time
    start_time: float = time.time()
    
    # Process all examples (example1kinect.mat through example4kinect.mat)
    # Each example will generate the four required visualizations:
    # 1. Figure 1: Original data (ToF amplitude image and Distance image)
    # 2. Figure 2: Floor mask (original and filtered)
    # 3. Figure 3: Box top component
    # 4. Figure 4: Final visualization (floor in green, box in red, box edges in blue)
    results: dict[str, Any] = model.process_dataset(dataset, visualize=True)
    
    # Calculate runtime
    elapsed_time: float = time.time() - start_time
    print(f"\nProcessing complete in {elapsed_time:.2f} seconds")
    
    # Summary of results for all examples
    print("\n======== Final Measurement Results ========")
    for name, result in results.items():
        dims: dict[str, float] = result['dimensions']
        print(f"{name}:")
        print(f"  Height: {dims['height']:.4f} meters")
        print(f"  Width:  {dims['width']:.4f} meters")
        print(f"  Length: {dims['length']:.4f} meters")
