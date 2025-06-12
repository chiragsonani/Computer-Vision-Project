import os
import scipy.io
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import label, binary_dilation, binary_erosion
from mpl_toolkits.mplot3d import Axes3D
np.random.seed(42)

def load_data(filepath, index):
    data = scipy.io.loadmat(filepath)
    A = data.get(f'amplitudes{index}')
    D = data.get(f'distances{index}')
    PC = data.get(f'cloud{index}')
    return A, D, PC

def ransac_plane(points, threshold=0.01, max_iter=1000):
    best_model = None
    max_inliers = 0
    best_inlier_mask = None
    N = points.shape[0]

    for _ in range(max_iter):
        idx = np.random.choice(N, 3, replace=False)
        p1, p2, p3 = points[idx]
        normal = np.cross(p2 - p1, p3 - p1)                                      # normal vector (perpendicular direction) of the plane.
        if np.linalg.norm(normal) == 0:
            continue
        d = np.dot(normal, p1)
        distances = np.abs(np.dot(points, normal) - d) / np.linalg.norm(normal)  # distance from every point to the current plane.
        inliers = distances < threshold                                          # boolean mask True/False
        if np.sum(inliers) > max_inliers:
            max_inliers = np.sum(inliers)
            best_model = (normal, d)
            best_inlier_mask = inliers
    return best_model, best_inlier_mask

def create_mask(shape, indices):
    mask = np.zeros(shape, dtype=np.uint8)
    mask[indices[:, 0], indices[:, 1]] = 1
    return mask

def process_file(filepath, index, save_root):
    save_dir = os.path.join(save_root, f"example{index}kinect")
    os.makedirs(save_dir, exist_ok=True)

    print(f"\nProcessing: example{index}kinect.mat")
    A, D, PC = load_data(filepath, index)

    # Show Amplitude, Distance image and point cloud
    fig = plt.figure(figsize=(18, 5))
    ax1 = fig.add_subplot(1, 3, 1)
    ax1.imshow(A.squeeze(), cmap='gray')
    ax1.set_title('Amplitude Image')
    fig.colorbar(ax1.imshow(A.squeeze(), cmap='gray'), ax=ax1)

    ax2 = fig.add_subplot(1, 3, 2)
    ax2.imshow(D.squeeze(), cmap='inferno')
    ax2.set_title('Distance Image')
    fig.colorbar(ax2.imshow(D.squeeze(), cmap='inferno'), ax=ax2)

    ax3 = fig.add_subplot(1, 3, 3, projection='3d')
    z_valid = PC[:, :, 2] != 0
    y, x = np.where(z_valid)
    sampled_indices = np.random.choice(len(y), size=5000, replace=False)
    x_sampled = x[sampled_indices]
    y_sampled = y[sampled_indices]
    xyz = PC[y_sampled, x_sampled, :]
    ax3.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], c=xyz[:, 2], cmap='cool', s=1)
    ax3.set_title('Subsampled Point Cloud')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, '01_initial_visuals.png'))
    plt.close()

    # Floor detection
    z_valid = PC[:, :, 2] != 0
    y, x = np.where(z_valid)
    valid_points = PC[y, x]
    valid_indices = np.stack((y, x), axis=1)

    floor_model, inlier_mask = ransac_plane(valid_points, threshold=0.05)
    floor_indices = valid_indices[inlier_mask]
    floor_mask_raw = create_mask((PC.shape[0], PC.shape[1]), floor_indices)

    kernel = np.ones((3, 3), dtype=bool)
    floor_mask_dilated = binary_dilation(floor_mask_raw, structure=kernel)
    floor_mask = binary_erosion(floor_mask_dilated, structure=kernel).astype(np.uint8)

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(floor_mask_raw, cmap='gray')
    plt.title("Floor Mask")
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(floor_mask, cmap='gray')
    plt.title("Floor Mask (Applied Closing)")
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, '02_floor_masks.png'))
    plt.close()

    # Box detection
    non_floor_mask = np.logical_not(floor_mask)
    y_nf, x_nf = np.where(non_floor_mask)
    non_floor_points = PC[y_nf, x_nf]

    box_model, box_mask_inl = ransac_plane(non_floor_points, threshold=0.01)
    box_indices = np.stack((y_nf[box_mask_inl], x_nf[box_mask_inl]), axis=1)
    box_mask_raw = create_mask((PC.shape[0], PC.shape[1]), box_indices)

    labeled, _ = label(box_mask_raw)
    sizes = np.bincount(labeled.ravel())
    sizes[0] = 0
    max_label = np.argmax(sizes)
    box_top = (labeled == max_label).astype(np.uint8)

    plt.imshow(box_top, cmap='gray')
    plt.title("Box Top Component")
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, '03_box_top.png'))
    plt.close()

    # Dimensions
    ys, xs = np.where(box_top == 1)
    p1 = PC[ys.min(), xs.min()]
    p2 = PC[ys.min(), xs.max()]
    p3 = PC[ys.max(), xs.min()]
    p4 = PC[ys.max(), xs.max()]

    height = abs(box_model[1] - floor_model[1]) / np.linalg.norm(floor_model[0])
    length = np.linalg.norm(p1 - p2)
    width = np.linalg.norm(p1 - p3)

    vis = np.zeros((PC.shape[0], PC.shape[1], 3), dtype=np.uint8)
    vis[floor_mask == 1] = [0, 255, 0]
    vis[box_top == 1] = [255, 0, 0]

    plt.imshow(vis)
    plt.title(f"example{index}kinect.mat: Floor (green), Box Top (red)")

    #box dimensions as text overlay
    dim_text = f"H: {height:.2f} m\nL: {length:.2f} m\nW: {width:.2f} m"
    img_height = vis.shape[0]
    plt.text(10, img_height-10, dim_text, color='white', fontsize=12, bbox=dict(facecolor='black', alpha=0.6))

    plt.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, '04_final_result.png'))
    plt.close()

    print(f"Height: {height:.3f} m | Length: {length:.3f} m | Width: {width:.3f} m")

#Main Block
if __name__ == "__main__":
    base_path = "/home/chirag/Desktop/Computer Vision Projekt/EX1/data/"
    output_root = "/home/chirag/Desktop/Computer Vision Projekt/EX1/Figures"

    for i in range(1, 5):
        filename = f"{base_path}example{i}kinect.mat"
        process_file(filename, i, output_root)

