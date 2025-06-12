# Computer-Vision-Project :: EX1

Tasks Completed
📂 Loaded Amplitude, Distance, and Point Cloud Data from MATLAB .mat files using scipy.io.loadmat.

🖼️ Visualized:

Amplitude images

Distance images

3D point clouds (subsampled)

🛠️ Implemented Custom RANSAC algorithm (no external libraries used) to:

Detect the dominant plane corresponding to the floor

Detect a second dominant plane corresponding to the top of the box

🧼 Filtered the floor mask using morphological operations (dilation + erosion) to reduce noise and fill gaps.

🧱 Isolated the box top:

Created a mask of the box’s top plane

Labeled and extracted the largest connected component to ensure accurate dimension detection

📐 Estimated box dimensions:

Height from the distance between floor and top planes

Length and width from the 3D coordinates of corners of the box top

🌈 Created final visualizations:

Overlaid masks showing floor (green) and box top (red)

Annotated final images with box dimensions

💾 Saved all output images in organized per-example folders under a output_images/ directory
