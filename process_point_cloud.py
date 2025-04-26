import os
import numpy as np
from PIL import Image

def write_ply(filename, points, colors):
    """
    Write a point cloud to a PLY file in ASCII format.

    Args:
        filename (str): Path to the output PLY file.
        points (ndarray): Array of shape (N, 3) containing 3D point coordinates.
        colors (ndarray): Array of shape (N, 3) containing RGB colors (0-255) for each point.
    """
    with open(filename, 'w') as f:
        # PLY header
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(points)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("end_header\n")
        # Write each point with its color
        for p, c in zip(points, colors):
            f.write(f"{p[0]} {p[1]} {p[2]} {int(c[0])} {int(c[1])} {int(c[2])}\n")

# Directory containing the processed Hypersim data
data_dir = "C:/Users/34328/Desktop/processed_hypersim/ai_001_001/cam_00"

# Collect frame IDs by listing all raw color PNG files
frame_ids = sorted([
    f.split("_")[0] for f in os.listdir(data_dir)
    if f.endswith("_rawcolor.png")
])

for frame_id in frame_ids:
    print(f"Processing {frame_id}...")

    # Load RGB image (uint8) and depth map (float) for the frame
    rgb = np.array(Image.open(f"{data_dir}/{frame_id}_rawcolor.png")).astype(np.uint8)
    depth = np.load(f"{data_dir}/{frame_id}_depth.npy")

    # Load camera intrinsics and pose from the .npz file
    cam = np.load(f"{data_dir}/{frame_id}_cam.npz")
    K = cam["intrinsics"]   # 3×3 intrinsic matrix
    pose = cam["pose"]      # 4×4 camera-to-world transformation

    H, W = depth.shape
    fx, fy = K[0, 0], K[1, 1]  # focal lengths
    cx, cy = K[0, 2], K[1, 2]  # principal point

    # Create a meshgrid of pixel coordinates
    i, j = np.meshgrid(np.arange(W), np.arange(H), indexing="xy")
    z = depth
    # Back-project pixels to camera coordinates
    x = (i - cx) * z / fx
    y = (j - cy) * z / fy
    points_cam = np.stack((x, y, z), axis=2).reshape(-1, 3)

    # Filter out invalid points where depth == 0
    valid = z.reshape(-1) > 0
    points_cam = points_cam[valid]
    colors = rgb.reshape(-1, 3)[valid]

    # Transform points from camera space to world space
    R = pose[:3, :3]
    t = pose[:3, 3]
    points_world = (R @ points_cam.T).T + t

    # Write the world-coordinate point cloud to a PLY file
    ply_path = f"{data_dir}/{frame_id}.ply"
    write_ply(ply_path, points_world, colors)
    print(f"Saved: {ply_path}")
