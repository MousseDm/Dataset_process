import numpy as np
import matplotlib.pyplot as plt
import os

def load_and_align_frames(data_dir, num_frames=100):
    """
    Load up to num_frames PLY files from data_dir, normalize each frame so its
    lowest Y-value is at 0, and return a list of point arrays.

    Args:
        data_dir (str): Directory containing frame PLY files named 000000.ply, etc.
        num_frames (int): Number of frames to attempt to load.

    Returns:
        List[np.ndarray]: Each entry is an (N,3) array of XYZ points for a frame.
    """
    frame_points = []
    for i in range(num_frames):
        path = os.path.join(data_dir, f"{i:06d}.ply")
        if not os.path.exists(path):
            print(f"[WARN] skipping missing file: {path}")
            continue

        pts = []
        with open(path, 'r') as f:
            in_header = True
            # Skip PLY header lines until "end_header"
            for line in f:
                if in_header:
                    if line.strip() == "end_header":
                        in_header = False
                    continue
                # Parse X, Y, Z from each data line
                x, y, z = map(float, line.split()[:3])
                pts.append([x, y, z])

        pts = np.array(pts)
        # Shift all points so that the minimum Y-coordinate becomes 0
        pts[:, 1] -= pts[:, 1].min()
        frame_points.append(pts)

    assert frame_points, "No PLY files loaded—please check data_dir!"
    return frame_points

def find_ground_peak(frame_points, max_h=0.5, bins=100):
    """
    Merge all frames into one giant point set, histogram the Y-values,
    and find the peak bin center within Y ≤ max_h.

    Args:
        frame_points (List[np.ndarray]): List of (N,3) point arrays.
        max_h (float): Upper bound on Y to consider for the “ground” peak.
        bins (int): Number of histogram bins.

    Returns:
        ground_peak_h (float): Y-value at the most frequent bin ≤ max_h.
        edges (np.ndarray): Bin edge array from the histogram.
    """
    all_pts = np.vstack(frame_points)
    counts, edges = np.histogram(all_pts[:, 1], bins=bins)
    centers = 0.5 * (edges[:-1] + edges[1:])
    # Mask out bins above max_h
    mask = centers <= max_h
    peak_idx = np.argmax(counts[mask])
    ground_peak_h = centers[mask][peak_idx]
    print(f"[INFO] ground_peak_h = {ground_peak_h:.3f}")
    return ground_peak_h, edges

def adjust_frames(frame_points, ground_peak_h):
    """
    For each frame, if its lowest point is below the ground peak,
    shift the entire frame upward so its min Y equals ground_peak_h.

    Args:
        frame_points (List[np.ndarray]): List of point arrays to modify in-place.
        ground_peak_h (float): Target minimum Y for any frame.
    """
    for pts in frame_points:
        frame_min = pts[:, 1].min()
        if frame_min < ground_peak_h:
            shift = ground_peak_h - frame_min
            pts[:, 1] += shift

def compute_ratios(points, edges, grid=0.05):
    """
    For each vertical bin defined by edges, compute the ratio of occupied grid
    cells in that layer to the total cells of its enclosing square.

    Args:
        points (np.ndarray): All points concatenated, shape (M,3).
        edges (np.ndarray): Bin edges along Y from the histogram.
        grid (float): Grid cell size for occupancy.

    Returns:
        np.ndarray: Array of area-occupation ratios per Y-bin.
    """
    ratios = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        # Select points in this height slice
        layer = points[(points[:,1] >= lo) & (points[:,1] < hi)]
        if layer.size == 0:
            ratios.append(0.0)
            continue

        # Project to XZ plane
        xz = layer[:, [0, 2]]
        xmin, xmax = xz[:,0].min(), xz[:,0].max()
        zmin, zmax = xz[:,1].min(), xz[:,1].max()
        # Determine side length of the minimal enclosing square
        side = max(xmax - xmin, zmax - zmin)
        cx, cz = (xmin + xmax) / 2, (zmin + zmax) / 2
        X0, X1 = cx - side/2, cx + side/2
        Z0, Z1 = cz - side/2, cz + side/2

        # Compute integer grid indices for occupied cells
        xi = np.floor((xz[:,0] - X0) / grid).astype(int)
        zi = np.floor((xz[:,1] - Z0) / grid).astype(int)
        occ = len(set(zip(xi, zi)))
        tot = int(np.ceil(side / grid)) ** 2
        ratios.append(occ / tot if tot > 0 else 0.0)

    return np.array(ratios)

def plot_ratios(centers, ratios, ground_peak_h):
    """
    Plot a bar chart of area-occupation ratio vs. height.

    Args:
        centers (np.ndarray): Bin centers along Y.
        ratios (np.ndarray): Occupation ratios for each bin.
        ground_peak_h (float): Y-value to mark with a vertical line.
    """
    width = centers[1] - centers[0]
    plt.figure(figsize=(10, 5))
    plt.bar(centers, ratios, width=width, edgecolor='black')
    plt.axvline(ground_peak_h, color='red', linestyle='--',
                label=f'Ground peak @ {ground_peak_h:.2f}')
    plt.xlabel('Height (Y)')
    plt.ylabel('Occupied Area Ratio')
    plt.title('Area Occupation Ratio vs. Height')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    data_dir = "C:/Users/34328/Desktop/processed_hypersim/ai_001_001/cam_00"

    # 1) Load all frames and normalize their bottoms to Y=0
    frame_points = load_and_align_frames(data_dir, num_frames=100)

    # 2) Merge and find the ground peak within Y ≤ 0.5
    ground_peak_h, edges = find_ground_peak(frame_points, max_h=0.5, bins=100)

    # 3) Shift any frame whose min-Y is below the peak up to ground_peak_h
    adjust_frames(frame_points, ground_peak_h)

    # 4) Concatenate all points again, compute area ratios per height bin
    all_pts2 = np.vstack(frame_points)
    centers2 = 0.5 * (edges[:-1] + edges[1:])
    ratios2 = compute_ratios(all_pts2, edges, grid=0.05)

    # 5) Plot the final area-occupation ratio histogram
    plot_ratios(centers2, ratios2, ground_peak_h)
