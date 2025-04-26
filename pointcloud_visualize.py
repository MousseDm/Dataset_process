import trimesh

# Path to your PLY file (replace with your actual file path)
ply_path = "C:/Users/34328/Desktop/processed_hypersim/ai_001_001/cam_00/000000.ply"

# Load the point cloud from the PLY file
# trimesh.load will automatically detect the file type and construct
# a PointCloud object when given a .ply
cloud = trimesh.load(ply_path)

# Open an interactive viewer window where you can rotate, pan, and zoom
# the point cloud to inspect it from different angles
cloud.show()
