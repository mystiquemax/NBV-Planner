import open3d as o3d
import numpy as np
import os
import glob
import re
from scipy.spatial.transform import Rotation as R

def load_pose(filename):
    print(f"Loading pose from {filename}")
    with open(filename, 'r') as file:
        data = file.read()

    position = re.search(r"Position: x=([\d\.\-]+), y=([\d\.\-]+)", data)
    orientation = re.search(r"Orientation: x: ([\d\.\-]+)\ny: ([\d\.\-]+)\nz: ([\d\.\-]+)\nw: ([\d\.\-]+)", data)

    if not position or not orientation:
        print(f"Error: Failed to extract pose data from {filename}")
        return np.eye(4) 

    x_pos = float(position.group(1))
    y_pos = float(position.group(2))

    x_orient = float(orientation.group(1))
    y_orient = float(orientation.group(2))
    z_orient = float(orientation.group(3))
    w_orient = float(orientation.group(4))

    q_g = np.array([x_orient, y_orient, z_orient, w_orient])
    
    roll, pitch, yaw = R.from_quat(q_g).as_euler('xyz', degrees=True)

    q_g = R.from_euler('xyz', [np.radians(roll-90), np.radians(pitch),  np.radians(yaw-90)]).as_quat()
    
    rotation = R.from_quat(q_g)  
    rotation_matrix = rotation.as_matrix()
    
    transformation_matrix = np.eye(4) 
    transformation_matrix[:3, :3] = rotation_matrix  
    transformation_matrix[0, 3] = x_pos 
    transformation_matrix[1, 3] = y_pos
    return transformation_matrix

def colored_icp(source, target, transform, dist=0.04):
    print("Applying ICP...")

    try:
        source = source.voxel_down_sample(dist)
        target = target.voxel_down_sample(dist)

        if len(source.points) == 0 or len(target.points) == 0:
            print("Error: One of the point clouds is empty after downsampling.")
            return transform

        source.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=dist * 2, max_nn=50))
        target.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=dist * 2, max_nn=50))

        reg_icp = o3d.pipelines.registration.registration_colored_icp(
            source,
            target,
            dist,
            transform)
        
        print(f"ICP transformation:\n{reg_icp.transformation}")
        return reg_icp.transformation

    except Exception as e:
        print(f"ICP failed: {e}")
        return transform 

def merge_point_clouds(scan_folder, icp=False):
    pcd_combined = o3d.geometry.PointCloud()
    
    pcd_files = sorted(glob.glob(os.path.join(scan_folder, "*.ply")),
                        key=lambda x: int(re.search(r'(\d+)', os.path.basename(x)).group(1)))

    pose_files = [os.path.splitext(pcd)[0] + "_pose.txt" for pcd in pcd_files]
    counter = 0
    
    prev_pcd = None
    
    for pcd_file, pose_file in zip(pcd_files, pose_files):
        print(f"Processing file: {pcd_file}")
        pcd = o3d.io.read_point_cloud(pcd_file)
        
        if pcd.is_empty():
            print(f"Warning: Point cloud {pcd_file} is empty. Skipping.")
            continue

        cloud_transform = load_pose(pose_file)
        print(f"Loaded pose transformation:\n{cloud_transform}")
        pcd.transform(cloud_transform)
        
        if icp:
            if prev_pcd is not None:
                transform = colored_icp(pcd, prev_pcd, np.eye(4))
                pcd.transform(transform)

        pcd_combined += pcd
        prev_pcd = pcd
        counter += 1

    return pcd_combined

if __name__ == "__main__":
    scan_folder = "scans/27_11_2024"
    combined_pcd = merge_point_clouds(scan_folder)
    
    output_filename = "combined_point_cloud_latest.ply"
    o3d.io.write_point_cloud(output_filename, combined_pcd)
    print(f"Combined point cloud saved as '{output_filename}'")
    
    axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0, 0, 0])

    o3d.visualization.draw_geometries([combined_pcd, axis])