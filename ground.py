import pyrealsense2 as rs
import numpy as np
import open3d as o3d

# Configure depth stream
pipeline = rs.pipeline()
config = rs.config()

config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

# Start streaming
pipeline.start(config)


# # Initialize Open3D visualizer
# vis = o3d.visualization.Visualizer()
# vis.create_window("Segmented Ground and Objects")

ground_cloud = o3d.geometry.PointCloud()
objects_cloud = o3d.geometry.PointCloud()
# vis.add_geometry(ground_cloud)
# vis.add_geometry(objects_cloud)

try:
    while True:
        # Wait for a coherent depth frame
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        if not depth_frame:
            continue

        # Convert depth image to numpy array
        depth_image = np.asanyarray(depth_frame.get_data())
        # Apply threshold filter to show only depth values less than 3 meters (3000 mm)
        max_distance = 5000  # 5 meters in millimeterss
        depth_image[depth_image > max_distance] = 0

        # Convert depth image to point cloud
        intrinsics = depth_frame.profile.as_video_stream_profile().intrinsics
        pcd = o3d.geometry.PointCloud.create_from_depth_image(
            o3d.geometry.Image(depth_image),
            o3d.camera.PinholeCameraIntrinsic(
                intrinsics.width, intrinsics.height,
                intrinsics.fx, intrinsics.fy,
                intrinsics.ppx, intrinsics.ppy
            )
        )
        # pcd_downsampled = pcd.voxel_down_sample(voxel_size=0.01)
        pcd.transform([[1,0,0,0],[0,-1,0,0],[0,0,-1,0],[0,0,0,1]])
        
    # Segment plane using RANSAC
        plane_model, inliers = pcd.segment_plane(distance_threshold=0.05, ransac_n=3, num_iterations=1000)

        # Extract inlier and outlier point clouds
        inlier_cloud = pcd.select_by_index(inliers)
        outlier_cloud = pcd.select_by_index(inliers, invert=True)

        # Color the segmented ground plane differently
        inlier_cloud.paint_uniform_color([1.0, 0, 0])  # Red
        outlier_cloud.paint_uniform_color([0, 1.0, 0])  # Green

        # Update the visualizer with the new point clouds
        ground_cloud.points = inlier_cloud.points
        ground_cloud.colors = inlier_cloud.colors
        objects_cloud.points = outlier_cloud.points
        objects_cloud.colors = outlier_cloud.colors

        # vis.update_geometry(ground_cloud)
        # vis.update_geometry(objects_cloud)
        # vis.poll_events()
        # vis.update_renderer()
        # # Downsample the point cloud for better performance
        # pcd_downsampled = pcd.voxel_down_sample(voxel_size=0.05)
        # Visualize the point cloud
        o3d.visualization.draw_geometries([ground_cloud, objects_cloud])
        # o3d.visualization.draw_geometries([objects_cloud])

finally:
    # Stop streaming
    pipeline.stop()
