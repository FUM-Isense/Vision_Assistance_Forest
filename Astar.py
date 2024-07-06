import pyrealsense2 as rs
import numpy as np
import open3d as o3d
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
import heapq
from collections import deque


def a_star(grid, start, goal):
    # Grid dimensions
    rows, cols = len(grid), len(grid[0])
    safe_distance = 20  # Minimum distance from obstacles

    # Precompute the distance to the nearest obstacle for each cell
    def compute_distances_to_obstacles(grid):
        distances = [[float('inf')] * cols for _ in range(rows)]
        queue = deque()

        # Enqueue all obstacles with a distance of 0
        for i in range(rows):
            for j in range(cols):
                if grid[i][j] != 0:
                    distances[i][j] = 0
                    queue.append((i, j))

        # Perform BFS to compute distances
        while queue:
            x, y = queue.popleft()
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < rows and 0 <= ny < cols:
                    if distances[nx][ny] > distances[x][y] + 1:
                        distances[nx][ny] = distances[x][y] + 1
                        queue.append((nx, ny))

        return distances

    distances_to_obstacles = compute_distances_to_obstacles(grid)

    # Define the heuristic function (prefer cells with higher y values)
    def heuristic(cell):
        x, y = cell
        gx, gy = goal
        return abs(x - gx) + abs(y - gy) - (gy - y)  # Manhattan distance with a preference for increasing y

    # Priority queue to store (cost, cell) pairs
    open_set = []
    heapq.heappush(open_set, (0, start))

    # Dictionaries to store the cost to reach each cell and the path
    g_costs = {start: 0}
    came_from = {start: None}

    while open_set:
        current_cost, current_cell = heapq.heappop(open_set)

        # If we reached the goal, reconstruct the path
        if current_cell == goal:
            path = []
            while current_cell:
                path.append(current_cell)
                current_cell = came_from[current_cell]
            return path[::-1]  # Return reversed path

        x, y = current_cell
        # Explore neighbors (up, down, left, right)
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            neighbor = (x + dx, y + dy)
            nx, ny = neighbor

            # Check if the neighbor is within bounds and is an empty cell (0)
            if 0 <= nx < rows and 0 <= ny < cols and grid[nx][ny] == 0:
                new_cost = g_costs[current_cell] + 1  # Assuming a cost of 1 for each step
                
                # Add additional cost if the cell is too close to an obstacle
                distance = distances_to_obstacles[nx][ny]
                if distance < safe_distance:
                    new_cost += (safe_distance - distance)

                if neighbor not in g_costs or new_cost < g_costs[neighbor]:
                    g_costs[neighbor] = new_cost
                    priority = new_cost + heuristic(neighbor)
                    heapq.heappush(open_set, (priority, neighbor))
                    came_from[neighbor] = current_cell

    # If there's no path, return None
    return None



# Configure depth stream
pipeline = rs.pipeline()
config = rs.config()

config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

# Start streaming
pipeline.start(config)


ground_cloud = o3d.geometry.PointCloud()
objects_cloud = o3d.geometry.PointCloud()

# # # Parameters for the occupancy grid
# grid_resolution = 0.1  # Resolution of the grid in meters
# grid_width = 5  # Width of the grid in meters
# grid_height = 5  # Height of the grid in meters

try:
    while True:
        # Wait for a coherent depth frame
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        if not depth_frame:
            continue

        # Convert depth image to numpy array
        depth_image = np.asanyarray(depth_frame.get_data())
        # Apply threshold filter to show only depth values less than 5 meters (5000 mm)
        max_distance = 5000  # 5 meters in millimeters
        depth_image[depth_image > max_distance] = 0

        # Convert depth image to point cloud
        intrinsics = depth_frame.profile.as_video_stream_profile().intrinsics
        pcd = o3d.geometry.PointCloud.create_from_depth_image(
            o3d.geometry.Image(depth_image),
            o3d.camera.PinholeCameraIntrinsic(
                intrinsics.width,
                intrinsics.height,
                intrinsics.fx,
                intrinsics.fy,
                intrinsics.ppx,
                intrinsics.ppy,
            ),
        )
        pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])

        # Segment plane using RANSAC
        plane_model, inliers = pcd.segment_plane(
            distance_threshold=0.05, ransac_n=3, num_iterations=1000
        )

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

        occupancy_grid = np.zeros((500, 400), dtype=np.int8)

        # # Iterate over the object points and mark the grid cells as occupied
        for point in np.asarray(outlier_cloud.points):
            x, y, z = point
            try:
                occupancy_grid[int((-z) * 100)][int((x + 2) * 100)] = 1
            except:
                continue

        # Perform DBSCAN clustering
        # Find the coordinates of occupied cells
        occupied_cells = np.argwhere(occupancy_grid == 1)
        if len(occupied_cells) > 0:
            dbscan = DBSCAN(eps=5, min_samples=50).fit(occupied_cells)
            labels = dbscan.labels_

            # Create an array to visualize the clusters
            cluster_grid = np.zeros_like(occupancy_grid, dtype=int)
            for idx, label in enumerate(labels):
                if label != -1:  # Ignore noise points
                    cluster_grid[occupied_cells[idx][0], occupied_cells[idx][1]] = label + 1

            
            for i in range(cluster_grid.shape[0]):
                for j in range(cluster_grid.shape[1]):
                    if cluster_grid[i][j] > 0: # check if it belogns to a cluster
                        try:
                            cluster_grid[i-5:i+5, j-5:j+5] = -1
                        except:
                            continue

            path = a_star(cluster_grid, (0, 200), (499, 200))
            if path:
                for (x, y) in path:
                    cluster_grid[x][y] = max(labels)+1

                
            # Display the clusters
            plt.imshow(cluster_grid, origin='lower', cmap='tab20')
            plt.title('DBSCAN Clusters')
            plt.xlabel('X (grid cells)')
            plt.ylabel('Y (grid cells)')
            plt.colorbar(label='Cluster ID')
            plt.show(block=False)  # Show the plot without blocking
            plt.pause(1)           # Pause for 5 seconds
            plt.close() 

        # Visualize the point cloud
        # o3d.visualization.draw_geometries([ground_cloud, objects_cloud])

finally:
    # Stop streaming
    pipeline.stop()
