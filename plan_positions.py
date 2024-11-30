import cv2
import numpy as np 
import pickle 
import common 
import math
from joblib import Parallel, delayed

BLACK = common.BLACK
GRAY = common.GRAY
WHITE = common.WHITE

def parallel_raycast(points, map, visited, field_of_view=np.pi, max_range=5, resolution=0.05, num_rays=180):
    results = Parallel(n_jobs=-1)(
        delayed(raycast)(
            map, 
            orientation, 
            point, 
            visited, 
            field_of_view, 
            max_range, 
            resolution, 
            num_rays
        )
        for point, _, orientation in points
    )
    return results

def raycast(map, orientation, start, visited, field_of_view=np.pi, max_range=5, resolution=0.05, num_rays=180):
    max_distance = int(max_range / resolution)
    x_robot, y_robot = start[1], start[0]

    # Precompute all angles and distances
    angles = np.linspace(orientation - field_of_view / 2, orientation + field_of_view / 2, num_rays)
    distances = np.arange(1, max_distance + 1)

    # Compute all potential (x, y) coordinates for the rays
    rays_x = (x_robot + np.outer(distances, np.cos(angles))).astype(int)
    rays_y = (y_robot + np.outer(distances, np.sin(angles))).astype(int)

    # Clip coordinates to ensure they are within bounds
    rays_x = np.clip(rays_x, 0, map.shape[1] - 1)
    rays_y = np.clip(rays_y, 0, map.shape[0] - 1)

    new_visible_cells = 0
    for i in range(num_rays):
        for j in range(max_distance):
            x, y = rays_x[j, i], rays_y[j, i]

            if map[y, x] != WHITE:
                break 
            
            if visited[y, x]:
                continue

            new_visible_cells += 1

    return new_visible_cells

def calculate_orientation(last_position, current_position):
    x_last, y_last = last_position[1],last_position[0]
    x_current, y_current = current_position[1], current_position[0]
    
    delta_x = x_current - x_last
    delta_y = y_current - y_last
    
    orientation = math.atan2(delta_y, delta_x)
    
    return orientation 

def mark_visited_half_circle(visited, robot_pos, orientation, map, max_range=5):
    max_range = int(max_range/0.05)
    x_robot, y_robot = robot_pos[1], robot_pos[0]
    map_height, map_width = visited.shape
    
    half_circle = []
    for angle in np.linspace(orientation - np.pi/2, orientation + np.pi/2, num=180): 
        for r in range(1, max_range + 1):
            x = int(x_robot + r * np.cos(angle))
            y = int(y_robot + r * np.sin(angle))
            
            if map[y,x] != WHITE:
                break
            if 0 <= x < map_width and 0 <= y < map_height:
                half_circle.append([y,x])
                visited[y, x] = True  # Mark as visited
    
    return visited, half_circle

def mark_visited_in_circle(map, start, visited, max_range=5, resolution=0.05):
    max_distance = int(max_range / resolution)
    x_robot, y_robot = start[1], start[0]

    # Precompute all angles and distances
    angles = np.linspace(0, 2 * np.pi, num=360)
    distances = np.arange(1, max_distance + 1)

    # Compute all coordinates for the circle
    circle_x = (x_robot + np.outer(distances, np.cos(angles))).astype(int)
    circle_y = (y_robot + np.outer(distances, np.sin(angles))).astype(int)

    # Clip coordinates
    circle_x = np.clip(circle_x, 0, map.shape[1] - 1)
    circle_y = np.clip(circle_y, 0, map.shape[0] - 1)

    full_circle = []
    for i in range(360):
        for j in range(max_distance):
            x, y = circle_x[j, i], circle_y[j, i]
            if map[y, x] != WHITE:
                break
            if not visited[y, x]:
                visited[y, x] = True
                full_circle.append([y, x])

    return visited, full_circle

def is_black_gray_pixel_nearby(map, x, y, radius):
    x_min = max(0, x-radius)
    x_max = min(map.shape[1] - 1, x+radius)
    y_min = max(0, y-radius)
    y_max = min(map.shape[0] - 1, y+radius)

    return np.any((map[y_min:y_max + 1, x_min:x_max + 1] == BLACK) | (map[y_min:y_max + 1, x_min:x_max + 1] == GRAY))

def possible_next_best_views(map, start, max_range=1, resolution=0.05):
    possible_next_best_views = []
    max_range = int(max_range/resolution)
    for angle in range(0, 360, 6): # Consider 60 new next postions
        rad = np.radians(angle)
        
        x = int(start[1] + max_range * np.cos(rad))
        y = int(start[0] + max_range * np.sin(rad))
        
        if 0 <= x < map.shape[1] and 0 <= y < map.shape[0] and map[y,x] == WHITE and not is_black_gray_pixel_nearby(map,x,y, 14): # And check if there are any black points in radius of 30 cm?
           possible_next_best_views.append([y,x])
                       
    return possible_next_best_views     

def cluster_by_distance(start, possible_next_bests, directions, current_angle_of_the_robot, max_range=1, resolution=0.05):
    direction_clusters = [[] for _ in range(directions)]
    angle_increment = 2 * np.pi / directions
    current_angle_of_the_robot %= 2 * np.pi

    # Precompute direction points
    directions_angles = current_angle_of_the_robot + np.arange(directions) * angle_increment
    direction_points = np.array([
        [
            start[0] + (max_range / resolution) * np.sin(angle),
            start[1] + (max_range / resolution) * np.cos(angle)
        ]
        for angle in directions_angles
    ])

    # Convert points to array for broadcasting
    points = np.array(possible_next_bests)

    # Compute distances using broadcasting
    distances = np.linalg.norm(points[:, None] - direction_points, axis=2)
    nearest_direction = np.argmin(distances, axis=1)

    # Assign points to clusters
    for i, point in enumerate(possible_next_bests):
        direction_clusters[nearest_direction[i]].append(point)

    return direction_clusters

def reevaluate(map, points, visited):
   
    visible_cells = parallel_raycast(points, map, visited)
    
    updated_worker = []
    for i, (point, _, orientation) in enumerate(points):
        if visible_cells[i] > 0:
            updated_worker.append([point, visible_cells[i], orientation])
    return updated_worker    

def next_best_view_planner(map, start, num_directions=4, visualize=False):
    if map[start[0], start[1]] != WHITE:
        print("Starting point is not a white point. Provide a valid starting point!")
        return []

    visited_cells = np.zeros(map.shape, dtype=bool)
    visited_cells[(map == BLACK) | (map == GRAY)] = True

    unvisited_threshold = int(0.03 * np.sum(~visited_cells))

    planned_path = [start]
    reevaluation_candidates = []
    is_initial_step = True

    while np.sum(~visited_cells) > unvisited_threshold:
        print("Unvisited pixels remaining:", np.sum(~visited_cells))

        if is_initial_step:     
            visited_cells, visible_area = mark_visited_in_circle(map, start, visited_cells)
            candidate_positions = possible_next_best_views(map, start)
            direction_clusters = cluster_by_distance(start, candidate_positions, num_directions, 0)
            is_initial_step = False   
        else:  
            visited_cells, visible_area = mark_visited_half_circle(visited_cells, start, orientation, map)
            candidate_positions = possible_next_best_views(map, start)
            direction_clusters = cluster_by_distance(start, candidate_positions, num_directions, orientation)
        
        if visualize:
            common.visualize_points(map, planned_path, view=visible_area)

        potential_positions = []
        
        for direction_cluster in direction_clusters:
            if not direction_cluster:
                continue  # Skip empty clusters

            # Parallelize raycasting for all points in the current cluster
            raycast_results = Parallel(n_jobs=-1)(
                delayed(raycast)(
                    map, 
                    calculate_orientation(start, candidate_point), 
                    candidate_point, 
                    visited_cells
                ) for candidate_point in direction_cluster
            )

            # Find the best candidate in this cluster
            best_candidate_index = np.argmax(raycast_results)
            max_visible_cells = raycast_results[best_candidate_index]

            if max_visible_cells > 0:
                best_candidate = direction_cluster[best_candidate_index]
                candidate_orientation = calculate_orientation(start, best_candidate)
                potential_positions.append([best_candidate, max_visible_cells, candidate_orientation])
            
        valid_candidates = [candidate for candidate in potential_positions if candidate[1] > 0]

        if len(valid_candidates) > 0:
            best_candidate = max(valid_candidates, key=lambda x: x[1])

            if best_candidate[1] <= 50 and len(reevaluation_candidates) > 0:
                reevaluation_candidates = reevaluate(map, reevaluation_candidates, visited_cells)
                top_reevaluation_candidate = max(reevaluation_candidates, key=lambda x: x[1])

                if best_candidate[1] < top_reevaluation_candidate[1]:
                    reevaluation_candidates.remove(top_reevaluation_candidate)
                    start = top_reevaluation_candidate[0]
                    orientation = top_reevaluation_candidate[2]
                    planned_path.append(start)
                    print("New position added from reevaluation:", start)
                else:
                    break    
            else:       
                valid_candidates.remove(best_candidate)
                reevaluation_candidates += valid_candidates
                start = best_candidate[0]
                orientation = best_candidate[2]
                planned_path.append(start)
                print("New position added:", start)

        elif len(reevaluation_candidates) > 0:            
            reevaluation_candidates = reevaluate(map, reevaluation_candidates, visited_cells)
            top_reevaluation_candidate = max(reevaluation_candidates, key=lambda x: x[1])   
            reevaluation_candidates.remove(top_reevaluation_candidate)
            start = top_reevaluation_candidate[0]
            orientation = top_reevaluation_candidate[2]
            planned_path.append(start)
            print("New position added from reevaluation:", start)

        else:
            print("Stopping: No valid candidates or reevaluation points remaining.")
            break    

    return planned_path

if __name__ == "__main__":
    map = cv2.imread('maps/maps-xiaoliang/fused_map_updated.pgm', cv2.IMREAD_GRAYSCALE)
    cv2.imshow('Image', map)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # xiaolang map
    init_loc_x = 190 # extract from the acml
    init_loc_y = 250

    # small map
    # init_loc_x = 220 # extract from the acml
    # init_loc_y = 145
    # other 
    # init_loc_x = 470 # extract from the acml
    # init_loc_y = 600

    with common.Timer():
        waypoints = next_best_view_planner(map, [init_loc_y, init_loc_x])
    
    common.calculate_tour_size(waypoints, verbose=True)

    with open("maps/maps-xiaoliang/data/latest_positions_26_11_2024_next.pkl", 'wb') as f: 
         pickle.dump(waypoints, f)

    common.visualize_points(map, waypoints)