import cv2
import numpy as np 
import heapq
from concurrent.futures import ProcessPoolExecutor
import pickle 
import os
import common

# values obtained from the pgm map
BLACK = common.BLACK
GRAY = common.GRAY
WHITE = common.WHITE

def astar(start, goal, image):
    print("Start point:" +  str(start) + ", Goal point:" + str(goal))
    rows, cols = image.shape
    open_set = []
    heapq.heappush(open_set, (0 + common.minkowski_distance(start, goal), 0, start))
    came_from = {}
    g_score = {start: 0}
    f_score = {start: common.minkowski_distance(start, goal)}
    
    while open_set:
        _, current_g, current = heapq.heappop(open_set)
        
        if current == goal:
            return g_score[goal]
        
        step = int(1)
        for d in [(-step, 0), (step, 0), (0, -step), (0, step)]:
            neighbor = (current[0] + d[0], current[1] + d[1])
            if 0 <= neighbor[0] < rows and 0 <= neighbor[1] < cols:
                if image[neighbor] == WHITE:
                    tentative_g_score = g_score[current] + 1
                    if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                        came_from[neighbor] = current
                        g_score[neighbor] = tentative_g_score
                        f_score[neighbor] = tentative_g_score + common.minkowski_distance(neighbor, goal)
                        heapq.heappush(open_set, (f_score[neighbor], tentative_g_score, neighbor))
    
    return float('inf')

def compute_distance(i, j, waypoints, map):
    start = tuple(waypoints[i])
    end = tuple(waypoints[j])
    return (i, j, astar(start, end, map))

def compute_distance2(i, j, waypoints, map):
    start = tuple(waypoints[i])
    end = tuple(waypoints[j])
    
    def bresenham_line(x0, y0, x1, y1):
        points = []
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx - dy

        while True:
            points.append((x0, y0))
            if x0 == x1 and y0 == y1:
                break
            e2 = err * 2
            if e2 > -dy:
                err -= dy
                x0 += sx
            if e2 < dx:
                err += dx
                y0 += sy
        return points

    line_points = bresenham_line(start[1], start[0], end[1], end[0])
    obstacle_detected = any(map[point[1], point[0]] != WHITE for point in line_points)
    
    if obstacle_detected:
        distance = common.minkowski_distance(end, start, p=1)
    else: 
        distance = common.minkowski_distance(end, start)    
    return (i, j, distance)

def worker(pair, waypoints, map):
    return compute_distance(pair[0], pair[1], waypoints, map)

def create_distance_matrix(waypoints, map):
    n = len(waypoints)
    distance_matrix = np.zeros((n, n))

    # Create pairs for distance computation
    pairs = [(i, j) for i in range(n) for j in range(i+1, n)]
    
    # Use ProcessPoolExecutor for parallel processing in chunks
    chunk_size = max(1, len(pairs) // (4 * os.cpu_count()))  # Adjust chunk size for your system
    with ProcessPoolExecutor() as executor:
        results = executor.map(worker, pairs, [waypoints]*len(pairs), [map]*len(pairs), chunksize=chunk_size)

    # Fill the distance matrix with results
    for i, j, distance in results:
        distance_matrix[i, j] = distance
        distance_matrix[j, i] = distance  # Symmetric matrix => compute only the upper triangle

    with open("distance_matrix.pkl", "wb") as f:
        pickle.dump(distance_matrix, f)
    print("Distance matrix saved as 'distance_matrix.pkl'.")

    return distance_matrix    

# an approximation of the TSP solution
def two_opt_swap(tour, i, j):
    return tour[:i] + tour[i:j+1][::-1] + tour[j+1:]

def calculate_tour_length(tour, dist_matrix):
    return sum(dist_matrix[tour[i], tour[i+1]] for i in range(len(tour) - 1)) + dist_matrix[tour[-1], tour[0]]

def lin_kernighan(points, map):
    print("Lin-Kernighan Algorithm started.")
    
    dist_matrix = create_distance_matrix(points, map)
    
    improvement = True
    while improvement:
        improvement = False
        for i in range(1, len(points) - 1):
            for j in range(i + 1, len(points)):
                new_tour = two_opt_swap(tour, i, j)
                if calculate_tour_length(new_tour, dist_matrix) < calculate_tour_length(tour, dist_matrix):
                    tour = new_tour
                    improvement = True
    
    return tour

if __name__ == '__main__':
    map = cv2.imread('maps/maps-xiaoliang/fused_map_updated.pgm', cv2.IMREAD_GRAYSCALE)
    
    with open("maps/maps-xiaoliang/data/latest_positions_26_11_2024_next.pkl", 'rb') as file:
        waypoints = pickle.load(file)
    print(len(waypoints))

    with common.Timer():
        visit_order = lin_kernighan(waypoints, map)
        sorted_waypoints = [waypoints[i] for i in visit_order]
    print(sorted_waypoints)

    with open("path_astar_points.pkl", 'wb') as f: 
         pickle.dump(sorted_waypoints, f)

    with open("path_astar_order.pkl", 'wb') as f: 
         pickle.dump(visit_order, f)

    common.visualize_path(map, sorted_waypoints)