import numpy as np
import cv2
import pickle
import time

BLACK = 0
GRAY = 205
WHITE = 254

# paths to maps and yaml files
max_range = 5 # m
pos_increment = 1 # m
fov = 2*np.pi
resolution = 0.05 # 1 pixel = 0.05 m

# Timer to mesure the performance of the code
class Timer:
    def __enter__(self):
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, *args):
        self.end_time = time.perf_counter()
        self.elapsed = self.end_time - self.start_time
        print(f"Elapsed time: {self.elapsed:.4f} seconds")

# Different distances for different cases
def minkowski_distance(a,b, p=2):
   a = np.array(a)
   b = np.array(b)
   return np.sum(np.abs(a-b) ** p) ** (1/p)

# Caluculate how many point a tour contains
def calculate_tour_size(waypoints, verbose=False):
    if len(waypoints) == 0:
      print("List of points is empty!")
      return
    if verbose:
       print("Tour total size: " + str(len(waypoints)) + " points.")
    return len(waypoints)

# Calculate how long in meters the tour is
def calculate_tour_length(path_to_visit_order, path_to_distance_matrix, waypoints=None, verbose=False):
    with open(path_to_distance_matrix, 'rb') as f:
        distance_matrix = pickle.load(f)

    distance_matrix = np.array(distance_matrix)
   
    if distance_matrix.shape[0] != distance_matrix.shape[1]:
        raise ValueError("The distance matrix must be square (same number of rows and columns).")
    
    if waypoints is None:
       with open(path_to_visit_order, 'rb') as f:
         waypoints = pickle.load(f)
    
    total_length = 0
    
    for i in range(len(waypoints) - 1):
        distance = distance_matrix[waypoints[i], waypoints[i + 1]]
        total_length += distance 
        if verbose:
            print(f"Distance from {waypoints[i]} to {waypoints[i + 1]}: {distance}")

    if verbose:        
       print("Tour total length: " + str(total_length * resolution) + " meter.")

    return total_length * resolution   

# Visualize generated points on the map
def visualize_points(map, waypoints, safe=False, view=None):
   if len(waypoints) == 0:
      print("List of points is empty!")
      return
   color_image = cv2.cvtColor(map, cv2.COLOR_GRAY2BGR)
   for y,x in waypoints:
      if 0 <= y < color_image.shape[0] and 0 <= x < color_image.shape[1]:
         color_image[y, x] = [0, 0, 255] # mark points with red

   first_y, first_x = waypoints[0][0], waypoints[0][1]
   last_y, last_x = waypoints[-1][0], waypoints[-1][1]
   cv2.circle(color_image, (first_x, first_y), 4, (0,255,0), -1) # the green point is start
   cv2.circle(color_image, (last_x, last_y) , 4, (255,0,0), -1) # the blue point is end

   if view is not None:
      for y,x in view:
         if 0 <= y < color_image.shape[0] and 0 <= x < color_image.shape[1]:
            color_image[y, x] = [255, 0, 255] # mark points with purple

   cv2.namedWindow('Image', cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO | cv2.WINDOW_GUI_EXPANDED)
   cv2.imshow('Image', color_image)
   cv2.waitKey(0)
   cv2.destroyAllWindows()

   if safe:   
      cv2.imwrite('points.jpg', color_image)

# Visualize the path the robot will take
def visualize_path(map, waypoints, safe=False):
   if len(waypoints) == 0:
      print("List of points is empty!")
      return
   color_image = cv2.cvtColor(map, cv2.COLOR_GRAY2BGR)

   first_y, first_x = waypoints[0][0], waypoints[0][1]
   last_y, last_x = waypoints[-1][0], waypoints[-1][1]
   cv2.circle(color_image, (first_x, first_y), 4, (0,255,0), -1) # the green point is start
   cv2.circle(color_image, (last_x, last_y) , 4, (255,0,0), -1) # the blue point is end

   for i in range(len(waypoints) - 1):
      y_start, x_start = waypoints[i]
      y_end, x_end = waypoints[i + 1]
      cv2.arrowedLine(color_image, (x_start,y_start), (x_end, y_end), (0, 0, 255), 1, tipLength=0.01)

   
   cv2.namedWindow('Image', cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO | cv2.WINDOW_GUI_EXPANDED)
   cv2.imshow('Image', color_image)
   cv2.waitKey(0)
   cv2.destroyAllWindows()

   if safe:   
      cv2.imwrite('points.jpg', color_image)

if __name__ == '__main__':
   calculate_tour_length("paths/astar/path_astar_order.pkl", "paths/astar/distance_matrix.pkl" , verbose=True)
   calculate_tour_length("paths/euclidain_manhat_path/path_euclidian_manh_order.pkl", "paths/astar/distance_matrix.pkl" , verbose=True)
   calculate_tour_length("", "paths/astar/distance_matrix.pkl" ,waypoints=[i for i in range(139)], verbose=True )
    