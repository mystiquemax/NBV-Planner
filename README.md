# NBV-Planner

## Motivation

The motivation of this project is to let the robot drive autonomously along a predefined trajectory, and take scans at some certain waypoints. At the end, all scans should be merged to a sigle pointcloud to reconstruct a specific environment.

The design of this system can be divided into two parts: driving to the waypoints and taking new scans.

The project uses Next-Best-View (NBV) planner to extract the next best positons on the map. Then, the points are processed with an approximate Travelling Salesman Problem (TSP) solution to find a good order to visit the points. After all positions are visited and scans are obtained, the scans are merged to a single pointcloud using transformation matrixes of the positions and ICP algorithms. 

## Implementation

Files and folders in the project:

* **plan_positions.py**: The file contains the code of the Next-Best-View planner. The camera has (simulated) 180 degree FOV and 5 meter range. Between two consecutive positions there should be at most 1 meter distance. At each position, the robot chooses the next by taking into account 60 positions that are 1 meter away from its current position. The one with the most visible pixels is chosen. If non of the near points has more than 50 pixels visible, a new point is choosen from previously considered points. Best performance: 53 seconds.
 
* **traverse_positions.py**: The code finds the distance matrix using A* algorithm, makes an approximatation of the TSP using the Lin-Kernighan approach (applying Chained Lin-Kernighan approach is currently being considered). Because this solution wasn't very performant, an approximation of the A* algorithm was used. The alternative uses the following approach: If no obstacle lies between two points, Euclidian distance is used. If an obstacle lies between two points, it's penaltized by Manhattan distance. Because of the map used, the approach gives good solution. The non-processed path (i.e. just the generated points) is 223.5 meters long. The TSP approximation using the alternative distance metric produces path that is 220 meters long and requires 2-4 seconds to find the solution. The TSP approximation using A* produces path that is 216 meters long and requires ~1 hour to find the solution. 

* **merge_pcd.py**: The code uses the transformation matrices of the individual positions to bring then to a common coordinate system and to construct a pointcloud. At the end, the pointcloud is refined using Colored ICP.

* **robot_localization.py**: The code uses ACML algorithm to find the approx. position of the robot and track it during the whole process.

* **make_pcd.py**: The code obtains a pointcloud and saves it.

* **common.py**: The code is used for common functions, visualization and benchmarking.
