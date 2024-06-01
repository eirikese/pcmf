import numpy as np
import folium

def generate_random_trajectory(start_lat, start_lon, num_points=100):
    # Generate random steps
    steps_lat = np.random.uniform(-0.0005, 0.0005, num_points)
    steps_lon = np.random.uniform(-0.0005, 0.0005, num_points)
    
    # Calculate the trajectory
    trajectory_lat = np.cumsum(np.insert(steps_lat, 0, start_lat))
    trajectory_lon = np.cumsum(np.insert(steps_lon, 0, start_lon))
    
    return trajectory_lat, trajectory_lon

def plot_trajectory_on_map(trajectory_lat, trajectory_lon):
    # Calculate center of the trajectory for map initialization
    map_center = [np.mean(trajectory_lat), np.mean(trajectory_lon)]
    m = folium.Map(location=map_center, zoom_start=15, tiles='OpenStreetMap')
    
    # Add points to the map
    for lat, lon in zip(trajectory_lat, trajectory_lon):
        folium.CircleMarker(location=[lat, lon], radius=1, color='red').add_to(m)
    
    # Save the map to an HTML file
    m.save('/home/mr_fusion/lidar_ws/src/pcmf/maps/trajectory_map.html')
    print("Map has been saved to trajectory_map.html")

if __name__ == "__main__":
    # Set the seed for reproducibility
    np.random.seed(0)

    # Starting point
    start_lat, start_lon = 63.438702, 10.3970369

    # Generate a random trajectory
    trajectory_lat, trajectory_lon = generate_random_trajectory(start_lat, start_lon)

    # Plot the trajectory on the map
    plot_trajectory_on_map(trajectory_lat, trajectory_lon)
