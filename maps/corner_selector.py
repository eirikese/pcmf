import ezdxf
import numpy as np
import matplotlib.pyplot as plt
import signal
import sys

# Set the path to the DXF file
# DXF_FILE_PATH = '/home/eirik/lidar_ws/src/pcmf/maps/bridge_map.dxf'
DXF_FILE_PATH = '/home/eirik/lidar_ws/src/pcmf/maps/bay_lines_v3.dxf'

# Define the number of points per meter
POINTS_PER_METER = 0.5  # Adjust this value as needed

def signal_handler(sig, frame):
    # Save data when Ctrl+C is pressed
    # np.save('all_points.npy', points)
    np.save('all_points_brattora.npy', points)
    # np.save('selected_points.npy', selected_points)
    np.save('selected_points_brattora.npy', selected_points)
    print('Files saved. Exiting.')
    sys.exit(0)

def on_click(event):
    if event.inaxes is not None:
        distances = np.sqrt((points[:, 0] - event.xdata) ** 2 + (points[:, 1] - event.ydata) ** 2)
        index = np.argmin(distances)
        if distances[index] < 0.8:  # Adjust the threshold based on your scale
            selected_points.append(points[index])
            ax.plot(points[index, 0], points[index, 1], 'ro')
            fig.canvas.draw()

def sample_points_on_line(start, end, length, points_per_meter):
    num_points = max(2, int(np.ceil(length * points_per_meter)))  # Ensure at least two points
    return np.array([np.linspace(start[i], end[i], num_points) for i in range(2)]).T

def main():
    # Load DXF file
    doc = ezdxf.readfile(DXF_FILE_PATH)
    msp = doc.modelspace()

    # Sample points from lines
    global points
    points = np.array([[0, 0]])  # Dummy initialization to define dtype
    for line in msp.query('LINE'):
        start = np.array(line.dxf.start.xyz[:2])
        end = np.array(line.dxf.end.xyz[:2])
        length = np.linalg.norm(end - start)
        line_points = sample_points_on_line(start, end, length, POINTS_PER_METER)
        points = np.vstack([points, line_points]) if points.size else line_points

    # Set up plot
    global fig, ax, selected_points
    selected_points = []
    fig, ax = plt.subplots()
    ax.plot(points[:, 0], points[:, 1], 'bo')
    fig.canvas.mpl_connect('button_press_event', on_click)

    # Handle Ctrl+C
    signal.signal(signal.SIGINT, signal_handler)

    plt.show()

if __name__ == '__main__':
    main()
