import numpy as np
import matplotlib.pyplot as plt

def on_click(event):
    global selected_point, previous_position
    if event.inaxes is not None and event.button == 1:  # Only respond to left mouse clicks
        x, y = event.xdata, event.ydata
        if selected_point is not None:
            # Move the previously selected point to the new position
            moved_points[selected_point] = [x, y]
            red_line.set_data(moved_points[:, 0], moved_points[:, 1])

            # Mark the old position grey if it exists
            if previous_position is not None:
                ax.plot(previous_position[0], previous_position[1], 'o', color='grey', markersize=5)

            # Update the plot
            fig.canvas.draw()

            # Reset the selected point
            previous_position = [x, y]  # Update previous position to new location
            selected_point = None
        else:
            distances = np.sqrt((moved_points[:, 0] - x) ** 2 + (moved_points[:, 1] - y) ** 2)
            if distances.min() < 0.5:  # 0.1 is a sensitivity threshold, adjust as needed
                selected_point = distances.argmin()
                previous_position = moved_points[selected_point].copy()
                # Mark the selected point yellow
                ax.plot(moved_points[selected_point, 0], moved_points[selected_point, 1], 'o', color='yellow', markersize=7)
                fig.canvas.draw()

def on_close(event):
    np.save('updated_selected_points.npy', moved_points)
    print('Updated points saved to "updated_selected_points.npy".')

def main():
    global moved_points, red_line, fig, ax, selected_point, previous_position
    selected_point = None
    previous_position = None

    # Load the selected points and all points from files
    all_points = np.load('/home/eirik/lidar_ws/src/pcmf/maps/all_points.npy')
    moved_points = np.load('/home/eirik/lidar_ws/src/pcmf/maps/selected_points.npy')

    # swapped 
    # moved_points = np.load('/home/eirik/lidar_ws/src/pcmf/maps/all_points.npy')
    # all_points = np.load('/home/eirik/lidar_ws/src/pcmf/maps/selected_points4.npy')

    # Create a plot
    fig, ax = plt.subplots()
    ax.plot(all_points[:, 0], all_points[:, 1], 'bo', markersize=5)  # Plot all points in blue
    red_line, = ax.plot(moved_points[:, 0], moved_points[:, 1], 'ro', markersize=10)

    ax.set_title("Click to select and move red points")
    ax.set_xlim([min(all_points[:, 0]) - 1, max(all_points[:, 0]) + 1])
    ax.set_ylim([min(all_points[:, 1]) - 1, max(all_points[:, 1]) + 1])

    # Connect event handlers
    fig.canvas.mpl_connect('button_press_event', on_click)
    fig.canvas.mpl_connect('close_event', on_close)

    plt.show()

if __name__ == '__main__':
    main()
