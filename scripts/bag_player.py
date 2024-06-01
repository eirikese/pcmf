import os
import subprocess

def play_rosbags(folder_path):
    # List all bag files in the given folder
    bag_files = [f for f in os.listdir(folder_path) if f.endswith('.bag')]

    # Sort the bag files alphabetically
    bag_files.sort()

    for bag_file in bag_files:
        # Construct the full path to the bag file
        full_path = os.path.join(folder_path, bag_file)
        print(f"Playing {full_path}")

        # Execute the rosbag play command
        subprocess.run(["rosbag", "play", full_path])
        print(f"Finished playing {full_path}")
        
if __name__ == "__main__":
    # Replace this with the path to your folder containing ROS bag files
    folder_path_1 = "/home/eirik/Downloads/mr_testing_271023"
    folder_path_2 = "/home/eirik/mr_rosbag_150424"
    folder_path_3 = "/home/eirik/Downloads/mr_testing_050424"
    

    # print folder paths, ask for select folder
    print(f"Folder 1: {folder_path_1}")
    print(f"Folder 2: {folder_path_2}")
    print(f"Folder 3: {folder_path_3}")

    folder_path = input("Select folder (1/2/3): ")
    if folder_path == "1":
        folder_path = folder_path_1
    elif folder_path == "2":
        folder_path = folder_path_2
    elif folder_path == "3":
        folder_path = folder_path_3
    else:
        print("Invalid folder selection")
        exit()
    play_rosbags(folder_path)
