# Setup Instructions

## Start the ROS Core
- Open a new terminal and start ROS master: `roscore`.

## Run the Flask Server
- Navigate to the directory containing `app.py`.
- Run the Flask server: `python app.py`.
- The server will start on `http://127.0.0.1:5000/`.

## Run the ROS Node
- Open a new terminal.
- Ensure your ROS workspace is sourced.
- Run the ROS node: `python src/ros_to_flask_node.py`.

## Open the Web Page
- Open your web browser.
- Navigate to `http://127.0.0.1:5000/`.
- The web page should now display the map, which will update in real-time with data from the ROS node.
