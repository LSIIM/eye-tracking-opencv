# receives using argaparse the path to the file with the data to be visualized (csv), the visualizations will be saved in the same path in a folder called "visualizations"

# the csv file has the following columns:
# ,frame,height,width,left_iris_x,left_iris_y,left_iris_r,right_iris_x,right_iris_y,right_iris_r,left_pupil_x,left_pupil_y,left_pupil_r,right_pupil_x,right_pupil_y,right_pupil_r,head_orientation_x,head_orientation_y,head_orientation_z,left_eye_gaze_x,left_eye_gaze_y,right_eye_gaze_x,right_eye_gaze_y, nose_tip_x, nose_tip_y
"""
frame: frame number
height: height of the image
width: width of the image
left_iris_x: x coordinate of the left iris
left_iris_y: y coordinate of the left iris
left_iris_r: radius of the left iris
right_iris_x: x coordinate of the right iris
right_iris_y: y coordinate of the right iris
right_iris_r: radius of the right iris
left_pupil_x: x coordinate of the left pupil
left_pupil_y: y coordinate of the left pupil
left_pupil_r: radius of the left pupil
right_pupil_x: x coordinate of the right pupil
right_pupil_y: y coordinate of the right pupil
right_pupil_r: radius of the right pupil
head_orientation_x: x coordinate of the head orientation
head_orientation_y: y coordinate of the head orientation
head_orientation_z: z coordinate of the head orientation
left_eye_gaze_x: x coordinate of the left eye gaze
left_eye_gaze_y: y coordinate of the left eye gaze
right_eye_gaze_x: x coordinate of the right eye gaze
right_eye_gaze_y: y coordinate of the right eye gaze
nose_tip_x: x coordinate of the nose tip
nose_tip_y: y coordinate of the nose tip

- head pose is a vector the shows the direction of the head
- eye gaze are two points that show a 2d projection of the vector that shows the direction of the eyes
"""

# the visualizations are:
# 1. Eye fixation: a plot for each eye that show throu time the position of the iris and the pupil. this will help to see if the eyes are fixating (stoping the movement) on a point
#    - the plot can be: a line plot for each axis (x,y) so we can see the movement of the eyes in the x and y axis through time
# 2. Eye range of motion: A plot that will show the difference between the vectors of the head orientation and the eye gaze through time. this will help to see how much the eyes are moving in relation to the head
#   - the plot can be: a line plot with the y axis being the angle between the head orientation and the eye gaze and the x axis being the time


import argparse
import pandas as pd
# use plotly to create the visualizations
import plotly.express as px
import os

verbose = False

def generate_eye_fixation_visualization(data_df, save_path="visualizations"):
    # create a line plot for the iris and pupil position through time

    if(verbose):
        print("Generating eye fixation visualization")
        print("Saving in:", save_path)
        print("Data columns:", data_df.columns)
        print(data_df.head())

    # get only the data needed for the plot (only pupil)
    pupil_data = data_df[['frame', 'left_pupil_x', 'left_pupil_y', 'right_pupil_x', 'right_pupil_y', 'nose_tip_x', 'nose_tip_y']]

    if(verbose):
        print("Data for the plot filtered")
    # first remove the lines with no data (it will assume the value of the last known value)
    filled_data = pupil_data.ffill()
    if(verbose):
        print("data filled")
        print(filled_data.head())

    # remove the head movement from the pupil position using the nose tip as a reference
    filled_data['left_pupil_x'] = filled_data['left_pupil_x'] - filled_data['nose_tip_x']
    filled_data['left_pupil_y'] = filled_data['left_pupil_y'] - filled_data['nose_tip_y']

    # moves the data so the mean is 0
    filled_data['left_pupil_x'] = filled_data['left_pupil_x'] - filled_data['left_pupil_x'].mean()
    filled_data['left_pupil_y'] = filled_data['left_pupil_y'] - filled_data['left_pupil_y'].mean()

    

    fig = px.line(filled_data, x='frame', y=['left_pupil_x', 'left_pupil_y'], title='Left eye fixation')
    if(verbose):
        print("plot created")
    # saves as 1000x1000 image
    fig.write_image(f"{save_path}/left_eye_fixation.png", width=1000, height=500 , format='png',engine='kaleido')
    if(verbose):
        print("plot saved")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate visualizations for eye tracking data')
    parser.add_argument('data_path', type=str, help='path to the csv file with the data')
    # verbose argument
    parser.add_argument('--verbose', action='store_true', help='print more information')
    args = parser.parse_args()

    if args.verbose:
        verbose = True

    data_df = pd.read_csv(args.data_path)
    dir_path = os.path.dirname(args.data_path)
    save_path = f"{dir_path}/visualizations"
    # create a folder to save the visualizations
    if not os.path.exists(f"{save_path}"):
        os.makedirs(f"{save_path}")

    
    generate_eye_fixation_visualization(data_df, save_path)
    