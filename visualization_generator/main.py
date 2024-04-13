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
import numpy as np

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

    # save the data with the left pupil position
    filled_data.to_csv(f"{save_path}/left_pupil_position.csv", index=False)

def remove_outliers_and_smooth_data(data, column_name, outlier_top_threshold=0.01, window_size=10):
    # remove outliers and smooth data
    # outliers in this context are the top N% of the data
    # the remotion will be done nullifying the values and then using the ffill method to fill the null values
    data.loc[data[column_name] > data[column_name].quantile(1 - outlier_top_threshold), column_name] = None
    data = data.ffill()

    # smooth the data using a rolling mean
    data[column_name] = data[column_name].rolling(window_size).mean()

    return data

def get_angles_between_vectors(v1, v2):
    # get the angle between two vectors
    # v1 and v2 are numpy arrays
    # the angle is in degrees

    # <u,v> = x1*x2 + y1*y2 + z1*z2
    # |v| = sqrt(x2^2 + y2^2 + z2^2)
    # <u,v> = cos(phi) * |u| * |v|
    # cos(phi) = <u,v> / (|u| * |v|) = (x1*x2 + y1*y2 + z1*z2 )/ sqrt(x1^2 + y1^2 + z1^2) * sqrt(x2^2 + y2^2 + z2^2)
    # phi = arccos((x1*x2 + y1*y2 + z1*z2 )/ sqrt(x1^2 + y1^2 + z1^2) * sqrt(x2^2 + y2^2 + z2^2))

    angle = np.arccos(np.divide(np.sum(v1 * v2, axis=1), np.linalg.norm(v1, axis=1) * np.linalg.norm(v2, axis=1))) * 180 / np.pi
    # if phi > 90, phi = -(180 - phi)
    angle = np.where(angle > 90, -(180 - angle), angle)
    return angle

def draw_eye_gaze_visualizations(eye, data, save_path):
    # create a line plot for the iris and pupil position through time

    if(verbose):
        print("Generating eye fixation visualization")
        print("Saving in:", save_path)
        print("Data columns:", data_df.columns)
        print(data_df.head())

    # create plot for {eye}_eye

    fig_ocular_angle = px.line(data, x='frame', y=[f'{eye}_ocular_degree'], title=f'{eye} eye ocular movement range')
    fig_ocular_angle.write_image(f"{save_path}/{eye}_eye_ocular_movement_range.png", width=1000, height=500, format='png', engine='kaleido')

    fig_vertical_angle = px.line(data, x='frame', y=[f'vertical_angle_{eye}_degree'], title=f'{eye} eye vertical ocular movement range')
    fig_vertical_angle.write_image(f"{save_path}/{eye}_eye_vertical_ocular_movement_range.png", width=1000, height=500, format='png', engine='kaleido')

    fig_horizontal_angle = px.line(data, x='frame', y=[f'horizontal_angle_{eye}_degree'], title=f'{eye} eye horizontal ocular movement range')
    fig_horizontal_angle.write_image(f"{save_path}/{eye}_eye_horizontal_ocular_movement_range.png", width=1000, height=500, format='png', engine='kaleido')

def process_eye_movement_range(data, eye, save_path):
    filled_data = data.ffill()
    if verbose:
        print(f'{eye} data filled')

    # Adjust gaze vectors based on pupil positions
    filled_data[f'{eye}_eye_gaze_x'] -= filled_data[f'{eye}_pupil_x']
    filled_data[f'{eye}_eye_gaze_y'] -= filled_data[f'{eye}_pupil_y']

    # divide by 10
    filled_data[f'{eye}_eye_gaze_x'] /= 10
    filled_data[f'{eye}_eye_gaze_y'] /= 10

    # Append the head orientation Z to gaze data to make 3D vectors
    filled_data[f'{eye}_eye_gaze_z'] = filled_data['head_orientation_z']
    filled_data[f'{eye}_eye_gaze_z'] = filled_data['head_orientation_z']

    # Normalize the gaze and head orientation vectors
    eye_gaze = filled_data[[f'{eye}_eye_gaze_x', f'{eye}_eye_gaze_y', f'{eye}_eye_gaze_z']].values
    head_orientation = filled_data[['head_orientation_x', 'head_orientation_y', 'head_orientation_z']].values

    eye_gaze /= np.linalg.norm(eye_gaze, axis=1)[:, np.newaxis]
    head_orientation /= np.linalg.norm(head_orientation, axis=1)[:, np.newaxis]

    # Calcula o ângulo entre a orientação da cabeça e a direção do olhar para cada frame, convertendo para graus
    filled_data[f'{eye}_ocular_degree'] = get_angles_between_vectors(eye_gaze, head_orientation)

    # agora pega somente a componente vertical da diferença angular (ou seja projeta o vetor no plano y,z)
    filled_data[f'vertical_angle_{eye}_degree'] = get_angles_between_vectors(eye_gaze[:, 1:], head_orientation[:, 1:]) # pega a componente vertical da diferença angular (ou seja projeta o vetor no plano y,z)

    # pega a componente horizontal da diferença angular (ou seja projeta o vetor no plano x,z)
    filled_data[f'horizontal_angle_{eye}_degree'] = get_angles_between_vectors(eye_gaze[:, [0, 2]], head_orientation[:, [0, 2]]) # pega a componente horizontal da diferença angular (ou seja projeta o vetor no plano x,z)

    # remove outliers and smooth data
    filled_data = remove_outliers_and_smooth_data(filled_data, f'{eye}_ocular_degree', outlier_top_threshold=0.01, window_size=10)

    filled_data.to_csv(f"{save_path}/{eye}_angles.csv", index=False)

    draw_eye_gaze_visualizations(eye, filled_data, save_path)


    


def generate_ocular_movement_range_vizualization(data, save_path="visualizations"):
    # create a line plot for the angle between the head orientation and the eye gaze through time

    if verbose:
        print("Generating ocular movement range visualization")
        print("Saving in:", save_path)
        print("Data columns:", data.columns)
        print(data.head())

    # get only the data needed for the plot (gaze and head orientation)
    gaze_data = data[['frame', 'left_eye_gaze_x', 'left_eye_gaze_y', 'right_eye_gaze_x', 'right_eye_gaze_y',
                      'head_orientation_x', 'head_orientation_y', 'head_orientation_z',
                      'left_pupil_x', 'left_pupil_y', 'right_pupil_x', 'right_pupil_y']]

    if verbose:
        print("Data for the plot filtered")
   
    process_eye_movement_range(gaze_data, 'left', save_path)
    process_eye_movement_range(gaze_data, 'right', save_path)

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
    generate_ocular_movement_range_vizualization(data_df, save_path)
    