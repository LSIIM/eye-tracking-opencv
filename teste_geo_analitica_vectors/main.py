from Face import Face
import cv2
import definitions

# import plotly for 3D visualization
import plotly.express as px


if __name__ == '__main__':
    face = Face()
    cap = cv2.VideoCapture(0)
    while cv2.waitKey(1) & 0xFF != ord('q'):
        success, img = cap.read()
        face.detect_face(img)
        print(face.lms_3d)
        cv2.imshow("Image", img)


        face.init_eye_module()
        face.detect_iris()
        face.detect_pupil()

        print(face.get_position_data_as_dict())


        # plot the 3D face mesh
        # face.lms_3d is a numpy array with shape (468, 3)
        fig = px.scatter_3d(x=face.lms_3d[:, 0], y=face.lms_3d[:, 1], z=face.lms_3d[:, 2])
        # change the size of the points
        fig.update_traces(marker=dict(size=2, color='black'))  # Default color

        # Update eye points to be blue
        left_eye_idx = definitions.LEFT_EYE
        right_eye_idx = definitions.RIGHT_EYE
        eye_color = 'blue'  # Color for eye points

        # Create a color array
        colors = ['black'] * face.lms_3d.shape[0]
        for idx in left_eye_idx + right_eye_idx:
            colors[idx] = eye_color

        # update iris points to red
        left_iris_idx = definitions.LEFT_IRIS
        right_iris_idx = definitions.RIGHT_IRIS
        iris_color = 'red'

        for idx in left_iris_idx + right_iris_idx:
            colors[idx] = iris_color

        fig.update_traces(marker=dict(color=colors))


        



        fig.show()

        
        # wait for the user to press 'space' to go to the next frame
        quit = False
        # wait for the user to press 'q' to quit the program
        while not quit:
            key = cv2.waitKey(1) & 0xFF
            if key == ord(' '):
                break
            elif key == ord('q'):
                quit = True
                break
        
        if quit:
            break

    cap.release()
    cv2.destroyAllWindows()
