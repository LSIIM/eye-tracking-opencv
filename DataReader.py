
from tqdm import tqdm
import os
import pickle
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    path = './vds/prc/'
    for video_path in os.listdir(path):
        file = open(path + video_path+"/data.pickle", 'rb')

        # dump information to that file
        data = pickle.load(file)

        # close the file
        data = data['data']
        file.close()
        for dt in data:
            #print(dt)
            left_raw = dt['left_eye']['raw']
            left_blur = dt['left_eye']['blurred']
            left_bin = dt['left_eye']['binary']
            left_row_count = dt['left_eye']['histograms']['rows']
            left_col_count = dt['left_eye']['histograms']['columns']

            right_raw = dt['right_eye']['raw']
            right_blur = dt['right_eye']['blurred']
            right_bin = dt['right_eye']['binary']
            right_row_count = dt['right_eye']['histograms']['rows']
            right_col_count = dt['right_eye']['histograms']['columns']

            #print(left_blur)

            try:
                #print(left_raw.shape)
                #print(left_raw)
                cv.imshow("Raw Left Eye",left_raw)
                cv.imshow("blurred Left Eye",left_blur)
                cv.imshow("Binary Left Eye",left_bin)

                cv.imshow("Raw Right Eye",right_raw)
                cv.imshow("blurred Right Eye",right_blur)
                cv.imshow("Binary Right Eye",right_bin)
                
                print(int(right_blur.mean()),int(right_blur.max()))
                ret, bin_img = cv.threshold(right_blur,int((right_blur.max() - right_blur.mean())),255,cv.THRESH_BINARY) #isso não faz sentido kkkk
                th2 = cv.adaptiveThreshold(right_blur,255,cv.ADAPTIVE_THRESH_MEAN_C,cv.THRESH_BINARY,11,2)
                
                cv.imshow("Binary Test Right Eye",bin_img)
                cv.imshow("AdapTh Right Eye",th2)
                '''plt.plot(right_row_count)
                plt.title(label="right row")
                plt.show()
                plt.plot(left_row_count)
                plt.title(label="left row")
                plt.show()

                plt.plot(right_col_count)
                plt.title(label="right col")
                plt.show()
                plt.plot(left_col_count)
                plt.title(label="left col")
                plt.show()'''
                key = cv.waitKey(10)

                if key == ord('q'):
                    break
                #print("\n\n")
            except:
                print("Invalid image")
        print("\n\n\n\n")