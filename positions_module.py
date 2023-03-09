import cv2
import numpy as np
import pandas as pd
from definitions import *


class PositionsModule():
    def __init__(self):
        self._data = []

    def add_positions(self, eyeData):
        self._data.append(eyeData)

    def get_past_n_positions(self, n):
        left_eye = []
        right_eye = []
        for i in range(len(self._data)):
            eyeData = self._data[len(self._data)-(1+i)]
            if (i > n):
                break
            left_eye.append([eyeData._left_iris["x"], eyeData._left_iris['y']])
            right_eye.append(
                [eyeData._right_iris["x"], eyeData._right_iris['y']])
        return left_eye, right_eye

    def save_data(self, path):
        df = pd.DataFrame()
        left_iris_x = []
        left_iris_y = []
        left_iris_r = []

        right_iris_x = []
        right_iris_y = []
        right_iris_r = []

        left_pupil_x = []
        left_pupil_y = []
        left_pupil_r = []

        right_pupil_x = []
        right_pupil_y = []
        right_pupil_r = []

        original_left_iris_x = []
        original_left_iris_y = []
        original_left_iris_r = []

        original_right_iris_x = []
        original_right_iris_y = []
        original_right_iris_r = []

        original_left_pupil_x = []
        original_left_pupil_y = []
        original_left_pupil_r = []

        original_right_pupil_x = []
        original_right_pupil_y = []
        original_right_pupil_r = []

        frame = []
        for i in range(len(self._data)):
            left_iris_x.append(
                self._data[i]._left_iris['x'] if self._data[i]._left_iris else "")
            left_iris_y.append(
                self._data[i]._left_iris['y'] if self._data[i]._left_iris else "")
            left_iris_r.append(
                self._data[i]._left_iris['r'] if self._data[i]._left_iris else "")

            right_iris_x.append(
                self._data[i]._right_iris['x'] if self._data[i]._right_iris else "")
            right_iris_y.append(
                self._data[i]._right_iris['y'] if self._data[i]._right_iris else "")
            right_iris_r.append(
                self._data[i]._right_iris['r'] if self._data[i]._right_iris else "")

            left_pupil_x.append(
                self._data[i]._left_pupil['x'] if self._data[i]._left_pupil else "")
            left_pupil_y.append(
                self._data[i]._left_pupil['y'] if self._data[i]._left_pupil else "")
            left_pupil_r.append(
                self._data[i]._left_pupil['r'] if self._data[i]._left_pupil else "")

            right_pupil_x.append(
                self._data[i]._right_pupil['y'] if self._data[i]._right_pupil else "")
            right_pupil_y.append(
                self._data[i]._right_pupil['y'] if self._data[i]._right_pupil else "")
            right_pupil_r.append(
                self._data[i]._right_pupil['r'] if self._data[i]._right_pupil else "")

            original_left_iris_x.append(
                self._data[i]._original_left_iris['x'] if self._data[i]._original_left_iris else "")
            original_left_iris_y.append(
                self._data[i]._original_left_iris['y'] if self._data[i]._original_left_iris else "")
            original_left_iris_r.append(
                self._data[i]._original_left_iris['r'] if self._data[i]._original_left_iris else "")

            original_right_iris_x.append(
                self._data[i]._original_right_iris['x'] if self._data[i]._original_right_iris else "")
            original_right_iris_y.append(
                self._data[i]._original_right_iris['y'] if self._data[i]._original_right_iris else "")
            original_right_iris_r.append(
                self._data[i]._original_right_iris['r'] if self._data[i]._original_right_iris else "")

            original_left_pupil_x.append(
                self._data[i]._original_left_pupil['x'] if self._data[i]._original_left_pupil else "")
            original_left_pupil_y.append(
                self._data[i]._original_left_pupil['y'] if self._data[i]._original_left_pupil else "")
            original_left_pupil_r.append(
                self._data[i]._original_left_pupil['r'] if self._data[i]._original_left_pupil else "")

            original_right_pupil_x.append(
                self._data[i]._original_right_pupil['x'] if self._data[i]._original_right_pupil else "")
            original_right_pupil_y.append(
                self._data[i]._original_right_pupil['y'] if self._data[i]._original_right_pupil else "")
            original_right_pupil_r.append(
                self._data[i]._original_right_pupil['r'] if self._data[i]._original_right_pupil else "")

            frame.append(
                self._data[i]._frame if self._data[i]._frame != None else "")

        print(original_left_iris_r)

        df["frame"] = frame

        df["left_iris_x"] = left_iris_x
        df["left_iris_y"] = left_iris_y
        df["left_iris_r"] = left_iris_r

        df["right_iris_x"] = right_iris_x
        df["right_iris_y"] = right_iris_y
        df["right_iris_r"] = right_iris_r

        df["left_pupil_x"] = left_pupil_x
        df["left_pupil_y"] = left_pupil_y
        df["left_pupil_r"] = left_pupil_r

        df["right_pupil_x"] = right_pupil_x
        df["right_pupil_y"] = right_pupil_y
        df["right_pupil_r"] = right_pupil_r

        df["original_left_iris_x"] = original_left_iris_x
        df["original_left_iris_y"] = original_left_iris_y
        df["original_left_iris_r"] = original_left_iris_r

        df["original_right_iris_x"] = original_right_iris_x
        df["original_right_iris_y"] = original_right_iris_y
        df["original_right_iris_r"] = original_right_iris_r

        df["original_left_pupil_x"] = original_left_pupil_x
        df["original_left_pupil_y"] = original_left_pupil_y
        df["original_left_pupil_r"] = original_left_pupil_r

        df["original_right_pupil_x"] = original_right_pupil_x
        df["original_right_pupil_y"] = original_right_pupil_y
        df["original_right_pupil_r"] = original_right_pupil_r

        df.to_csv(path)


class EyeDataModule():
    def __init__(self, frame):
        self._left_iris = None
        self._right_iris = None
        self._left_pupil = None
        self._right_pupil = None
        self._frame = frame

    def print_data(self):
        print(f'\nFrame: {self._frame}')
        print("\nLeft Iris:")
        print(self._left_iris)
        print("\nRight Iris:")
        print(self._right_iris)
        print("\nLeft Pupil:")
        print(self._left_pupil)
        print("\nRight Pupil:")
        print(self._right_pupil)
        print("\n-------------------------------------")

    def add_left_iris(self, left_iris):
        if (left_iris):
            (x, y), r = left_iris

            self._left_iris = {
                "x": x,
                "y": y,
                "r": r
            }
        else:
            self._left_iris = {
                "x": None,
                "y": None,
                "r": None
            }

    def add_right_iris(self, right_iris):
        if (right_iris):
            (x, y), r = right_iris

            self._right_iris = {
                "x": x,
                "y": y,
                "r": r
            }
        else:
            self._right_iris = {
                "x": None,
                "y": None,
                "r": None
            }

    def add_left_pupil(self, left_pupil):
        if (left_pupil):
            (x, y), r = left_pupil

            self._left_pupil = {
                "x": x,
                "y": y,
                "r": r
            }
        else:
            self._left_pupil = {
                "x": None,
                "y": None,
                "r": None
            }

    def add_right_pupil(self, right_pupil):
        if (right_pupil):
            (x, y), r = right_pupil

            self._right_pupil = {
                "x": x,
                "y": y,
                "r": r
            }
        else:
            self._right_pupil = {
                "x": None,
                "y": None,
                "r": None
            }

    def add_original_left_iris(self, original_left_iris):
        if (original_left_iris):
            (x, y), r = original_left_iris

            self._original_left_iris = {
                "x": x,
                "y": y,
                "r": r
            }
        else:
            self._original_left_iris = {
                "x": None,
                "y": None,
                "r": None
            }

    def add_original_right_iris(self, original_right_iris):
        if (original_right_iris):
            (x, y), r = original_right_iris

            self._original_right_iris = {
                "x": x,
                "y": y,
                "r": r
            }
        else:
            self._original_right_iris = {
                "x": None,
                "y": None,
                "r": None
            }

    def add_original_left_pupil(self, original_left_pupil):
        if (original_left_pupil):
            (x, y), r = original_left_pupil

            self._original_left_pupil = {
                "x": x,
                "y": y,
                "r": r
            }
        else:
            self._original_left_pupil = {
                "x": None,
                "y": None,
                "r": None
            }

    def add_original_right_pupil(self, original_right_pupil):
        if (original_right_pupil):
            (x, y), r = original_right_pupil

            self._original_right_pupil = {
                "x": x,
                "y": y,
                "r": r
            }
        else:
            self._original_right_pupil = {
                "x": None,
                "y": None,
                "r": None
            }
