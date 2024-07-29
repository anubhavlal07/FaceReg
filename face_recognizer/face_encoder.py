# import the necessary packages
from face_recognizer.detect_faces import face_detection
from imutils.face_utils import FaceAligner
from imutils import paths
import face_recognition
import pandas as pd
import argparse
import imutils
import pickle
import cv2
import os

class FaceEncoder():
    def __init__(self, facesPath, encodings, attendance, prototxt, model):
        self.facesPath = facesPath
        self.encodings = encodings
        self.attendance = attendance
        self.prototxt = prototxt
        self.model = model

        self.knowEncodings = []
        self.KnowNames = []
        
    def encode_faces(self):
        # extract image paths and initialize empty encoding and names array.
        image_paths = list(paths.list_images(self.facesPath))

        # loop over image paths
        for (i, image_path) in enumerate(image_paths):
            print("[INFO] processing image {}/{}".format(i + 1, len(image_paths)))
            name = image_path.split(os.path.sep)[-2]

            image = cv2.imread(image_path)

            # Check if the image is loaded correctly
            if image is None:
                print(f"[WARNING] could not load image at path: {image_path}")
                continue

            # Ensure the image is 8-bit and has 3 channels (RGB)
            if image.dtype != 'uint8':
                image = (image * 255).astype('uint8')

            if len(image.shape) == 2:
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            elif image.shape[2] == 4:
                image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)

            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            # boxes = face_recognition.face_locations(rgb, model=args["detection_method"])
            (boxes, _) = face_detection(image, self.prototxt, self.model)
            boxes = [(box[1], box[2], box[3], box[0]) for box in boxes]
            encodings = face_recognition.face_encodings(rgb, boxes)

            for encoding in encodings:
                self.knownEncodings.append(encoding)
                self.knownNames.append(name)

    def save_face_encodings(self):
        # if there are registers faces
        if os.path.exists(self.attendance) and os.path.exists(self.attendance):
            # append new encodings to the existing one
            with open(self.encodings, "rb") as f:
                print("[INFO] loading encodings...")
                data = pickle.loads(f.read())
                data["names"].extend(self.KnowNames)
                data["encodings"].extend(self.knowEncodings)
            with open(self.encodings, "wb") as f:
                print("[INFO] serialize encodings to disk...")
                f.write(pickle.dumps(data))

            # append new dataframe to the existing one.
            df = pd.read_csv(self.attendance, index_col=0)
            new_df = pd.DataFrame(columns=df.columns)
            new_df['names'] = list(set(self.KnowNames))
            new_df = new_df.fillna(0)

            df = df.append(new_df)
            df = df.sort_values(by='names')
            df = df.reset_index(drop=True)
            print("[INFO] storing additional student names in a dataframe...")
            df.to_csv(self.attendance)
        else:
            # serialize encodings to disk
            print("[INFO] serialize encodings to disk...")
            data = {"names": self.KnowNames, "encodings": self.knowEncodings}
            f = open(self.encodings, "wb")
            f.write(pickle.dumps(data))
            f.close()

            # stroring the names in dataframe
            print("[INFO] storing student names in a dataframe...")
            df = pd.DataFrame({"names": sorted(list(set(self.KnowNames)))})
            df.to_csv(self.attendance)
