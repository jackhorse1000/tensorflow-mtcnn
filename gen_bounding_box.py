from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import os
import argparse
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np
import detect_face
import cv2
import csv
import pandas as pd

def main():
    sess = tf.Session()
    pnet, rnet, onet = detect_face.create_mtcnn(sess, None)
    
    minsize = 40 # minimum size of face
    threshold = [ 0.6, 0.7, 0.9 ]  # three steps's threshold
    factor = 0.709 # scale factor
    MTFL_folder = '/home/jack/Documents/Year-4/Project/Proj-continual-learning/datasets/MTFL'
    filename = 'training.txt'
    input_csv = os.path.join(MTFL_folder, filename)
    output_csv = os.path.join(MTFL_folder, 'bb_' + filename)

    names = ["#image path", "#x1","#x2","#x3","#x4","#x5","#y1","#y2","#y3",
                 "#y4","#y5","#gender","#smile", "#wearing glasses", "#head pose"]
    in_df = pd.read_csv(input_csv, delim_whitespace=True, header=None, names=names, engine='python')
    out_df = None
    remove_rows = []
    for idx, row in in_df.iterrows():
        img_filepath = os.path.join(MTFL_folder, row[0]).replace('\\', '/')
        print(img_filepath)
        draw = cv2.imread(img_filepath)

        img=cv2.cvtColor(draw,cv2.COLOR_BGR2RGB)

        bounding_boxes, points = detect_face.detect_face(img, minsize, pnet, rnet, onet, threshold, factor)
        try:
            nrof_faces = bounding_boxes.shape[0]
            bounding_box = list(bounding_boxes[0])
        except:
            remove_rows.append(idx)
            continue
    
        if nrof_faces > 1:
            # TODO: maybe show img
            remove_rows.append(idx)
            continue

        # print(bounding_box)
        if out_df is None:
            # series = pd.Series(bounding_box)
            out_df = pd.DataFrame()
            series = pd.Series(bounding_box)
            out_df = out_df.append(series, ignore_index=True)
        else:
            # app_df = pd.DataFrame(bounding_box)
            series = pd.Series(bounding_box)
            out_df = out_df.append(series, ignore_index=True)
        # add_bb = pd.DataFrame(bounding_box)
        # out_df.append(add_bb)
        # TODO: Write line to out csv
    print(in_df)

    print(out_df)

    in_df.drop(remove_rows, inplace=True)

    print(in_df)

    in_df.to_csv(input_csv, sep=' ', index=False, header=False)
    out_df.to_csv(output_csv, sep=' ', index=False, header=False)

if __name__ == '__main__':
    main()