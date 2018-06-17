#usage:  python remove_test_fakes.py --source_folder /media/david/Samsung1/data/yt8m/frame-test2 --dest_folder /media/david/Samsung1/data/yt8m/frame-test-slim --csv_file ./vids_by_std.csv

import pandas as pd
import os
import tensorflow as tf
import numpy as np
import glob
from tqdm import tqdm
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-s", "--source_folder", required=True, help="source folder of test files")
ap.add_argument("-d", "--dest_folder", required=True, help="dest folder of test files")
ap.add_argument("-c", "--csv_file", required=True, help="csv file containing vids and std metric")
args = vars(ap.parse_args())
source_folder = args["source_folder"]
dest_folder = args["dest_folder"]

if not os.path.exists(dest_folder):
    os.makedirs(dest_folder)

df = pd.read_csv(args["csv_file"])

good_vids = df['vids'].loc[df['std'] >0.1]  #get ids of only real data

s = set(good_vids.unique())  #convert to set for faster lookup

for filename in tqdm(glob.iglob(source_folder + '/*.tfrecord')):
    val_basename = os.path.basename(filename)  # address to save the TFRecords file
    writer = tf.python_io.TFRecordWriter(dest_folder + '/' + val_basename)
    for example in tf.python_io.tf_record_iterator(filename):        
        tf_seq_example = tf.train.SequenceExample.FromString(example)
        vid_id = tf_seq_example.context.feature['id'].bytes_list.value[0].decode(encoding='UTF-8')
        if vid_id in s:
            writer.write(tf_seq_example.SerializeToString())
    writer.close()
