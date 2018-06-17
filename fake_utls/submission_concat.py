#usage: python submission_concat.py --good_file ./netvald.csv --fake_file ./badreplace.csv --submit_file ./submission.csv
import pandas as pd
import argparse
import os


ap = argparse.ArgumentParser()
ap.add_argument("-o", "--good_file", required=True, help="file with good values only")
ap.add_argument("-f", "--fake_file", required=True, help="file with fakes")
ap.add_argument("-s", "--submit_file", required=True, help="submission ready file")
args = vars(ap.parse_args())

df_orig = pd.read_csv(args["good_file"])
df_fake = pd.read_csv(args["fake_file"])

frames = [df_orig, df_fake]

df_submit = pd.concat(frames)
df_submit.to_csv(args["submit_file"], index=False)


