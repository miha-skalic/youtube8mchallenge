
# remove_test_fakes.py

- this file will loop through the frames-test folder, select which records to keep, and then write the new tfrecords file using the same basename to a new directory.
- vids_by_std.csv file contains the mapping from video id's to the std metric of video level rgb
- note that this does NOT delete your previously saved test tfrecords, so you'll need an additional 149GB of storage for the newly created files

python remove_test_fakes.py --source_folder /media/david/Samsung1/data/yt8m/frame-test --dest_folder /media/david/Samsung1/data/yt8m/frame-test-slim --csv_file ./vids_by_std.csv


# submission_concat.py

- this file will take a submission using only the good data and concat it with a file containing only the fake data, creating a new file that can be used for final submission
- fake rows = 576949, good rows = 556374
- badreplace.csv contains the fake rows along with dummy values for submission

python submission_concat.py --good_file ./netvlad.csv --fake_file ./badreplace.csv --submit_file ./submission.csv
