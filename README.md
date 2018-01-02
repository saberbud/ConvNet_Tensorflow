CNN_prad.py: training ConvNet using train_good.txt and train_bad.txt, storing trained model into ckpt files.

check_err_CNN_prad.py: read ckpt files and check classification precision with user-defined file(s).

input_xy_data.py: put 2157-dimensional pixel data into user-defined format for ConvNet to process, using numpy.

Example training and testing files:
Each row contains 2157 "pixel" reading from Hybrid Calorimeter, each row is for an event or "picture". Number of rows = number of events.

train_good.txt: pure electron-proton scattering events.

train_bad.txt: cosmic-ray events (~95% purity).

test_good.txt: pure electron-proton scattering events.

test_bad.txt: pure cosmic-ray events.

