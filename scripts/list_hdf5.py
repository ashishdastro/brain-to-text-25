import glob, os

for path in glob.glob("data/raw/**/data_test.hdf5", recursive=True):
    print(path)
