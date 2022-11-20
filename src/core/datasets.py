import os
from abc import ABC
from torch_geometric.data import Dataset
import numpy as np
import pandas as pd
from torchvision.transforms import ToTensor, Compose, Normalize, Resize
import cv2
from math import isnan


class EchoNetEfDataset(Dataset, ABC):
    def __init__(self,
                 dataset_path,
                 max_frames,
                 frame_size,
                 mean=0.1289,
                 std=0.1911):
        super().__init__()

        # CSV file containing file names and labels
        filelist_df = pd.read_csv(os.path.join(dataset_path, 'FileList.csv'))

        # Extract Split information
        splits = np.array(filelist_df['Split'].tolist())
        self.train_idx = np.where(splits == 'TRAIN')[0]
        self.val_idx = np.where(splits == 'VAL')[0]
        self.test_idx = np.where(splits == 'TEST')[0]

        # Extract ES and ED frame indices
        es_frames = list(filelist_df['ESFrame'])
        self.es_frames = [0 if isnan(es_frame) else int(es_frame) for es_frame in es_frames]
        ed_frames = list(filelist_df['EDFrame'])
        self.ed_frames = [0 if isnan(ed_frame) else int(ed_frame) for ed_frame in ed_frames]

        # Extract video file names
        filenames = np.array(filelist_df['FileName'].tolist())

        # All file paths
        self.patient_data_dirs = [os.path.join(dataset_path,
                                               'Videos',
                                               file_name + '.avi')
                                  for file_name
                                  in filenames.tolist()]

        # Get the labels
        self.labels = list()
        for patient, _ in enumerate(self.patient_data_dirs):
            self.labels.append(filelist_df['EF'].tolist()[patient] / 100)

        # Extract the number of available data samples
        self.num_samples = len(self.patient_data_dirs)

        # Normalization operation
        self.trans = Compose([ToTensor(),
                              Resize((frame_size, frame_size)),
                              Normalize((mean), (std))])

        # Other attributes
        self.max_frames = max_frames

    def get(self, idx):
        # Get the labels
        label = self.labels[idx]

        # Get the video
        cine_vid = self._loadvideo(self.patient_data_dirs[idx])

        # Transform video
        cine_vid = self.trans(cine_vid)

        if cine_vid.shape[0] > self.max_frames:
            cine_vid = cine_vid[0:self.max_frames]

        # Get the ED/ES frames
        ed_frame = cine_vid[self.ed_frames[idx] if self.ed_frames[idx] < self.max_frames else 0]
        es_frame = cine_vid[self.es_frames[idx] if self.es_frames[idx] < self.max_frames else 0]

        return cine_vid.unsqueeze(0), ed_frame.unsqueeze(0), es_frame.unsqueeze(0), label

    def len(self):

        return self.num_samples

    @staticmethod
    def _loadvideo(filename):

        if not os.path.exists(filename):
            raise FileNotFoundError(filename)
        capture = cv2.VideoCapture(filename)

        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

        v = np.zeros((frame_height, frame_width, frame_count), np.uint8)

        for count in range(frame_count):
            ret, frame = capture.read()
            if not ret:
                raise ValueError("Failed to load frame #{} of {}.".format(count, filename))

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            v[:, :, count] = frame

        return v
