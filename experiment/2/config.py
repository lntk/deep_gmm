import torch

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
SEQUENCE_LENGTH = 1
INPUT_SIZE = (128, 128)
BATCH_SIZE = 8
PATH_TO_CD2014 = "data/CDnet2014"
TRAIN_FILE = "data/CDnet2014_train.txt"
NUM_EPOCH = 50
NUM_COMPONENT = 2
FOREGROUND_THRESHOLD = 0.001
SCALING_FACTOR = 100
MAIN_DIR = "..."