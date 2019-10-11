import torch

DEVICE = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
SEQUENCE_LENGTH = 20
INPUT_SIZE = (128, 128)
BATCH_SIZE = 4
PATH_TO_CD2014 = "data/CDnet2014"
TRAIN_FILE = "data/CDnet2014_train.txt"
NUM_EPOCH = 50
NUM_COMPONENT = 2
FOREGROUND_THRESHOLD = 0.001
SCALING_FACTOR = 100
MAIN_DIR = "..."
TEST_THRESHOLD = 0.5