import argparse

parser = argparse.ArgumentParser(description="Visualize outputs of a model")
parser.add_argument("--result_file", help="Path of Keras model")
parser.add_argument("--to_gif", type=bool, default=False, help="Whether or not get a gif result")
parser.add_argument("--gif_path", default="result.gif")
parser.add_argument("--data_dir", default="")
parser.add_argument("--start_frame", type=int, default=0)

args = parser.parse_args()

"""
OUTPUT TO FRAMES
"""
to_frames = True

result = args.result_file

if to_frames:
    import pickle
    import cv2
    from tqdm import tqdm
    import numpy as np
    from util import general_utils

    with open(result, 'rb') as handle:
        outputs = pickle.load(handle)

    S, _, H, W = outputs.shape

    general_utils.create_directory("tmp")
    general_utils.delete_files_in_dir("tmp")

    for i in tqdm(range(S)):
        output = outputs[i, 0, :, :]
        output[output > 0.9] = 255.
        output = output.astype('uint8')
        cv2.imwrite(f"tmp/{'{:06d}'.format(i + 1)}.jpg", output)

"""
FRAMES TO VIDEO
"""
to_video = False

if to_video:
    import utils

    utils.frame_files_to_video(frames_path="result_3/", video_path="video/result_3.avi")

"""
FRAMES TO GIF

Resources:
- https://gifcompressor.com/
"""
to_gif = args.to_gif

view_train = False
# result = "result_16"

# data_dir = "/home/lntk/Desktop/VinAI/Computer vision/Background subtraction/code/data/CDnet2014_dataset"

# results_to_dirs = {
#     "result_13": "baseline/highway",
#     "result_14": "cameraJitter/badminton",
#     "result_15": "baseline/pedestrians",
#     "result_16": "badWeather/skating",
#     "result_17": "badWeather/snowFall",
# }

# dirs_to_first_idx = {
#     "baseline/highway": 300,
#     "cameraJitter/badminton": 800,
#     "baseline/pedestrians": 300,
#     "badWeather/skating": 800,
#     "badWeather/snowFall": 800
# }


if to_gif:
    from util import general_utils
    import imageio
    import numpy as np
    import cv2

    test_dir = args.data_dir

    start_idx = args.start_frame
    concat_axis = 0

    result_files = general_utils.get_all_files("tmp", keep_dir=True)
    result_files = sorted(result_files)

    if view_train:
        num_file = 240
    else:
        num_file = len(result_files)

    frames = list()
    for i in range(num_file):
        original = imageio.imread(f"{test_dir}/input/in{'{:06d}'.format(i + start_idx)}.jpg")
        original = cv2.resize(original, (128, 128))

        if view_train:
            foreground = imageio.imread(f"{test_dir}/groundtruth/gt{'{:06d}'.format(i + start_idx)}.png")
            foreground = cv2.resize(foreground, (128, 128))
            foreground = np.stack([foreground, foreground, foreground], axis=-1)
        else:
            foreground = imageio.imread(result_files[i])
            foreground = np.stack([foreground, foreground, foreground], axis=-1)

        # print(original.shape)
        # print(foreground.shape)

        frame = np.concatenate((original, foreground), axis=concat_axis)
        frames.append(frame)

    imageio.mimsave(args.gif_path, frames, fps=24, loop=0)
