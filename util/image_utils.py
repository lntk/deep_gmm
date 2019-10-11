import os
from os.path import isfile, join

import cv2
import numpy as np
from matplotlib import pyplot as plt


def frames_to_video(frames, video_path, fps=24):    
    height, width = frames[0].shape[:2]
    size = (width, height)
    out = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'DIVX'), fps, size)    
    for i in range(len(frames)):
        # writing to an image array        
        out.write(frames[i].astype('uint8'))
    out.release()


def frame_files_to_video(frames_path, video_path, fps=24):
    files = [f for f in os.listdir(frames_path) if isfile(join(frames_path, f))]    

    # for sorting the file names properly
    # files.sort(key = lambda x: x[5:-4])
    files.sort()
    frame_array = []
    files = [f for f in os.listdir(frames_path) if isfile(join(frames_path, f))]

    # for sorting the file names properly
    files.sort()
    for i in range(len(files)):
        filename = frames_path + files[i]
        # reading each files
        img = cv2.imread(filename)
        height, width, layers = img.shape
        size = (width, height)

        # inserting the frames into an image array
        frame_array.append(img)

    out = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'DIVX'), fps, size)
    for i in range(len(frame_array)):
        # writing to a image array
        out.write(frame_array[i])
    out.release()


def show_images(images, cmaps=None, titles=None, show_ticks=False, show_grids=False, fig_size=(10, 8)):
    num_plot = len(images)

    plt.figure(figsize=fig_size)
    for i in range(num_plot):
        plt.subplot(1, num_plot, i + 1)

        if cmaps is None:
            plt.imshow(images[i])
        else:
            plt.imshow(images[i], cmap=cmaps[i])

        if not show_ticks:
            plt.xticks([])
            plt.yticks([])

        if not show_grids:
            plt.grid(False)

        if titles is not None:
            plt.title(titles[i])

    plt.show()
    plt.close()


def last_axis_reorder_by_indices(a, indices):
    shape = a.shape
    num_element = np.prod(shape[1:]).item()

    shift = np.array(range(num_element)).reshape(shape[1:])
    sorted_indices = np.array([shift + sorted_indices_patch * num_element for sorted_indices_patch in indices])

    a = a.flatten()
    sorted_indices = sorted_indices.flatten()
    a = a[sorted_indices]

    return a.reshape(shape)


def clear_dir(folder):
    for the_file in os.listdir(folder):
        file_path = os.path.join(folder, the_file)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
            # elif os.path.isdir(file_path): shutil.rmtree(file_path)
        except Exception as e:
            print(e)
