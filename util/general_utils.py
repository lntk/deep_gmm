import csv
import json
import os
import pickle
import random
import string
from shutil import copyfile, rmtree
from os.path import dirname, basename


def copy_file(src, dst):
    copyfile(src, dst)


def move_file(src, dst):
    copyfile(src, dst)
    os.remove(src)


def move_dir(src, dst, verbose=False):
    src_base = basename(src)
    dst_base = basename(dst)
    if src_base != dst_base:
        raise Exception("Two directories do not have a same name.")

    # MAKE SURE DESTINATION EXIST
    create_directory(dst)

    src_files = get_all_files(src)
    dst_files = [f"{dst}/{src_file}" for src_file in src_files]
    src_files = [f"{src}/{src_file}" for src_file in src_files]

    for src_file, dst_file in zip(src_files, dst_files):
        move_file(src_file, dst_file)
        if verbose:
            print(f"Moved {src_file} to {dst_file}.")

    # DELETE SOURCE
    rmtree(src)


def create_directory(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
    return directory


def write_list(l, file_name):
    with open(file_name, 'w') as file_handle:
        json.dump(l, file_handle)


def read_list(file_name):
    with open(file_name, 'r') as file_handle:
        l = json.load(file_handle)
        return l


def get_all_files(directory, keep_dir=False):
    """
    :param keep_dir:
    :param directory: A directory
    :return: List of file names in the directory
    """

    if keep_dir:
        files = [f"{directory}/{f}" for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]
    else:
        files = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]

    return files


def get_all_paths(directory, keep_dir=False):
    if keep_dir:
        paths = [f"{directory}/{d}" for d in os.listdir(directory)]
    else:
        paths = [d for d in os.listdir(directory)]
    return paths


def rename_files_in_directory(directory, prefix, suffix):
    """
    This function renames (by enumerating) all files in a directory
    E.g.:
    If:
    - prefix = 'karyotype'
    - suffix = '.bmp'
    then:
        '123132', '12312', '2132' --> karyotype_1.bmp, karyotype_2.bmp, karyotype_3.bmp

    :param directory:
    :param prefix:
    :param suffix:
    :return:
    """
    files = get_all_files(directory)
    for i in range(len(files)):
        file_name = directory + "/" + files[i]
        new_file_name = directory + "/" + prefix + "_" + str('{:05}'.format(i + 1)) + suffix
        os.rename(file_name, new_file_name)


def delete_file(filename):
    os.remove(filename)


def delete_files_in_dir(directory):
    files = get_all_files(directory)
    for file in files:
        delete_file(directory + "/" + file)


def read_lines(file_name):
    with open(file_name) as f:
        lines = f.readlines()
    lines = [x.strip() for x in lines]
    return lines


def write_lines(l, file_name):
    with open(file_name, 'w') as f:
        for item in l:
            f.write("%s\n" % item)


def write_list_to_csv(csvData, file_name):
    with open(file_name, 'w', newline='') as csvFile:
        writer = csv.writer(csvFile)
        writer.writerows(csvData)

    csvFile.close()


def random_string(n):
    return ''.join(random.choices(string.ascii_uppercase + string.digits, k=n))


def path_leaf(path):
    return os.path.basename(path)

    # head, tail = ntpath.split(path)
    # return tail or ntpath.basename(head)


def save_object(obj, file_name):
    # CREATE DIRECTORY IF NOT EXIST
    curr_dir = dirname(file_name)
    create_directory(curr_dir)

    with open(file_name, "wb") as file_out:
        pickle.dump(obj, file_out, pickle.HIGHEST_PROTOCOL)


def read_object(file_name):
    with open(file_name, "rb") as file_in:
        return pickle.load(file_in)


def get_all_files_in_tree(dirName):
    """
    For the given path, get the List of all files in the directory tree

    :param dirName:
    :return:
    """
    # create a list of file and sub directories
    # names in the given directory
    listOfFile = os.listdir(dirName)
    allFiles = list()
    # Iterate over all the entries
    for entry in listOfFile:
        # Create full path
        fullPath = os.path.join(dirName, entry)
        # If entry is a directory then get the list of files in this directory
        if os.path.isdir(fullPath):
            allFiles = allFiles + get_all_files_in_tree(fullPath)
        else:
            allFiles.append(fullPath)

    return allFiles