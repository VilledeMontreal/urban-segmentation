# Copyright (c) Ville de Montreal. All rights reserved.
# Licensed under the MIT license.
# See LICENSE file in the project root for full license information.

import os
import cv2
import numpy as np
from PIL import Image
import pyperclip
from labels import CITYSCAPE_LABELS, CARLA_LABELS, CGMU_LABELS


WAITKEY = 10
DIFF_COLOR = [128, 255, 0]


def print_help():
    print('*** Key mapping ***')
    print('Displaying images')
    print('\tI to show Image')
    print('\tL to show Labels')
    print('\tO to show overlap of Labels and Image')
    print('\tD to show changed pixels since first QA round')
    print('Resizing images')
    print('\t1 to show the normal size')
    print('\t2 to resize the image to 200%')
    print('\t3 to resize the image to 300%')
    print('Video playback')
    print('\tX to go back 10 images')
    print('\tC to toggle video playback')
    print('\tV to skip 10 images')
    print('\tM to skip to the next motherframe')
    print('\tQ to quit')
    print('Press H in image window to display this message again')


def load_pair(img_path, sem_path):
    rgb = np.array(Image.open(img_path))[:, :, 0:3][:, :, ::-1]
    sem = np.array(Image.open(sem_path))[:, :, 0:3][:, :, ::-1]

    cv2.namedWindow('image')
    cv2.imshow('image', rgb)

    while(True):
        keyPress = cv2.waitKey(0)

        if keyPress == ord('q'):
            break

        elif keyPress == ord('m'):
            cv2.imshow('image', sem)

        elif keyPress == ord('o'):
            overlapped = np.array((rgb / 2) + (sem / 2), dtype="uint8")
            cv2.imshow('image', overlapped)

        else:
            cv2.imshow('image', rgb)

    cv2.destroyAllWindows


def load_vid():
    image_list = np.loadtxt('Images/RGB/data.txt', str)
    i = 1000

    showMask = False
    showOverlap = False
    waitKey = 100

    cv2.namedWindow('image')

    while(True):
        if i >= len(image_list):
            i = 0

        elif i < 0:
            i = 0

        rgb = np.array(Image.open('Images/RGB/' + image_list[i]))[:, :, 0:3][:, :, ::-1]
        msk = np.zeros(rgb.shape, dtype='uint8')

        if showMask or showOverlap:
            ground_truth = np.array(Image.open('Images/Semantic/' + image_list[i]))[:, :, 0]

            for label in range(len(CARLA_LABELS)):
                ix = ground_truth[0:ground_truth.shape[0], 0:ground_truth.shape[1]] == label
                msk[ix] = CARLA_LABELS[label][2:5]

            msk = msk[:, :, ::-1]

        if showMask:
            cv2.imshow('image', msk)
        elif showOverlap:
            red = (rgb / 2 + msk / 2).astype(dtype='uint8')
            overlapped = red + 0
            cv2.imshow('image', overlapped)
        else:
            cv2.imshow('image', rgb)

        keyPress = cv2.waitKey(waitKey)

        if keyPress == ord('q'):
            break

        elif keyPress == ord('m'):
            showMask = not showMask
            showOverlap = False
            waitKey = 1

        elif keyPress == ord('o'):
            showOverlap = not showOverlap
            showMask = False
            waitKey = 1

        elif keyPress == ord('i'):
            showMask = False
            showOverlap = False
            waitKey = 100

        elif keyPress == ord('l'):
            i += 200

        elif keyPress == ord('j'):
            i -= 200

        i += 1

    cv2.destroyAllWindows


def load_label(name, labels):

    gTruth = np.array(Image.open(name))

    rgb = make_rgb(gTruth, labels)

    cv2.namedWindow('image')
    cv2.imshow('image', rgb)

    while(True):
        keyPress = cv2.waitKey(0)

        if keyPress == ord('q'):
            break

    cv2.destroyAllWindows


def make_rgb(image, labels):
    rgb = np.zeros(image.shape + (3,))

    for label in range(len(labels)):
        rgb[image == label] = labels[label][2:5][::-1]

    return rgb.astype('uint8')


def filter_files(filename):
    filtered = False

    if filename[-8:] == "fuse.png":
        filtered = True

    if "CamOcclude" in filename:
        filtered = True

    if "MotherFrame" in filename:
        filtered = True

    return not filtered


def file_pairs(oldpath):
    for oldroot, _, oldfiles in os.walk(oldpath):
        for oldfile in oldfiles:
            yield oldroot, oldfile


def load_SA(path, oldpath):
    for root, dirs, files in os.walk(path):
        dirs.sort()
        if len(files) > 0:
            files.sort()
            image_list = [filename for filename in files if filter_files(filename)]
            print(root[-5:])

            i = 0
            showMask = False
            showOverlap = False
            showDiff = False
            waitKey = 0
            mega_break = False
            size_factor = 1

            cv2.namedWindow('image')

            while(True):
                if i >= len(image_list):
                    i = 0

                elif i < 0:
                    i = 0

                rgb = np.array(Image.open(root + "/" + image_list[i]))[:, :, 0:3][:, :, ::-1]
                pyperclip.copy(image_list[i])
                rgb = resize_n(rgb, size_factor)
                msk = np.zeros(rgb.shape, dtype='uint8')
                ground_truth = np.array(Image.open(root + "/" + image_list[i] + "___fuse.png"))[:, :, 0:3]
                ground_truth = resize_n(ground_truth, size_factor)

                if showMask or showOverlap:
                    for label in range(len(CGMU_LABELS)):
                        msk[class_mask(ground_truth, label)] = CGMU_LABELS[label][2:5]

                    msk = msk[:, :, ::-1]

                if showDiff:
                    for oldroot, oldfile in file_pairs(oldpath):
                        if image_list[i][-26:] == oldfile[-26:]:
                            old_gt = np.array(Image.open(oldroot + "/" + oldfile + "___fuse.png"))[:, :, 0:3]
                            old_gt = resize_n(old_gt, size_factor)
                            break

                    diff_msk = np.invert(np.equal(np.sum(ground_truth, axis=2),
                                                  np.sum(old_gt[:, :], axis=2)))

                    rgb[diff_msk] = DIFF_COLOR
                    if showMask or showOverlap:
                        msk[diff_msk] = DIFF_COLOR

                if showMask:
                    cv2.imshow('image', msk)
                elif showOverlap:
                    red = (rgb / 2 + msk / 2).astype(dtype='uint8')
                    overlapped = red + 0
                    cv2.imshow('image', overlapped)
                else:
                    cv2.imshow('image', rgb)

                keyPress = cv2.waitKey(waitKey)

                if keyPress == ord('q'):
                    mega_break = True
                    break

                if keyPress == ord('m'):
                    break

                if keyPress == ord('h'):
                    i -= 1
                    print_help()

                elif keyPress == ord('l'):
                    showMask = not showMask
                    showOverlap = False
                    i -= 1

                elif keyPress == ord('o'):
                    showOverlap = not showOverlap
                    showMask = False
                    i -= 1

                elif keyPress == ord('i'):
                    showMask = False
                    showOverlap = False
                    i -= 1

                elif keyPress == ord('d') and oldpath is not None:
                    showDiff = not showDiff
                    i -= 1

                elif keyPress == ord('v'):
                    i += 9

                elif keyPress == ord('x'):
                    i -= 11

                elif keyPress == ord('c'):
                    waitKey = WAITKEY - waitKey

                elif keyPress == ord('1'):
                    size_factor = 1
                    i -= 1

                elif keyPress == ord('2'):
                    size_factor = 2
                    i -= 1

                elif keyPress == ord('3'):
                    size_factor = 3
                    i -= 1

                i += 1

            cv2.destroyAllWindows

            if mega_break:
                break


def class_mask(ground_truth, label):
    label_rgb = CGMU_LABELS[label][2:5]

    bool_rgb = np.split((ground_truth == label_rgb), 3, axis=2)

    class_mask = np.squeeze(np.logical_and(np.logical_and(bool_rgb[0], bool_rgb[1]),
                                           bool_rgb[2]))

    return class_mask


def compare_duplicates(path1, path2):
    img1 = np.array(Image.open(path1))[:, :, 0:3]
    img2 = np.array(Image.open(path2))[:, :, 0:3]

    bool_rgb = np.split(img1[:, :] != img2[:, :], 3, axis=2)

    comp = 255 * np.squeeze(np.logical_and(np.logical_and(bool_rgb[0], bool_rgb[1]),
                                           bool_rgb[2])).astype('uint8')

    cv2.namedWindow('image')
    cv2.imshow('image', comp)

    while(True):
        keyPress = cv2.waitKey(0)

        if keyPress == ord('q'):
            break

    cv2.destroyAllWindows


def resize_n(img, factor):
    if factor != 1:
        new_img = np.zeros((img.shape[0] * factor, img.shape[1] * factor, 3))
        for i in range(factor):
            new_img[i::factor, 0::factor, :] = img

        for j in range(1, factor):
            new_img[:, j::factor, :] = new_img[:, 0::factor, :]

        return new_img.astype('uint8')

    else:
        return img


def target_str(string):
    return string.replace("RGB", "Semantic").replace("jpeg", "png")


def split_CGMU(datadir):
    sizes = np.array([0, 0])
    for root, dirs, files in os.walk(datadir):
        for f in files:
            filepath = os.path.join(root, f)
            size = Image.open(filepath).size
            sizes = np.vstack([sizes, np.array(size)])

            if size[0] == 704:
                os.rename(filepath, filepath.replace("CGMU", "CGMU_L"))
                os.rename(target_str(filepath),
                          target_str(filepath).replace("CGMU", "CGMU_L"))

    print(np.unique(sizes, axis=0))


if __name__ == "__main__":
    # Select a dir where images are located
    rgb_dir = "Images/CGMU/RGB"
    sem_dir = os.path.join(rgb_dir, "Input")

    files_rgb = [filename for filename in os.listdir(rgb_dir) if os.path.isfile(os.path.join(rgb_dir, filename))]
    files_sem = [filename for filename in os.listdir(sem_dir) if os.path.isfile(os.path.join(sem_dir, filename))]

    for filename in files_rgb:
        path_rgb = os.path.join(rgb_dir, filename) 
        path_sem = os.path.join(sem_dir, [file_sem for file_sem in files_sem if filename in file_sem][0])

        load_pair(path_sem, path_rgb)
