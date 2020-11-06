# Copyright (c) Ville de Montreal. All rights reserved.
# Licensed under the MIT license.
# See LICENSE file in the project root for full license information.

import os
import shutil
import argparse
import numpy as np
from PIL import Image
from labels import CGMU_LABELS


def move_images(source, destination):
    for root, dirs, files in os.walk(source):
        if len(dirs) == 0 and files[0] != "classes.json":
            if root.split("_")[-1] != "CamOcclude" and root.split("_")[-1] != "MotherFrame":
                motherframe = root.split("/")[-1]
                prefix = motherframe.split(".")[1] + "_" + motherframe.split(".")[0] + "_"
                suffix = ".jpeg___fuse"

                if "train" in root.lower():
                    split = "train"
                elif "valid" in root.lower():
                    split = "valid"
                elif "test" in root.lower():
                    split = "test"
                else:
                    print('Cannot find split for ' + root)
                    break

                destination_dir = os.path.join(destination, "Semantic/" + split + "/" + motherframe.split(".")[0])

                if os.path.isdir(destination_dir) is False:
                    os.mkdir(destination_dir)

                for filename in files:
                    if filename.split(".")[-1] == "png":
                        source_path = os.path.join(root, filename)

                        new_filename = filename.replace(prefix, "").replace(suffix, "")
                        destination_path = os.path.join(destination_dir, new_filename)

                        shutil.copy(source_path, destination_path)


def validate(dirpath):
    print("These files do not have a semantic equivalent:\n")

    for root, dirs, files in os.walk(os.path.join(dirpath, "RGB")):
        if len(files) > 0:
            for filename in files:
                semantic_path = os.path.join(root.replace("RGB", "Semantic"),
                                             filename.replace(".jpeg", ".png"))

                if os.path.isfile(semantic_path) is False:
                    print(os.path.join(root, filename))


def unpack_cgmu(dirpath):
    i = 0

    for root, dirs, files in os.walk(os.path.join(dirpath)):
        if len(files) > 0:
            for filename in files:
                rgb_path = os.path.join(root, filename)
                new_filename = str(i).zfill(6)

                path_elements = root.split("/")[0:-1]
                new_rgb_path = os.path.join(*path_elements, new_filename + ".jpeg")

                shutil.copy(rgb_path, new_rgb_path)

                i += 1


def redscaleify(dirpath):
    """
    Transforms all the RGB Semantic images into their Redshade version.
    """

    for root, dirs, files in os.walk(dirpath):
        if len(files) > 0:
            for filename in files:
                filepath = os.path.join(root, filename)

                img = np.array(Image.open(filepath))

                if img.shape[2] > 3:
                    img = img[:, :, 0:3]

                for label in CGMU_LABELS:
                    img[np.all(img == label[2:5], axis=2)] = [label[1], 0, 0]

                img = Image.fromarray(img)

                img.save(filepath)


if __name__ == "__main__":
    SA_PATH = 'Images/CGMU_Final/'
    CGMU_PATH = 'Images/CGMU/'

    parser = argparse.ArgumentParser(
        description='Unpack the CGMU images to extract Semantic labels')
    parser.add_argument('-s', '--source', default=SA_PATH, type=str,
                        help='Path the annotated data dir (ex: Images/SA_Montreal_Submission/)')
    parser.add_argument('-d', '--destination', default=CGMU_PATH, type=str,
                        help='Path the CGMU data dir (default: Images/CGMU/)')
    parser.add_argument('-v', '--validate', action='store_true',
                        help='Only valides if all files match between RGB and Semantic in Destination')

    args = parser.parse_args()

    # redscaleify("Images/CGMU/Semantic/")
    unpack_cgmu('Images/CGMU/RGB/extra')


