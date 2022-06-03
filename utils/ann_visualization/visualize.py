# -*- coding: utf-8 -*-
# ---------------------

import json
import os
import sys
from typing import *

import click
import imageio
import matplotlib.pyplot as plt
import numpy as np
from path import Path

from utils.ann_visualization.joint import Joint
from utils.ann_visualization.pose import Pose
import cv2

from utils.ann_visualization.utils_imavis import parse_cvat_images_xml

MAX_COLORS = 42

# check python version
assert sys.version_info >= (3, 6), '[!] This script requires Python >= 3.6'


def get_colors(number_of_colors, cmap_name='rainbow'):
    # type: (int, str) -> List[List[int]]
    """
    :param number_of_colors: number of colors you want to get
    :param cmap_name: name of the colormap you want to use
    :return: list of 'number_of_colors' colors based on the required color map ('cmap_name')
    """
    colors = plt.get_cmap(cmap_name)(np.linspace(0, 1, number_of_colors))[:, :-1] * 255
    return colors.astype(int).tolist()


def get_pose(frame_data, person_id):
    # type: (np.ndarray, int) -> Pose
    """
    :param frame_data: data of the current frame
    :param person_id: person identifier
    :return: list of joints in the current frame with the required person ID
    """
    pose = [Joint(j) for j in frame_data[frame_data[:, 1] == person_id]]
    pose.sort(key=(lambda j: j.type))
    return Pose(pose)


H1 = 'path of the video you want to visualize annotations'
H2 = 'path of JSON containing the annotations you want to visualize'
H3 = 'path of the output video with the annotations'
H4 = 'if `hide` the annotations of people completely occluded by objects will not be displayed in the output video'


# @click.command()
# @click.option('--in_mp4_file_path', type=click.Path(exists=True), prompt='Enter \'in_mp4_file_path\'', help=H1)
# @click.option('--json_file_path', type=click.Path(exists=True), prompt='Enter \'json_file_path\'', help=H2)
# @click.option('--out_mp4_file_path', type=click.Path(), prompt='Enter \'out_mp4_file_path\'', help=H3)
# @click.option('--hide/--no-hide', default=True, help=H4)
def visualize(in_mp4_file_path, xml_file_path, out_mp4_file_path, hide, plot_bbox=False):
    """
    Script that provides a visual representation of the annotations
    """
    out_mp4_file_path = Path(out_mp4_file_path)
    if not out_mp4_file_path.parent.exists() and out_mp4_file_path.parent != Path(''):
        out_mp4_file_path.parent.makedirs()

    reader = imageio.get_reader(in_mp4_file_path)
    writer = imageio.get_writer(out_mp4_file_path, fps=20)

    # with open(json_file_path, 'r') as json_file:
    #    data = json.load(json_file)
    #    data = np.array(data)

    colors = get_colors(number_of_colors=MAX_COLORS, cmap_name='jet')

    detections_by_frame = parse_cvat_images_xml(Path(xml_file_path))

    print(f'▸ visualizing annotations of \'{Path(in_mp4_file_path).abspath()}\'')
    for frame_number, image in enumerate(reader):

        for det in detections_by_frame[frame_number]:

            # select pose color base on its unique identifier
            color = colors[int(25) % len(colors)]

            # draw pose on image

            if plot_bbox:
                # bbox = np.array(pose.bbox_2d_padded).astype(int)

                bbox = np.array(det.points).astype(int)
                image = cv2.rectangle(image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)

            else:
                pass  # image = pose.draw(image=image, color=color)

        writer.append_data(np.vstack([image, image[-8:, :]]))
        print(f'\r▸ progress: {100 * (frame_number / 1800):6.2f}%', end='')

    writer.close()
    print(f'\n▸ video with annotations: \'{out_mp4_file_path.abspath()}\'\n')


if __name__ == '__main__':

    for j in range(0, 16):
        seq_path = f"C:\\Users\\simoc\\Desktop\\Synthetic Data IMAVIS\\seq_{j}"
        visualize(in_mp4_file_path=os.path.join(seq_path, f"seq_{j}.mp4"),
                  xml_file_path=os.path.join(seq_path, f"seq_{j}_imavis.xml"),
                  out_mp4_file_path=os.path.join(seq_path, f"res_seq_{j}_PROVA.mp4"), hide=True, plot_bbox=True)

        exit()
