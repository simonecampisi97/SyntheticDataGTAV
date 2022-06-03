import json
import os

import numpy as np
from utils.ann_visualization.joint import Joint
from utils.ann_visualization.pose import Pose
from xml.dom import minidom
from path import Path
import cv2

import xml.etree.ElementTree as gfg
from xml.etree import ElementTree

from utils.ann_visualization.visualize import get_colors

MAX_COLORS = 42

LABEL_MAP = dict(
    person=1,
    dog=2,
    car=3,
    truck=4,
    motorcycle=5,
    bicycle=6,
)


def prettify(elem):
    """Return a pretty-printed XML string for the Element.
    """
    rough_string = ElementTree.tostring(elem, 'utf-8')
    reparsed = minidom.parseString(rough_string)
    return reparsed.toprettyxml(indent="\t")


def create_xml_root(name_root):
    return gfg.Element(name_root)


def get_image_node(id_frame: int, frame_name: str, frame_width: int, frame_height: int):
    image_node = gfg.Element("image")
    image_node.set("id", str(id_frame))
    image_node.set("name", frame_name)
    image_node.set("width", str(frame_width))
    image_node.set("height", str(frame_height))

    return image_node


def get_box_node(top_elem, label: int, xtl: int, ytl: int, xbr: int, ybr: int):
    box_node = gfg.SubElement(top_elem, "box")
    box_node.set("label", str(label))
    box_node.set("xtl", str(xtl))
    box_node.set("ytl", str(ytl))
    box_node.set("xbr", str(xbr))
    box_node.set("ybr", str(ybr))

    return box_node


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


def json_imavis_style_conversion(json_file_path, seq_path):
    """
    Script that provides a visual representation of the annotations
    """

    with open(json_file_path, 'r') as json_file:
        data = json.load(json_file)
        data = np.array(data)

    n_frames = int(data[-1][0])

    colors = get_colors(number_of_colors=MAX_COLORS, cmap_name='jet')

    new_data = []

    for frame_number in range(n_frames + 1):

        # NOTE: frame #0 does NOT exist: first frame is #1
        frame_data = data[data[:, 0] == frame_number]  # type: np.ndarray

        for p_id in set(frame_data[:, 1]):
            pose = get_pose(frame_data=frame_data, person_id=p_id)

            # get bbox of the ped:  ( x, y, width, height )
            bbox = np.array(pose.bbox_2d_padded).astype(int)
            row = np.concatenate([[frame_number, p_id], bbox])

            new_data.append(list(row))

        print(f'\r▸"Annotation seq_{j} progress: {100 * (frame_number / n_frames):6.2f}%', end='')

    with open(os.path.join(seq_path, f"seq_{j}_imavis.json"), "w") as f:
        json.dump(new_data, f)


if __name__ == "__main__":
    root = create_xml_root("annotations")

    img = get_image_node(0, "prova1", 1920, 1080)
    img.append(get_box_node(img, 0, 12, 45, 300, 450))
    img.append(get_box_node(img, 0, 12, 45, 300, 450))
    img.append(get_box_node(img, 0, 12, 45, 300, 450))

    root.append(img)

    xml_str = prettify(root)

    with open("prova.xml", "w") as f:
        f.write(xml_str)
