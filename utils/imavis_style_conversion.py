import json
import os

import numpy as np
from utils.ann_visualization.joint import Joint
from utils.ann_visualization.pose import Pose
from xml.dom import minidom

import xml.etree.ElementTree as gfg
from xml.etree import ElementTree
from path import Path

MAX_COLORS = 42

FRAME_WIDTH = 1920
FRAME_HEIGHT = 1080

LABEL_MAP = {
    1: "person",
    2: "dog",
    3: "car",
    4: "truck",
    5: "motorcycle",
    8: "bicycle",
}


# ["id", "name", "overlap", "bugtracker", "created", "updated", "frame_filter", "segments", "owner", "assignee"]:

def prettify(elem):
    """Return a pretty-printed XML string for the Element.
    """
    rough_string = ElementTree.tostring(elem, encoding='utf8')
    reparsed = minidom.parseString(rough_string)
    return reparsed.toprettyxml(indent="\t")


def create_xml_root(name_root: str):

    root = gfg.Element(name_root)
    root.append(single_text_elem("version", 1.1))

    return root


def single_text_elem(name_elem, text):
    elem = gfg.Element(name_elem)
    elem.text = str(text)
    return elem


def label_elem_meta(name):
    label = gfg.Element("label")
    label.append(single_text_elem("name", name))
    label.append(single_text_elem("attributes", ""))

    return label


def create_meta_xml(n_frames: int):

    meta_node = gfg.Element("meta")

    task_node = gfg.Element("task")
    meta_node.append(task_node)

    task_node.append(single_text_elem("size", n_frames))  # Size Filed
    task_node.append(single_text_elem("mode", "annotation"))  # Size Filed
    task_node.append(single_text_elem("start_frame", 0))  # Start Frame Filed
    task_node.append(single_text_elem("end_frame", n_frames - 1))  # End Frame Filed

    # Labels
    labels = gfg.Element("labels")

    task_node.append(labels)

    for e in LABEL_MAP.values():
        print(e)
        labels.append(label_elem_meta(e))



    return meta_node


def get_image_node(id_frame: int, frame_name: str, frame_width: int, frame_height: int):
    image_node = gfg.Element("image")
    image_node.set("id", str(id_frame))
    image_node.set("name", frame_name)
    image_node.set("width", str(frame_width))
    image_node.set("height", str(frame_height))

    return image_node


def get_box_node(label: str, xtl: int, ytl: int, xbr: int, ybr: int, occluded=0):
    box_node = gfg.Element("box")
    box_node.set("label", label)
    box_node.set("occluded", str(occluded))
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


def json_imavis_style_conversion(json_file_path, out_folder):
    """
    Script that provides a visual representation of the annotations
    """

    with open(json_file_path, 'r') as json_file:
        data = json.load(json_file)
        data = np.array(data)

    n_seq = json_file_path.split(os.sep)[-1].split(".")[0].split("_")[1]

    n_frames = int(data[-1][0]) + 1

    xml_root = create_xml_root("annotations")
    xml_root.append(create_meta_xml(n_frames))

    for frame_number in range(n_frames):

        # Get all the data for a given frame
        frame_data = data[data[:, 0] == frame_number]  # type: np.ndarray

        image_node = get_image_node(id_frame=frame_number,
                                    frame_name=f"{frame_number}.jpg",
                                    frame_width=FRAME_WIDTH,
                                    frame_height=FRAME_HEIGHT)

        for p_id in set(frame_data[:, 1]):

            pose = get_pose(frame_data=frame_data, person_id=p_id)

            if pose.head_not_visible or pose.half_not_visible:
                continue

            bbox = np.array(pose.bbox_2d).astype(int)
            x, y, width, height = bbox

            image_node.append(get_box_node(LABEL_MAP[1], x, y, x+width, y+width))

        xml_root.append(image_node)
        print(f'\r▸"Annotation seq_{n_seq} progress: {100 * (frame_number / (n_frames - 1)):6.2f}%', end='')

    xml_str = prettify(xml_root)

    with open(os.path.join(out_folder, f"seq_{n_seq}_imavis.xml"), "w") as f:
        f.write(xml_str)


if __name__ == "__main__":
    folder_data = "C:\\Users\\simoc\\Desktop\\Synthetic Data IMAVIS\\"

    for dir in Path(folder_data).dirs():

        name_dir = dir.split(os.sep)[-1]

        for file in dir.files():
            if name_dir + ".json" in file:
                json_imavis_style_conversion(file, dir)

        exit()
