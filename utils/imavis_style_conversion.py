import json
import os

import numpy as np
from utils.ann_visualization.joint import Joint
from utils.ann_visualization.pose import Pose
from xml.dom import minidom

import xml.etree.ElementTree as gfg
from xml.etree import ElementTree

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


def prettify(elem):
    """Return a pretty-printed XML string for the Element.
    """
    rough_string = ElementTree.tostring(elem, 'utf-8')
    reparsed = minidom.parseString(rough_string)
    return reparsed.toprettyxml(indent="\t")


def create_xml_root(name_root: str):
    return gfg.Element(name_root)


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


def json_imavis_style_conversion(json_file_path):
    """
    Script that provides a visual representation of the annotations
    """

    with open(json_file_path, 'r') as json_file:
        data = json.load(json_file)
        data = np.array(data)

    n_seq = json_file_path.split(os.sep)[-1].split(".")[0].split("_")[1]

    n_frames = int(data[-1][0])

    xml_root = create_xml_root("annotations")

    for frame_number in range(n_frames + 1):

        # Get all the data for a given frame
        frame_data = data[data[:, 0] == frame_number]  # type: np.ndarray

        image_node = get_image_node(id_frame=frame_number,
                                    frame_name=f"{frame_number}.jpg",
                                    frame_width=FRAME_WIDTH,
                                    frame_height=FRAME_HEIGHT)

        for p_id in set(frame_data[:, 1]):
            # if the "hide" flag is set, ignore the "invisible" poses
            # (invisible pose = pose of which I do not see any joint)
            # if hide and pose.invisible:
            #    continue

            pose = get_pose(frame_data=frame_data, person_id=p_id)

            if pose.head_not_visible:
                occluded = 0
            else:
                occluded = 1


            # exit()
            # get bbox of the ped:  ( x, y, width, height )
            bbox = np.array(pose.bbox_2d_padded).astype(int)
            x, y, width, height = bbox

            image_node.append(get_box_node(LABEL_MAP[1], x, y, x + width, y + height, occluded=occluded))

        xml_root.append(image_node)
        print(f'\r▸"Annotation seq_{n_seq} progress: {100 * (frame_number / n_frames):6.2f}%', end='')

    xml_str = prettify(xml_root)

    with open("prova.xml", "w") as f:
        f.write(xml_str)


if __name__ == "__main__":
    json_imavis_style_conversion("C:\\Users\\simoc\\Desktop\\Synthetic Data IMAVIS\\seq_0\\seq_0.json")
