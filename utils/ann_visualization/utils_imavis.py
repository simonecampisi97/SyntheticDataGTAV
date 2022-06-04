import codecs
from typing import Tuple, List
from venv import logger

from xml.etree import ElementTree as ET
from path import Path

LABEL_MAP = dict(
    person=1,
    dog=2,
    car=3,
    truck=4,
    motorcycle=5,
    bicycle=6,
)


class Detection(object):
    _tl_x: float
    _tl_y: float
    _br_x: float
    _br_y: float
    _centroid_x: float
    _centroid_y: float
    _label: str
    _invisible: bool
    _occluded: bool

    def __init__(self, tl_x: float, tl_y: float, br_x: float, br_y: float, label: str, invisible: bool, occluded: bool):
        assert 0 <= tl_x, tl_x
        assert 0 <= tl_y, tl_y
        assert 0 <= br_x, br_x
        assert 0 <= br_y, br_y
        self._centroid_x = ((tl_x + br_x) / 2.)
        self._centroid_y = ((tl_y + br_y) / 2.)
        self._tl_x = tl_x
        self._tl_y = tl_y
        self._br_x = br_x
        self._br_y = br_y
        width = self._br_x - self._tl_x
        height = self._br_y - self._tl_y
        assert width >= 0, width
        assert height >= 0, height

        try:
            assert width * height >= 0, width * height
        except:
            print("width * height: ", width * height)
            print(self._tl_x, self._tl_y, self._br_x, self._br_y)
            exit()

        if label == "cat" and "cat" not in LABEL_MAP:
            label = "dog"
        if label == "bus":
            label = "truck"
        if label == "motorbike":
            label = "motorcycle"
        assert label in LABEL_MAP
        self._label = label
        self._invisible = invisible
        self._occluded = occluded

    def is_occluded(self) -> bool:
        return self._occluded

    def is_invisible(self) -> bool:
        return self._invisible

    def get_centroid(self) -> Tuple[float, float]:
        return self._centroid_x, self._centroid_y

    @property
    def points(self) -> Tuple[float, float, float, float]:
        return self._tl_x, self._tl_y, self._br_x, self._br_y

    @property
    def label(self) -> str:
        return self._label

    @property
    def label_id(self) -> int:
        return LABEL_MAP[self._label]


def parse_cvat_images_xml(xml_path: Path) -> List[List[Detection]]:
    logger.info(f"processing xml {xml_path.name}")

    with codecs.open(str(xml_path), 'r', encoding='utf-8', errors='replace') as fh:
        xml = fh.read()

    xml = xml[xml.index('<?xml'):xml.index('</annotations>') + len('</annotations>')]

    root = ET.XML(xml)

    frame_count = int(root.find("meta/task/size").text)

    detections_by_frame = [[] for _ in range(frame_count)]

    # all the images must have same dimension
    dimensions = set()
    for image_tag in root.findall('image'):
        frame_index = int(image_tag.attrib["id"])
        frame_width = int(image_tag.attrib["width"])
        frame_height = int(image_tag.attrib["height"])
        dimensions.add((frame_width, frame_height))

        for box_tag in image_tag.findall('box'):
            f1 = float(box_tag.attrib["xtl"])
            f2 = float(box_tag.attrib["ytl"])
            f3 = float(box_tag.attrib["xbr"])
            f4 = float(box_tag.attrib["ybr"])

            if f1 < 0:
                print("frame width f1 < 0: ", f1)

            if f3 < 0:
                print("frame width f3 < 0: ", f3)

            if f2 < 0:
                print("frame height f2 < 0: ", f2)

            if f4 < 0:
                print("frame height f4 < 0: ", f4)

            if f1 > frame_width:
                print("frame width f1 > width: ", f1)

            if f3 > frame_width:
                print("frame width f3 > width: ", f3)

            if f2 > frame_height:
                print("frame height f2 > height: ", f2)

            if f4 > frame_height:
                print("frame height f4 > height: ", f4)

            if not (0 <= f1 <= frame_width) or not (0 <= f3 <= frame_width) or not (0 <= f2 <= frame_height) or not (
                    0 <= f4 <= frame_height):
                logger.warning(f"task {str(xml_path)} has not valid bounding boxes")
                return [[] for _ in range(frame_count)]

            label = box_tag.attrib["label"]
            occluded = int(box_tag.attrib["occluded"])
            outside = False
            detections_by_frame[frame_index].append(Detection(f1, f2, f3, f4, label, outside, occluded == 1))

    if len(dimensions) != 1:
        # logger.warning(f"task {str(xml_path)} has images with different sizes")
        return [[] for _ in range(frame_count)]

    return detections_by_frame
