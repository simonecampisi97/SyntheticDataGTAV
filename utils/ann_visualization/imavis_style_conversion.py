import json

import imageio
import numpy as np
from utils.ann_visualization.joint import Joint
from utils.ann_visualization.pose import Pose
from path import Path

from utils.ann_visualization.visualize import get_colors

MAX_COLORS = 42


def json_imavis_style_conversion(in_mp4_file_path, json_file_path, out_mp4_file_path, hide, bbox=False):
    """
    Script that provides a visual representation of the annotations
    """
    out_mp4_file_path = Path(out_mp4_file_path)
    if not out_mp4_file_path.parent.exists() and out_mp4_file_path.parent != Path(''):
        out_mp4_file_path.parent.makedirs()

    reader = imageio.get_reader(in_mp4_file_path)

    with open(json_file_path, 'r') as json_file:
        data = json.load(json_file)
        data = np.array(data)

    colors = get_colors(number_of_colors=MAX_COLORS, cmap_name='jet')

    print(f'▸ visualizing annotations of \'{Path(in_mp4_file_path).abspath()}\'')
    for frame_number, image in enumerate(reader):

        # NOTE: frame #0 does NOT exists: first frame is #1
        frame_data = data[data[:, 0] == frame_number]  # type: np.ndarray

        for p_id in set(frame_data[:, 1]):
            pose = get_pose(frame_data=frame_data, person_id=p_id)

            # if the "hide" flag is set, ignore the "invisible" poses
            # (invisible pose = pose of which I do not see any joint)
            if hide and pose.invisible:
                continue

            # select pose color base on its unique identifier
            color = colors[int(p_id) % len(colors)]

            # draw pose on image

            if bbox:
                bbox = np.array(pose.bbox_2d_padded).astype(int)
                image = cv2.rectangle(image, (bbox[0], bbox[1]), (bbox[0] + bbox[2], bbox[1] + bbox[3]), color, 2)
            else:
                image = pose.draw(image=image, color=color)

        writer.append_data(np.vstack([image, image[-8:, :]]))
        print(f'\r▸ progress: {100 * (frame_number / 1800):6.2f}%', end='')

    writer.close()
    print(f'\n▸ video with annotations: \'{out_mp4_file_path.abspath()}\'\n')


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


if __name__ == "__main__":
    pass
