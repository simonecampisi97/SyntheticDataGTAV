import os

import cv2


def sort_by_filename_number(e):
    return int(e.split(os.sep)[-1].split("-")[0].split(".")[0])


def get_file_folder_list(video_folder):
    video_path_list = []
    for root, dirs, files in os.walk(video_folder):
        for file in files:

            if file.endswith(".csv") or file.endswith(".txt") or file.endswith(".mp4"):
                continue
            else:
                video_path_list.append(os.path.join(root, file))

    video_path_list.sort(key=sort_by_filename_number)

    return video_path_list


if __name__ == "__main__":

    for i in range(14, 16):

        seq_path = f"C:\\Users\\simoc\\Desktop\\Synthetic Data IMAVIS\\seq_{i}"

        width = 1920
        height = 1080
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fps_video = 20

        out_video = cv2.VideoWriter(os.path.join(seq_path, f"seq_{i}.mp4"), fourcc, fps_video, (width, height))
        seq_path = get_file_folder_list(seq_path)

        for path in seq_path:

            frame = cv2.imread(path)
            out_video.write(frame)

            cv2.imshow("Display_Image", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        out_video.release()
        cv2.destroyAllWindows()
