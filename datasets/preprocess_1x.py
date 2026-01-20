import os
import numpy as np
import cv2 as cv
import json
import argparse
from tqdm import tqdm
import mmengine

# Index-to-State Mapping (NEW):
#            {
#                0: HIP_YAW
#                1: HIP_ROLL
#                2: HIP_PITCH
#                3: KNEE_PITCH
#                4: ANKLE_ROLL
#                5: ANKLE_PITCH
#                6: LEFT_SHOULDER_PITCH
#                7: LEFT_SHOULDER_ROLL
#                8: LEFT_SHOULDER_YAW
#                9: LEFT_ELBOW_PITCH
#                10: LEFT_ELBOW_YAW
#                11: LEFT_WRIST_PITCH
#                12: LEFT_WRIST_ROLL
#                13: RIGHT_SHOULDER_PITCH
#                14: RIGHT_SHOULDER_ROLL
#                15: RIGHT_SHOULDER_YAW
#                16: RIGHT_ELBOW_PITCH
#                17: RIGHT_ELBOW_YAW
#                18: RIGHT_WRIST_PITCH
#                19: RIGHT_WRIST_ROLL
#                20: NECK_PITCH
#                21: Left hand closure state (0 = open, 1 = closed)
#                22: Right hand closure state (0 = open, 1 = closed)
#                23: Linear Velocity
#                24: Angular Velocity
#            }

origin_fps = 30

def process_train_data(root_dir, args):
    metadata_dir = os.path.join(root_dir, 'metadata')
    robot_states_dir = os.path.join(root_dir, 'robot_states')
    segment_indices_dir = os.path.join(root_dir, 'segment_indices')
    video_dir = os.path.join(root_dir, 'videos')

    save_index = args.save_index

    file_list = os.listdir(video_dir)
    # sort the file list
    file_list = sorted(file_list, key=lambda x: int(x.split('.')[0].split('_')[-1]))
    for index, file_name in enumerate(file_list):
        file_index = file_name.split('.')[0].split('_')[-1]

        # define path
        metadata_path = os.path.join(metadata_dir, f'metadata_{file_index}.json')
        segment_idx_path = os.path.join(segment_indices_dir, f'segment_idx_{file_index}.bin')
        robot_states_path = os.path.join(robot_states_dir, f'states_{file_index}.bin')
        video_path = os.path.join(video_dir, f'video_{file_index}.mp4')

        # load data
        metadata = json.load(open(metadata_path, 'r'))
        shard_num_frames = metadata['shard_num_frames']
        segment_idx = np.memmap(segment_idx_path, dtype=np.int32, mode='r', shape=(shard_num_frames,))
        robot_state = np.memmap(robot_states_path, dtype=np.float32, mode='r', shape=(shard_num_frames, 25))

        frame_index = 0
        video = cv.VideoCapture(video_path)
        frames = []
        print('Loading video:', video_path)
        for _ in mmengine.track_iter_progress(range(shard_num_frames)):
            ret, frame = video.read()
            if not ret:
                break
            frames.append(frame)
            frame_index = frame_index + 1
        video.release()

        # skip according to the process_fps
        segment_idx = segment_idx[::int(origin_fps / args.process_fps)]
        robot_state = robot_state[::int(origin_fps / args.process_fps)]
        frames = frames[::int(origin_fps / args.process_fps)]

        # save according to the segment_idx
        print('segment_idx:', segment_idx)
        max_segment_idx = segment_idx.max()     # because the last segment is the same as next segment
        min_segment_idx = segment_idx.min()
        print('Save samples:', max_segment_idx-min_segment_idx)
        for seg_i in mmengine.track_iter_progress(range(min_segment_idx, max_segment_idx)):
            idx = np.where(segment_idx == seg_i)[0]
            # filter too short segment
            if len(idx) < args.filter_short_frame_number:
                continue

            # each save_video contain 30 each_sample_frame
            segment_video = frames[idx[0]:idx[-1]]
            save_frames = [segment_video[j:j + args.each_sample_frame] for j in range(0, len(idx), args.each_sample_frame)]
            save_robot_state = [robot_state[j:j + args.each_sample_frame] for j in range(0, len(idx), args.each_sample_frame)]
            for j in range(len(save_frames)):
                save_frame = np.array(save_frames[j])
                if save_frame.shape[0] < 15:
                    continue
                save_state = save_robot_state[j]
                save_file_path = os.path.join(args.train_save_path, f'train_{str(save_index).zfill(6)}.npz')
                np.savez_compressed(save_file_path, **{'image':save_frame, 'action':save_state})
                save_index = save_index + 1

    return save_index

def process_val_data(root_dir, args):
    save_index = args.save_index

    # define path
    metadata_path = os.path.join(root_dir, f'metadata_{0}.json')
    segment_idx_path = os.path.join(root_dir, f'segment_idx_{0}.bin')
    robot_states_path = os.path.join(root_dir, f'states_{0}.bin')
    video_path = os.path.join(root_dir, f'video_{0}.mp4')

    # load data
    metadata = json.load(open(metadata_path, 'r'))
    shard_num_frames = metadata['shard_num_frames']
    segment_idx = np.memmap(segment_idx_path, dtype=np.int32, mode='r', shape=(shard_num_frames,))
    robot_state = np.memmap(robot_states_path, dtype=np.float32, mode='r', shape=(shard_num_frames, 25))

    frame_index = 0
    video = cv.VideoCapture(video_path)
    frames = []
    print('Loading video:', video_path)
    for _ in mmengine.track_iter_progress(range(shard_num_frames)):
        ret, frame = video.read()
        if not ret:
            break
        frames.append(frame)
        frame_index = frame_index + 1
    video.release()

    # skip according to the process_fps
    segment_idx = segment_idx[::int(origin_fps / args.process_fps)]
    robot_state = robot_state[::int(origin_fps / args.process_fps)]
    frames = frames[::int(origin_fps / args.process_fps)]

    # save according to the segment_idx
    print('segment_idx:', segment_idx)
    max_segment_idx = segment_idx.max()     # because the last segment is the same as next segment
    min_segment_idx = segment_idx.min()
    print('Save samples:', max_segment_idx-min_segment_idx)
    for seg_i in mmengine.track_iter_progress(range(min_segment_idx, max_segment_idx)):
        idx = np.where(segment_idx == seg_i)[0]
        # filter too short segment
        if len(idx) < args.filter_short_frame_number:
            continue

        # each save_video contain 30 each_sample_frame
        segment_video = frames[idx[0]:idx[-1]]
        save_frames = [segment_video[j:j + args.each_sample_frame] for j in range(0, len(idx), args.each_sample_frame)]
        save_robot_state = [robot_state[j:j + args.each_sample_frame] for j in range(0, len(idx), args.each_sample_frame)]
        for j in range(len(save_frames)):
            save_frame = np.array(save_frames[j])
            if save_frame.shape[0] < 15:
                continue
            save_state = save_robot_state[j]
            save_file_path = os.path.join(args.val_save_path, f'val_{str(save_index).zfill(6)}.npz')
            np.savez_compressed(save_file_path, **{'image':save_frame, 'action':save_state})
            save_index = save_index + 1

    return save_index

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_root', type=str, required=True)
    parser.add_argument('--val_root', type=str, required=True)
    parser.add_argument('--save_path', type=str, required=True)
    parser.add_argument('--process_fps', type=int, default=30)
    parser.add_argument('--each_sample_frame', type=int, default=30)
    parser.add_argument('--filter_short_frame_number', type=int, default=17*3)
    args = parser.parse_args()

    train_save_path = os.path.join(args.save_path, "train")
    val_save_path = os.path.join(args.save_path, "val")
    os.makedirs(train_save_path, exist_ok=True)
    os.makedirs(val_save_path, exist_ok=True)

    args.save_index = 0
    args.train_save_path = train_save_path
    args.val_save_path = val_save_path

    # process train data
    save_index = process_train_data(args.train_root, args)
    print('Done processing train data')
    args.save_index = save_index

    # process val data
    process_val_data(args.val_root, args)
    print('Done processing val data')