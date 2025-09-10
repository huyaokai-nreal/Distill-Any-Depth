import datetime
import json
import os

import click
import cv2
import lmdb
import numpy as np
from loguru import logger
from tqdm import tqdm

from nreal_data_tool.utils import (cross_merge_list, is_img_path, mkdir_or_exist, task_func,
                     track_parallel_progress)
from nreal_data_tool.utils.image import resize_with_short_edge
from nreal_data_tool.utils.path import get_filepath_list
from dataclasses import dataclass, field
from typing import List, Optional

from dataclasses_json import dataclass_json


@dataclass_json
@dataclass
class LMDBMeta:
    lmdb_path: str  # 采样处理时间
    file_name_list: List[str]  # lmdb 中文件名字，即key的列表

def get_data_list_from_dir(data_path: str, interval: int):
    image_list = []
    image_list = get_filepath_list(data_path, [])
    image_list = [
        filepath for filepath in image_list if is_img_path(filepath)
    ]
    image_list = image_list[::interval]
    return image_list


def get_frame_list_from_video(data_path: str, interval: int):
    cap = cv2.VideoCapture(data_path)
    if not cap.isOpened():
        return None, []
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    image_list = np.arange(0, total_frames, interval)
    return cap, image_list


@task_func
def warp_run_task(image_file_path: str,
                  interval: int,
                  output_dir: str,
                  is_video: bool = False,
                  short_edge_size: int = 0,
                  output_name=None,
                  save_meta: bool = True):
    return run_task(image_file_path, interval, output_dir, is_video,
                    short_edge_size, output_name, save_meta)


def run_task(
    image_file_path: str,
    interval: int,
    output_dir: str,
    is_video: bool = False,
    short_edge_size: int = 0,
    output_name=None,
    save_meta: bool = True,
):
    if output_name is None:
        lmdb_path = os.path.join(output_dir,
                                 f'{os.path.basename(image_file_path)}_lmdb')
    else:
        lmdb_path = os.path.join(output_dir, f'{output_name}_lmdb')
    mkdir_or_exist(lmdb_path)
    image_keys = []
    image_paths = []
    image_height = 0
    image_width = 0
    db = lmdb.open(lmdb_path, map_size=1099511627776)
    logger.info(
        f'sample {image_file_path} to {lmdb_path} with interval {interval}')
    if short_edge_size > 0:
        logger.info(f'target short edge size is {short_edge_size}')
    if is_video:
        cap, frame_list = get_frame_list_from_video(image_file_path, interval)
        if cap is None:
            return LMDBMeta(process_time='', file_name_list=[])
        video_name, _ = os.path.splitext(os.path.basename(image_file_path))
        with db.begin(write=True) as txn:
            for frame_id in tqdm(frame_list):
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
                # 读取一帧图像，ret表示是否成功，frame表示图像数据
                ret, frame = cap.read()
                # 如果成功读取，则保存该帧
                if ret:
                    if short_edge_size > 0:
                        frame = resize_with_short_edge(frame, short_edge_size)
                    image_height = frame.shape[0]
                    image_width = frame.shape[1]
                    # 使用cv2.imencode()方法将图像数据编码为jpeg格式的字节对象
                    _, frame_data = cv2.imencode(
                        '.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
                    # 使用视频文件名和当前帧数作为键，图像数据作为值，写入lmdb数据库
                    key = f'{video_name}_{frame_id}'
                    image_keys.append(key)
                    image_paths.append(f'{image_file_path}:{frame_id}')
                    txn.put(key.encode(), frame_data)
                else:
                    logger.warning(
                        f'parse frame {frame_id} failed on {image_file_path}')
    else:
        max_side = 1000
        data_list = get_data_list_from_dir(image_file_path, interval)
        with db.begin(write=True) as txn:
            for i, data_name in enumerate(tqdm(data_list)):
                # 原图保存为jpg，gt保存为png
                if '/disp/' in data_name:
                    frame = cv2.imread(data_name, cv2.IMREAD_GRAYSCALE)
                    _, image_data = cv2.imencode('.png', frame, [cv2.IMWRITE_PNG_COMPRESSION, 0])
                else:
                    frame = cv2.imread(data_name)
                    _, image_data = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
                key = os.path.relpath(data_name, image_file_path)
                key = '__'.join(key.split('.')[:-1])
                image_keys.append(key)
                image_paths.append(data_name)
                txn.put(key.encode(), image_data)
    db.close()
    meta_info = LMDBMeta(
        lmdb_path=os.path.abspath(lmdb_path),
        file_name_list=image_keys)
    if save_meta:
        with open(os.path.join(lmdb_path, 'meta.json'), 'w') as f:
            json.dump(meta_info.to_dict(), f)
    return meta_info


def process(image_file_dir: str, interval: int, output_dir: str, json_dir: str, output_name: str, 
            nr_process: int, merge: bool, is_video: bool,
            short_edge_size: int):
    image_file_dir = os.path.abspath(image_file_dir)
    if merge and not is_video:
        image_file_list = [image_file_dir]
    else:
        image_file_list = [
            os.path.join(image_file_dir, filename)
            for filename in os.listdir(image_file_dir)
        ]
    tasks = [(tar_file_name, interval, output_dir, is_video,
              short_edge_size, output_name) for tar_file_name in image_file_list]
    if merge:
        lmdb_name = f'{output_name}_lmdb'
        lmdb_path = os.path.join(output_dir, lmdb_name)
        logger.info(
            f'sample {len(tasks)} file/dir in {image_file_dir} with interval \
             {interval} and write to {lmdb_path}')
        all_meta_info = LMDBMeta(
            lmdb_path='',
            file_name_list=[])
        for task in tasks:
            meta_info = run_task(*task, save_meta=False)
            all_meta_info.file_name_list += meta_info.file_name_list
            all_meta_info.lmdb_path = meta_info.lmdb_path
        with open(os.path.join(json_dir, f'{output_name}.json'), 'w') as f:
            json.dump(all_meta_info.to_dict(), f)
    else:
        track_parallel_progress(warp_run_task, tasks=tasks, nproc=nr_process)


@click.command()
@click.argument('image_file_dir')
@click.option('--interval', '-i', default=1, help='数据采样的帧间隔')
@click.option(
    '--output-dir', '-o', default='sample_output', help='采样数据lmdb存储目标文件夹')
@click.option(
    '--json_dir', default='', help='采样数据json存储目标文件夹')
@click.option('--output-name', '-on', default='sample', help='采样数据存储目标文件名称')
@click.option(
    '--merge', '-m', is_flag=True, help='将文件夹下所有文件夹的采样数据合并到一个lmdb文件中')
@click.option(
    '--nr-process',
    '-nr',
    type=int,
    default=2,
    help='number of processor for parallel processing')
@click.option('--is-video', '-iv', is_flag=True, help='原始数据是否是视频')
@click.option(
    '--short-edge-size', '-ses', type=int, default=0, help='缩放短边到指定尺寸， 大于0时有效')
def main(image_file_dir: str, interval: int, output_dir: str, json_dir: str, output_name: str, nr_process: int,
         merge: bool, is_video: bool, short_edge_size: int):
    process(image_file_dir, interval, output_dir, json_dir, output_name, nr_process, merge,
            is_video, short_edge_size)


if __name__ == '__main__':
    main()
    