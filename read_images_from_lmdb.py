import json
import cv2
from nreal_data_tool import LmdbClient

def read_images_from_lmdb(meta_path):
    # 读取JSON文件内容
    with open(meta_path, 'r') as f:
        meta_data = json.load(f)
    
    # 提取lmdb_path和file_name_list
    lmdb_path = meta_data.get('lmdb_path')
    file_name_list = meta_data.get('files')
    
    # 初始化LMDB客户端
    lmdb_client = LmdbClient('unchanged')
    images = []
    for file_name in file_name_list:
        import ipdb;ipdb.set_trace()
        key = lmdb_path + ':' + file_name
        img = lmdb_client.get(key)
        images.append(img)
    
    return images, file_name_list

images, file_name_list = read_images_from_lmdb("/data/AI_DATA/byzhou/datasets/mono_depth/train_data/VG_100K_2.json")

import ipdb;ipdb.set_trace()
print()