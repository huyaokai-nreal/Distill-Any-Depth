import torch
import os, cv2, glob, argparse
from PIL import Image
import PIL.Image as pil
import numpy as np
from tqdm import tqdm
from distillanydepth.modeling.archs.dam.dam import DepthAnything
from distillanydepth.midas.transforms import Resize, NormalizeImage, PrepareForNet
from torchvision.transforms import Compose
from safetensors.torch import load_file
import torch.multiprocessing as mp

# 显卡使用数量
num_devices = 8  # 可根据实际情况调整

def parse_args():
    parser = argparse.ArgumentParser(description='generate disp gt value.')
    parser.add_argument('--input_path', type=str, help='input image path', required=True)
    parser.add_argument('--out_path', type=str, help='save disp gt path', required=True)
    parser.add_argument("--input_type", type=str, default="video", choices=["image", "video"])
    parser.add_argument('--ext', type=str, help='image extension', default="png")
    parser.add_argument('--lmdb_name', type=str, required=True, help='lmdb name')
    return parser.parse_args()

def inference(image, model, device):
    if model is None:
        return None
    image_np = np.array(image)[..., ::-1] / 255  # BGR转RGB并归一化
    transform = Compose([
        Resize(700, 700, resize_target=False, keep_aspect_ratio=False, ensure_multiple_of=14, 
               resize_method='lower_bound', image_interpolation_method=cv2.INTER_CUBIC),
        NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        PrepareForNet()
    ])
    image_tensor = transform({'image': image_np})['image']
    image_tensor = torch.from_numpy(image_tensor).unsqueeze(0).to(device)
    with torch.no_grad():
        pred_disp, _ = model(image_tensor)
    torch.cuda.empty_cache()
    pred_disp_np = pred_disp.cpu().detach().numpy()[0, 0, :, :]
    pred_disp = pred_disp_np / pred_disp_np.max()  # 归一化
    h, w = image_np.shape[:2]
    disp_resized = cv2.resize(pred_disp, (w, h), cv2.INTER_LINEAR)
    return (disp_resized * 255).astype(np.uint8)

def process_image(model, device, image_paths, output_dir, start_idx):
    """处理图片，按全局索引保存（保证顺序）"""
    disp_save_dir = os.path.join(output_dir, "disp")
    # disp_gt_save_dir = os.path.join(output_dir, "disp_gt")
    image_save_dir = os.path.join(output_dir, "image")
    os.makedirs(disp_save_dir, exist_ok=True)
    # os.makedirs(disp_gt_save_dir, exist_ok=True)
    os.makedirs(image_save_dir, exist_ok=True)
    max_side = 1000
    # movie原图上下去黑边
    # movie_num = int(image_paths[0].split('/')[-2].split('_')[-1])
    pad_info = [0,0]
    # if movie_num < 26:
    #     if movie_num in [2, 7, 8, 14, 15, 16, 17, 19, 21, 23]:
    #         pad_info = [0, 38]  # 底部38行黑边
    # else:
    #     if movie_num not in [32, 37, 40, 42, 43, 46, 50, 57, 63, 65, 68, 72, 73, 77, 81, 85, 94]:
    #         pad_info = [64, 64] # 上下各有至少64行黑边
            
    for local_idx, path_ in enumerate(tqdm(image_paths)):
        # 全局序号 =  chunk起始索引 + 本地索引
        global_idx = start_idx + local_idx
        frame = pil.open(path_).convert('RGB')
        w, h = frame.size
        top_pad, bottom_pad = pad_info
        frame = frame.crop((0, top_pad, w, h - bottom_pad)) # 原图上下去黑边

        # 缩放处理（保持原逻辑）
        if w / h < 2 or w / h > 0.5:  # 原逻辑中的宽高比判断
            current_max = max(h, w)
            scale = max_side / current_max if current_max > max_side else 1.0
            new_w, new_h = int(w * scale), int(h * scale)
            frame = frame.resize((new_w, new_h), Image.Resampling.BILINEAR)
        
        disp_image = inference(frame, model, device)
        # 按全局序号保存
        cv2.imwrite(os.path.join(disp_save_dir, f"{global_idx:010d}.png"), disp_image)
        cv2.imwrite(os.path.join(image_save_dir, f"{global_idx:010d}.png"), 
                    cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR))
        # path_ = path_.replace('images', 'depths')
        # path_ = path_.replace('_l_', '_d_')
        # os.system(f'cp {path_} {os.path.join(disp_gt_save_dir, f"{global_idx:010d}.png")}')

def process_video(model, device, video_path, out_path, start_frame):
    """处理视频，按预分配的起始序号保存（保证顺序）"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"警告：无法打开视频 {video_path}")
        return

    video_name = os.path.splitext(os.path.basename(video_path))[0]
    output_dir = os.path.join(out_path, video_name)
    os.makedirs(output_dir, exist_ok=True)
    frame_idx = 0  # 视频内的本地帧索引

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        # 全局序号 = 视频起始序号 + 视频内帧索引
        global_idx = start_frame + frame_idx
        disp_image = inference(frame, model, device)
        cv2.imwrite(os.path.join(output_dir, f"{global_idx:010d}.png"), disp_image)
        frame_idx += 1

    cap.release()
    print(f"视频 {video_name} 处理完成，共 {frame_idx} 帧")

def get_video_paths(directory, recursive=True):
    """获取所有视频路径（保持原逻辑）"""
    video_extensions = ['*.mp4', '*.avi', '*.mov', '*.mkv', '*.flv', '*.wmv', '*.rmvb', '*.mpeg', '*.mpg', '*.webm', '*.3gp']
    video_paths = []
    for ext in video_extensions:
        pattern = os.path.join(directory, '**', ext) if recursive else os.path.join(directory, ext)
        video_paths.extend(glob.glob(pattern, recursive=recursive))
    return list(sorted(set(video_paths)))  # 去重并排序

def worker(device_id, input_type, chunk_data, out_path, ext, start_info):
    """工作进程：处理分配的图片/视频chunk"""
    device = torch.device(f"cuda:{device_id}" if torch.cuda.is_available() else "cpu")
    # 加载模型（保持原逻辑）
    model_dict = dict(
        encoder="vitl", features=256, out_channels=[256, 512, 1024, 1024],
        use_bn=False, use_clstoken=False, max_depth=150.0, mode='disparity',
        pretrain_type='dinov2', del_mask_token=False
    )
    model = DepthAnything(** model_dict).to(device)
    model_weights = load_file("models/large.safetensors")
    model.load_state_dict(model_weights)
    model.eval()

    if input_type == "image":
        # 处理图片：chunk_data是图片路径列表，start_info是chunk起始索引
        process_image(model, device, chunk_data, out_path, start_info)
    elif input_type == "video":
        # 处理视频：chunk_data是视频路径列表，start_info是每个视频的起始帧序号列表
        for video_path, video_start in zip(chunk_data, start_info):
            process_video(model, device, video_path, out_path, video_start)

import json
import re

def sort_json_file(json_path):
    """保持原JSON排序逻辑"""
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    file_list = data.get('file_name_list', [])
    image_files = [f for f in file_list if f.startswith('image/')]
    disp_files = [f for f in file_list if f.startswith('disp/')]
    def extract_number(filename):
        match = re.search(r'\d+', filename)
        return int(match.group()) if match else 0
    sorted_image_files = sorted(image_files, key=extract_number)
    sorted_disp_files = sorted(disp_files, key=extract_number)
    data['file_name_list'] = sorted_image_files + sorted_disp_files
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
        
    with open(os.path.join(json_path), 'r')  as f:
        annos = json.load(f)

    lmdb_path = annos["lmdb_path"]
    file_name_list = annos["file_name_list"]
    save_file_list = []
    save_file_dict = {}
    cam_in = [100.0, 100.0, 100.0, 100.0]

    for file_name in file_name_list:
        if "disp/" in file_name:
            continue
        curr_file = [{'rgb':file_name+'.jpg', 'depth':file_name.replace("image/", "disp/")+'.jpg', 'cam_in':cam_in}]
        save_file_list = save_file_list + curr_file

    save_file_dict["files"] = save_file_list
    save_file_dict["lmdb_path"] = lmdb_path
    save_file_dict.pop("file_name_list", None)
    with open(json_path, 'w') as fj:
        json.dump(save_file_dict, fj)
    print(f"已排序JSON文件: {json_path}")

if __name__ == "__main__":
    if mp.get_start_method(allow_none=True) is None:
        mp.set_start_method('spawn')
    args = parse_args()
    out_path = os.path.join(args.out_path, args.lmdb_name)
    processes = []

    # 主进程：统一处理输入路径并排序
    if args.input_type == "image":
        # 获取并排序所有图片路径
        if os.path.isfile(args.input_path):
            all_paths = [args.input_path]
        elif os.path.isdir(args.input_path):
            all_paths = glob.glob(os.path.join(args.input_path, f'*.{args.ext}'))
        else:
            raise ValueError(f"无效图片路径: {args.input_path}")
        all_paths = sorted(all_paths)  # 关键：按路径排序
        total = len(all_paths)

        # 分割路径给多个进程
        chunk_size = total // num_devices
        for device_id in range(num_devices):
            start = device_id * chunk_size
            end = start + chunk_size if device_id < num_devices - 1 else total
            chunk_paths = all_paths[start:end]
            # 传递chunk路径和起始索引（用于计算全局序号）
            p = mp.Process(target=worker, args=(device_id, args.input_type, chunk_paths, out_path, args.ext, start))
            processes.append(p)
            p.start()

    elif args.input_type == "video":
        # 获取并排序所有视频路径
        all_videos = get_video_paths(args.input_path)
        total_videos = len(all_videos)
        if total_videos == 0:
            raise ValueError(f"未找到视频文件: {args.input_path}")

        # 预先计算每个视频的帧数和起始帧序号（关键：保证视频间序号连续）
        video_frame_counts = []
        for vid_path in all_videos:
            cap = cv2.VideoCapture(vid_path)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            video_frame_counts.append(frame_count)
            cap.release()

        # 计算每个视频的起始帧序号（累计前序视频的总帧数）
        video_start_indices = [0]
        for cnt in video_frame_counts[:-1]:
            video_start_indices.append(video_start_indices[-1] + cnt)

        # 分割视频给多个进程
        chunk_size = total_videos // num_devices
        for device_id in range(num_devices):
            start = device_id * chunk_size
            end = start + chunk_size if device_id < num_devices - 1 else total_videos
            chunk_videos = all_videos[start:end]
            # 传递该chunk视频对应的起始帧序号
            chunk_starts = video_start_indices[start:end]
            p = mp.Process(target=worker, args=(device_id, args.input_type, chunk_videos, out_path, args.ext, chunk_starts))
            processes.append(p)
            p.start()

    # 等待所有进程完成
    for p in processes:
        p.join()

    # 后续LMDB打包和JSON排序（保持原逻辑）
    json_dir = '/data/AI_DATA/byzhou/datasets/mono_depth/train_data'
    lmdb_run_cmd = f"python image_to_lmdb.py {out_path} -m -o /data/AI_DATA/byzhou/datasets/mono_depth/lmdb_data/ -on {args.lmdb_name} --json_dir {json_dir}"
    # os.system(lmdb_run_cmd)
    meta_json_path = os.path.join(json_dir, f'{args.lmdb_name}.json')
    # sort_json_file(meta_json_path)
    