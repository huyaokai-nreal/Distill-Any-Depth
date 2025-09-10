import cv2
import os
import argparse
import numpy as np
from pathlib import Path
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading


# 只处理视频文件（假设后缀为.mp4，可根据实际情况添加其他格式）
def rename_dir(dir_):
    video_extensions = ('.mp4', '.mov', '.avi', '.mkv')
    videos = [
        f for f in os.listdir(f"./{dir_}") 
        if os.path.isfile(os.path.join(f"./{dir_}", f))  # 只处理文件（排除子目录）
        and f.lower().endswith(video_extensions)    # 只处理视频格式
    ]

    # 按文件名排序（确保顺序一致）
    videos_sorted = sorted(videos)

    # 重命名，避免覆盖（若担心已有movie_xxx.mp4，可先检查或加前缀）
    for idx, movie in enumerate(videos_sorted):
        old_path = os.path.join(f"./{dir_}", movie)
        new_name = f"movie_{idx}.mp4"
        new_path = os.path.join(f"./{dir_}", new_name)
        
        # 避免覆盖已有文件（若存在则跳过或加后缀）
        if os.path.exists(new_path):
            new_name = f"movie_{idx}_dup.mp4"
            new_path = os.path.join(f"./{dir_}", new_name)
        
        os.rename(old_path, new_path)
        print(f"重命名: {movie} -> {new_name}")


def merge_image_folders(source_dir, target_dir, start_number=0):
    """
    合并多个文件夹中的图像，保持编号连续
    
    参数:
        source_dir: 包含所有movie_xxx文件夹的父目录
        target_dir: 合并后的图像保存目录
        start_number: 起始编号，默认为0
    """
    # 创建目标目录（如果不存在）
    Path(target_dir).mkdir(parents=True, exist_ok=True)
    
    # 收集所有图像路径
    image_paths = []
    
    # 遍历所有movie_xxx文件夹
    for folder_name in sorted(os.listdir(source_dir)):
        folder_path = os.path.join(source_dir, folder_name)
        
        # 只处理以'movie_'开头的文件夹
        if os.path.isdir(folder_path) and folder_name.startswith('movie_'):
            print(f"正在处理文件夹: {folder_name}")
            
            # 收集该文件夹中的所有png图像
            for filename in sorted(os.listdir(folder_path)):
                if filename.endswith('.png'):
                    # 确保是数字命名的图像文件
                    try:
                        # 提取文件名中的数字部分
                        file_num = int(filename.split('.')[0])
                        image_paths.append(os.path.join(folder_path, filename))
                    except ValueError:
                        print(f"跳过非数字命名的文件: {filename}")
    
    # 按原文件编号排序（确保顺序正确）
    image_paths.sort(key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
    
    # 复制并重新编号图像
    current_number = start_number
    # 使用线程池加速复制操作
    with ThreadPoolExecutor() as executor:
        futures = []
        for img_path in image_paths:
            new_filename = f"{current_number:010d}.png"
            new_path = os.path.join(target_dir, new_filename)
            # 提交复制任务
            futures.append(executor.submit(shutil.copy2, img_path, new_path))
            current_number += 1
            
            # 打印进度
            if current_number % 100 == 0:
                print(f"已处理 {current_number} 张图像")
        
        # 等待所有任务完成
        for future in as_completed(futures):
            try:
                future.result()
            except Exception as e:
                print(f"复制文件时出错: {e}")
    
    print(f"合并完成！总共处理了 {current_number - start_number} 张图像")
    print(f"合并后的图像保存在: {target_dir}")


def extract_frames(video_path, output_folder, fps=30, frame_interval=1, ext="png", max_workers=4):
    """
    从视频中按指定帧率提取帧并保存为10位数字命名的图像
    
    参数:
    video_path: 输入视频路径
    output_folder: 输出文件夹路径
    fps: 目标帧率，默认为30fps
    ext: 输出图像扩展名，默认为"png"
    max_workers: 线程池最大工作线程数，默认为4
    """
    # 确保输出文件夹存在
    os.makedirs(output_folder, exist_ok=True)
    
    # 打开视频文件
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"错误: 无法打开视频文件 {video_path}")
        return False
    
    # 获取视频原始帧率和总帧数
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / original_fps
    
    print(f"视频信息:")
    print(f"  路径: {video_path}")
    print(f"  原始帧率: {original_fps:.2f} fps")
    print(f"  总帧数: {total_frames}")
    print(f"  抽帧: 1/{frame_interval}")
    print(f"  时长: {duration:.2f} 秒")
    print(f"目标参数:")
    print(f"  输出帧率: {fps} fps")
    print(f"  输出文件夹: {output_folder}")
    print(f"  图像格式: {ext}")
    print(f"  命名格式: 10位数字（如0000000000.{ext}）")
    print(f"  线程数: {max_workers}")
    
    expected_frames = int(total_frames / frame_interval)
    print(f"预计提取帧数: {expected_frames}")
    
    # 开始提取帧
    frame_count = 0
    output_count = 0
    max_side = 640
    # 创建线程池
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        # 线程安全的计数器（虽然这里主要用主线程计数）
        lock = threading.Lock()
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            # 跳过前5秒和后20秒的帧（保留原有逻辑）
            # if frame_count < 10 * original_fps or frame_count > total_frames - original_fps * 60:
            # # if frame_count < 10 * original_fps:
            #     frame_count += 1
            #     continue
            frame = frame[:,frame.shape[1]//2:]
            h,w = frame.shape[:2]
            # 如果两边比例相差过大跳过
            # if h/w<0.5 or w/h>2:
            #     frame_count += 1
            #     continue
            # 按间隔提取帧
            if frame_count % frame_interval == 0:
                # 构建输出路径
                output_name = f"{output_count:010d}.{ext}"
                output_path = os.path.join(output_folder, output_name)
                
                # 使用双线性插值缩放帧
                current_max = max(h, w)
                scale = max_side / current_max if current_max > max_side else 1.0
                new_w = int(w * scale)
                new_h = int(h * scale)
                # print(new_w, new_h)
                resized_frame = cv2.resize(
                    frame, 
                    (new_w, new_h),  # 目标尺寸 (宽, 高)
                    interpolation=cv2.INTER_LINEAR  # 指定双线性插值
                )
                
                # 提交保存任务到线程池
                futures.append(executor.submit(
                    cv2.imwrite, 
                    output_path, 
                    resized_frame  # 保留原有帧处理逻辑（当前为原始帧）
                ))
                
                # 打印进度（主线程操作，线程安全）
                with lock:
                    output_count += 1
                    if output_count % 100 == 0:
                        print(f"已提取 {output_count}/{expected_frames} 帧 - {output_name}")
            
            frame_count += 1
        
        # 等待所有保存任务完成
        for future in as_completed(futures):
            try:
                # 检查保存结果
                if not future.result():
                    print(f"警告: 帧保存失败")
            except Exception as e:
                print(f"保存帧时出错: {e}")
    
    # 释放资源
    cap.release()
    
    print(f"\n帧提取完成!")
    print(f"共处理 {frame_count} 帧")
    print(f"成功提交 {output_count} 帧保存任务到线程池")
    
    # 验证输出文件夹中的文件数量
    actual_files = len([f for f in os.listdir(output_folder) if f.endswith(f".{ext}")])
    if actual_files == output_count:
        print(f"✅ 验证通过: 输出文件夹包含 {actual_files} 个图像文件")
    else:
        print(f"⚠️ 警告: 输出文件夹包含 {actual_files} 个图像文件，与预期的 {output_count} 不符")
    
    return True

if __name__ == "__main__":
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser(description='将视频按指定帧率提取帧并保存为10位数字命名的图像')
    parser.add_argument('video_path', help='输入视频路径')
    parser.add_argument('output_folder', help='输出文件夹路径')
    parser.add_argument('--fps', type=int, default=30, help='目标帧率，默认为30fps')
    parser.add_argument('--frame_interval', type=int, default=1, help='抽帧间隔，默认1帧间隔')
    parser.add_argument('--ext', type=str, choices=['png', 'jpg', 'jpeg'], default='png', 
                        help='输出图像扩展名，默认为"png"')
    parser.add_argument('--max_workers', type=int, default=4, help='线程池最大工作线程数，默认为4')
    
    # 解析参数
    args = parser.parse_args()
    
    # 检查输入视频是否存在
    if not os.path.isfile(args.video_path):
        print(f"错误: 视频文件 {args.video_path} 不存在")
    else:
        # 提取帧
        extract_frames(
            args.video_path, 
            args.output_folder, 
            args.fps, 
            args.frame_interval, 
            args.ext,
            args.max_workers
        )