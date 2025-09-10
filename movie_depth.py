import os,cv2
from extract_img_speed import merge_image_folders, rename_dir

# 批量改名
# rename_dir('aa')



# 合并图像文件夹
out_movie = 'movie_101'
source_directory = "./output_img_tmp"  # 包含所有movie_xxx文件夹的目录
target_directory = f"./output_img/{out_movie}"  # 合并后的目录
# merge_image_folders(source_directory, target_directory)

# 生成真值，打包lmdb

for movie in sorted(os.listdir("./output_img")):
    movie_num = int(movie.split('_')[-1])
    if movie_num in [7, 14, 15, 16, 17, 19, 2, 21, 23, 26]:
        continue
    if movie_num < 26:
        if movie_num not in [2, 7, 8, 14, 15, 16, 17, 19, 21, 23]: 
            continue
    else:
        if movie_num in [32, 37, 40, 42, 43, 46, 50, 57, 63, 65, 68, 72, 73, 77, 81, 85, 94]:
            continue

    depth_lmdb_cmd = f"python ./generate_depth_data_speed_seq.py --input_path ./output_img/{movie} --out_path /data/AI_DATA/byzhou/datasets/mono_depth/depth_raw_data/  --input_type image --ext png --lmdb_name {movie}"
    # os.system(depth_lmdb_cmd)    