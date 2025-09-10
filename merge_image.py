import os
import shutil

def merge_jpg_folders(folder1, folder2, output_folder):
    os.makedirs(output_folder, exist_ok=True)

    counter = 0  # 全局计数器

    def copy_images(src_folder, dst_folder, prefix, counter):
        for idx, file in enumerate(os.listdir(src_folder)):
            if file.lower().endswith(".jpg"):
                src_path = os.path.join(src_folder, file)
                dst_name = f"{prefix}_{idx:010d}.jpg"
                dst_path = os.path.join(dst_folder, dst_name)
                shutil.copy2(src_path, dst_path)

                counter += 1
                if counter % 10000 == 0:
                    print(f"已处理 {counter} 张图片...")

        return counter

    # 拷贝两个文件夹
    counter = copy_images(folder1, output_folder, "A", counter)
    counter = copy_images(folder2, output_folder, "B", counter)

    print(f"✅ 合并完成，总共处理了 {counter} 张图片，结果保存在: {output_folder}")


# 使用示例
folder1 = "/data/AI_DATA/ykhu/public/Objects365/image_test"
folder2 = "/data/AI_DATA/ykhu/public/Objects365/image_val"
output = "/data/AI_DATA/ykhu/public/Objects365/image"

merge_jpg_folders(folder1, folder2, output)
