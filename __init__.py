import os
import shutil

# 假设你的图片文件夹路径是 'path_to_your_images_folder'
# 请将 'path_to_your_images_folder' 替换成实际的图片文件夹路径
images_folder = r'F:\MACR\train'

# 获取图片文件夹中所有文件的列表
files = os.listdir(images_folder)

# 创建一个字典来存储每个类别的图片路径
category_dict = {}

# 遍历文件列表，使用正则表达式来匹配并提取类别
for file in files:
    if file.endswith('.jpg'):
        # 使用正则表达式匹配文件名中的汉字部分
        import re

        match = re.search(r'[\u4e00-\u9fa5]+', file)
        if match:
            category = match.group()  # 获取匹配到的汉字作为类别
            if category not in category_dict:
                category_dict[category] = []
            category_dict[category].append(file)  # 将文件名添加到对应类别的列表中

# 根据类别创建文件夹并移动图片
for category, file_list in category_dict.items():
    # 创建以类别命名的文件夹
    category_folder = os.path.join(images_folder, category)
    if not os.path.exists(category_folder):
        os.makedirs(category_folder)

    # 移动图片到新创建的文件夹
    for file in file_list:
        src_path = os.path.join(images_folder, file)
        dst_path = os.path.join(category_folder, file)
        shutil.move(src_path, dst_path)

print("图片分类完成！")