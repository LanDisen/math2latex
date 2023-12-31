import os
import shutil

label_dir = '../../images/formulas_labels_fmm/'
# label_dir = './data/math_210421/formula_labels_210421_no_chinese/'
image_dir = '../../images/output_images/'

output_dir ='../../images/formulas_images_fmm/'

label_name_list = os.listdir(label_dir)
if not os.path.exists(output_dir):
    os.mkdir(output_dir)
for i in range(len(label_name_list)):
    label_name_list[i] = label_name_list[i][:-4]

# print(label_list)

image_name_list = os.listdir(image_dir)

for image_name in image_name_list:
    if image_name[:-4] in label_name_list:
        print(image_name)
        shutil.copy(image_dir + image_name, output_dir + image_name)

# 以下是新增代码
def align_images_labels(input_path, output_path):
    '''
    对齐图片和标签文件的数量(一般来说标签文件数量会少于图片数量, 根据标签提取图片)
    '''
    label_dir = input_path + 'labels'
    # label_dir = './data/math_210421/formula_labels_210421_no_chinese/'
    image_dir = input_path + 'images'

    output_dir = output_path + 'images'

    label_name_list = os.listdir(label_dir)
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    for i in range(len(label_name_list)):
        label_name_list[i] = label_name_list[i][:-4]

    # print(label_list)

    image_name_list = os.listdir(image_dir)

    for image_name in image_name_list:
        if image_name[:-4] in label_name_list:
            print(image_name)
            shutil.copy(image_dir + image_name, output_dir + image_name)
