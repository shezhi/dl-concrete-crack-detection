import os, os.path
from random import shuffle
from shutil import copyfile
# SENET2018 SDNET2018是带注释的图像数据集，用于训练，验证和确定基于人工智能的混凝土裂缝检测算法。SDNET2018包含超过56,000张开裂和未开裂的混凝土（桥面板，墙壁和人行道的图像）。
# D：deck 桥面
#P：pavements人行道
#W：wall墙
# specify raw dataset paths 原始数据集路径
non_cracked_images_path_D = r"SDNET2018\D\UD//"
cracked_images_path_D = r"SDNET2018\D\CD//"
non_cracked_images_path_P = r"SDNET2018\P\UP//"
cracked_images_path_P = r"SDNET2018\P\CP//"
non_cracked_images_path_W = r"SDNET2018\W\UW//"
cracked_images_path_W = r"SDNET2018\W\CW//"


# load names of files present in raw dataset paths into a list, and randomly shuffle 将原始数据集路径中存在的文件名加载到列表中，然后随机洗牌
non_cracked_images_path_contents_D = os.listdir(non_cracked_images_path_D)
shuffle(non_cracked_images_path_contents_D)
cracked_images_path_contents_D = os.listdir(cracked_images_path_D)
shuffle(cracked_images_path_contents_D)

non_cracked_images_path_contents_P = os.listdir(non_cracked_images_path_P)
shuffle(non_cracked_images_path_contents_P)
cracked_images_path_contents_P = os.listdir(cracked_images_path_P)
shuffle(cracked_images_path_contents_P)

non_cracked_images_path_contents_W = os.listdir(non_cracked_images_path_W)
shuffle(non_cracked_images_path_contents_W)
cracked_images_path_contents_W = os.listdir(cracked_images_path_W)
shuffle(cracked_images_path_contents_W)


# get number of images in each list
num_of_non_cracked_images_D = len(non_cracked_images_path_contents_D)
num_of_cracked_images_D = len(cracked_images_path_contents_D)

num_of_non_cracked_images_P = len(non_cracked_images_path_contents_P)
num_of_cracked_images_P = len(cracked_images_path_contents_P)

num_of_non_cracked_images_W = len(non_cracked_images_path_contents_W)
num_of_cracked_images_W = len(cracked_images_path_contents_W)


# select the number of items in non-dominant class for each of the 3 types of concrete 三种混泥土中分别选择少的部分，比如D（crack：6，no-crack：4），那么从D中最多拿出4个出来
max_num_of_images_to_select_from_D = min(num_of_cracked_images_D, num_of_non_cracked_images_D)
max_num_of_images_to_select_from_P = min(num_of_cracked_images_P, num_of_non_cracked_images_P)
max_num_of_images_to_select_from_W = min(num_of_cracked_images_W, num_of_non_cracked_images_W)


# select the files to be copied into training and test sets for each of the 3 types of concrete
non_cracked_images_to_copy_D = non_cracked_images_path_contents_D[:max_num_of_images_to_select_from_D]# 三种混泥土中分别选择少的部分，比如D（crack：6，no-crack：4），那么从D中最多拿出4个出来
cracked_images_to_copy_D = cracked_images_path_contents_D[:max_num_of_images_to_select_from_D]

non_cracked_images_to_copy_P = non_cracked_images_path_contents_P[:max_num_of_images_to_select_from_P]
cracked_images_to_copy_P = cracked_images_path_contents_P[:max_num_of_images_to_select_from_P]

non_cracked_images_to_copy_W = non_cracked_images_path_contents_W[:max_num_of_images_to_select_from_W]
cracked_images_to_copy_W = cracked_images_path_contents_W[:max_num_of_images_to_select_from_W]


# compute test train (+ validation from train) split thresholds
train_to_total_ratio = 0.8
valid_to_train_ratio = 0.25

non_cracked_split_num_train_test_D = int(round(train_to_total_ratio*len(non_cracked_images_to_copy_D)))
cracked_split_num_train_test_D = int(round(train_to_total_ratio*len(cracked_images_to_copy_D)))
non_cracked_split_num_valid_train_D = int(round(valid_to_train_ratio*non_cracked_split_num_train_test_D))
cracked_split_num_valid_train_D = int(round(valid_to_train_ratio*cracked_split_num_train_test_D))

non_cracked_split_num_train_test_P = int(round(train_to_total_ratio*len(non_cracked_images_to_copy_P)))
cracked_split_num_train_test_P = int(round(train_to_total_ratio*len(cracked_images_to_copy_P)))
non_cracked_split_num_valid_train_P = int(round(valid_to_train_ratio*non_cracked_split_num_train_test_P))
cracked_split_num_valid_train_P = int(round(valid_to_train_ratio*cracked_split_num_train_test_P))

non_cracked_split_num_train_test_W = int(round(train_to_total_ratio*len(non_cracked_images_to_copy_W)))
cracked_split_num_train_test_W = int(round(train_to_total_ratio*len(cracked_images_to_copy_W)))
non_cracked_split_num_valid_train_W = int(round(valid_to_train_ratio*non_cracked_split_num_train_test_W))
cracked_split_num_valid_train_W = int(round(valid_to_train_ratio*cracked_split_num_train_test_W))


# create test, train, validation splits
def create_splits(list_to_split, split_num_train_test, split_num_valid_train):
  images_train = list_to_split[:split_num_train_test]
  images_test = list_to_split[split_num_train_test:]
  images_valid = images_train[:split_num_valid_train]
  images_train = images_train[split_num_valid_train:]
  return images_train, images_test, images_valid

non_cracked_images_train_D, non_cracked_images_test_D, non_cracked_images_valid_D = create_splits(non_cracked_images_to_copy_D, non_cracked_split_num_train_test_D, non_cracked_split_num_valid_train_D)
cracked_images_train_D, cracked_images_test_D, cracked_images_valid_D = create_splits(cracked_images_to_copy_D, cracked_split_num_train_test_D, cracked_split_num_valid_train_D)

non_cracked_images_train_P, non_cracked_images_test_P, non_cracked_images_valid_P = create_splits(non_cracked_images_to_copy_P, non_cracked_split_num_train_test_P, non_cracked_split_num_valid_train_P)
cracked_images_train_P, cracked_images_test_P, cracked_images_valid_P = create_splits(cracked_images_to_copy_P, cracked_split_num_train_test_P, cracked_split_num_valid_train_P)

non_cracked_images_train_W, non_cracked_images_test_W, non_cracked_images_valid_W = create_splits(non_cracked_images_to_copy_W, non_cracked_split_num_train_test_W, non_cracked_split_num_valid_train_W)
cracked_images_train_W, cracked_images_test_W, cracked_images_valid_W = create_splits(cracked_images_to_copy_W, cracked_split_num_train_test_W, cracked_split_num_valid_train_W)


# copy images to train / test / validate folders
test_folder_path = r"SDNET2018\test//"
train_folder_path = r"SDNET2018\train//"
valid_folder_path = r"SDNET2018\valid//"
#分割数据集流程：
#首先取出D，P，W数据，一共六个文件夹，每个文件夹都打乱，接着分别获取每个文件夹的数量，然后在同一类（分别D，P，W）中获取数量最少的数，比如D（crack：6，no-crack：4），那么从D中最多拿出4个出来；
#接着分别取出作为M，然后设置分割阈值（训练样本数占总的同一类总数的比例，验证集占训练集的比例），然后求出训练样本的个数，验证集的个数，然后从M划分验证集，测试集，训练集。
#然后把这些文件放到根目录下。
# 总体来说，就是先打乱，然后按照比例取出数据，作为验证集，测试集，训练集。（这里还没有把DPW合并，所以每个操作都是分别针对同一类做的）





# SDNET2018/test/0/image
def copy_images(base_path_to_images, image_list_test, image_list_train, image_list_valid, label):
  for image in image_list_test:
    image_full_path_src = base_path_to_images + image
    image_full_path_dest = test_folder_path + label + r"//" + image
    copyfile(image_full_path_src, image_full_path_dest)
  for image in image_list_train:
    image_full_path_src = base_path_to_images + image
    image_full_path_dest = train_folder_path + label + r"//" + image
    copyfile(image_full_path_src, image_full_path_dest)
  for image in image_list_valid:
    image_full_path_src = base_path_to_images + image
    image_full_path_dest = valid_folder_path + label + r"//" + image
    copyfile(image_full_path_src, image_full_path_dest)

copy_images(non_cracked_images_path_D, non_cracked_images_test_D, non_cracked_images_train_D, non_cracked_images_valid_D, r"0")
copy_images(cracked_images_path_D, cracked_images_test_D, cracked_images_train_D, cracked_images_valid_D, r"1")

copy_images(non_cracked_images_path_P, non_cracked_images_test_P, non_cracked_images_train_P, non_cracked_images_valid_P, r"0")
copy_images(cracked_images_path_P, cracked_images_test_P, cracked_images_train_P, cracked_images_valid_P, r"1")

copy_images(non_cracked_images_path_W, non_cracked_images_test_W, non_cracked_images_train_W, non_cracked_images_valid_W, r"0")
copy_images(cracked_images_path_W, cracked_images_test_W, cracked_images_train_W, cracked_images_valid_W, r"1")
