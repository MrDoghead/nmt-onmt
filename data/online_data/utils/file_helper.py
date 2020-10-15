# coding=utf8

import os
import shutil


def get_data_folder():
    cur_dir, _ = os.path.split(__file__)
    return os.path.join(cur_dir, os.pardir, "data")


def get_data_file(file_name):
    data_dir = get_data_folder()
    return os.path.join(data_dir, file_name)


def get_project_file(file_name):
    cur_dir, _ = os.path.split(__file__)
    return os.path.join(cur_dir, os.pardir, file_name)


def get_config_file(file_name):
    cur_dir, _ = os.path.split(__file__)
    return os.path.join(cur_dir, os.pardir, file_name)


def is_f_exists(file_name):
    return os.path.exists(file_name)


def mk_folder_for_file(file_name):
    """
    为file_name创建其文件夹
    :param file_name:
    :return:
    """
    dir_name = os.path.dirname(file_name)
    os.makedirs(dir_name, exist_ok=True)


def mk_dir(dir_name):
    os.makedirs(dir_name, exist_ok=True)


def rm_f(dir_or_file):
    if os.path.exists(dir_or_file):
        shutil.rmtree(dir_or_file)


def rename(src, dst):
    os.rename(src, dst)


def copy(src, dst):
    shutil.copy(src, dst)


def get_dir(f_path):
    """
    :param f_path:
    :return:
    """
    return os.path.dirname(f_path)


def split(f_path):
    return os.path.split(f_path)


def get_online_data(f_path):
    cur_dir, _ = os.path.split(__file__)
    abs_dir = os.path.join(cur_dir, os.pardir, "online_data")
    return os.path.join(abs_dir, f_path)


if __name__ == "__main__":
    print(("data_folder", get_data_folder()))
    print(get_project_file("project_conf.ini"))
