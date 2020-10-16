# coding=utf8

import os
import shutil


def mk_data_folder():
    data_folder = get_data_folder()
    os.makedirs(data_folder, exist_ok=True)


def get_data_folder():
    cur_dir, _ = os.path.split(__file__)
    return os.path.join(cur_dir, os.pardir, "data")


def get_data_file(file_name):
    data_dir = get_data_folder()
    return os.path.join(data_dir, file_name)


def is_file_exists(file_name):
    return os.path.exists(file_name)


def mk_folder_for_file(file_name):
    """
    为file_name创建其文件夹
    :param file_name:
    :return:
    """
    dir_name = os.path.dirname(file_name)
    os.makedirs(dir_name, exist_ok=True)


def get_conf_file(file_name):
    cur_dir, _ = os.path.split(__file__)
    return os.path.join(cur_dir, os.pardir, "conf", file_name)


def get_project_path():
    cur_dir, _ = os.path.split(__file__)
    return os.path.join(cur_dir, os.pardir, os.pardir)


def get_real_path(f_path):
    """
    本函数主要是为了找到文件的地址而构建的
    :param f_path: 目标文件地址
    :return:
    """
    if os.path.exists(f_path):
        return f_path
    res_path = os.path.abspath(f_path)
    if os.path.exists(res_path):
        return res_path
    res_path = os.path.join(get_project_path(), f_path)
    if os.path.exists(res_path):
        return res_path
    return f_path


def get_abs_path(f_path):
    if os.path.exists(f_path):
        return f_path
    if os.path.isabs(f_path):
        return f_path
    res_path = os.path.join(get_project_path(), f_path)
    return res_path


def get_online_data(f_path):
    _dir = get_abs_path("data/online_data")
    return os.path.join(_dir, f_path)


def cp_dir(src_dir, dst_dir):
    if os.path.exists(dst_dir):
        shutil.rmtree(dst_dir)
    shutil.copytree(src_dir, dst_dir)


def cp_f(src_file, dst_dir):
    shutil.copy(src_file, dst_dir)


if __name__ == "__main__":
    print(("project_path", get_project_path()))
    print(os.listdir(get_project_path()))



