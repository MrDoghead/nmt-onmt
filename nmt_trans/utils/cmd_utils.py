# coding=utf8

"""
本文件主要封住用来执行命令行的工具
"""

import subprocess
import shlex
from subprocess import Popen


def exe_cmd(cmd_line):
    cmd_args = shlex.split(cmd_line)
    Popen(cmd_args)


def exe_cmd_raw(cmd_line):
    """
    :param
    """
    Popen(cmd_line, shell=True)


def get_cmd_args(cmd_line):
    return shlex.split(cmd_line)


def exe_c_args(args):
    return subprocess.call(args)
