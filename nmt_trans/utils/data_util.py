# coding=utf8

import hashlib


def uid_to_hash_value(uid):
    """
    通过将uid经过md5进行hash， 取16个字节, 得到最终的字符
    :return:
    """
    uid = str(uid).encode()
    hash_obj = hashlib.md5(uid)
    hex_str = hash_obj.hexdigest()
    return int(hex_str[:4], 16)


def str2md5(str_):
    str_ = str_.encode()
    hash_obj = hashlib.md5(str_)
    return hash_obj.hexdigest()


if __name__ == "__main__":
    print(1, uid_to_hash_value(1))
    print(2, uid_to_hash_value(2))
    print(3, uid_to_hash_value(3))
    print(4, uid_to_hash_value(4))
    print(5, uid_to_hash_value(5))
    print(6, uid_to_hash_value(6))
