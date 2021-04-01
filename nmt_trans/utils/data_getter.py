# coding=utf8

import os
from nmt_trans.utils import mysql_data_getter
from nmt_trans.utils import file_helper


host = os.getenv("DICT_HOST", "rm-uf6it2u39agqevqh8.mysql.rds.aliyuncs.com")
port = os.getenv("DICT_PORT", 3306)
uname = os.getenv("DICT_UNAME", "spider")
pwd = os.getenv("DICT_PWD", "spider_bot")
db = os.getenv("DICT_DB", "tgy_spider")
table_name = os.getenv("DICT_TABLE", "word_dict")
type_name = os.getenv("DICT_TYPE", None)
status = os.getenv("STATUS", None)

dict_path = file_helper.get_online_data("caijing_clean.csv")


def get_data():
    if status is not None and len(status.strip()) > 0:
        dict_path_value = "select en_words as en, cn_words as cn from {0} where status={1}".format(table_name, status)
    elif type_name is not None and len(type_name.strip()) > 0:
        dict_path_value = "select en, cn from {0} where type={1}".format(table_name, type_name)
    else:
        dict_path_value = "select en, cn from {0} ".format(table_name)

    sqls = {
        dict_path: dict_path_value,
    }
    mysql_data_getter.BaseDataGetter.download(host, port, uname, pwd, db, sqls)


if __name__ == "__main__":
    get_data()
