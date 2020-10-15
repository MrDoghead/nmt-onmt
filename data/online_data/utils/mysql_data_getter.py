# coding=utf8

import pandas as pd
import pymysql.cursors


class BaseDataGetter(object):
    """
    基础数据拉取类
    """
    @classmethod
    def _dump_sql_data(cls, sqls, conn):
        """
        功能：将sql查询结构保存到本地文件
        :param sqls:  {file_name:sql} , sql 是要执行的sql查询， file_name 是待查询的语句保存到文件
        :param conn:  具体的数据库连接
        :return:
        """
        try:
            for file_name, sql in sqls.items():
                print(sql)
                df = pd.read_sql(sql, conn)
                df.to_csv(file_name)
        finally:
            conn.close()

    @classmethod
    def _init_connect(cls, host, port, uname, pwd, db):
        """
        用来构建mysql connect 连接对象
        :param host:
        :param port: 端口号， 必须是int
        :param uname:
        :param pwd:
        :param db:
        :return: connect 对象
        """
        if isinstance(port, str):
            port = int(port)
        conn = pymysql.connect(host=host, port=port, user=uname,
                               password=pwd, db=db, charset="utf8",
                               cursorclass=pymysql.cursors.DictCursor)
        return conn

    @classmethod
    def download(cls, host, port, uname, pwd, db, sqls):
        conn = cls._init_connect(host=host, port=int(port), uname=uname, pwd=pwd, db=db)
        cls._dump_sql_data(sqls, conn)
