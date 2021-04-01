# coding=utf8

from flask import Flask
from flask_cors import *
from nmt_trans.web.info_ext_service import route_entity
from nmt_trans.web import info_ext_service


app = Flask(__name__)
CORS(app, supports_credentials=True)
app.register_blueprint(route_entity, url_prefix="/new_trans")


def load_app(conf_path, zh2en_conf_path):
    info_ext_service.init_predictor(conf_path, zh2en_conf_path)
    return app


if __name__ == "__main__":
    import sys
    f_conf = sys.argv[1]
    zh_f_conf = sys.argv[2]  # 中译英配置文档
    load_app(f_conf, zh_f_conf)
    app.run('0.0.0.0', port=8867)
