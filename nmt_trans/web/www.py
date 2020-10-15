# coding=utf8

from flask import Flask
from flask_cors import *
from hb_chat.web.info_ext_service import route_entity
from hb_chat.web import info_ext_service


app = Flask(__name__)
CORS(app, supports_credentials=True)
# app.register_blueprint(route_entity, url_prefix='/alg_NT')
app.register_blueprint(route_entity, url_prefix="/chat")


def load_app(conf_path):
    info_ext_service.init_predictor(conf_path)
    return app


if __name__ == "__main__":
    import sys
    f_conf = sys.argv[1]
    load_app(f_conf)
    app.run('0.0.0.0', port=8867)
