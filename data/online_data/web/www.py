# coding=utf8

from flask import Flask
from flask_cors import *
from hb_translate.web.trans_service import route_enzh
from hb_translate.web.tag_service import route_tag


app = Flask(__name__)
CORS(app, supports_credentials=True)
app.register_blueprint(route_enzh, url_prefix='/new_trans')
app.register_blueprint(route_tag, url_prefix='/new_trans')

if __name__ == "__main__":
    app.run('0.0.0.0', port=8864)
    # app.run('0.0.0.0', port=8865)
