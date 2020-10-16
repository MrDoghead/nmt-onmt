# coding=utf8

import re
import json
from flask import request, Blueprint
from nmt_trans.utils import conf_parser
from nmt_trans import predictor

sen_sep = re.compile("([。？;\n])+")


route_entity = Blueprint('chat', __name__)
# do ner initial

# do initial
predictor = None


def init_predictor(conf_path):
    global predictor
    config = conf_parser.parse_conf(conf_path)
    predictor = predictor.Predictor(config)


@route_entity.route('/en_trans', methods=['POST', 'GET'])
def get_filter_info():
    sen_triple_info = request.json
    sen_info_arr = sen_triple_info.get("text_list")
    if sen_info_arr is None:
        result = {
            "status": 1,
            "msg": "没有解析到text_list, 请检查传入的参数"
        }
        return json.dumps(result, ensure_ascii=False)

    infos = predictor.predict(sen_info_arr)
    result = {
        "status": 0,
        "msg": infos
    }
    return json.dumps(result, ensure_ascii=False)


