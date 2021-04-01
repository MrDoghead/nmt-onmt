# coding=utf8

import re
import json
from flask import request, Blueprint
from nmt_trans.utils import conf_parser, file_helper
from nmt_trans import predictor as nmt_pred
from nmt_trans import predictor_zh2en as zh2en_pred
# from apscheduler.schedulers.background import BackgroundScheduler

sen_sep = re.compile("([。？;\n])+")


route_entity = Blueprint('chat', __name__)

# do initial
predictor = None
zh2en_predictor = None


def init_predictor(conf_path, zh2en_path):
    global predictor
    global zh2en_predictor
    dict_path = file_helper.get_online_data("caijing_clean.csv")
    if conf_path:
        config = conf_parser.parse_conf(conf_path)
        predictor = nmt_pred.Predictor(config, dict_path)
    else:
        predictor = None
    
    if zh2en_path:
        zh_config = conf_parser.parse_conf(zh2en_path)
        zh2en_predictor = zh2en_pred.Predictor(zh_config, dict_path)
    else:
        zh2en_predictor = None


@route_entity.route('/en2zh', methods=['POST', 'GET'])
def get_filter_info():
    sen_triple_info = request.json
    sen_info_arr = sen_triple_info.get("text_list")
    if sen_info_arr is None:
        result = {
            "status": 1,
            "msg": "没有解析到text_list, 请检查传入的参数"
        }
        return json.dumps(result)

    infos = predictor.predict(sen_info_arr)
    result = {
        "status": 0,
        "msg": infos
    }
    return json.dumps(result)


@route_entity.route('/zh2en', methods=['POST', 'GET'])
def get_zh2en():
    sen_triple_info = request.json
    sen_info_arr = sen_triple_info.get("text_list")
    if sen_info_arr is None:
        result = {
            "status": 1,
            "msg": "没有解析到text_list, 请检查传入的参数"
        }
        return json.dumps(result)

    infos, states = zh2en_predictor.predict(sen_info_arr)
    result = {
        "status": 0,
        "msg": infos,
        "selfcheck": states
    }
    return json.dumps(result)


@route_entity.route('/change_word', methods=['POST', 'GET'])
def update_words():
    raw_dict = request.json
    if predictor:
        predictor.update_tag_words(raw_dict)
    if zh2en_predictor:
        zh2en_predictor.update_tag_words(raw_dict)
    result = {
        "status": 0,
        "msg": "succ"
    }
    return json.dumps(result)


def update_batchly():
    print("starting to update words")
    predictor.update_tag_batchly()
    zh2en_predictor.update_tag_batchly()


"""
sched = BackgroundScheduler(daemon=True)
sched.add_job(update_batchly, 'interval', minutes=30)
sched.start()
"""
