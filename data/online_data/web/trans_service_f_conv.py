# coding=utf8

# coding=utf8

import json
from flask import request, Blueprint
from fairseq import options
from hb_translate.predictor_online import Predictor
from hb_translate.web import en_ch_config_f_conv as en_ch_conf
# from apscheduler.schedulers.background import BackgroundScheduler

route_enzh = Blueprint('en2zh_trans', __name__)

# do initial
parser = options.get_generation_parser(interactive=True)
input_args = en_ch_conf.get_args_arr()
cus_dict_path = en_ch_conf.align_dict_path
args = options.parse_args_and_arch(parser, input_args)

predictor = Predictor(args, cus_dict_path)


@route_enzh.route('/en2zh', methods=['POST', 'GET'])
def get_filter_info():
    raw_dict = request.json
    sen_list = raw_dict.get("text_list")
    trans_result = predictor.predict(sen_list)
    res_dict = {"text_list": trans_result}

    result = {
        "status": 0,
        "msg": res_dict
    }
    return json.dumps(result)


@route_enzh.route('/change_word', methods=['POST', 'GET'])
def update_words():
    raw_dict = request.json
    predictor.update_tag_words(raw_dict)
    result = {
        "status": 0,
        "msg": "succ"
    }
    return json.dumps(result)


def update_batchly():
    print("starting to update words")
    predictor.update_tag_batchly()

"""
sched = BackgroundScheduler(daemon=True)
sched.add_job(update_batchly, 'interval', minutes=30)
sched.start()
"""
