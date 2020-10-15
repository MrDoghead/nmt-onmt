# coding=utf8

import json
from flask import request, Blueprint
import requests
from hb_translate.web import en_ch_config_tag
from hb_translate.pre_process import custom_tag


route_tag = Blueprint('tag_service', __name__)

# do initial
cus_dict_path = en_ch_config_tag.align_dict_path
tag_helper = custom_tag.TagHelper(cus_dict_path)


@route_tag.route('/mod_words', methods=['POST', 'GET'])
def notify_mod_words():
    """
        mod_info = {"add":[{zh: w1, en:ew1,}, ... ],
                    "mod":[{zh:w1, en:w2}], 
                    "del":[{zh: w1, en:ew1,}]}
    """
    raw_dict = request.json
    tag_helper.update(raw_dict)
    result = {
        "status": 0,
        "msg": "词典更新成功!"
    }
    return json.dumps(result)
