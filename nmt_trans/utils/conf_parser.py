# coding=utf8

import json


def parse_conf(f_path):
    return Config(config_file=f_path)


def conf2args(conf):
    dict_ = conf2dict(conf)
    return dict2args(dict_)


def conf2dict(conf):
    res = conf.__dict__.copy()
    if "dict" in res:
        del res["dict"]
    return res


def dict2args(dict_):
    res = []
    for key, val in dict_.items():
        res.append("--" + key)
        if val is not None:
            if not isinstance(val, list):
                res.append(str(val))
            else:
                for elem in val:
                    res.append(str(elem))
    return res


class Config(object):
    """Config load from json file
    """

    def __init__(self, config=None, config_file=None):
        if config_file:
            with open(config_file, 'r') as fin:
                config = json.load(fin)

        self.dict = config
        if config:
            self._update(config)

    def __getitem__(self, key):
        return self.dict[key]

    def __contains__(self, item):
        return item in self.dict

    def items(self):
        return self.dict.items()

    def add(self, key, value):
        """Add key value pair
        """
        self.__dict__[key] = value

    def _update(self, config):
        if not isinstance(config, dict):
            return

        for key in config:
            if isinstance(config[key], dict):
                config[key] = Config(config[key])

            if isinstance(config[key], list):
                config[key] = [Config(x) if isinstance(x, dict) else x for x in
                               config[key]]

        self.__dict__.update(config)


if __name__ == "__main__":
    t_path = "../conf/chat_config.json"
    test_conf = parse_conf(t_path)
    evl_conf = test_conf.train_info
    t_d = conf2dict(evl_conf)
    t_args = conf2args(evl_conf)
    print(t_args)
