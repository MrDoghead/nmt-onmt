# coding=utf8

from onmt.bin import train
from nmt_trans.utils import file_helper, conf_parser


class Trainer(object):
    def __init__(self, conf):
        self.conf = conf

    """
    # faq 配置
 29 CUDA_VISIBLE_DEVICES="2,3,4,5,6,7" onmt_train -data ${dest_data} -save_model ${mode_path} \
 30         -layers 6 -rnn_size 512 -word_vec_size 512 -transformer_ff 2048 -heads 8  \
 31         -encoder_type transformer -decoder_type transformer -position_encoding \
 32         -train_steps 200000  -max_generator_batches 2 -dropout 0.1 \
 33         -batch_size 2048 -batch_type tokens -normalization tokens  -accum_count 2 \
 34         -optim adam -adam_beta2 0.998 -decay_method noam -warmup_steps 16000 -learning_rate 2 \
 35         -max_grad_norm 0 -param_init 0  -param_init_glorot \
 36         -label_smoothing 0.1 -valid_steps 10000 -save_checkpoint_steps 5000 \
 37         -world_size 6 -gpu_ranks 0 1 2 3 4 5
    """

    def get_args(self, parser):
        train_path = file_helper.get_abs_path(self.conf.train_info.train_fmt_path)
        model_path = file_helper.get_abs_path(self.conf.model_path)
        file_helper.mk_folder_for_file(model_path)
        args_dict = {
            "data": train_path,
            "save_model": model_path,
        }
        conf_dict = conf_parser.conf2dict(self.conf.train_info)
        args_dict.update(conf_dict)
        args_arr = conf_parser.dict2args(args_dict)
        args, _ = parser.parse_known_args(args_arr)
        return args

    def train(self):
        parser = train._get_parser()
        args = self.get_args(parser)
        train.train(args)


if __name__ == "__main__":
    import sys
    t_conf_path = sys.argv[1]
    # t_conf_path = file_helper.get_conf_file("chat_config.json")
    conf = conf_parser.parse_conf(t_conf_path)
    t_trainer = Trainer(conf)
    t_trainer.train()
