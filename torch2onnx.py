import torch
import torch.onnx
import onmt
from onmt.bin import train
from onmt.encoders import str2enc
from onmt.decoders import str2dec
from onmt.utils.parse import ArgumentParser
from onmt.model_builder import build_model
from onmt.inputters.inputter import old_style_vocab
from nmt_trans.utils import file_helper, conf_parser
import sys

def load_model(opt,model_path):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f'loading checkpoint from {device}')
    checkpoint = torch.load(model_path,map_location=device)
    model_opt = ArgumentParser.ckpt_model_opts(checkpoint["opt"])
    ArgumentParser.update_model_opts(model_opt)
    ArgumentParser.validate_model_opts(model_opt)
    vocab = checkpoint['vocab']

    if old_style_vocab(vocab):
        fields = load_old_vocab(
            vocab, opt.model_type, dynamic_dict=opt.copy_attn)
    else:
        fields = vocab

    print('building model...')
    model = build_model(model_opt, opt, fields, checkpoint)
    
    return model

def run(conf,pt_path,onnx_path):
    parser = train._get_parser()
    conf_dict = conf_parser.conf2dict(conf.train_info)
    train_path = file_helper.get_abs_path(conf.train_info.train_fmt_path)
    args_dict = {
            "data": train_path,
            "save_model": pt_path,
        }
    conf_dict = conf_parser.conf2dict(conf.train_info)
    args_dict.update(conf_dict)
    args_arr = conf_parser.dict2args(args_dict)
    print('args parsing...')
    args, _ = parser.parse_known_args(args_arr)

    #import os 
    #if not os.path.exists(onnx_path): 
    #    os.makedirs(onnx_path)

    # load model
    model = load_model(args,pt_path)
    model.eval()
    
    # Input to the model
    batch_size = 1
    seq_len = 200
    # input size batch_size * seq_len
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dummy_src = torch.randint(0,100,(seq_len, batch_size, 1),device=device)
    dummy_tgt = torch.randint(0,100,(seq_len, batch_size, 1),device=device)
    dummy_length = torch.tensor([batch_size,],device=device)

    #data = torch.load('test_in.pt')
    #dummy_src,dummy_tgt,dummy_length = data

    print(f"exporting onnx to {onnx_path}")
    # Export the model
    torch.onnx.export(model,               # model being run
            (dummy_src, dummy_tgt, dummy_length),               # model input (or a tuple for multiple inputs)
            onnx_path,   # where to save the model (can be a file or file-like object)
            export_params=True,        # store the trained parameter weights inside the model file
            opset_version=12,          # the ONNX version to export the model to
            do_constant_folding=True,  # whether to execute constant folding for optimization
            input_names = ['input'],   # the model's input names
            output_names = ['output'], # the model's output names
            dynamic_axes={'input' : {1 : 'batch_size'},    # variable lenght axes
                'output' : {1 : 'batch_size'}
                }
            )

if __name__=='__main__':
    #pt_path = './data/offline_data/model_bin/nmt_en_zh/cht_avg.pt'
    pt_path = './data/offline_data/model_bin/nmt_en_zh/cht_step_80000.pt'
    onnx_path = '/home/ubuntu/caodongnan/work/nmt_opennmt/data/offline_data/model_bin/onnx/nmt_en_zh.onnx'
    conf_path = '/home/ubuntu/caodongnan/work/nmt_opennmt/nmt_trans/conf/en_zh_config.json'
    conf = conf_parser.parse_conf(conf_path)
    run(conf,pt_path,onnx_path)

