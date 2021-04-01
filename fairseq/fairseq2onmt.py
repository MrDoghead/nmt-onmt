#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
use CUDA_VISIBLE_DEVICES to run the test

'''
from argparse import Namespace
import torch
#from onmt.model_builder import build_base_model
#import fairseq
from fairseq import tasks
#from numpy.testing import assert_allclose

import torch.nn as nn
'''
notice that: layernorm epsilon is 1e-6 in onmt and 1e-5 in fairseq/ctranslate2
'''
from apex.normalization import FusedLayerNorm
class FixLayerNorm(FusedLayerNorm):
    def __init__(self, shape, eps=1e-5):
        super().__init__(shape, 1e-5)
nn.LayerNorm = FixLayerNorm

torch.manual_seed(123456)

use_large = False
def onmt_cfg():
    opt = Namespace(
        aan_useffn=False, alignment_heads=0, alignment_layer=-3, apex_opt_level='O2', attention_dropout=[0.1], audio_enc_pooling='1', average_decay=0, average_every=1, batch_size=2048, batch_type='tokens', bridge=False, brnn=False, cnn_kernel_width=3, config=None, context_gate=None, copy_attn=False, copy_attn_force=False, copy_attn_type='general', copy_loss_by_seqlength=False, coverage_attn=False, data_weights=[1], decay_method='noam', decay_steps=10000, decoder_type='transformer', dropout=[0.1], dropout_steps=[0], early_stopping=0, early_stopping_criteria=None, encoder_type='transformer', epochs=0, exp='', exp_host='', feat_merge='concat', feat_vec_exponent=0.7, feat_vec_size=-1, fix_word_vecs_dec=False, fix_word_vecs_enc=False, full_context_alignment=False, generator_function='softmax', global_attention='general', global_attention_function='softmax', gpu_backend='nccl', gpu_ranks=[0, 1, 2, 3], gpu_verbose_level=0, gpuid=[], image_channel_size=3, input_feed=1, keep_checkpoint=20, label_smoothing=0.1, lambda_align=0.0, lambda_coverage=0.0, layers=6, learning_rate=1.0, learning_rate_decay=0.5, log_file='', log_file_level='0', loss_scale=0, master_ip='localhost', master_port=10000, max_generator_batches=2, max_grad_norm=0.0, max_relative_positions=0, model_dtype='fp16', model_type='text', normalization='tokens', optim='fusedadam', param_init=0.0, param_init_glorot=True, pool_factor=8192, pre_word_vecs_dec=None, pre_word_vecs_enc=None, queue_size=40, report_every=50, reset_optim='none', reuse_copy_attn=False, sample_rate=16000, seed=-1, self_attn_type='scaled-dot', share_decoder_embeddings=False, share_embeddings=False, single_pass=False, src_noise=[], src_noise_prob=[], start_decay_steps=50000, tensorboard=False, tensorboard_log_dir='runs/onmt', truncated_decoder=0, valid_batch_size=32, valid_steps=4000, warmup_steps=8000, window_size=0.02, world_size=4,
        position_encoding=True,
        dec_layers=6, enc_layers=6, heads=8, transformer_ff=2048,
        rnn_size=512, dec_rnn_size=512, enc_rnn_size=512,
        word_vec_size=512, tgt_word_vec_size=512, src_word_vec_size=512)
    if use_large:
        opt.heads = 16,
        opt.transformer_ff = 4096,
        opt.rnn_size = opt.dec_rnn_size = opt.enc_rnn_size = \
        opt.word_vec_size = opt.tgt_word_vec_size = opt.src_word_vec_size = 1024
        return opt
    return opt

def transfer_model(fairseq_ckpt):
    print('transfering the model')
    onmt_transformer = {}
    transfer_map_0 = [
        ('encoder.embed_positions.weight', 'encoder.embeddings.make_embedding.pe.pe'),
        ('decoder.embed_positions.weight', 'decoder.embeddings.make_embedding.pe.pe'),
    ]
    for k, v in transfer_map_0:
        onmt_transformer[v] = fairseq_ckpt[k][2:].unsqueeze(1)
    
    
    transfer_map_1 = [
        ('encoder.embed_tokens.weight', 'encoder.embeddings.make_embedding.emb_luts.0.weight'),
        ('decoder.embed_tokens.weight', 'decoder.embeddings.make_embedding.emb_luts.0.weight'),
        ('encoder.layer_norm.weight', 'encoder.layer_norm.weight'),
        ('encoder.layer_norm.bias', 'encoder.layer_norm.bias'),
        ('decoder.layer_norm.weight', 'decoder.layer_norm.weight'),
        ('decoder.layer_norm.bias', 'decoder.layer_norm.bias'),
    ]
    for k, v in transfer_map_1:
        onmt_transformer[v] = fairseq_ckpt[k]
    
    for layer_num in range(onmt_cfg().enc_layers):
        transfer_map_2 = [
            (f'encoder.layers.{layer_num}.self_attn.k_proj.weight', f'encoder.transformer.{layer_num}.self_attn.linear_keys.weight'),
            (f'encoder.layers.{layer_num}.self_attn.k_proj.bias', f'encoder.transformer.{layer_num}.self_attn.linear_keys.bias'),
            (f'encoder.layers.{layer_num}.self_attn.v_proj.weight', f'encoder.transformer.{layer_num}.self_attn.linear_values.weight'),
            (f'encoder.layers.{layer_num}.self_attn.v_proj.bias', f'encoder.transformer.{layer_num}.self_attn.linear_values.bias'),
            (f'encoder.layers.{layer_num}.self_attn.q_proj.weight', f'encoder.transformer.{layer_num}.self_attn.linear_query.weight'),
            (f'encoder.layers.{layer_num}.self_attn.q_proj.bias', f'encoder.transformer.{layer_num}.self_attn.linear_query.bias'),
            (f'encoder.layers.{layer_num}.self_attn.out_proj.weight', f'encoder.transformer.{layer_num}.self_attn.final_linear.weight'),
            (f'encoder.layers.{layer_num}.self_attn.out_proj.bias', f'encoder.transformer.{layer_num}.self_attn.final_linear.bias'),
            (f'encoder.layers.{layer_num}.self_attn_layer_norm.weight', f'encoder.transformer.{layer_num}.layer_norm.weight'),
            (f'encoder.layers.{layer_num}.self_attn_layer_norm.bias', f'encoder.transformer.{layer_num}.layer_norm.bias'),
            (f'encoder.layers.{layer_num}.fc1.weight', f'encoder.transformer.{layer_num}.feed_forward.w_1.weight'),
            (f'encoder.layers.{layer_num}.fc1.bias', f'encoder.transformer.{layer_num}.feed_forward.w_1.bias'),
            (f'encoder.layers.{layer_num}.fc2.weight', f'encoder.transformer.{layer_num}.feed_forward.w_2.weight'),
            (f'encoder.layers.{layer_num}.fc2.bias', f'encoder.transformer.{layer_num}.feed_forward.w_2.bias'),
            (f'encoder.layers.{layer_num}.final_layer_norm.weight', f'encoder.transformer.{layer_num}.feed_forward.layer_norm.weight'),
            (f'encoder.layers.{layer_num}.final_layer_norm.bias', f'encoder.transformer.{layer_num}.feed_forward.layer_norm.bias'),
    
            (f'decoder.layers.{layer_num}.self_attn.k_proj.weight', f'decoder.transformer_layers.{layer_num}.self_attn.linear_keys.weight'),
            (f'decoder.layers.{layer_num}.self_attn.k_proj.bias', f'decoder.transformer_layers.{layer_num}.self_attn.linear_keys.bias'),
            (f'decoder.layers.{layer_num}.self_attn.v_proj.weight', f'decoder.transformer_layers.{layer_num}.self_attn.linear_values.weight'),
            (f'decoder.layers.{layer_num}.self_attn.v_proj.bias', f'decoder.transformer_layers.{layer_num}.self_attn.linear_values.bias'),
            (f'decoder.layers.{layer_num}.self_attn.q_proj.weight', f'decoder.transformer_layers.{layer_num}.self_attn.linear_query.weight'),
            (f'decoder.layers.{layer_num}.self_attn.q_proj.bias', f'decoder.transformer_layers.{layer_num}.self_attn.linear_query.bias'),
            (f'decoder.layers.{layer_num}.self_attn.out_proj.weight', f'decoder.transformer_layers.{layer_num}.self_attn.final_linear.weight'),
            (f'decoder.layers.{layer_num}.self_attn.out_proj.bias', f'decoder.transformer_layers.{layer_num}.self_attn.final_linear.bias'),
            (f'decoder.layers.{layer_num}.self_attn_layer_norm.weight', f'decoder.transformer_layers.{layer_num}.layer_norm_1.weight'),
            (f'decoder.layers.{layer_num}.self_attn_layer_norm.bias', f'decoder.transformer_layers.{layer_num}.layer_norm_1.bias'),
    
            (f'decoder.layers.{layer_num}.encoder_attn.k_proj.weight', f'decoder.transformer_layers.{layer_num}.context_attn.linear_keys.weight'),
            (f'decoder.layers.{layer_num}.encoder_attn.k_proj.bias', f'decoder.transformer_layers.{layer_num}.context_attn.linear_keys.bias'),
            (f'decoder.layers.{layer_num}.encoder_attn.v_proj.weight', f'decoder.transformer_layers.{layer_num}.context_attn.linear_values.weight'),
            (f'decoder.layers.{layer_num}.encoder_attn.v_proj.bias', f'decoder.transformer_layers.{layer_num}.context_attn.linear_values.bias'),
            (f'decoder.layers.{layer_num}.encoder_attn.q_proj.weight', f'decoder.transformer_layers.{layer_num}.context_attn.linear_query.weight'),
            (f'decoder.layers.{layer_num}.encoder_attn.q_proj.bias', f'decoder.transformer_layers.{layer_num}.context_attn.linear_query.bias'),
            (f'decoder.layers.{layer_num}.encoder_attn.out_proj.weight', f'decoder.transformer_layers.{layer_num}.context_attn.final_linear.weight'),
            (f'decoder.layers.{layer_num}.encoder_attn.out_proj.bias', f'decoder.transformer_layers.{layer_num}.context_attn.final_linear.bias'),
            (f'decoder.layers.{layer_num}.encoder_attn_layer_norm.weight', f'decoder.transformer_layers.{layer_num}.layer_norm_2.weight'),
            (f'decoder.layers.{layer_num}.encoder_attn_layer_norm.bias', f'decoder.transformer_layers.{layer_num}.layer_norm_2.bias'),
    
            (f'decoder.layers.{layer_num}.fc1.weight', f'decoder.transformer_layers.{layer_num}.feed_forward.w_1.weight'),
            (f'decoder.layers.{layer_num}.fc1.bias', f'decoder.transformer_layers.{layer_num}.feed_forward.w_1.bias'),
            (f'decoder.layers.{layer_num}.fc2.weight', f'decoder.transformer_layers.{layer_num}.feed_forward.w_2.weight'),
            (f'decoder.layers.{layer_num}.fc2.bias', f'decoder.transformer_layers.{layer_num}.feed_forward.w_2.bias'),
            (f'decoder.layers.{layer_num}.final_layer_norm.weight', f'decoder.transformer_layers.{layer_num}.feed_forward.layer_norm.weight'),
            (f'decoder.layers.{layer_num}.final_layer_norm.bias', f'decoder.transformer_layers.{layer_num}.feed_forward.layer_norm.bias'),
        ]
        for k, v in transfer_map_2:
            onmt_transformer[v] = fairseq_ckpt[k]

    assert torch.all(torch.eq(fairseq_ckpt['decoder.output_projection.weight'],
                              fairseq_ckpt['decoder.embed_tokens.weight']))
    onmt_generator = {
        '0.weight' : fairseq_ckpt['decoder.embed_tokens.weight'],
        '0.bias' : torch.zeros(fairseq_ckpt['decoder.embed_tokens.weight'].size()[0],
                               dtype=fairseq_ckpt['decoder.embed_tokens.weight'].dtype)
    }
    return onmt_transformer, onmt_generator



from onmt.inputters.text_dataset import TextMultiField
from torchtext.data.field import Field
from torchtext.vocab import Vocab
from collections import Counter
def transfer_vocab(fairseq_ckpt):
    print('transfering the vocabulary')
    task = tasks.setup_task(fairseq_ckpt['args'])
    src_dict = task.source_dictionary
    tgt_dict = task.target_dictionary
    
    src_field = TextMultiField('src', Field(eos_token='</s>', pad_token='<blank>', unk_token='<unk>'), [])
    tgt_field = TextMultiField('src', Field(eos_token='</s>', pad_token='<blank>', unk_token='<unk>'), [])
    def to_onmt(fairseq_vocab):
        specials=['<s>', '<blank>', '</s>', '<unk>']
        onmt_vocab = Vocab(Counter(), specials=specials)
        onmt_vocab.stoi = fairseq_vocab.indices
        onmt_vocab.itos = fairseq_vocab.symbols
        
        onmt_vocab.stoi['<blank>'] = onmt_vocab.stoi['<pad>']
        del onmt_vocab.stoi['<pad>']
        onmt_vocab.itos[onmt_vocab.stoi['<blank>']] = '<blank>'
        
        onmt_vocab.freqs = {s:freq for s, freq in zip(onmt_vocab.itos, fairseq_vocab.count)}
        assert onmt_vocab.UNK == '<unk>'
        assert onmt_vocab.unk_index == 3
        return onmt_vocab
    src_field.base_field.vocab = to_onmt(src_dict)
    tgt_field.base_field.vocab = to_onmt(tgt_dict)
    
    return {'src' : src_field,
            'tgt' : tgt_field}

"""
from onmt.utils.misc import sequence_mask
def test_model(fairseq_ckpt, onmt_ckpt):
    # TODO test 未完成
    print('testing the model')
    device = torch.device('cuda')
    batch_size=16
    seq_len=20
    prev_seq_len=8
    src_tokens = torch.randint(4, 10000, [batch_size, seq_len], device=device)
    src_lengths = torch.full([batch_size], seq_len, dtype=src_tokens.dtype, device=device)
    prev_out_tokens = torch.randint(4, 10000, [batch_size, prev_seq_len], device=device)
    
    src = src_tokens.T.unsqueeze(2)
    tgt = prev_out_tokens.T.unsqueeze(2)
    tgt = torch.cat([tgt, torch.zeros([1, batch_size, 1], dtype=src_tokens.dtype, device=device)])
    
    '''
    fairseq model
    
    '''
    args = fairseq_ckpt['args']
    task = tasks.setup_task(args)
    fairseq_model = fairseq.models.transformer.TransformerModel.build_model(args, task)
    fairseq_model.load_state_dict(fairseq_ckpt['model'], strict=False, args=args)
    fairseq_model.to(device)
    fairseq_model.half()
    fairseq_model.eval()
    
    
#    fairseq_out, _ = fairseq_model(src_tokens, src_lengths, prev_out_tokens)
#    fairseq_out, _ = fairseq_model(src_tokens, src_lengths, prev_out_tokens, features_only=True)
    '''
    onmt model
    '''
    
    onmt_model = build_base_model(
        onmt_cfg(),
        onmt_ckpt['vocab'],
        False,
        checkpoint=onmt_ckpt
    )
    onmt_model.to(device)
    onmt_model.half()
    onmt_model.eval()
    
#    onmt_out, onmt_attn = onmt_model(
#        src,
#        tgt,
#        src_lengths
#    )
#    onmt_out = onmt_model.generator[0](onmt_out)
#    onmt_out = onmt_out.transpose(0,1)
#    
    
    fs_e_o = fairseq_model.encoder(src_tokens, src_lengths)
    fairseq_encoder_out = fs_e_o.encoder_out
    onmt_encoder_emb, onmt_encoder_out, lengths = onmt_model.encoder(src, src_lengths)
    assert_allclose(fairseq_encoder_out.detach().cpu().numpy(),
                    onmt_encoder_out.detach().cpu().numpy())
    
    
    # Decoder
    fs_de_embed = fairseq_model.decoder.embed_scale * fairseq_model.decoder.embed_tokens(prev_out_tokens) + \
                  fairseq_model.decoder.embed_positions(prev_out_tokens, incremental_state=None)
    fs_de_embed = fs_de_embed.transpose(0,1)
    fs_future_mask = fairseq_model.decoder.buffered_future_mask(fs_de_embed)
    fs = fairseq_model.decoder.layers[0]
    fs_input = fs.self_attn_layer_norm(fs_de_embed)
    fs_self_out, _ = fs.self_attn(
            query=fs_input,
            key=fs_input,
            value=fs_input,
            key_padding_mask=None,
            incremental_state=None,
            need_weights=False,
            attn_mask=fs_future_mask,
    )
    fs_input2 = fs.encoder_attn_layer_norm(fs.residual_connection(fs_self_out, fs_input))
    fs_out2, fs_att2 = fs.encoder_attn(
                query=fs_input2,
                key=fairseq_encoder_out,
                value=fairseq_encoder_out,
                key_padding_mask=fs_e_o.encoder_padding_mask,
                incremental_state=None,
                static_kv=True,
                need_weights=False,
                need_head_weights=False,
            )
    
    
    l1, fs_att, _ = fairseq_model.decoder.layers[0](
        fs_de_embed,
        fairseq_encoder_out,
        fs_e_o.encoder_padding_mask,
        incremental_state=None,
        self_attn_mask=fs_future_mask,
    )
    
    
    
    in_dec = tgt[:-1]
    onmt_model.decoder.init_state(src, onmt_encoder_out, onmt_encoder_emb)
    on_de_embed = onmt_model.decoder.embeddings(in_dec).transpose(0,1).contiguous()
    on_tgt_pad_mask = in_dec[:,:,0].transpose(0,1).data.eq(1).unsqueeze(1)
    on_tgt_len = on_tgt_pad_mask.size(-1)
    on_future_mask = torch.ones(
                    [on_tgt_len, on_tgt_len],
                    device=on_tgt_pad_mask.device,
                    dtype=torch.uint8,
                )
    on_future_mask = on_future_mask.triu_(1).view(1, on_tgt_len, on_tgt_len).bool()
    on_dec_mask = torch.gt(on_tgt_pad_mask + on_future_mask, 0)
    on = onmt_model.decoder.transformer_layers[0]
    on_input = on.layer_norm_1(on_de_embed)
    on_self_out, _ = on.self_attn(
                on_input,
                on_input,
                on_input,
                mask=on_dec_mask,
                layer_cache=None,
                attn_type="self",
    )
    on_input2 = on.layer_norm_2(on_self_out + on_input)
    ####
    on_out2, on_att2 = on.context_attn(
            onmt_encoder_out.transpose(0,1).contiguous(),
            onmt_encoder_out.transpose(0,1).contiguous(),
            on_input2,
            mask=~sequence_mask(lengths, onmt_model.decoder.state["src"].shape[0]).unsqueeze(1),
            layer_cache=None,
            attn_type="context",
        )
    
    on_l1, on_att, _ = onmt_model.decoder.transformer_layers[0](
        on_de_embed,
        onmt_encoder_out.transpose(0,1).contiguous(),
        ~sequence_mask(lengths, onmt_model.decoder.state["src"].shape[0]).unsqueeze(1),
        in_dec[:,:,0].transpose(0,1).data.eq(1).unsqueeze(1)
    )
    

    
    assert_allclose(fs_de_embed.detach().cpu().numpy(),
                    on_de_embed.transpose(0,1).detach().cpu().numpy())
    assert_allclose(fs_input.detach().cpu().numpy(),
                    on_input.transpose(0,1).detach().cpu().numpy())
    assert_allclose(fs_self_out.detach().cpu().numpy(),
                    on_self_out.transpose(0,1).detach().cpu().numpy())
    assert_allclose(fs_input2.detach().cpu().numpy(),
                    on_input2.transpose(0,1).detach().cpu().numpy())
    assert_allclose(fs_out2.detach().cpu().numpy(),
                    on_out2.transpose(0,1).detach().cpu().numpy())

    assert_allclose(l1.detach().cpu().numpy(),
                    on_l1.transpose(0,1).detach().cpu().numpy())
    assert_allclose(fs_att.detach().cpu().numpy(),
                    on_att.detach().cpu().numpy())
    '''
    notice that fairseq_out.shape = bs * prev_len * vocab, but
                onmt_out.shape = bs * prev_len-1 * vocab
    '''
    
#    for i in range(prev_seq_len):
#        print(i)
#        assert_allclose(fairseq_out[:,i,:].detach().cpu().numpy(),
#                        onmt_out[:,i,:].detach().cpu().numpy())
"""

    
def transfer(fairseq_path, onmt_path):
    '''
    只保留ctranslate2 需要的部分
    '''
    fairseq_ckpt = torch.load(fairseq_path, map_location='cpu')
    
    onmt_vocab = transfer_vocab(fairseq_ckpt)
    onmt_transformer, onmt_generator = transfer_model(fairseq_ckpt['model'])
    onmt_ckpt = {'model' : onmt_transformer,
                 'generator' : onmt_generator,
                 'vocab' : onmt_vocab}
#    test_model(fairseq_ckpt, onmt_ckpt)
    print('saving the model')
    torch.save(onmt_ckpt, onmt_path)


if __name__ == '__main__':
    import sys
    # TODO parseargs
    from_ = sys.argv[1]
    to_ = sys.argv[2]
    if 'large' in sys.argv:
        use_large = True
    transfer(from_, to_)

