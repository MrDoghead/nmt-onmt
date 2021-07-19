import onnx
import onnxruntime

model_path = '/home/ubuntu/caodongnan/work/nmt_opennmt/data/offline_data/model_bin/onnx/nmt_en_zh.onnx'
onnx_model = onnx.load(model_path)
onnx.checker.check_model(onnx_model)

#nmt_session = onnxruntime.InferenceSession(model_path)
