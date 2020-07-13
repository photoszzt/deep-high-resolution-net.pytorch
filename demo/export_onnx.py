import onnxruntime as rt
import onnx.utils
import onnx

import sys
sys.path.append("../lib")

from config import update_config
from config import cfg
import torch
import models
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description='Train keypoints network')
    # general
    parser.add_argument('--cfg', type=str, required=True)

    parser.add_argument('opts',
                        help='Modify config options using the command-line',
                        default=None,
                        nargs=argparse.REMAINDER)

    args = parser.parse_args()

    # args expected by supporting codebase
    args.modelDir = ''
    args.logDir = ''
    args.dataDir = ''
    args.prevModelDir = ''
    return args


args = parse_args()
update_config(cfg, args)

pose_model = models.pose_hrnet.get_pose_net(
    cfg, is_train=False
)
pose_model.load_state_dict(torch.load('model/pose_hrnet_w32_256x192.pth'),
                           strict=False)
onnx_file_name = 'pose_hrnet_w32_256x192.onnx'
batch_size = 10
x = torch.randn((batch_size, 3, 256, 192), requires_grad=True)
print('Export the onnx model ...')
torch.onnx.export(pose_model,
                  x,
                  onnx_file_name,
                  export_params=True,
                  opset_version=11,
                  do_constant_folding=True,
                  input_names=['input'],
                  output_names=['output'],
                  dynamic_axes={'input': {0: 'batch_size'},
                                'output': {0: 'batch_size'}}
                  )
print("done exporting; load onnx model and optimize it")
model = onnx.load(onnx_file_name)
onnx.checker.check_model(model)

sess_op_cpu = rt.SessionOptions()
sess_op_cpu.graph_optimization_level = rt.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
sess_op_cpu.intra_op_num_threads = 1
sess_op_cpu.execution_mode = rt.ExecutionMode.ORT_SEQUENTIAL
providers = ['CPUExecutionProvider']
sess_op_cpu.optimized_model_filepath = "cpu_pose_hrnet_w32_256x192.onnx"
sess_cpu = rt.InferenceSession(onnx_file_name, providers=providers,
                               sess_options=sess_op_cpu)

sess_op_gpu = rt.SessionOptions()
sess_op_gpu.graph_optimization_level = rt.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
sess_op_gpu.intra_op_num_threads = 1
sess_op_gpu.execution_mode = rt.ExecutionMode.ORT_SEQUENTIAL
providers = ['CUDAExecutionProvider']
sess_op_gpu.optimized_model_filepath = "gpu_pose_hrnet_w32_256x192.onnx"
sess_gpu = rt.InferenceSession(onnx_file_name, providers=providers,
                               sess_options=sess_op_gpu)

model = onnx.load("cpu_pose_hrnet_w32_256x192.onnx")
onnx.checker.check_model(model)
model = onnx.load("gpu_pose_hrnet_w32_256x192.onnx")
onnx.checker.check_model(model)