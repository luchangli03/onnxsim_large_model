import sys
import logging
import json
import onnx
from onnx_utils import get_onnx_tensor_proto_dtype, get_onnx_tensor_proto_shape
from onnx_utils import DTYPE_BYTES, del_onnx_initializers, shape_elem_num

ELEM_TH = 768 * 768


def compress_onnx_model(onnx_model):
    graph = onnx_model.graph
    initializer = graph.initializer

    name_2_init_map = {}
    for init in initializer:
        name_2_init_map[init.name] = init

    replaced_inits = []
    for init in graph.initializer:
        dtype = get_onnx_tensor_proto_dtype(init)
        shape = get_onnx_tensor_proto_shape(init)

        if dtype not in DTYPE_BYTES:
            continue

        shape_elem = shape_elem_num(shape)
        if shape_elem <= ELEM_TH:
            continue
        replaced_inits.append(init)

    replaced_tensor_names = [init.name for init in replaced_inits]
    print("replaced_tensor_names:", replaced_tensor_names)
    del_onnx_initializers(graph, replaced_tensor_names)

    new_inputs = []
    for init in replaced_inits:
        dtype = get_onnx_tensor_proto_dtype(init)
        shape = get_onnx_tensor_proto_shape(init)
        _input = onnx.helper.make_tensor_value_info(name=init.name, elem_type=dtype, shape=shape)
        new_inputs.append(_input)
    graph.input.extend(new_inputs)
    return onnx_model, replaced_inits


def uncompress_onnx_model(onnx_model, replaced_inits):
    graph = onnx_model.graph
    graph.initializer.extend(replaced_inits)
    replaced_tensor_names = [init.name for init in replaced_inits]

    in_num = len(graph.input)
    for idx in reversed(range(in_num)):
        if graph.input[idx].name in replaced_tensor_names:
            del graph.input[idx]
    return onnx_model
