import argparse
import os
import onnx
from onnxsim import simplify
from onnx_utils import set_onnx_input_shape
from compress_model import SIZE_1MB, compress_onnx_model, uncompress_onnx_model


def simplify_large_onnx(args):
    in_model_path = args.in_model_path
    out_model_path = args.out_model_path
    if not out_model_path:
        out_model_path = in_model_path[:-5] + ".sim.onnx"
    if os.path.isdir(out_model_path):
        out_model_path = os.path.join(out_model_path, os.path.basename(in_model_path))

    onnx_model = onnx.load(in_model_path)
    print(f"load model from {in_model_path} success")

    size_th_bytes = args.size_th_kb * 1024

    onnx_model, removed_inits = compress_onnx_model(onnx_model, size_th_bytes=size_th_bytes)
    print(f"compress model success")

    onnx_model = set_onnx_input_shape(onnx_model, args.input_shape)

    tensor_size_threshold = f"{args.size_th_kb}KB"
    skipped_optimizers = args.skip.split(";")
    onnx_model, check = simplify(onnx_model, skipped_optimizers=skipped_optimizers,
                                 tensor_size_threshold=tensor_size_threshold)
    if not check:
        raise ValueError(f"simplify compressed model {in_model_path} failed")

    print(f"simplify model success")

    onnx_model = uncompress_onnx_model(onnx_model, removed_inits)
    print(f"uncompress model success")

    save_extern = True if args.save_extern_data else False
    onnx.save(onnx_model, out_model_path, save_as_external_data=save_extern)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='export chatglm2',
    )
    parser.add_argument('-m', '--in_model_path', required=True, type=str)
    parser.add_argument('-o', '--out_model_path', required=False, type=str, default="")
    parser.add_argument('--size_th_kb', required=False, type=int, default="1024")
    parser.add_argument('--save_extern_data', required=False, type=int, default=1)
    parser.add_argument('--input_shape', required=False, type=str, default="")
    parser.add_argument('--skip', required=False, type=str, default="")

    args = parser.parse_args()

    if args.size_th_kb <= 1:
        raise ValueError("invalid size_th")

    print(args.input_shape)
    simplify_large_onnx(args)
