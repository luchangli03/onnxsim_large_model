# onnxsim_large_model

simplify large onnx model (>2GB)

Can be used to simplify large onnx exported from stable diffusion or large language models.

## Examples

```python
python simplify_large_onnx.py -m model.onnx
```

-o can be used to set output file path or output dir.

you can set model input shape before onnx sim by setting a json str:

```
python simplify_large_onnx.py -m model.onnx --input_shape  '{"input":[1,1,4096]}'
```

--skip can be used to disable optimizations for onnxsim

This project also works for small onnx models and you can set --save_extern_data 0 to avoid save the weight into extern data file.

## How this project works

1. we replace large initializers in onnx models by const of shape nodes.

2. then simplify onnx model and avoid simplifing these const of shape nodes.
3. remove const of shape nodes and recover original initializers.

## Note

Please install onnxsim>=0.4.24

The project (all versions) and its developers are not responsible for the correctness of the exported models, and any consequences arising from the use of the project and exported models.
