from sparseml.pytorch.utils.quantization import quantize_torch_qat_export

model = "arunet_1.1.0a_pruned_85qat.onnx"
out = "arunet_1.1.0a_pruned_85qat_converted.onnx"

m = quantize_torch_qat_export(model, out)
