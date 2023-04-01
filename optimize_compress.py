import onnxruntime as rt
import xxhash
import gzip

MODEL = "arunet_2.1.1_bag4.onnx"
MODEL_OPT = MODEL[:-5] + "-optimized.onnx"


def get_hash(fname: str) -> str:
    """
    Get xxHash3 of a file
    Args:
        fname (str): Path to file
    Returns:
        str: xxHash3 of file
    """
    xxh = xxhash.xxh3_64()
    with open(fname, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            xxh.update(chunk)
    return xxh.hexdigest()


sess_options = rt.SessionOptions()
# Set graph optimization level
sess_options.graph_optimization_level = rt.GraphOptimizationLevel.ORT_ENABLE_BASIC

sess_options.optimized_model_filepath = MODEL_OPT

session = rt.InferenceSession(MODEL, sess_options)

print("xxh3_64:", get_hash(MODEL_OPT))

# # test
# import deepsparse
# f = gzip.GzipFile(MODEL_OPT + ".gz", "rb")
# data = f.read()
# # gzip.decompress(data)
# fp = open("uncompressed.onnx", "wb")
# fp.write(data)
# fp.close()
# f.close()
# deepsparse.compile_model("uncompressed.onnx")
