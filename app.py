import os
import numpy as np
import onnx
import onnxruntime as ort

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_24_PATH = os.path.join(BASE_DIR, "pangu_weather_24.onnx")
MODEL_6_PATH  = os.path.join(BASE_DIR, "pangu_weather_6.onnx")

INPUT_DIR = os.path.join(BASE_DIR, "input_data")

# === SAFETY CHECK ===
if not os.path.exists(MODEL_24_PATH):
    raise FileNotFoundError("pangu_weather_24.onnx not found")

if not os.path.exists(MODEL_6_PATH):
    raise FileNotFoundError("pangu_weather_6.onnx not found")

# Load models
model_24 = onnx.load(MODEL_24_PATH)
model_6 = onnx.load(MODEL_6_PATH)

# ONNX Runtime CPU ONLY
options = ort.SessionOptions()
options.intra_op_num_threads = 1

ort_session_24 = ort.InferenceSession(
    MODEL_24_PATH,
    sess_options=options,
    providers=["CPUExecutionProvider"]
)

ort_session_6 = ort.InferenceSession(
    MODEL_6_PATH,
    sess_options=options,
    providers=["CPUExecutionProvider"]
)

# Load input data
input_upper = np.load(os.path.join(INPUT_DIR, "input_upper.npy")).astype(np.float32)
input_surface = np.load(os.path.join(INPUT_DIR, "input_surface.npy")).astype(np.float32)

# Run inference
input_24, input_surface_24 = input_upper, input_surface

for i in range(28):
    if (i + 1) % 4 == 0:
        output, output_surface = ort_session_24.run(
            None,
            {"input": input_24, "input_surface": input_surface_24}
        )
        input_24, input_surface_24 = output, output_surface
    else:
        output, output_surface = ort_session_6.run(
            None,
            {"input": input_upper, "input_surface": input_surface}
        )

    input_upper, input_surface = output, output_surface
