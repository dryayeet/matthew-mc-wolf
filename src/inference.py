import numpy as np

try:
    from tflite_runtime.interpreter import Interpreter
except ImportError:
    from tensorflow.lite.python.lite import Interpreter

def load_model(path='models/agent.tflite'):
    interp = Interpreter(model_path=path)
    interp.allocate_tensors()
    return interp

def get_action(interp, state):
    inp = interp.get_input_details()
    out = interp.get_output_details()
    interp.set_tensor(inp[0]['index'], state.reshape(1, -1).astype(np.float32))
    interp.invoke()
    q = interp.get_tensor(out[0]['index'])
    return int(np.argmax(q[0]))
