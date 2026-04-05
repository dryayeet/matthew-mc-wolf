import numpy as np

def _get_interpreter_class():
    try:
        from tflite_runtime.interpreter import Interpreter
        return Interpreter
    except ImportError:
        pass
    try:
        from tensorflow.lite.python.lite import Interpreter
        return Interpreter
    except ImportError:
        pass
    raise ImportError(
        'Neither tflite-runtime nor tensorflow is installed. '
        'Install one: pip install tflite-runtime  OR  pip install tensorflow'
    )

def load_model(path='models/agent.tflite'):
    Interpreter = _get_interpreter_class()
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
