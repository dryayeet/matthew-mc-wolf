import numpy as np
import onnxruntime as ort

def load_model(path='models/trading_model.onnx'):
    sess = ort.InferenceSession(path)
    return sess

def get_action(sess, prices, threshold=0.005):
    inp_name = sess.get_inputs()[0].name
    x = prices.reshape(1, -1, 1).astype(np.float32)
    pred = sess.run(None, {inp_name: x})[0][0, 0]
    last = prices[-1]
    diff = (pred - last) / (abs(last) + 1e-8)
    if diff > threshold:
        return 1  # Buy
    elif diff < -threshold:
        return 2  # Sell
    return 0  # Hold
