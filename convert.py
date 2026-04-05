# Requires full tensorflow -- run on Colab/remote T4, NOT locally.
import tensorflow as tf
import os, sys

def convert(saved_model_dir, out='models/agent.tflite'):
    os.makedirs(os.path.dirname(out), exist_ok=True)
    c = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
    c.optimizations = [tf.lite.Optimize.DEFAULT]
    c.target_spec.supported_types = [tf.float16]
    buf = c.convert()
    with open(out, 'wb') as f:
        f.write(buf)
    print(f'Saved {len(buf)} bytes to {out}')

if __name__ == '__main__':
    d = sys.argv[1] if len(sys.argv) > 1 else 'models/saved_model'
    convert(d)
