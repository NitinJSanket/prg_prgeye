#!/usr/bin/env python

import numpy
import tensorflow as tf

def export_tflite_from_session(session, input_nodes, output_nodes, tflite_filename):
    print("Converting to tflite...")
    converter = tf.lite.TFLiteConverter.from_session(session, input_nodes, output_nodes)
    tflite_model = converter.convert()
    with open(tflite_filename, "wb") as f:
        f.write(tflite_model)
    print("Converted %s." % tflite_filename)

def test_tflite_model(tflite_filename, examples):
    print("Loading TFLite interpreter for %s..." % tflite_filename)
    interpreter = tf.lite.Interpreter(model_path=tflite_filename)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    print("input details: %s" % input_details)
    print("output details: %s" % output_details)

    for i, input_tensor in enumerate(input_details):
        interpreter.set_tensor(input_tensor['index'], examples[i])
    interpreter.invoke()
    model_output = []
    for i, output_tensor in enumerate(output_details):
        model_output.append(interpreter.get_tensor(output_tensor['index']))
    return model_output

def main():
    tflite_filename = "model.tflite"
    shape_a = (2, 3, 4)
    shape_b = (2, 4, 5)

    a = tf.placeholder(dtype=tf.float32, shape=shape_a, name="A")
    b = tf.placeholder(dtype=tf.float32, shape=shape_b, name="B")
    c = tf.matmul(a, b, name="output")

    numpy.random.seed(1234)
    a_ = numpy.random.rand(*shape_a).astype(numpy.float32)
    b_ = numpy.random.rand(*shape_b).astype(numpy.float32)
    with tf.Session() as session:
        session_output = session.run(c, feed_dict={a: a_, b: b_})
        export_tflite_from_session(session, [a, b], [c], tflite_filename)

    tflite_output = test_tflite_model(tflite_filename, [a_, b_])
    tflite_output = tflite_output[0]

    print("Input example:")
    print(a_)
    print(a_.shape)
    print(b_)
    print(b_.shape)
    print("Session output:")
    print(session_output)
    print(session_output.shape)
    print("TFLite output:")
    print(tflite_output)
    print(tflite_output.shape)
    print(numpy.allclose(session_output, tflite_output))

if __name__ == '__main__':
    main()