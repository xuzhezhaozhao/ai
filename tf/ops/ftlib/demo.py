
import tensorflow as tf

test_line = [
    '03959e48258956ah 6635a24a17e289ae 2615a3b2c54096aj 0005a21359a912ae']
fasttext_model = tf.load_op_library('fasttext_example_generate_ops.so')
(records, labels) = fasttext_model.fasttext_example_generate(
    train_data_path='train_data.in',
    input=test_line)


sess = tf.Session()
records = sess.run(records)
labels = sess.run(labels)

print("records = \n{}\nlabels = \n{}\n".format(records, labels))
