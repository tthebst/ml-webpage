import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.saved_model import simple_save

export_dir = "/tmp/tfserving/biggan/00000001"
with tf.Session(graph=tf.Graph()) as sess:
    module = hub.Module('https://tfhub.dev/deepmind/biggan-512/2')

    inputs = {k: tf.placeholder(v.dtype, v.get_shape().as_list(), k)
              for k, v in module.get_input_info_dict().items()}

    sess.run(tf.global_variables_initializer())

    embeddings = module(gen_class)

    simple_save(sess,
                export_dir,
                inputs={'class': gen_class},
                outputs={'image': image},
                legacy_init_op=tf.tables_initializer())
