import tensorflow as tf
from model import RDN

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_integer("epoch", 100, "Number of epoch")
flags.DEFINE_integer("image_size", 32, "The size of image input")
flags.DEFINE_integer("c_dim", 3, "The size of channel")
flags.DEFINE_boolean("is_train", True, "if the train")
flags.DEFINE_integer("scale", 3, "the size of scale factor for preprocessing input image")
flags.DEFINE_integer("stride", 16, "the size of stride")
flags.DEFINE_string("checkpoint_dir", "checkpoint", "Name of checkpoint directory")
flags.DEFINE_float("learning_rate", 1e-5 , "The learning rate")
flags.DEFINE_integer("batch_size", 64, "the size of batch")
flags.DEFINE_string("result_dir", "result", "Name of result directory")
flags.DEFINE_string("test_img", "", "test_img")
flags.DEFINE_integer("D", 5, "D")
flags.DEFINE_integer("C", 3, "C")
flags.DEFINE_integer("G", 64, "G")
flags.DEFINE_integer("G0", 64, "G0")
flags.DEFINE_integer("kernel_size", 3, "the size of kernel")



def main(_):
    rdn = RDN(tf.Session(),
              image_size = FLAGS.image_size,
              is_train = FLAGS.is_train,
              scale = FLAGS.scale,
              c_dim = FLAGS.c_dim,
              batch_size = FLAGS.batch_size,
              test_img = FLAGS.test_img,
              D = FLAGS.D,
              C = FLAGS.C,
              G = FLAGS.G,
              G0 = FLAGS.G0,
              kernel_size = FLAGS.kernel_size
              )

    if rdn.is_train:
        rdn.train(FLAGS)
    else:
        rdn.test(FLAGS)

if __name__=='__main__':
    tf.app.run()
