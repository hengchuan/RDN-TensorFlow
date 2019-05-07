import tensorflow as tf
from model import RDN

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_boolean("is_train", True, "if the train")
flags.DEFINE_boolean("matlab_bicubic", False, "using bicubic interpolation in matlab")
flags.DEFINE_integer("image_size", 32, "the size of image input")
flags.DEFINE_integer("c_dim", 3, "the size of channel")
flags.DEFINE_integer("scale", 3, "the size of scale factor for preprocessing input image")
flags.DEFINE_integer("stride", 16, "the size of stride")
flags.DEFINE_integer("epoch", 50, "number of epoch")
flags.DEFINE_integer("batch_size", 16, "the size of batch")
flags.DEFINE_float("learning_rate", 1e-4 , "the learning rate")
flags.DEFINE_float("lr_decay_steps", 10 , "steps of learning rate decay")
flags.DEFINE_float("lr_decay_rate", 0.5 , "rate of learning rate decay")
flags.DEFINE_boolean("is_eval", True, "if the evaluation")
flags.DEFINE_string("test_img", "", "test_img")
flags.DEFINE_string("checkpoint_dir", "checkpoint", "name of the checkpoint directory")
flags.DEFINE_string("result_dir", "result", "name of the result directory")
flags.DEFINE_string("train_set", "DIV2K_train_HR", "name of the train set")
flags.DEFINE_string("test_set", "Set5", "name of the test set")
flags.DEFINE_integer("D", 16, "D")
flags.DEFINE_integer("C", 8, "C")
flags.DEFINE_integer("G", 64, "G")
flags.DEFINE_integer("G0", 64, "G0")
flags.DEFINE_integer("kernel_size", 3, "the size of kernel")



def main(_):
    rdn = RDN(tf.Session(),
              is_train = FLAGS.is_train,
              is_eval = FLAGS.is_eval,
              image_size = FLAGS.image_size,
              c_dim = FLAGS.c_dim,
              scale = FLAGS.scale,
              batch_size = FLAGS.batch_size,
              D = FLAGS.D,
              C = FLAGS.C,
              G = FLAGS.G,
              G0 = FLAGS.G0,
              kernel_size = FLAGS.kernel_size
              )

    if rdn.is_train:
        rdn.train(FLAGS)
    else:
        if rdn.is_eval:
            rdn.eval(FLAGS)
        else:
            rdn.test(FLAGS)

if __name__=='__main__':
    tf.app.run()
