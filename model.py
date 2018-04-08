import tensorflow as tf
import numpy as np
import time
import os
import h5py

from utils import (
    input_setup,
    get_data_dir,
    get_data_num,
    get_batch,
    checkimage,
    imsave,
    imread,
    load_data,
    psnr
)

class RDN(object):

    def __init__(self,
                 sess,
                 image_size,
                 is_train,
                 scale,
                 batch_size,
                 c_dim,
                 test_img,
                 D,
                 C,
                 G,
                 G0,
                 kernel_size
                 ):

        self.sess = sess
        self.image_size = image_size
        self.is_train = is_train
        self.c_dim = c_dim
        self.scale = scale
        self.batch_size = batch_size
        self.test_img = test_img
        self.D = D
        self.C = C
        self.G = G
        self.G0 = G0
        self.kernel_size = kernel_size

        self.build_model()

    def SFEParams(self):
        G = self.G
        G0 = self.G0
        ks = self.kernel_size
        weightsS = {
            'w_S_1': tf.Variable(tf.random_normal([ks, ks, self.c_dim, G0], stddev=np.sqrt(2.0/ks**2/3)), name='w_S_1'),
            'w_S_2': tf.Variable(tf.random_normal([ks, ks, G0, G], stddev=np.sqrt(2.0/ks**2/64)), name='w_S_2')
        }
        biasesS = {
            'b_S_1': tf.Variable(tf.zeros([G0], name='b_S_1')),
            'b_S_2': tf.Variable(tf.zeros([G], name='b_S_2'))
        }

        return weightsS, biasesS
    
    def RDBParams(self):
        weightsR = {}
        biasesR = {}
        D = self.D
        C = self.C
        G = self.G
        G0 = self.G0
        ks = self.kernel_size

        for i in range(1, D+1):
            for j in range(1, C+1):
                weightsR.update({'w_R_%d_%d' % (i, j): tf.Variable(tf.random_normal([ks, ks, G * j, G], stddev=np.sqrt(2.0/ks**2/(G * j))), name='w_R_%d_%d' % (i, j))}) 
                biasesR.update({'b_R_%d_%d' % (i, j): tf.Variable(tf.zeros([G], name='b_R_%d_%d' % (i, j)))})
            weightsR.update({'w_R_%d_%d' % (i, C+1): tf.Variable(tf.random_normal([1, 1, G * (C+1), G], stddev=np.sqrt(2.0/1/(G * (C+1)))), name='w_R_%d_%d' % (i, C+1))})
            biasesR.update({'b_R_%d_%d' % (i, C+1): tf.Variable(tf.zeros([G], name='b_R_%d_%d' % (i, C+1)))})

        return weightsR, biasesR

    def DFFParams(self):
        D = self.D
        C = self.C
        G = self.G
        G0 = self.G0
        ks = self.kernel_size
        weightsD = {
            'w_D_1': tf.Variable(tf.random_normal([1, 1, G * D, G0], stddev=np.sqrt(2.0/1/(G * D))), name='w_D_1'),
            'w_D_2': tf.Variable(tf.random_normal([ks, ks, G0, G0], stddev=np.sqrt(2.0/ks**2/G0)), name='w_D_2')
        }
        biasesD = {
            'b_D_1': tf.Variable(tf.zeros([G0], name='b_D_1')),
            'b_D_2': tf.Variable(tf.zeros([G0], name='b_D_2'))
        }

        return weightsD, biasesD

    def UPNParams(self):
        G0 = self.G0
        weightsU = {
            'w_U_1': tf.Variable(tf.random_normal([5, 5, G0, 64], stddev=np.sqrt(2.0/25/G0)), name='w_U_1'),
            'w_U_2': tf.Variable(tf.random_normal([3, 3, 64, 32], stddev=np.sqrt(2.0/9/64)), name='w_U_2'),
            'w_U_3': tf.Variable(tf.random_normal([3, 3, 32, self.c_dim * self.scale * self.scale ], stddev=np.sqrt(2.0/9/32)), name='w_U_3')
        }
        biasesU = {
            'b_U_1': tf.Variable(tf.zeros([64], name='b_U_1')),
            'b_U_2': tf.Variable(tf.zeros([32], name='b_U_2')),
            'b_U_3': tf.Variable(tf.zeros([self.c_dim * self.scale * self.scale ], name='b_U_3'))
        }

        return weightsU, biasesU
    
    def build_model(self):
        if self.is_train:
            self.images = tf.placeholder(tf.float32, [None, self.image_size, self.image_size, self.c_dim], name='images')
            self.labels = tf.placeholder(tf.float32, [None, self.image_size * self.scale, self.image_size * self.scale, self.c_dim], name='labels')
        else:
            '''
                Because the test need to put image to model,
                so here we don't need do preprocess, so we set input as the same with preprocess output
            '''
            data = load_data(self.is_train, self.test_img)
            input_ = imread(data[0])
            self.h, self.w, c = input_.shape
            self.images = tf.placeholder(tf.float32, [None, self.h, self.w, self.c_dim], name='images')
            self.labels = tf.placeholder(tf.float32, [None, self.h * self.scale, self.w * self.scale, self.c_dim], name='labels')

        self.weightsS, self.biasesS = self.SFEParams()
        
        self.weightsR, self.biasesR = self.RDBParams()

        self.weightsD, self.biasesD = self.DFFParams()

        self.weightsU, self.biasesU = self.UPNParams()

        self.weight_final = tf.Variable(tf.random_normal([self.kernel_size, self.kernel_size, self.c_dim, self.c_dim], stddev=np.sqrt(2.0/9/3)), name='w_f')
        self.bias_final = tf.Variable(tf.zeros([self.c_dim], name='b_f')),
        
        self.pred = self.model()
        
        self.loss = tf.reduce_mean(tf.square(self.labels - self.pred))

        self.summery = tf.summary.scalar('loss', self.loss)

        self.saver = tf.train.Saver() # To save checkpoint

    def UPN(self, input_layer):
        x = tf.nn.conv2d(input_layer, self.weightsU['w_U_1'], strides=[1,1,1,1], padding='SAME') + self.biasesU['b_U_1']
        x = tf.nn.relu(x)
        x = tf.nn.conv2d(x, self.weightsU['w_U_2'], strides=[1,1,1,1], padding='SAME') + self.biasesU['b_U_2']
        x = tf.nn.relu(x)
        x = tf.nn.conv2d(x, self.weightsU['w_U_3'], strides=[1,1,1,1], padding='SAME') + self.biasesU['b_U_3']
       
        x = self.PS(x, self.scale)
       
        return x
        
    def RDBs(self, input_layer):
        rdb_concat = list()
        rdb_in = input_layer
        for i in range(1, self.D+1):
            x = rdb_in
            for j in range(1, self.C+1):
                tmp = tf.nn.conv2d(x, self.weightsR['w_R_%d_%d' %(i, j)], strides=[1,1,1,1], padding='SAME') + self.biasesR['b_R_%d_%d' % (i, j)]
                tmp = tf.nn.relu(tmp)
                x = tf.concat([x, tmp], axis=3)

            x = tf.nn.conv2d(x, self.weightsR['w_R_%d_%d' % (i, self.C+1)], strides=[1,1,1,1], padding='SAME') +  self.biasesR['b_R_%d_%d' % (i, self.C+1)]
            rdb_in = tf.add(x, rdb_in)
            rdb_concat.append(rdb_in)

        return tf.concat(rdb_concat, axis=3)
    
    def model(self):
        F_1 = tf.nn.conv2d(self.images, self.weightsS['w_S_1'], strides=[1,1,1,1], padding='SAME') + self.biasesS['b_S_1']
        F0 = tf.nn.conv2d(F_1, self.weightsS['w_S_2'], strides=[1,1,1,1], padding='SAME') + self.biasesS['b_S_2']

        FD = self.RDBs(F0)

        FGF1 = tf.nn.conv2d(FD, self.weightsD['w_D_1'], strides=[1,1,1,1], padding='SAME') + self.biasesD['b_D_1']
        FGF2 = tf.nn.conv2d(FGF1, self.weightsD['w_D_2'], strides=[1,1,1,1], padding='SAME') + self.biasesD['b_D_2']

        FDF = tf.add(FGF2, F_1)

        FU = self.UPN(FDF)
        IHR = tf.nn.conv2d(FU, self.weight_final, strides=[1,1,1,1], padding='SAME') + self.bias_final

        return IHR

    # NOTE: train with batch size 
    def _phase_shift(self, I, r):
        # Helper function with main phase shift operation
        bsize, a, b, c = I.get_shape().as_list()
        X = tf.reshape(I, (self.batch_size, a, b, r, r))
        X = tf.split(X, a, 1)  # a, [bsize, b, r, r]
        X = tf.concat([tf.squeeze(x) for x in X], 2)  # bsize, b, a*r, r
        X = tf.split(X, b, 1)  # b, [bsize, a*r, r]
        X = tf.concat([tf.squeeze(x) for x in X], 2)  # bsize, a*r, b*r
        return tf.reshape(X, (self.batch_size, a*r, b*r, 1))

    # NOTE: test without batchsize
    def _phase_shift_test(self, I ,r):
        bsize, a, b, c = I.get_shape().as_list()
        X = tf.reshape(I, (1, a, b, r, r))
        X = tf.split(X, a, 1)  # a, [bsize, b, r, r]
        X = tf.concat([tf.squeeze(x) for x in X], 1)  # bsize, b, a*r, r
        X = tf.split(X, b, 0)  # b, [bsize, a*r, r]
        X = tf.concat([tf.squeeze(x) for x in X], 1)  # bsize, a*r, b*r
        return tf.reshape(X, (1, a*r, b*r, 1))

    def PS(self, X, r):
        # Main OP that you can arbitrarily use in you tensorflow code
        Xc = tf.split(X, 3, 3)
        if self.is_train:
            X = tf.concat([self._phase_shift(x, r) for x in Xc], 3) # Do the concat RGB
        else:
            X = tf.concat([self._phase_shift_test(x, r) for x in Xc], 3) # Do the concat RGB
        return X

    def train(self, config):
        print("\nPrepare Data...\n")
        input_setup(config)
        data_dir = get_data_dir(config)
        data_num = get_data_num(data_dir)

        self.train_op = tf.train.AdamOptimizer(learning_rate=config.learning_rate).minimize(self.loss)
        tf.initialize_all_variables().run() 

        # merged_summary_op = tf.summary.merge_all()
        # summary_writer = tf.summary.FileWriter(config.checkpoint_dir, self.sess.graph)

        counter = 0
        time_ = time.time()

        self.load(config.checkpoint_dir)
        # Train
        if config.is_train:
            print("\nNow Start Training...\n")
            for ep in range(config.epoch):
                # Run by batch images
                batch_idxs = data_num // config.batch_size
                for idx in range(0, batch_idxs):
                    batch_images, batch_labels = get_batch(data_dir, idx, config.batch_size)
                    counter += 1
                    _, err = self.sess.run([self.train_op, self.loss], feed_dict={self.images: batch_images, self.labels: batch_labels})

                    if counter % 10 == 0:
                        print("Epoch: [%2d], step: [%2d], time: [%4.4f], loss: [%.8f]" % ((ep+1), counter, time.time()-time_, err))
                    if counter % 100 == 0:
                        self.save(config.checkpoint_dir, counter)
                        # summary_str = self.sess.run(merged_summary_op)
                        # summary_writer.add_summary(summary_str, counter)
        # Test
        else:
            print("\nNow Start Testing...\n")
            time_ = time.time()
            input_, label_ = get_batch(data_dir, 0, 1)
            result = self.sess.run([self.pred], feed_dict={self.images: input_[0].reshape(1, self.h, self.w, self.c_dim)})
            print "time:", (time.time() - time_)
            x = np.squeeze(result)
            checkimage(x)
            print "shape:", x.shape
            imsave(x, config.result_dir+'/result.png', config)

    def load(self, checkpoint_dir):
        print("\nReading Checkpoints.....\n")
        model_dir = "%s_%s_%s_%s_x%s" % ("rdn", self.D, self.C, self.G, self.scale)
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
 
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_path = str(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(os.getcwd(), ckpt_path))
            print("\nCheckpoint Loading Success! %s\n" % ckpt_path)
        else:
            print("\nCheckpoint Loading Failed! \n")

    def save(self, checkpoint_dir, step):
        model_name = "RDN.model"
        model_dir = "%s_%s_%s_%s_x%s" % ("rdn", self.D, self.C, self.G, self.scale)
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

        if not os.path.exists(checkpoint_dir):
             os.makedirs(checkpoint_dir)

        self.saver.save(self.sess,
                        os.path.join(checkpoint_dir, model_name),
                        global_step=step)
