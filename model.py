import tensorflow as tf
import numpy as np
import time
import os

from utils import (
    input_setup,
    get_data_dir,
    get_data_num,
    get_batch,
    get_image,
    checkimage,
    imsave,
    imread,
    prepare_data,
    PSNR
)

class RDN(object):

    def __init__(self,
                 sess,
                 is_train,
                 is_eval,
                 image_size,
                 c_dim,
                 scale,
                 batch_size,
                 D,
                 C,
                 G,
                 G0,
                 kernel_size
                 ):

        self.sess = sess
        self.is_train = is_train
        self.is_eval = is_eval
        self.image_size = image_size
        self.c_dim = c_dim
        self.scale = scale
        self.batch_size = batch_size
        self.D = D
        self.C = C
        self.G = G
        self.G0 = G0
        self.kernel_size = kernel_size

    # def SFEParams(self):
    #     G = self.G
    #     G0 = self.G0
    #     ks = self.kernel_size
    #     weightsS = {
    #         'w_S_1': tf.Variable(tf.random_normal([ks, ks, self.c_dim, G0], stddev=np.sqrt(2.0/ks**2/3)), name='w_S_1'),
    #         'w_S_2': tf.Variable(tf.random_normal([ks, ks, G0, G], stddev=np.sqrt(2.0/ks**2/64)), name='w_S_2')
    #     }
    #     biasesS = {
    #         'b_S_1': tf.Variable(tf.zeros([G0], name='b_S_1')),
    #         'b_S_2': tf.Variable(tf.zeros([G], name='b_S_2'))
    #     }

    #     return weightsS, biasesS
    
    # def RDBParams(self):
    #     weightsR = {}
    #     biasesR = {}
    #     D = self.D
    #     C = self.C
    #     G = self.G
    #     G0 = self.G0
    #     ks = self.kernel_size

    #     for i in range(1, D+1):
    #         for j in range(1, C+1):
    #             weightsR.update({'w_R_%d_%d' % (i, j): tf.Variable(tf.random_normal([ks, ks, G * j, G], stddev=np.sqrt(2.0/ks**2/(G * j))), name='w_R_%d_%d' % (i, j))}) 
    #             biasesR.update({'b_R_%d_%d' % (i, j): tf.Variable(tf.zeros([G], name='b_R_%d_%d' % (i, j)))})
    #         weightsR.update({'w_R_%d_%d' % (i, C+1): tf.Variable(tf.random_normal([1, 1, G * (C+1), G], stddev=np.sqrt(2.0/1/(G * (C+1)))), name='w_R_%d_%d' % (i, C+1))})
    #         biasesR.update({'b_R_%d_%d' % (i, C+1): tf.Variable(tf.zeros([G], name='b_R_%d_%d' % (i, C+1)))})

    #     return weightsR, biasesR

    # def DFFParams(self):
    #     D = self.D
    #     C = self.C
    #     G = self.G
    #     G0 = self.G0
    #     ks = self.kernel_size
    #     weightsD = {
    #         'w_D_1': tf.Variable(tf.random_normal([1, 1, G * D, G0], stddev=np.sqrt(2.0/1/(G * D))), name='w_D_1'),
    #         'w_D_2': tf.Variable(tf.random_normal([ks, ks, G0, G0], stddev=np.sqrt(2.0/ks**2/G0)), name='w_D_2')
    #     }
    #     biasesD = {
    #         'b_D_1': tf.Variable(tf.zeros([G0], name='b_D_1')),
    #         'b_D_2': tf.Variable(tf.zeros([G0], name='b_D_2'))
    #     }

    #     return weightsD, biasesD

    # def UPNParams(self):
    #     G0 = self.G0
    #     weightsU = {
    #         'w_U_1': tf.Variable(tf.random_normal([5, 5, G0, 64], stddev=np.sqrt(2.0/25/G0)), name='w_U_1'),
    #         'w_U_2': tf.Variable(tf.random_normal([3, 3, 64, 32], stddev=np.sqrt(2.0/9/64)), name='w_U_2'),
    #         'w_U_3': tf.Variable(tf.random_normal([3, 3, 32, self.c_dim * self.scale * self.scale ], stddev=np.sqrt(2.0/9/32)), name='w_U_3')
    #     }
    #     biasesU = {
    #         'b_U_1': tf.Variable(tf.zeros([64], name='b_U_1')),
    #         'b_U_2': tf.Variable(tf.zeros([32], name='b_U_2')),
    #         'b_U_3': tf.Variable(tf.zeros([self.c_dim * self.scale * self.scale ], name='b_U_3'))
    #     }

    #     return weightsU, biasesU

    def SFEParams(self):
        G = self.G
        G0 = self.G0
        ks = self.kernel_size
        weightsS = {
            'w_S_1': tf.Variable(tf.random_normal([ks, ks, self.c_dim, G0], stddev=0.01), name='w_S_1'),
            'w_S_2': tf.Variable(tf.random_normal([ks, ks, G0, G], stddev=0.01), name='w_S_2')
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
                weightsR.update({'w_R_%d_%d' % (i, j): tf.Variable(tf.random_normal([ks, ks, G * j, G], stddev=0.01), name='w_R_%d_%d' % (i, j))}) 
                biasesR.update({'b_R_%d_%d' % (i, j): tf.Variable(tf.zeros([G], name='b_R_%d_%d' % (i, j)))})
            weightsR.update({'w_R_%d_%d' % (i, C+1): tf.Variable(tf.random_normal([1, 1, G * (C+1), G], stddev=0.01), name='w_R_%d_%d' % (i, C+1))})
            biasesR.update({'b_R_%d_%d' % (i, C+1): tf.Variable(tf.zeros([G], name='b_R_%d_%d' % (i, C+1)))})

        return weightsR, biasesR

    def DFFParams(self):
        D = self.D
        C = self.C
        G = self.G
        G0 = self.G0
        ks = self.kernel_size
        weightsD = {
            'w_D_1': tf.Variable(tf.random_normal([1, 1, G * D, G0], stddev=0.01), name='w_D_1'),
            'w_D_2': tf.Variable(tf.random_normal([ks, ks, G0, G0], stddev=0.01), name='w_D_2')
        }
        biasesD = {
            'b_D_1': tf.Variable(tf.zeros([G0], name='b_D_1')),
            'b_D_2': tf.Variable(tf.zeros([G0], name='b_D_2'))
        }

        return weightsD, biasesD

    def UPNParams(self):
        G0 = self.G0
        weightsU = {
            'w_U_1': tf.Variable(tf.random_normal([5, 5, G0, 64], stddev=0.01), name='w_U_1'),
            'w_U_2': tf.Variable(tf.random_normal([3, 3, 64, 32], stddev=0.01), name='w_U_2'),
            'w_U_3': tf.Variable(tf.random_normal([3, 3, 32, self.c_dim * self.scale * self.scale ], stddev=np.sqrt(2.0/9/32)), name='w_U_3')
        }
        biasesU = {
            'b_U_1': tf.Variable(tf.zeros([64], name='b_U_1')),
            'b_U_2': tf.Variable(tf.zeros([32], name='b_U_2')),
            'b_U_3': tf.Variable(tf.zeros([self.c_dim * self.scale * self.scale ], name='b_U_3'))
        }

        return weightsU, biasesU
    
    def build_model(self, images_shape, labels_shape):
        self.images = tf.placeholder(tf.float32, images_shape, name='images')
        self.labels = tf.placeholder(tf.float32, labels_shape, name='labels')

        self.weightsS, self.biasesS = self.SFEParams()
        self.weightsR, self.biasesR = self.RDBParams()
        self.weightsD, self.biasesD = self.DFFParams()
        self.weightsU, self.biasesU = self.UPNParams()
        self.weight_final = tf.Variable(tf.random_normal([self.kernel_size, self.kernel_size, self.c_dim, self.c_dim], stddev=np.sqrt(2.0/9/3)), name='w_f')
        self.bias_final = tf.Variable(tf.zeros([self.c_dim], name='b_f')),
        
        self.pred = self.model()
        self.loss = tf.reduce_mean(tf.square(self.labels - self.pred))
        self.summary = tf.summary.scalar('loss', self.loss)
        self.saver = tf.train.Saver()

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
        # FU = self.UPN(F_1)
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

        images_shape = [None, self.image_size, self.image_size, self.c_dim]
        labels_shape = [None, self.image_size * self.scale, self.image_size * self.scale, self.c_dim]
        self.build_model(images_shape, labels_shape)
        self.train_op = tf.train.AdamOptimizer(learning_rate=config.learning_rate).minimize(self.loss)
        tf.global_variables_initializer().run(session=self.sess) 
        # merged_summary_op = tf.summary.merge_all()
        # summary_writer = tf.summary.FileWriter(config.checkpoint_dir, self.sess.graph)

        counter = self.load(config.checkpoint_dir)
        time_ = time.time()
        print("\nNow Start Training...\n")
        for ep in range(config.epoch):
            # Run by batch images
            batch_idxs = data_num // config.batch_size
            for idx in range(0, batch_idxs):
                batch_images, batch_labels = get_batch(data_dir, data_num, config.batch_size)
                counter += 1

                _, err = self.sess.run([self.train_op, self.loss], feed_dict={self.images: batch_images, self.labels: batch_labels})

                if counter % 10 == 0:
                    print("Epoch: [%2d], batch: [%2d/%2d], step: [%2d], time: [%4.4f], loss: [%.8f]" % ((ep+1), idx, batch_idxs, counter, time.time()-time_, err))
                if counter % 100 == 0:
                    self.save(config.checkpoint_dir, counter)
                    # summary_str = self.sess.run(merged_summary_op)
                    # summary_writer.add_summary(summary_str, counter)
                
                if counter > 0 and counter == batch_idxs * config.epoch:
                    return

    def eval(self, config):
        print("\nPrepare Data...\n")
        paths = prepare_data(config)
        data_num = len(paths)

        avg_time = 0
        avg_pasn = 0
        print("\nNow Start Testing...\n")
        for idx in range(data_num):
            input_, label_ = get_image(paths[idx], config.scale, config.matlab_bicubic)

            images_shape = input_.shape
            labels_shape = label_.shape
            self.build_model(images_shape, labels_shape)
            tf.global_variables_initializer().run(session=self.sess) 

            self.load(config.checkpoint_dir)

            time_ = time.time()
            result = self.sess.run([self.pred], feed_dict={self.images: input_ / 255.0})
            avg_time += time.time() - time_

            # import matlab.engine
            # eng = matlab.engine.start_matlab()
            # time_ = time.time()
            # result = np.asarray(eng.imresize(matlab.double((input_[0, :] / 255.0).tolist()), config.scale, 'bicubic'))
            # avg_time += time.time() - time_

            self.sess.close()
            tf.reset_default_graph()
            self.sess = tf.Session()

            x = np.squeeze(result) * 255.0
            x = np.clip(x, 0, 255)
            psnr = PSNR(x, label_[0], config.scale)
            avg_pasn += psnr

            print "image: %d/%d, time: %.4f, psnr: %.4f" % (idx, data_num, time.time() - time_ , psnr)

            if not os.path.isdir(os.path.join(os.getcwd(),config.result_dir)):
                os.makedirs(os.path.join(os.getcwd(),config.result_dir))
            imsave(x[:, :, ::-1], config.result_dir+'/%d.png' % idx)

        print "Avg. Time:", avg_time / data_num
        print "Avg. PSNR:", avg_pasn / data_num

    def test(self, config):
        print("\nPrepare Data...\n")
        paths = prepare_data(config)
        data_num = len(paths)

        avg_time = 0
        print("\nNow Start Testing...\n")
        for idx in range(data_num):
            input_ = imread(paths[idx])
            input_ = input_[:, :, ::-1]
            input_ = input_[np.newaxis, :]

            images_shape = input_.shape
            labels_shape = input_.shape * np.asarray([1, self.scale, self.scale, 1])
            self.build_model(images_shape, labels_shape)
            tf.global_variables_initializer().run(session=self.sess) 

            self.load(config.checkpoint_dir)

            time_ = time.time()
            result = self.sess.run([self.pred], feed_dict={self.images: input_ / 255.0})
            avg_time += time.time() - time_

            self.sess.close()
            tf.reset_default_graph()
            self.sess = tf.Session()

            x = np.squeeze(result) * 255.0
            x = np.clip(x, 0, 255)
            x = x[:, :, ::-1]
            checkimage(np.uint8(x))

            if not os.path.isdir(os.path.join(os.getcwd(),config.result_dir)):
                os.makedirs(os.path.join(os.getcwd(),config.result_dir))
            imsave(x, config.result_dir+'/%d.png' % idx)

        print "Avg. Time:", avg_time / data_num

    def load(self, checkpoint_dir):
        print("\nReading Checkpoints.....\n")
        model_dir = "%s_%s_%s_%s_x%s" % ("rdn", self.D, self.C, self.G, self.scale)
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
 
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_path = str(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(os.getcwd(), ckpt_path))
            step = int(os.path.basename(ckpt_path).split('-')[1])
            print("\nCheckpoint Loading Success! %s\n" % ckpt_path)
        else:
            step = 0
            print("\nCheckpoint Loading Failed! \n")

        return step

    def save(self, checkpoint_dir, step):
        model_name = "RDN.model"
        model_dir = "%s_%s_%s_%s_x%s" % ("rdn", self.D, self.C, self.G, self.scale)
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

        if not os.path.exists(checkpoint_dir):
             os.makedirs(checkpoint_dir)

        self.saver.save(self.sess,
                        os.path.join(checkpoint_dir, model_name),
                        global_step=step)
