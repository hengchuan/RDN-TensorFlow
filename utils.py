import cv2
import numpy as np
import tensorflow as tf
import os 
import glob
import h5py

def psnr(target, ref, scale):
	# assume RGB image
    target_data = np.array(target, dtype=np.float64)
    ref_data = np.array(ref,dtype=np.float64)
    diff = ref_data - target_data
    diff = diff.flatten('C')
    rmse = math.sqrt(np.mean(diff ** 2.) )
    return 20*math.log10(255.0/rmse)

def imread(path):
    img = cv2.imread(path)
    return img

def imsave(image, path, config):
    if not os.path.isdir(os.path.join(os.getcwd(),config.result_dir)):
        os.makedirs(os.path.join(os.getcwd(),config.result_dir))

    cv2.imwrite(os.path.join(os.getcwd(),path),image * 255.)

def checkimage(image):
    cv2.imshow("test",image)
    cv2.waitKey(0)

def modcrop(img, scale =3):
    if len(img.shape) ==3:
        h, w, _ = img.shape
        h = (h / scale) * scale
        w = (w / scale) * scale
        img = img[0:h, 0:w, :]
    else:
        h, w = img.shape
        h = (h / scale) * scale
        w = (w / scale) * scale
        img = img[0:h, 0:w]
    return img


def preprocess(path ,scale = 3):
    img = imread(path)

    label_ = modcrop(img, scale)
    input_ = cv2.resize(label_,None,fx = 1.0/scale ,fy = 1.0/scale, interpolation = cv2.INTER_CUBIC)

    # kernel_size = (7, 7)
    # sigma = 3.0
    # input_ = cv2.GaussianBlur(input_, kernel_size, sigma);

    return input_, label_

def prepare_data(dataset="Train",Input_img=""):
    if dataset == "Train":
        data_dir = os.path.join(os.path.join(os.getcwd(), dataset), "DIV2K_train_HR") # Join the Train dir to current directory
        data = glob.glob(os.path.join(data_dir, "*.png"))
    else:
        if Input_img !="":
            data = [os.path.join(os.getcwd(),Input_img)]
        else:
            data_dir = os.path.join(os.path.join(os.getcwd(), dataset), "Set5")
            data = glob.glob(os.path.join(data_dir, "*.bmp"))
    return data

def load_data(is_train, test_img):
    if is_train:
        data = prepare_data(dataset="Train")
    else:
        if test_img != "":
            return prepare_data(dataset="Test",Input_img=test_img)
        data = prepare_data(dataset="Test")
    return data

def make_data_hf(input_, label_, config, times):
    if not os.path.isdir(os.path.join(os.getcwd(),config.checkpoint_dir)):
        os.makedirs(os.path.join(os.getcwd(),config.checkpoint_dir))

    if config.is_train:
        savepath = os.path.join(os.getcwd(), config.checkpoint_dir + '/train.h5')
    else:
        savepath = os.path.join(os.getcwd(), config.checkpoint_dir + '/test.h5')
    
    if times == 0:
        if os.path.exists(savepath):
            print "\n%s have existed!\n" % (savepath)
            return False
        else:
            hf = h5py.File(savepath, 'w')

            if config.is_train:
                input_h5 = hf.create_dataset("input", (1, config.image_size, config.image_size, config.c_dim), 
                                            maxshape=(None, config.image_size, config.image_size, config.c_dim), 
                                            chunks=(1, config.image_size, config.image_size, config.c_dim), dtype='float32')
                label_h5 = hf.create_dataset("label", (1, config.image_size*config.scale, config.image_size*config.scale, config.c_dim), 
                                            maxshape=(None, config.image_size*config.scale, config.image_size*config.scale, config.c_dim), 
                                            chunks=(1, config.image_size*config.scale, config.image_size*config.scale, config.c_dim),dtype='float32')
            else:
                input_h5 = hf.create_dataset("input", (1, input_.shape[0], input_.shape[1], input_.shape[2]), 
                                            maxshape=(None, input_.shape[0], input_.shape[1], input_.shape[2]), 
                                            chunks=(1, input_.shape[0], input_.shape[1], input_.shape[2]), dtype='float32')
                label_h5 = hf.create_dataset("label", (1, label_.shape[0], label_.shape[1], label_.shape[2]), 
                                            maxshape=(None, label_.shape[0], label_.shape[1], label_.shape[2]), 
                                            chunks=(1, label_.shape[0], label_.shape[1], label_.shape[2]),dtype='float32')
    else:
        hf = h5py.File(savepath, 'a')
        input_h5 = hf["input"]
        label_h5 = hf["label"]

    if config.is_train:
        input_h5.resize([times + 1, config.image_size, config.image_size, config.c_dim])
        input_h5[times : times+1] = input_
        label_h5.resize([times + 1, config.image_size*config.scale, config.image_size*config.scale, config.c_dim])
        label_h5[times : times+1] = label_
    else:
        input_h5.resize([times + 1, input_.shape[0], input_.shape[1], input_.shape[2]])
        input_h5[times : times+1] = input_
        label_h5.resize([times + 1, label_.shape[0], label_.shape[1], label_.shape[2]])
        label_h5[times : times+1] = label_

    hf.close()
    return True

def make_sub_data(data, config):
    times = 0
    for i in range(len(data)):
        input_, label_, = preprocess(data[i], config.scale)
        if len(input_.shape) == 3:
            h, w, c = input_.shape
        else:
            h, w = input_.shape
        
        if not config.is_train:
            input_ = imread(data[i])
            input_ = input_ / 255.0
            label_ = imread(data[i])
            label_ = label_ / 255.0
            make_data_hf(input_, label_, config, times)
            return data

        for x in range(0, h * config.scale - config.image_size * config.scale + 1, config.stride * config.scale):
            for y in range(0, w * config.scale - config.image_size * config.scale + 1, config.stride * config.scale):
                sub_label = label_[x: x + config.image_size * config.scale, y: y + config.image_size * config.scale]
                
                sub_label = sub_label.reshape([config.image_size * config.scale , config.image_size * config.scale, config.c_dim])

                t = cv2.cvtColor(sub_label, cv2.COLOR_BGR2YCR_CB)
                t = t[:, :, 0]
                gx = t[1:, 0:-1] - t[0:-1, 0:-1]
                gy = t[0:-1, 1:] - t[0:-1, 0:-1]
                Gxy = (gx**2 + gy**2)**0.5
                r_gxy = float((Gxy > 10).sum()) / ((config.image_size*config.scale)**2) * 100
                if r_gxy < 10:
                    continue             

                sub_label =  sub_label / 255.0

                x_i = x/config.scale
                y_i = y/config.scale
                sub_input = input_[x_i: x_i + config.image_size, y_i: y_i + config.image_size]
                sub_input = sub_input.reshape([config.image_size, config.image_size, config.c_dim])
                sub_input = sub_input / 255.0

                save_flag = make_data_hf(sub_input, sub_label, config, times)
                if not save_flag:
                    return data
                times += 1

        print("image: [%2d], total: [%2d]"%(i, len(data)))

    return data

def input_setup(config):
    """
        Read image files and make their sub-images and saved them as a h5 file format
    """

    data = load_data(config.is_train, config.test_img)
    make_sub_data(data, config)

def get_data_dir(config):
    if config.is_train:
        return os.path.join('./{}'.format(config.checkpoint_dir), "train.h5")
    else:
        return os.path.join('./{}'.format(config.checkpoint_dir), "test.h5")

def get_data_num(path):
     with h5py.File(path, 'r') as hf:
        input_ = hf['input']
        return input_.shape[0]

def get_batch(path, batch_idx, batch_size):
    with h5py.File(path, 'r') as hf:
        input_ = hf['input']
        label_ = hf['label']
        batch_images = input_[batch_idx * batch_size : (batch_idx + 1) * batch_size]
        batch_labels = label_[batch_idx * batch_size : (batch_idx + 1) * batch_size]
        return batch_images, batch_labels