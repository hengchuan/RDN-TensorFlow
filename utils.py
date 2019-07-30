import PIL
import cv2
import numpy as np
import h5py
import math
import glob
import os 

def PIL_resize(image, ratio, mode):
  PIL_image = PIL.Image.fromarray(image.astype(dtype=np.uint8))
  PIL_image_resize = PIL_image.resize((int(PIL_image.size[0] * ratio), int(PIL_image.size[1] * ratio)), mode)
  image_resize = (np.array(PIL_image_resize)).astype(dtype=np.uint8)
  return image_resize

def rgb2ycbcr(img):
    y = 16 + (65.481 * img[:, :, 0]) + (128.553 * img[:, :, 1]) + (24.966 * img[:, :, 2])
    return y / 255

def PSNR(target, ref, scale):
    target_data = np.array(target, dtype=np.float32)
    ref_data = np.array(ref, dtype=np.float32)

    target_y = rgb2ycbcr(target_data)
    ref_y = rgb2ycbcr(ref_data)
    diff = ref_y - target_y

    shave = scale
    diff = diff[shave:-shave, shave:-shave]

    mse = np.mean((diff / 255) ** 2)
    if mse == 0:
        return 100

    return -10 * math.log10(mse)

def imread(path):
    img = cv2.imread(path)
    return img

def imsave(image, path):
    cv2.imwrite(os.path.join(os.getcwd(),path),image)

def checkimage(image):
    cv2.imshow("test",image)
    cv2.waitKey(0)

def modcrop(img, scale =3):
    if len(img.shape) ==3:
        h, w, _ = img.shape
        h = int((h / scale)) * scale
        w = int((w / scale)) * scale
        img = img[0:h, 0:w, :]
    else:
        h, w = img.shape
        h = int((h / scale)) * scale
        w = int((w / scale)) * scale
        img = img[0:h, 0:w]
    return img

def preprocess(path, scale = 3, eng = None, mdouble = None):
    img = imread(path)
    label_ = modcrop(img, scale)
    if eng is None:
        # input_ = cv2.resize(label_, None, fx=1.0/scale, fy=1.0/scale, interpolation=cv2.INTER_CUBIC)
        input_ = PIL_resize(label_, 1.0/scale, PIL.Image.BICUBIC)
    else:
        input_ = np.asarray(eng.imresize(mdouble(label_.tolist()), 1.0/scale, 'bicubic'))

    input_ = input_[:, :, ::-1]
    label_ = label_[:, :, ::-1]

    return input_, label_

def make_data_hf(input_, label_, config, times):
    if not os.path.isdir(os.path.join(os.getcwd(),config.checkpoint_dir)):
        os.makedirs(os.path.join(os.getcwd(),config.checkpoint_dir))

    if config.is_train:
        savepath = os.path.join(os.path.join(os.getcwd(), config.checkpoint_dir), 'train_x%d.h5' % config.scale)
    else:
        savepath = os.path.join(os.path.join(os.getcwd(), config.checkpoint_dir), 'test.h5')

    if times == 0:
        if os.path.exists(savepath):
            print("\n%s have existed!\n" % (savepath))
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
    if config.matlab_bicubic:
        import matlab.engine
        eng = matlab.engine.start_matlab()
        mdouble = matlab.double
    else:
        eng = None
        mdouble = None

    times = 0
    for i in range(len(data)):
        input_, label_, = preprocess(data[i], config.scale, eng, mdouble)
        if len(input_.shape) == 3:
            h, w, c = input_.shape
        else:
            h, w = input_.shape

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

                x_i = int(x / config.scale)
                y_i = int(y / config.scale)
                sub_input = input_[x_i: x_i + config.image_size, y_i: y_i + config.image_size]
                sub_input = sub_input.reshape([config.image_size, config.image_size, config.c_dim])
                sub_input = sub_input / 255.0

                # checkimage(sub_input)
                # checkimage(sub_label)

                save_flag = make_data_hf(sub_input, sub_label, config, times)
                if not save_flag:
                    return
                times += 1

        print("image: [%2d], total: [%2d]"%(i, len(data)))

    if config.matlab_bicubic:
        eng.quit()

def prepare_data(config):
    if config.is_train:
        data_dir = os.path.join(os.path.join(os.getcwd(), "Train"), config.train_set)
        data = glob.glob(os.path.join(data_dir, "*.png"))
    else:
        if config.test_img != "":
            data = [os.path.join(os.getcwd(), config.test_img)]
        else:
            data_dir = os.path.join(os.path.join(os.getcwd(), "Test"), config.test_set)
            data = glob.glob(os.path.join(data_dir, "*.png"))
    return data

def input_setup(config):
    """
        Read image files and make their sub-images and saved them as a h5 file format
    """

    data = prepare_data(config)
    make_sub_data(data, config)
    return data

def augmentation(batch, random):
    if random[0] < 0.3:
        batch_flip = np.flip(batch, 1)
    elif random[0] > 0.7:
        batch_flip = np.flip(batch, 2)
    else:
        batch_flip = batch

    if random[1] < 0.5:
        batch_rot = np.rot90(batch_flip, 1, [1, 2])
    else:
        batch_rot = batch_flip

    return batch_rot

def get_data_dir(checkpoint_dir, is_train, scale):
    if is_train:
        return os.path.join(os.path.join(os.getcwd(), checkpoint_dir), 'train_x%d.h5' % scale)
    else:
        return os.path.join(os.path.join(os.getcwd(), checkpoint_dir), 'test.h5')

def get_data_num(path):
     with h5py.File(path, 'r') as hf:
        input_ = hf['input']
        return input_.shape[0]

def get_batch(path, data_num, batch_size):
    with h5py.File(path, 'r') as hf:
        input_ = hf['input']
        label_ = hf['label']

        random_batch = np.random.rand(batch_size) * (data_num - 1)
        batch_images = np.zeros([batch_size, input_[0].shape[0], input_[0].shape[1], input_[0].shape[2]])
        batch_labels = np.zeros([batch_size, label_[0].shape[0], label_[0].shape[1], label_[0].shape[2]])
        for i in range(batch_size):
            batch_images[i, :, :, :] = np.asarray(input_[int(random_batch[i])])
            batch_labels[i, :, :, :] = np.asarray(label_[int(random_batch[i])])

        random_aug = np.random.rand(2)
        batch_images = augmentation(batch_images, random_aug)
        batch_labels = augmentation(batch_labels, random_aug)
        return batch_images, batch_labels

def get_image(path, scale, matlab_bicubic):
    if matlab_bicubic:
        import matlab.engine
        eng = matlab.engine.start_matlab()
        mdouble = matlab.double
    else:
        eng = None
        mdouble = None
    
    image, label = preprocess(path, scale, eng, mdouble)
    image = image[np.newaxis, :]
    label = label[np.newaxis, :]

    if matlab_bicubic:
        eng.quit()

    return image, label
