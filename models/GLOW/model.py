import os

import numpy as np
import tensorflow as tf
from threading import Lock

lock = Lock()


class Glow_Model:
    def __init__(self, graph_dir, optimized=False):
        self.optimized = optimized
        if self.optimized:
            # Optimized model. Twice as fast as
            # 1. we freeze conditional network (label is always 0)
            # 2. we use fuseterd kernels
            import blocksparse
            self.graph_path = os.path.join(graph_dir, 'graph_optimized.pb')
            self.inputs = {
                'dec_eps_0': 'dec_eps_0',
                'dec_eps_1': 'dec_eps_1',
                'dec_eps_2': 'dec_eps_2',
                'dec_eps_3': 'dec_eps_3',
                'dec_eps_4': 'dec_eps_4',
                'dec_eps_5': 'dec_eps_5',
                'enc_x': 'input/enc_x',
            }
            self.outputs = {
                'self.dec_x': 'model_3/Cast_1',
                'enc_eps_0': 'model_2/pool0/truediv_1',
                'enc_eps_1': 'model_2/pool1/truediv_1',
                'enc_eps_2': 'model_2/pool2/truediv_1',
                'enc_eps_3': 'model_2/pool3/truediv_1',
                'enc_eps_4': 'model_2/pool4/truediv_1',
                'enc_eps_5': 'model_2/truediv_4'
            }

            def feed(feed_dict, bs):
                return feed_dict

            self.update_feed = feed
        else:
            self.graph_path = os.path.join(graph_dir, 'graph_unoptimized.pb')
            self.inputs = {
                'dec_eps_0': 'Placeholder',
                'dec_eps_1': 'Placeholder_1',
                'dec_eps_2': 'Placeholder_2',
                'dec_eps_3': 'Placeholder_3',
                'dec_eps_4': 'Placeholder_4',
                'dec_eps_5': 'Placeholder_5',
                'enc_x': 'input/image',
                'enc_x_d': 'input/downsampled_image',
                'enc_y': 'input/label'
            }
            self.outputs = {
                'dec_x': 'model_1/Cast_1',
                'enc_eps_0': 'model/pool0/truediv_1',
                'enc_eps_1': 'model/pool1/truediv_1',
                'enc_eps_2': 'model/pool2/truediv_1',
                'enc_eps_3': 'model/pool3/truediv_1',
                'enc_eps_4': 'model/pool4/truediv_1',
                'enc_eps_5': 'model/truediv_4'
            }

            def feed(feed_dict, bs):
                x_d = 128 * np.ones([bs, 128, 128, 3], dtype=np.uint8)
                y = np.zeros([bs], dtype=np.int32)
                feed_dict[enc_x_d] = x_d
                feed_dict[enc_y] = y
                return feed_dict

            self.update_feed = feed

        with tf.gfile.GFile(self.graph_path, 'rb') as f:
            graph_def_optimized = tf.GraphDef()
            graph_def_optimized.ParseFromString(f.read())

        self.sess = self.__tensorflow_session()
        tf.import_graph_def(graph_def_optimized)

        self.n_eps = 6

        # Encoder
        self.enc_x = self.get(self.inputs['enc_x'])
        self.enc_eps = [self.get(self.outputs['enc_eps_' + str(i)]) for i in range(self.n_eps)]
        if not self.optimized:
            enc_x_d = self.get(self.inputs['enc_x_d'])
            enc_y = self.get(self.inputs['enc_y'])

        # Decoder
        self.dec_x = self.get(self.outputs['dec_x'])
        self.dec_eps = [self.get(self.inputs['dec_eps_' + str(i)]) for i in range(self.n_eps)]
        self.eps_shapes = [(128, 128, 6), (64, 64, 12), (32, 32, 24),
                           (16, 16, 48), (8, 8, 96), (4, 4, 384)]
        self.eps_sizes = [np.prod(e) for e in self.eps_shapes]
        self.eps_size = 256 * 256 * 3 # Size of Latent Space
        self.dim_shift = self.eps_size
        self.dim_z = self.eps_size



    def get(self, name):
        return tf.get_default_graph().get_tensor_by_name('import/' + name + ':0')

    def __tensorflow_session(self):
        # Init session and params
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        # Pin GPU to local rank (one GPU per process)
        config.gpu_options.visible_device_list = str(0)
        sess = tf.Session(config=config)
        return sess

    # z_manipulate = np.load('z_manipulate.npy')
    #
    # _TAGS = "5_o_Clock_Shadow Arched_Eyebrows Attractive Bags_Under_Eyes Bald Bangs Big_Lips Big_Nose Black_Hair Blond_Hair Blurry Brown_Hair Bushy_Eyebrows Chubby Double_Chin Eyeglasses Goatee Gray_Hair Heavy_Makeup High_Cheekbones Male Mouth_Slightly_Open Mustache Narrow_Eyes No_Beard Oval_Face Pale_Skin Pointy_Nose Receding_Hairline Rosy_Cheeks Sideburns Smiling Straight_Hair Wavy_Hair Wearing_Earrings Wearing_Hat Wearing_Lipstick Wearing_Necklace Wearing_Necktie Young"
    # _TAGS = _TAGS.split()
    #
    # flip_tags = ['No_Beard', 'Young']
    # for tag in flip_tags:
    #     i = _TAGS.index(tag)
    #     z_manipulate[i] = -z_manipulate[i]
    #
    # scale_tags = ['Narrow_Eyes']
    # for tag in scale_tags:
    #     i = _TAGS.index(tag)
    #     z_manipulate[i] = 1.2*z_manipulate[i]
    #
    # z_sq_norms = np.sum(z_manipulate**2, axis=-1, keepdims=True)
    # z_proj = (z_manipulate / z_sq_norms).T

    def run(self, fetches, feed_dict):
        with lock:
            # Locked tensorflow so average server response time to user is lower
            result = self.sess.run(fetches, feed_dict)
        return result

    def __flatten_eps(self, eps):
        # [BS, eps_size]
        return np.concatenate([np.reshape(e, (e.shape[0], -1)) for e in eps], axis=-1)

    def __unflatten_eps(self, feps):
        index = 0
        eps = []
        bs = feps.shape[0]  # feps.size // eps_size
        for shape in self.eps_shapes:
            # eps.append(np.reshape(feps[:, index: index + np.prod(shape)], (bs, *shape)))
            eps.append(np.reshape(feps[:, index: index + np.prod(shape)].cpu(), (bs, *shape)))
            index += np.prod(shape)
        return eps

    def encode(self, img):
        if len(img.shape) == 3:
            img = np.expand_dims(img, 0)
        bs = img.shape[0]
        assert img.shape[1:] == (256, 256, 3)
        feed_dict = {self.enc_x: img}

        self.update_feed(feed_dict, bs)  # For unoptimized model
        return self.__flatten_eps(self.run(self.enc_eps, feed_dict))

    def decode(self, feps):
        if len(feps.shape) == 1:
            feps = np.expand_dims(feps, 0)
        bs = feps.shape[0]
        # assert len(eps) == n_eps
        # for i in range(n_eps):
        #     shape = (BATCH_SIZE, 128 // (2 ** i), 128 // (2 ** i), 6 * (2 ** i) * (2 ** (i == (n_eps - 1))))
        #     assert eps[i].shape == shape
        eps = self.__unflatten_eps(feps)

        feed_dict = {}
        for i in range(self.n_eps):
            feed_dict[self.dec_eps[i]] = eps[i]

        self.update_feed(feed_dict, bs)  # For unoptimized model
        return self.run(self.dec_x, feed_dict)

    # def project(z):
    #     return np.dot(z, z_proj)

    def __manipulate(self, z, dz, alpha):
        z = z + alpha * dz
        return self.decode(z), z

    def __manipulate_range(self, z, dz, points, scale):
        z_range = np.concatenate(
            [z + scale * (pt / (points - 1)) * dz for pt in range(0, points)], axis=0)
        return self.decode(z_range), z_range

    # alpha from [0,1]
    def mix(self, z1, z2, alpha):
        dz = (z2 - z1)
        return self.__manipulate(z1, dz, alpha)

    def mix_range(self, z1, z2, points=5):
        dz = (z2 - z1)
        return self.__manipulate_range(z1, dz, points, 1.)

    # alpha goes from [-1,1]
    def manipulate(self, z, dz, alpha):
        return self.__manipulate(z, dz, alpha)

    def manipulate_all(self, z, dz, alphas):
        for i in range(len(dz)):
            dz += alphas[i] * dz[i]
        return self.__manipulate(z, dz, 1.0)

    def manipulate_range(self, z, dz, points=5, scale=1):
        return self.__manipulate_range(z - dz, 2 * dz, points, scale)

    def random(self, bs=1, eps_std=0.7):
        feps = np.random.normal(scale=eps_std, size=[bs, self.eps_size])
        return self.decode(feps), feps
