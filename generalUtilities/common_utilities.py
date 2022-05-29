# -*- coding: utf-8 -*-

"""
FileName:               common_utilities
Author Name:            Arun M Saranathan
Description:            This file includes implementation of common utility functions needed to save images in the
                        training of network, saving network states, etc..

                        Some of the functions in this file are extracted from various sources such as:
                        1) Kushagra Pandey's github implementation of wGANS (github.com/kpandey008/wasserstein-gans)
                        
Date Created:           05th November 2019
Last Modified:          06th November 2019
"""

import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
import os
import shutil
from PIL import Image

class common_utilities:

    def cleanup(self,path):
        if not os.path.isdir(path):
            raise TypeError('Path provided to cleanup can only be directory')

        'Used to cleanup some resources'
        if os.path.exists(path):
            shutil.rmtree(path)

    def sort_by_time(self, folder, reverse=False):
        def getmtime(name):
            path = os.path.join(folder, name)
            return os.path.getmtime(path)

        return sorted(os.listdir(folder), key=getmtime, reverse=reverse)

    def generateSamples(self, m, noise_dim, scale):
        return np.random.normal(scale=scale, size=(m, noise_dim))

    def create_gif_from_images(self, src, dst, cleanup_path=None):
        """
        This functin creates a using images in src and stores into the dst

        :param src: Source location
        :param dst: Destination Location
        :param cleanup_path: Path to be removed
        :return:
        """

        dst_dir = dst if os.path.isdir(dst) else os.path.split(dst)[0]
        if not os.path.exists(src):
            raise OSError('No such path or directory.')

        'If destination directory does not exist create it'
        if not os.path.exists(dst_dir):
            print('Create destination directory')
            os.makedirs(dst_dir)

        print('Creating gif from the images')
        'Create a variable to hold the set of images that make up the GIF'
        images = []
        nImages = 0
        'Get the list of images we are adding to the GIF'
        imageList = [os.path.join(src, image) for image in self.sort_by_time(src, reverse=False) if os.path.isfile(os.path.join(src, image))]
        for image in imageList:
            img = Image.open(image)
            images.append(img)
            nImages += 1

        images[0].save(dst, save_all=True, append_images=images[1:], optimize=False, loop=0, duration=nImages)

        if cleanup_path != None:
            for path in cleanup_path:
                self.cleanup(path)


    def save_images(self, sample_images, img_dim_x=None, img_dim_y=None, shape=None, tmp_path=None, show=False,
                    save=False, id=None, **kwargs):
        """
        This function cab be used to save/display a set of images

        :param sample_images: The images to be displayed
        :param img_dim_x: Number of rows in each image
        :param img_dim_y: Number of columns in each image
        :param shape: Overall shape of the image
        :param tmp_path: Where the data is to be saved
        :param show: Flag -should images be displayed
        :param save: Flag -should images be saved
        :param id:
        :return:
        """

        'The input should be a matrix that is 2D (flattened) or 4D (collection of RGB images or 3D graysacale images)'
        img_shape_len = len(sample_images.shape)
        if (img_shape_len !=2) and (img_shape_len !=4):
            raise SyntaxError('Images must be a collection of 2D or DD with a channel last ordering for 4-dim images')

        'If 2D assume grayscale'
        numChannels =1 if img_shape_len ==2 else sample_images.shape[-1]
        'Get the size of the image'
        image_size = int(np.sqrt(sample_images.shape[1])) if img_shape_len ==2 else sample_images.shape[1]

        'Size of row and column (if not square must be provided by the user) '
        dim_x = img_dim_x or image_size
        dim_y = img_dim_y or image_size

        'Number of images in the provided sample'
        num_images = sample_images.shape[0]

        'Create an appropriate grid to display the images'
        grid_size =  int(np.ceil(np.sqrt(num_images)))
        fig = plt.figure(figsize=(grid_size, grid_size), **kwargs)
        idx = 1

        for _ in range(grid_size):
            for _ in range(grid_size):

                'Display the images in the plot'
                fig.add_subplot(grid_size, grid_size, idx)
                temp_img = np.reshape(sample_images[idx -1], (dim_x, dim_y)) if numChannels ==1 else sample_images[idx-1]

                plt.imshow(temp_img, cmap='binary')
                plt.gca().set_xticks([])
                plt.gca().set_yticks([])

                idx += 1

        'Save the image file locally'
        if show:
            plt.show()
        elif save and not show:
            tmp = tmp_path or os.path.join(os.getcwd(),'tmp')
            if not os.path.exists(tmp):
                os.makedirs(tmp)

            plt.savefig(os.path.join(tmp,'{}.png'.format(id)))
            plt.close()

    def restore_checkpoint_status_tf(self, saver, sess, path):
        """
        This function can be used to restore a tensorflow model from a saved checkpoint

        :param saver:
        :param sess:
        :param path:
        :return:
        """

        'Check if the checkpoint exists for this experiment'
        dir_path = os.path.split(path)[0] if os.path.splitext(path)[1] else path
        if not tf.train.latest_checkpoint(dir_path):
            print('No checkpoint found. Starting training....')
            return False

        'Else resume training'
        print('Checkpoint found for this experiment. Restoring variables....')
        tf.compat.v1.reset_default_graph()
        saver.restore(sess, path)
        return True

    def save_model_state(self, saver, sess, path):
        """
        This function saves the model state.

        :param saver:
        :param sess:
        :param path:
        :return:
        """

        'Save the model state'
        if not os.path.exists(os.path.split(path)[0]):
            os.makedirs(os.path.split(path)[0])
        else:
            saver.save(sess, path)


    """def save_model_state(self, model, path, modelName):
        
        This function is used to save the model state (assumes a keras path)
        :param model: The model which we want to save
        :param path: The folder where we want to save the data
        :param modelType: The folder name in which the model name is saved
        :param modelName: The name under which the model is stored
        :return:
        
        if not os.path.exists(os.path.join(os.path.split(path)[0], modelName)):
            os.makedirs(os.path.join(os.path.split(path)[0], modelName))

        'If the architecture has not been saved - save it'
        modelArch = os.path.join(os.path.split(path)[0], modelName, 'modelArch.json')
        if not os.path.isfile(modelArch):
            model_json = model.to_json()
            with open(modelArch, "w") as json_file:
                json_file.write(model_json)

        'Save the weights'
        str = ('weights_' + time.strftime("%Y%m%d-%H%M%S") + '.h5')
        model.save_weights(os.path.join(os.path.split(path)[0], modelName, str))





"""