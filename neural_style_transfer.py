# Project 5: Neural Style Transfer
# Due May 2nd

import keras.preprocessing as kp
import matplotlib.pyplot as plt
import numpy as np
import vgg
import keras.backend as K
import keras.layers as kl
import keras.models as km
import os
import sys

def content_layer_loss(Fp, Fx):
    # ! Change me
    _, h, w, d = Fp.get_shape()
    loss = 1

    return loss


def gram_matrix(f, M, N):
    # Accepts a (height,width,depth)-sized feature map,
    # reshapes to (M,N), then computes the inner product
    # !Change me
    return 1


def style_layer_loss(Fa, Fx):
    # ! Change me
    _, h, w, d = Fa.get_shape()
    loss = 0
    return loss

on_gpu_server = True
if (on_gpu_server == True):
    sys.path.append("./libs/GPUtil/GPUtil")
    import GPUtil
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    gpus = GPUtil.getAvailable(order="first",limit=1,maxLoad=.2,maxMemory=.2)
    if(len(gpus) > 0):
        os.environ["CUDA_VISIBLE_DEVICES"]=str(gpus[0])
    else:
        print("No free GPU")
        sys.exit()

content_path = './main_hall.jpg'
style_path = './starry_night.jpg'

# Load image to get geometry
temp_img = kp.image.load_img(content_path)
width, height = temp_img.size

# fix the number of rows, while adapting the aspect ratio
img_rows = 400
img_cols = int(width * img_rows / height)

# Load content image
content_img = kp.image.load_img(content_path, target_size=(img_rows, img_cols))
content_img = kp.image.img_to_array(content_img)
plt.figure()
plt.imshow(content_img.astype(int))

# Load style image
style_img = kp.image.load_img(style_path, target_size=(img_rows, img_cols))
style_img = kp.image.img_to_array(style_img)
plt.figure()
plt.imshow(style_img.astype(int))

plt.show()

content_img[:, :, 0] -= 103.939
content_img[:, :, 1] -= 116.779
content_img[:, :, 2] -= 123.68
content_img = np.expand_dims(content_img, axis=0)

style_img[:, :, 0] -= 103.939
style_img[:, :, 1] -= 116.779
style_img[:, :, 2] -= 123.68
style_img = np.expand_dims(style_img, axis=0)

# Note that we'll be working quite a bit with the TensorFlow objects that underlie Keras
content_model_input = kl.Input(tensor=K.tf.Variable(content_img))

content_base_model = vgg.VGG19(input_tensor=content_model_input)
evaluator = K.function([content_base_model.input], [content_base_model.output])
feature_maps = evaluator([content_img])
plt.imshow(feature_maps[0][0, :, :, 500])
plt.show()

# Define the layer outputs that we are interested in
content_layers = ['block4_conv2']

# Get the tensor outputs of those layers
content_outputs = [content_base_model.get_layer(n).output for n in content_layers]

# Instantiate a new model with those outputs as outputs
content_model = km.Model(inputs=content_base_model.inputs,
                         outputs=[content_base_model.get_layer(n).output for n in content_layers])

# This is not used any further, it's just for visualizing the features
evaluator = K.function([content_model.input], [content_model.output])
feature_maps = evaluator([content_img])
plt.imshow(feature_maps[0][0, :, :, 125])
plt.show()

# Please call this second network 'style_model'

# ! Change me
style_layers = []

style_base_model = vgg.VGG19(input_tensor=kl.Input(tensor=K.tf.Variable(style_img)))

# ! Change me
style_model = None

# What are we saying about the size of the input image here?
blended_model_input = kl.Input(shape=content_img.shape[1:])

# Please call this third network 'blend_model'

base_model = vgg.VGG19(input_tensor=blended_model_input)

# ! Change me
blend_model = None

# Separate the model outputs into those intended for comparison with the content layer and the style layer
blend_content_outputs = [blend_model.outputs[0]]
blend_style_outputs = blend_model.outputs[1:]

content_loss = content_layer_loss(content_model.output, blend_content_outputs[0])

# The correct output of this function is 195710720.0
np.random.seed(0)
input_img = np.random.randn(1, img_rows, img_cols, 3)
content_loss_evaluator = K.function([blend_model.input], [content_loss])
content_loss_evaluator([input_img])

# For a correctly implemented gram_matrix, the following code will produce 113934860.0
fmap = content_model.output
_, h, w, d = fmap.get_shape()
M = h * w
N = d
gram_matrix_evaluator = K.function([content_model.input], [gram_matrix(fmap, M, N)])
gram_matrix_evaluator([content_img])[0].mean()

style_loss_0 = style_layer_loss(style_model.output[0], blend_style_outputs[0])

# The correct output of this function is 220990.31
np.random.seed(0)
input_img = np.random.randn(1, img_rows, img_cols, 3)
content_loss_evaluator = K.function([blend_model.input], [style_loss_0])
content_loss_evaluator([input_img])

style_loss = 0
for i in range(5):
    style_loss += 0.2 * style_layer_loss(style_model.output[i], blend_style_outputs[i])

alpha = 5.0
beta = 1e4
total_loss = alpha * content_loss + beta * style_loss

# The correct output of this function is 1771575600000.0
np.random.seed(0)
input_img = np.random.randn(1, img_rows, img_cols, 3)
content_loss_evaluator = K.function([blend_model.input], [total_loss])
content_loss_evaluator([input_img])
