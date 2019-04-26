# Project 5: Neural Style Transfer
# Due May 2nd

from Evaluator import *
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

    _, h, w, d = Fp.get_shape().as_list()

    # Compute sum of residual squares (sse)
    sse = K.tf.reduce_sum((Fx - Fp)**2)

    # Note: to access underlying value: w.value
    M = w * h
    N = d

    # Compute scaling factor
    scale = 1.0 / (2 * (M**0.5) * (N**0.5))

    loss = scale * sse

    return loss


def gram_matrix(f):

    # Accepts a (height,width,depth)-sized feature map,
    # reshapes to (M,N), then computes the inner product

    _, h, w, d = f.get_shape().as_list()

    M = h * w
    N = d

    f = K.tf.reshape(f, shape=(M, N))

    return K.tf.tensordot(K.tf.transpose(f), f, 1)


def style_layer_loss(Fa, Fx):

    _, h, w, d = Fa.get_shape().as_list()

    # Calculate gram matrix of respective feature maps
    G_Fa = gram_matrix(Fa)
    G_Fx = gram_matrix(Fx)

    # Compute sse between gram matrices
    sse = K.tf.reduce_sum((G_Fa - G_Fx)**2)

    # Compute scaling factor
    M = w * h
    N = d
    scale = 1 / (4 * M**2 * N**2)

    loss = scale * sse

    return loss


def create_model(input_img, output_layers):

    # Instantiate full VGG model w/ input img
    base_model = vgg.VGG19(input_tensor=kl.Input(tensor=K.tf.Variable(input_img)))
    return km.Model(inputs=base_model.inputs, outputs=[base_model.get_layer(n).output for n in output_layers])


def pixel_means(img, add=False):

    if add:

        img[:, :, 0] += 103.939
        img[:, :, 1] += 116.779
        img[:, :, 2] += 123.68

    else:

        img[:, :, 0] -= 103.939
        img[:, :, 1] -= 116.779
        img[:, :, 2] -= 123.68

    return img


on_gpu_server = False
if on_gpu_server is True:
    sys.path.append("./libs/GPUtil/GPUtil")
    import GPUtil

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    gpus = GPUtil.getAvailable(order="first", limit=1, maxLoad=.2, maxMemory=.2)
    if len(gpus) > 0:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpus[0])
    else:
        print("No free GPU")
        sys.exit(1)

# Get image paths
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

# Load style image
style_img = kp.image.load_img(style_path, target_size=(img_rows, img_cols))
style_img = kp.image.img_to_array(style_img)

# Subtract mean pixel value of
# dataset used to train vgg19
# from both content and style image
content_img = np.expand_dims(pixel_means(content_img), axis=0)
style_img = np.expand_dims(pixel_means(style_img), axis=0)

# Define the layer outputs that we are interested in
content_layers = ['block4_conv2']

# Create content model
content_model = create_model(content_img, content_layers)

# Create style model
style_layers = ['block1_relu1', 'block2_relu1', 'block3_relu1', 'block4_relu1', 'block5_relu1']
style_model = create_model(style_img, style_layers)

# Instantiate blend model
# Note that the blend model input is same shape/size as content image
blend_base_model = vgg.VGG19(input_tensor=kl.Input(shape=content_img.shape[1:]))

# blend_outputs = content_outputs + style_outputs
blend_outputs = [blend_base_model.get_layer(n).output for n in content_layers] + [blend_base_model.get_layer(n).output for n in style_layers]

blend_model = km.Model(inputs=blend_base_model.inputs, outputs=blend_outputs)

# Separate the model outputs into those intended for comparison with the content layer and the style layer
blend_content_outputs = [blend_model.outputs[0]]
blend_style_outputs = blend_model.outputs[1:]

content_loss = content_layer_loss(content_model.output, blend_content_outputs[0])

content_loss_evaluator = K.function([blend_model.input], [content_loss])

# For a correctly implemented gram_matrix, the following code will produce 113934860.0
fmap = content_model.output

gram_matrix_evaluator = K.function([content_model.input], [gram_matrix(fmap)])

style_loss = 0
for i in range(5):
    style_loss += 0.2 * style_layer_loss(style_model.output[i], blend_style_outputs[i])

style_loss_evaluator = K.function([blend_model.input], [style_loss])

tv_loss = K.tf.image.total_variation(blend_model.input)

# Note: these parameters are arbitrarily chosen
alpha = 5.0
beta = 1e4
gamma = 1e-3

# Calculate total loss as a paramterized lc of content loss, style loss, and total variation loss
total_loss = alpha * content_loss + beta * style_loss + gamma * tv_loss

# Create total loss evaluator
total_loss_evaluator = K.function([blend_model.input], [total_loss])

# Create loss and gradient evaluator.
# Note that tensorflow performs automatic symbolic
# differentiation on the given inputs
grads = K.gradients(total_loss, blend_model.input)[0]
loss_and_grad_evaluator = K.function([blend_model.input], [total_loss, grads])

# Generate random data and perform optimization
input_img = np.random.randn(1, img_rows, img_cols, 3)
my_evaluator = Evaluator(loss_and_grad_evaluator)
blend_img = my_evaluator.optimize(input_img, img_rows, img_cols)

# Once optimization is complete,
# re-add band means we subtracted earlier,
# cast to integer, clip values greater than 255
blend_img = pixel_means(blend_img).astype(np.int32)

# Display and save image.
plt.imshow(blend_img)
plt.show()
