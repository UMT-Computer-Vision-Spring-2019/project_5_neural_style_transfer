import keras.preprocessing as kp
import matplotlib.pyplot as plt
import numpy as np

import vgg
import keras.backend as K
import keras.layers as kl
import keras.models as km
import tensorflow as tf

import scipy.optimize as sio

content_path = './main_hall.jpg'
style_path = './starry_night.jpg'

# Load image to get geometry
temp_img = kp.image.load_img(content_path)
width,height = temp_img.size

# fix the number of rows, while adapting the aspect ratio
img_rows = 400
img_cols = int(width * img_rows / height)

# Load content image
content_img = kp.image.load_img(content_path, target_size=(img_rows, img_cols))
content_img = kp.image.img_to_array(content_img)
# plt.figure()
# plt.imshow(content_img.astype(int))

# Load style image
style_img = kp.image.load_img(style_path, target_size=(img_rows, img_cols))
style_img = kp.image.img_to_array(style_img)
# plt.figure()
# plt.imshow(style_img.astype(int))

# plt.show()

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
evaluator = K.function([content_base_model.input],[content_base_model.output])
feature_maps = evaluator([content_img])
# plt.imshow(feature_maps[0][0,:,:,500])
# plt.show()

# Define the layer outputs that we are interested in
content_layers = ['block4_conv2']

# Get the tensor outputs of those layers
content_outputs = [content_base_model.get_layer(n).output for n in content_layers]

# Instantiate a new model with those outputs as outputs
content_model = km.Model(inputs=content_base_model.inputs,outputs=[content_base_model.get_layer(n).output for n in content_layers])

# This is not used any further, it's just for visualizing the features
evaluator = K.function([content_model.input],[content_model.output])
feature_maps = evaluator([content_img])
#plt.imshow(feature_maps[0][0,:,:,125])
#plt.show()

# Please call this second network 'style_model'

#! Change me
style_layers = ['block1_relu1', 'block2_relu1', 'block3_relu1', 'block4_relu1', 'block5_relu1']

style_base_model = vgg.VGG19(input_tensor=kl.Input(tensor=K.tf.Variable(style_img)))

#! Change me
style_model = km.Model(inputs=style_base_model.inputs,outputs=[style_base_model.get_layer(n).output for n in style_layers])

blended_model_input = kl.Input(shape=content_img.shape[1:])

# Please call this third network 'blend_model'

blend_layers = ['block4_conv2', 'block1_relu1', 'block2_relu1', 'block3_relu1', 'block4_relu1', 'block5_relu1']

blend_base_model = vgg.VGG19(input_tensor=blended_model_input)

#! Change me
blend_model = km.Model(inputs=blend_base_model.inputs,outputs=[blend_base_model.get_layer(n).output for n in blend_layers])

# Separate the model outputs into those intended for comparison with the content layer and the style layer
blend_content_outputs = [blend_model.outputs[0]]
blend_style_outputs = blend_model.outputs[1:]
def content_layer_loss(Fp, Fx):
    #! Change me
    _,h,w,d = Fp.get_shape().as_list()
    constant = 1/(2*(w*h)**.5*d**.5)
    suum = tf.reduce_sum(np.sum((Fx[0,:,:,l] - Fp[0,:,:,l])**2 for l in range(0, d)))
    loss = constant*suum
    return loss

content_loss = content_layer_loss(content_model.output,blend_content_outputs[0])

# The correct output of this function is 195710720.0
np.random.seed(0)
input_img = np.random.randn(1,img_rows,img_cols,3)
content_loss_evaluator = K.function([blend_model.input],[content_loss])
# print(content_loss_evaluator([input_img]))

def gram_matrix(f, M, N):
     # Accepts a (height,width,depth)-sized feature map,
     # reshapes to (M,N), then computes the inner product
     reshaped_f = kl.Reshape((M, N))(f)[0]
     reshaped_f_t = K.transpose(reshaped_f)
     gram_matrix = K.dot(reshaped_f_t, reshaped_f) 
     # !Change me
     return gram_matrix

# For a correctly implemented gram_matrix, the following code will produce 113934860.0
fmap = content_model.output
_,h,w,d = fmap.get_shape()
M = h*w
N = d
gram_matrix_evaluator = K.function([content_model.input],[gram_matrix(fmap,M,N)])
# print(gram_matrix_evaluator([content_img])[0].mean())

def style_layer_loss(Fa, Fx):
    #! Change me
    _, h, w, d = Fa.get_shape().as_list()
    M = h * w
    N = d
    constant = 1/(4*M**2*N**2)
    suum = tf.reduce_sum((gram_matrix(Fa, M, N) - gram_matrix(Fx, M, N))**2)
    loss = constant*suum  
    return loss

style_loss_0 = style_layer_loss(style_model.output[0],blend_style_outputs[0])

# The correct output of this function is 220990.31
np.random.seed(0)
input_img = np.random.randn(1,img_rows,img_cols,3)
content_loss_evaluator = K.function([blend_model.input],[style_loss_0])
# print(content_loss_evaluator([input_img]))

style_loss = 0
for i in range(5):
    style_loss += 0.2*style_layer_loss(style_model.output[i],blend_style_outputs[i])
    
# The correct output of this function is 177059700.0
np.random.seed(0)
input_img = np.random.randn(1,img_rows,img_cols,3)
style_loss_evaluator = K.function([blend_model.input],[style_loss])
# print (style_loss_evaluator([input_img]))

tv_loss = K.tf.image.total_variation(blend_model.input)

alpha = 5.0
beta = 2e3
gamma = 1e-3

total_loss = alpha*content_loss + beta*style_loss + gamma*tv_loss
    
# The correct output of this function is 1.7715756e+12
np.random.seed(0)
input_img = np.random.randn(1,img_rows,img_cols,3)
total_loss_evaluator = K.function([blend_model.input],[total_loss])
# print (total_loss_evaluator([input_img]))

grads = K.gradients(total_loss,blend_model.input)[0]

loss_and_grad_evaluator = K.function([blend_model.input],[total_loss,grads])

np.random.seed(0)
input_img = np.random.randn(1,img_rows,img_cols,3)
l0,g0 = loss_and_grad_evaluator([input_img])
# Correct value of l0 is 3.5509e11
# Correct value of first element in g0 is -7.28989e2

g = None
def internal_loss_and_grad_evaluator(x):
  loss_and_grad_evaluator = K.function([blend_model.input],[total_loss,grads])
  np.random.seed(0)
  input_img = np.random.randn(1,img_rows,img_cols,3)
  l,g = loss_and_grad_evaluator([input_img])
  return loss_and_grad_evaluator
def gradient(x):
  return g

sio.fmin_l_bfgs_b(internal_loss_and_grad_evaluator, [content_img], fprime=gradient)
