import keras.preprocessing as kp
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

content_path = './brock.jpg'
style_path = './anime.jpg'

# Load image to get geometry
temp_img = kp.image.load_img(content_path)
width,height = temp_img.size

# fix the number of rows, while adapting the aspect ratio
img_rows = 800
img_cols = int(width * img_rows / height)

# Load content image
content_img = kp.image.load_img(content_path, target_size=(img_rows, img_cols))
content_img = kp.image.img_to_array(content_img)
#plt.figure()
#plt.imshow(content_img.astype(int))

# Load style image
style_img = kp.image.load_img(style_path, target_size=(img_rows, img_cols))
style_img = kp.image.img_to_array(style_img)


content_img[:, :, 0] -= 103.939
content_img[:, :, 1] -= 116.779
content_img[:, :, 2] -= 123.68
content_img = np.expand_dims(content_img, axis=0)

style_img[:, :, 0] -= 103.939
style_img[:, :, 1] -= 116.779
style_img[:, :, 2] -= 123.68
style_img = np.expand_dims(style_img, axis=0)


# Next, let's instantiate a VGG19 model for the content image:

# In[6]:


import vgg
import keras.backend as K
import keras.layers as kl
import keras.models as km

# Note that we'll be working quite a bit with the TensorFlow objects that underlie Keras
content_model_input = kl.Input(tensor=K.tf.Variable(content_img))

content_base_model = vgg.VGG19(input_tensor=content_model_input)
evaluator = K.function([content_base_model.input],[content_base_model.output])
feature_maps = evaluator([content_img])


# Define the layer outputs that we are interested in
content_layers = ['block4_conv2']

# Get the tensor outputs of those layers
content_outputs = [content_base_model.get_layer(n).output for n in content_layers]

# Instantiate a new model with those outputs as outputs
content_model = km.Model(inputs=content_base_model.inputs,outputs=[content_base_model.get_layer(n).output for n in content_layers])


# In[8]:


# This is not used any further, it's just for visualizing the features
evaluator = K.function([content_model.input],[content_model.output])
feature_maps = evaluator([content_img])

style_layers = ['block1_relu1', 'block2_relu1', 'block3_relu1', 'block4_relu1', 'block5_relu1']

style_base_model = vgg.VGG19(input_tensor=kl.Input(tensor=K.tf.Variable(style_img)))

#! Change me
style_model = km.Model(inputs=style_base_model.inputs,outputs=[style_base_model.get_layer(n).output for n in style_layers])

blended_model_input = kl.Input(shape=content_img.shape[1:])



base_model = vgg.VGG19(input_tensor=blended_model_input)

#! Change me
blend_model = km.Model(inputs=blended_model_input, outputs=[base_model.get_layer(n).output for n in (['block4_conv2'] + style_layers)])

# Separate the model outputs into those intended for comparison with the content layer and the style layer
blend_content_outputs = [blend_model.outputs[0]]
blend_style_outputs = blend_model.outputs[1:]

def content_layer_loss(Fp, Fx):
    #! Change me
    _,h,w,d = Fp.get_shape()
    M =  w.value * h.value
    N = d.value
    
    err = (Fx - Fp)**2
    sse = K.tf.reduce_sum(err)
    
    loss = 1 / (2 * M ** (1/2) * N ** (1/2)) * sse
    return loss

content_loss = content_layer_loss(content_model.output,blend_content_outputs[0])

# The correct output of this function is 195710720.0
np.random.seed(0)
input_img = np.random.randn(1,img_rows,img_cols,3)
content_loss_evaluator = K.function([blend_model.input],[content_loss])
content_loss_evaluator([input_img])




def gram_matrix(f, M, N):
    f = K.tf.reshape(f, (M, N))
    G = K.tf.linalg.matmul(f, f, transpose_a=True)
    return G

# For a correctly implemented gram_matrix, the following code will produce 113934860.0
fmap = content_model.output
_,h,w,d = fmap.get_shape()
M = h*w
N = d
gram_matrix_evaluator = K.function([content_model.input],[gram_matrix(fmap,M,N)])
gram_matrix_evaluator([content_img])[0].mean()

def style_layer_loss(Fa, Fx):
    _, h, w, d = Fa.get_shape()
    M = h*w
    N = d
    Ga = gram_matrix(Fa, M, N)
    Gx = gram_matrix(Fx, M, N)
    
    err = (Ga - Gx)**2
    sse = K.tf.reduce_sum(err)
    
    loss = (1 / (4 * M.value ** 2 * N.value ** 2)) * sse
    return loss

style_loss_0 = style_layer_loss(style_model.output[0],blend_style_outputs[0])

# The correct output of this function is 220990.31
np.random.seed(0)
input_img = np.random.randn(1,img_rows,img_cols,3)
style_loss_evaluator = K.function([blend_model.input],[style_loss_0])
style_loss_evaluator([input_img])



style_loss = 0
for i in range(5):
    style_loss += 0.2*style_layer_loss(style_model.output[i],blend_style_outputs[i])
    
# The correct output of this function is 177059700.0
np.random.seed(0)
input_img = np.random.randn(1,img_rows,img_cols,3)
style_loss_evaluator = K.function([blend_model.input],[style_loss])
style_loss_evaluator([input_img])




tv_loss = K.tf.image.total_variation(blend_model.input)



alpha = 5
beta = 1000
gamma = 1

total_loss = alpha*content_loss + beta*style_loss + gamma*tv_loss
    
# The correct output of this function is 1.7715756e+12
np.random.seed(0)
input_img = np.random.randn(1,img_rows,img_cols,3)
total_loss_evaluator = K.function([blend_model.input],[total_loss])
total_loss_evaluator([input_img])



grads = K.gradients(total_loss,blend_model.input)[0]




loss_and_grad_evaluator = K.function([blend_model.input],[total_loss,grads])

np.random.seed(0)
input_img = np.random.randn(1,img_rows,img_cols,3)
l0,g0 = loss_and_grad_evaluator([input_img])
# Correct value of l0 is 3.5509e11
# Correct value of first element in g0 is -7.28989e2
print(l0, g0)


import scipy.optimize as sio

class cool():
    def __init__(self):
        self.current_grad = None
    
    def f(self, input_img):
        #print(input_img.shape)
        input_img = np.reshape(input_img, (1,img_rows,img_cols,3))
        #print("img", input_img.dtype)
        #print(input_img.shape)
        loss, self.current_grad = loss_and_grad_evaluator([input_img])
        #print("grad", self.current_grad.dtype)
        return loss
    
    def grad(self, _):
        return self.current_grad.astype('float64').flatten()
        


c = cool()

new_img = sio.fmin_l_bfgs_b(c.f, input_img, c.grad, iprint=5, maxiter=500)[0]

new_img = np.reshape(new_img, (img_rows, img_cols, 3))
new_img[:, :, 0] += 103.939
new_img[:, :, 1] += 116.779
new_img[:, :, 2] += 123.68

new_img = np.clip(new_img, 0, 255)

img = Image.fromarray(new_img.astype('uint8'), 'RGB')
img.save('anime-brock.png')
