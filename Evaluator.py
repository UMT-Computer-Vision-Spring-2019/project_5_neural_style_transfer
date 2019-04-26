import numpy as np
import scipy.optimize as sio


class Evaluator:

    def __init__(self, lg_evaluator):

        self._gradient = None
        self._lg_eval = lg_evaluator

    # inputs = [input_img]
    def optimize(self, input_img, img_rows, img_cols):

        def loss(inputs):

            inputs = np.reshape(inputs, newshape=(1, img_rows, img_cols, 3))
            l, g = self._lg_eval([inputs])

            self._gradient = g.astype(np.float64)
            return l.astype(np.float64)

        def gradient(inputs):
            return self._gradient.flatten()

        op = sio.fmin_l_bfgs_b(func=loss, fprime=gradient, x0=input_img, maxiter=500, iprint=10)
        print(op)

        op_x = np.reshape(op[0], (1, img_rows, img_cols, 3))

        return op_x[0]
