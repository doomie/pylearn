from itertools import izip
from theano import tensor
from .base import IterativeOptimizerBase, dict_merge

# This file has no valuable content yet

class SGDQN(IterativeOptimizerBase):
   """
   The SGD-QN (Quasi-Newton) algorithm of Antoine Bordes et al.

   There are two versions (TODO: cite the JMLR papers)

   """
   def __call__(self, parameters,
           cost,
           gradients=None,
           stop=None,
           updates=None,
           step_size=None):
 
        # TODO:
        # - how to have a state? Need to keep diag_W around, for each set of
        # parameters.
        # - How to do the whole computation as a symbolic graph? Seems a bit
        # difficult, not sure how to avoid compiling a theano function (perhaps
        # at __init__() time?).
        # - Need an Elemwise Op that takes two tensors, divides them and
        # replaces each occurence of 0/0 with a given number (lambda)
           
        iter = epoch * n_train_batches + minibatch_index
        t = iter + 1

        # w_t and g_t(w_t)
        W_prev = [W.value.copy() for W in parameters]
        gW_prev = [gW.value.copy) for gW in gradients]

        W.value -= diag_W * gW / (t + t0)
        b.value -= diag_b * gb / (t + t0)

        if updateB:

            # computing w_{t+1} and g_t(w_{t+1})
            W_next, b_next = W.value.copy(), b.value.copy()
            minibatch_avg_cost,gW_next,gb_next = train_model(minibatch_index)

            # updating things for W
            #grad_diff = gW_next - gW_prev
            # the whole gradient, including the regularization term
            grad_diff = (gW_next + mylambda * W_next) - (gW_prev + mylambda * W_prev)
            weight_diff = W_next - W_prev

            ri = grad_diff / weight_diff
            ri[numpy.isnan(ri)] = mylambda # if 0/0
            diag_W += 2.0/r * (1.0/ri - diag_W)
            diag_W = numpy.maximum(diag_W,0.01/mylambda)

            # updating things for b
            grad_diff = gb_next - gb_prev
            weight_diff = b_next - b_prev

            ri = grad_diff / weight_diff
            ri[numpy.isnan(ri)] = mylambda # if 0/0
            diag_b += 2.0/r * (1.0/ri - diag_b)
            diag_b = numpy.maximum(diag_b,0.01/mylambda)

            updateB = False
            r += 1         
            
        count -= 1    
        if count <= 0:
            count = skip
            updateB = True
            # regularization
            W.value -= skip*mylambda*diag_W*W.value / (t + t0)

sgdqn = SGDQN()
