"""
Build Your Own teenygrad: A Tiny Tensor Autograd Engine

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - prod
def prod(shape):
    result = 1
    for dim in shape:
        result *= dim
    return result

# Step 2 - argsort
def argsort(values):
    return sorted(range(len(values)), key=lambda i: values[i])

# Step 3 - make_op_enums (not yet solved)
# TODO: implement

# Step 4 - LazyBuffer (not yet solved)
# TODO: implement

# Step 5 - lazybuffer_const (not yet solved)
# TODO: implement

# Step 6 - rand (not yet solved)
# TODO: implement

# Step 7 - lazybuffer_unary_e (not yet solved)
# TODO: implement

# Step 8 - lazybuffer_binary_e (not yet solved)
# TODO: implement

# Step 9 - lazybuffer_r (not yet solved)
# TODO: implement

# Step 10 - lazybuffer_reshape (not yet solved)
# TODO: implement

# Step 11 - lazybuffer_expand (not yet solved)
# TODO: implement

# Step 12 - lazybuffer_permute (not yet solved)
# TODO: implement

# Step 13 - Function
class Function:
    def __init__(self, *tensors):
        self.needs_input_grad = [t.requires_grad for t in tensors]
        if any(self.needs_input_grad):
            self.requires_grad = True
        elif None in self.needs_input_grad:
            self.requires_grad = None
        else:
            self.requires_grad = False
        if self.requires_grad:
            self.parents = tensors

# Step 14 - function_forward_backward_stubs
def function_forward_backward_stubs():
    def forward(self, *args, **kwargs):
        raise NotImplementedError(f"forward not implemented for {type(self).__name__}")

    def backward(self, *args, **kwargs):
        raise NotImplementedError(f"backward not implemented for {type(self).__name__}")

    Function.forward = forward
    Function.backward = backward
    return Function

# Step 15 - apply
@classmethod
def apply(cls, *tensors, **kwargs):
    # Create the operation context
    ctx = cls(*tensors)

    # Run forward using the raw tensor buffers
    out = ctx.forward(*[t.data for t in tensors], **kwargs)

    # Output requires grad if any input requires grad
    requires_grad = any(t.requires_grad for t in tensors)

    # Wrap result back into a Tensor
    result = Tensor(out, requires_grad=requires_grad)

    # Link output tensor back to this operation for backward
    if requires_grad:
        result._ctx = ctx

    return result


# Attach apply onto Function so subclasses inherit it
Function.apply = apply

# Step 16 - Neg (not yet solved)
# TODO: implement

# Step 17 - Relu (not yet solved)
# TODO: implement

# Step 18 - Log (not yet solved)
# TODO: implement

# Step 19 - Exp (not yet solved)
# TODO: implement

# Step 20 - Sqrt (not yet solved)
# TODO: implement

# Step 21 - Sigmoid (not yet solved)
# TODO: implement

# Step 22 - Add (not yet solved)
# TODO: implement

# Step 23 - Sub (not yet solved)
# TODO: implement

# Step 24 - Mul (not yet solved)
# TODO: implement

# Step 25 - Div (not yet solved)
# TODO: implement

# Step 26 - sum_function_forward (not yet solved)
# TODO: implement

# Step 27 - sum_function_backward (not yet solved)
# TODO: implement

# Step 28 - max_function_forward (not yet solved)
# TODO: implement

# Step 29 - max_function_backward (not yet solved)
# TODO: implement

# Step 30 - Reshape (not yet solved)
# TODO: implement

# Step 31 - expand_function_forward (not yet solved)
# TODO: implement

# Step 32 - expand_function_backward (not yet solved)
# TODO: implement

# Step 33 - permute_function_forward_backward (not yet solved)
# TODO: implement

# Step 34 - Tensor (not yet solved)
# TODO: implement

# Step 35 - tensor_from_data (not yet solved)
# TODO: implement

# Step 36 - tensor_creation_helpers (not yet solved)
# TODO: implement

# Step 37 - tensor_randn (not yet solved)
# TODO: implement

# Step 38 - build_topological_order (not yet solved)
# TODO: implement

# Step 39 - tensor_backward (not yet solved)
# TODO: implement

# Step 40 - bind_unary_tensor_methods (not yet solved)
# TODO: implement

# Step 41 - broadcasted (not yet solved)
# TODO: implement

# Step 42 - bind_binary_tensor_methods (not yet solved)
# TODO: implement

# Step 43 - bind_movement_tensor_methods (not yet solved)
# TODO: implement

# Step 44 - bind_reduce_tensor_methods (not yet solved)
# TODO: implement

# Step 45 - tensor_mean (not yet solved)
# TODO: implement

# Step 46 - tensor_transpose (not yet solved)
# TODO: implement

# Step 47 - tensor_matmul_2d (not yet solved)
# TODO: implement

# Step 48 - tensor_softmax (not yet solved)
# TODO: implement

# Step 49 - tensor_log_softmax (not yet solved)
# TODO: implement

# Step 50 - sparse_categorical_cross_entropy (not yet solved)
# TODO: implement

# Step 51 - Linear (not yet solved)
# TODO: implement

# Step 52 - MLP (not yet solved)
# TODO: implement

# Step 53 - sgd_step (not yet solved)
# TODO: implement

# Step 54 - zero_grad (not yet solved)
# TODO: implement

# Step 55 - make_toy_digit_dataset (not yet solved)
# TODO: implement

# Step 56 - accuracy (not yet solved)
# TODO: implement

# Step 57 - train_mlp (not yet solved)
# TODO: implement

# Step 58 - evaluate_mlp (not yet solved)
# TODO: implement

