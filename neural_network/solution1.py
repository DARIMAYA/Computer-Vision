from interface import *

# ================================= 1.4.1 SGD ================================
class SGD(Optimizer):
    def __init__(self, lr):
        self.lr = lr
    def get_parameter_updater(self, parameter_shape):
        def updater(parameter, parameter_grad):
            return parameter - self.lr * parameter_grad
        return updater

# ============================= 1.4.2 SGDMomentum ============================
class SGDMomentum(Optimizer):
    def __init__(self, lr, momentum=0.0):
        self.lr = lr
        self.momentum = momentum
    def get_parameter_updater(self, parameter_shape):
        def updater(parameter, parameter_grad):
            updater.inertia = self.momentum * updater.inertia + self.lr * parameter_grad
            return parameter - updater.inertia
        updater.inertia = np.zeros(parameter_shape)
        return updater

# ================================ 2.1.1 ReLU ================================
class ReLU(Layer):
    def forward_impl(self, inputs):
        return np.maximum(0, inputs)
    def backward_impl(self, grad_outputs):
        return grad_outputs * (self.forward_inputs >= 0)

# =============================== 2.1.2 Softmax ==============================
class Softmax(Layer):
    def forward_impl(self, inputs):
        exps = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        self.forward_outputs = exps / np.sum(exps, axis=1, keepdims=True)
        return self.forward_outputs
    def backward_impl(self, grad_outputs):
        s = self.forward_outputs
        g = grad_outputs
        scalar_product = np.sum(g * s, axis=1, keepdims=True)
        return s * (g - scalar_product)

# ================================ 2.1.3 Dense ===============================
class Dense(Layer):
    def __init__(self, units, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.output_units = units
        self.weights, self.weights_grad = None, None
        self.biases, self.biases_grad = None, None

    def build(self, *args, **kwargs):
        super().build(*args, **kwargs)
        (input_units,) = self.input_shape
        self.weights, self.weights_grad = self.add_parameter(
            name="weights", shape=(self.output_units, input_units),
            initializer=he_initializer(input_units))
        self.biases, self.biases_grad = self.add_parameter(
            name="biases", shape=(self.output_units,), initializer=np.zeros)
        self.output_shape = (self.output_units,)

    def forward_impl(self, inputs):
        return inputs @ self.weights.T + self.biases

    def backward_impl(self, grad_outputs):
        self.weights_grad = grad_outputs.T @ self.forward_inputs
        self.biases_grad = np.sum(grad_outputs, axis=0)
        return grad_outputs @ self.weights

# ============================ 2.2.1 Crossentropy ============================
class CategoricalCrossentropy(Loss):
    def value_impl(self, y_gt, y_pred):
        n = y_gt.shape[0]
        clipped_y_pred = np.clip(y_pred, eps, 1.0 - eps)
        loss = -np.sum(y_gt * np.log(clipped_y_pred)) / n
        return np.array([loss])
    def gradient_impl(self, y_gt, y_pred):
        n = y_gt.shape[0]
        clipped_y_pred = np.clip(y_pred, eps, 1.0 - eps)
        return (-y_gt / clipped_y_pred) / n

# ======================== 2.3 Train and Test on MNIST =======================
def train_mnist_model(x_train, y_train, x_valid, y_valid):
    model = Model(
        loss=CategoricalCrossentropy(),
        optimizer=SGDMomentum(lr=0.01, momentum=0.9)
    )
    model.add(Dense(units=128, input_shape=(784,)))
    model.add(ReLU())
    model.add(Dense(units=64))
    model.add(ReLU())
    model.add(Dense(units=10))
    model.add(Softmax())
    print("Model architecture:")
    print(model)
    model.fit(
        x_train=x_train,
        y_train=y_train,
        x_valid=x_valid,
        y_valid=y_valid,
        epochs=10,
        batch_size=32
    )
    return model

# ============================== 3.3.2 convolve ==============================
def convolve(inputs, kernels, padding=0):
    """
    :param inputs: np.array((n, d, ih, iw)), input values
    :param kernels: np.array((c, d, kh, kw)), convolution kernels
    :param padding: int >= 0, the size of padding, 0 means 'valid'

    :return: np.array((n, c, oh, ow)), output values

        n - batch size
        d - number of input channels
        c - number of output channels
        (ih, iw) - input image shape
        (oh, ow) - output image shape
    """
    # !!! Don't change this function, it's here for your reference only !!!
    assert isinstance(padding, int) and padding >= 0
    assert inputs.ndim == 4 and kernels.ndim == 4
    assert inputs.shape[1] == kernels.shape[1]

    if os.environ.get("USE_FAST_CONVOLVE", False):
        return convolve_pytorch(inputs, kernels, padding)
    else:
        return convolve_numpy(inputs, kernels, padding)

def convolve_numpy(inputs, kernels, padding):
    """
    :param inputs: np.array((n, d, ih, iw)), input values
    :param kernels: np.array((c, d, kh, kw)), convolution kernels
    :param padding: int >= 0, the size of padding, 0 means 'valid'

    :return: np.array((n, c, oh, ow)), output values

        n - batch size
        d - number of input channels
        c - number of output channels
        (ih, iw) - input image shape
        (oh, ow) - output image shape
    """
    # your code here \/
    n, d, ih, iw = inputs.shape
    c, _, kh, kw = kernels.shape

    oh = ih + 2 * padding - kh + 1
    ow = iw + 2 * padding - kw + 1

    if padding > 0:
        inputs_padded = np.pad(
            inputs,
            pad_width=((0, 0), (0, 0), (padding, padding), (padding, padding)),
            mode='constant'
        )
    else:
        inputs_padded = inputs

    outputs = np.zeros((n, c, oh, ow))

    kernels_flipped = kernels[:, :, ::-1, ::-1]

    for y in range(oh):
        for x in range(ow):
            patch = inputs_padded[:, :, y:y + kh, x:x + kw]

            conv_result = np.tensordot(patch, kernels_flipped, axes=([1, 2, 3], [1, 2, 3]))

            outputs[:, :, y, x] = conv_result

    return outputs
    # your code here /\

# =============================== 4.1.1 Conv2D ===============================
class Conv2D(Layer):
    def __init__(self, output_channels, kernel_size=3, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert kernel_size % 2, "Kernel size should be odd"
        self.output_channels = output_channels
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2
        self.kernels, self.kernels_grad = None, None
        self.biases, self.biases_grad = None, None

    def build(self, *args, **kwargs):
        super().build(*args, **kwargs)
        ic, ih, iw = self.input_shape
        oc, ks = self.output_channels, self.kernel_size
        self.kernels, self.kernels_grad = self.add_parameter(
            name="kernels", shape=(oc, ic, ks, ks), initializer=he_initializer(ih * iw * ic))
        self.biases, self.biases_grad = self.add_parameter(
            name="biases", shape=(oc,), initializer=np.zeros)
        self.output_shape = (oc, ih, iw)

    def forward_impl(self, inputs):
        """Прямой проход: Y = convolve(X, K) + B"""
        return convolve(inputs, self.kernels, self.padding) + self.biases.reshape(1, -1, 1, 1)

    def backward_impl(self, grad_outputs):

        self.biases_grad = np.sum(grad_outputs, axis=(0, 2, 3))

        grad_outputs_swapped = grad_outputs.transpose(1, 0, 2, 3)
        inputs_swapped = self.forward_inputs.transpose(1, 0, 2, 3)
        inputs_swapped_rot180 = inputs_swapped[:, :, ::-1, ::-1]

        kernels_grad = convolve(grad_outputs_swapped, inputs_swapped_rot180, padding=self.padding)
        self.kernels_grad = kernels_grad

        kernels_swapped = self.kernels.transpose(1, 0, 2, 3)
        kernels_swapped_rot180 = kernels_swapped[:, :, ::-1, ::-1]

        grad_inputs = convolve(grad_outputs, kernels_swapped_rot180, padding=self.padding)

        return grad_inputs

# ============================== 4.1.2 Pooling2D =============================
class Pooling2D(Layer):
    def __init__(self, pool_size=2, pool_mode="max", *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert pool_mode in {"avg", "max"}
        self.pool_size, self.pool_mode, self.forward_idxs = pool_size, pool_mode, None

    def build(self, *args, **kwargs):
        super().build(*args, **kwargs)
        c, ih, iw = self.input_shape
        p = self.pool_size
        oh, rem_h = divmod(ih, p); ow, rem_w = divmod(iw, p)
        assert not rem_h and not rem_w, "Input shape not divisible by pool size"
        self.output_shape = (c, oh, ow)

    def forward_impl(self, inputs):
        n, d, ih, iw = inputs.shape
        p = self.pool_size
        oh, ow = ih // p, iw // p
        pools = inputs.reshape(n, d, oh, p, ow, p).transpose(0, 1, 2, 4, 3, 5)
        if self.pool_mode == "max":
            flat_pools = pools.reshape(n, d, oh, ow, p * p)
            self.forward_idxs = np.argmax(flat_pools, axis=-1)
            return np.max(flat_pools, axis=-1)
        else:
            return np.mean(pools, axis=(-2, -1))

    def backward_impl(self, grad_outputs):
        p = self.pool_size
        n, d, oh, ow = grad_outputs.shape
        ih, iw = oh * p, ow * p
        if self.pool_mode == "max":
            mask = (np.arange(p * p) == self.forward_idxs[..., None])
            mask = mask.reshape(n, d, oh, ow, p, p).transpose(0, 1, 2, 4, 3, 5).reshape(n, d, ih, iw)
            return grad_outputs.repeat(p, axis=2).repeat(p, axis=3) * mask
        else:
            return grad_outputs.repeat(p, axis=2).repeat(p, axis=3) / (p * p)

# ============================== 4.1.3 BatchNorm =============================
class BatchNorm(Layer):
    def __init__(self, momentum=0.9, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.momentum = momentum
        self.running_mean, self.running_var = None, None
        self.beta, self.beta_grad = None, None
        self.gamma, self.gamma_grad = None, None
        self.forward_inverse_std = None
        self.forward_normalized_inputs = None

    def build(self, *args, **kwargs):
        super().build(*args, **kwargs)
        if len(self.input_shape) == 3:
            channels, _, _ = self.input_shape
        elif len(self.input_shape) == 1:
            (channels,) = self.input_shape
        else:
            raise ValueError(f"BatchNorm does not support input shape {self.input_shape}")
        self.running_mean = np.zeros(channels)
        self.running_var = np.ones(channels)
        self.beta, self.beta_grad = self.add_parameter("beta", (channels,), np.zeros)
        self.gamma, self.gamma_grad = self.add_parameter("gamma", (channels,), np.ones)

    def forward_impl(self, inputs):
        is_conv = inputs.ndim == 4
        axes = (0, 2, 3) if is_conv else (0,)
        param_shape = (1, -1, 1, 1) if is_conv else (1, -1)
        if self.is_training:
            mean = np.mean(inputs, axis=axes)
            var = np.var(inputs, axis=axes)
            self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * mean
            self.running_var = self.momentum * self.running_var + (1 - self.momentum) * var
            mean_r, var_r = mean.reshape(param_shape), var.reshape(param_shape)
        else:
            mean_r, var_r = self.running_mean.reshape(param_shape), self.running_var.reshape(param_shape)
        self.forward_inverse_std = 1.0 / np.sqrt(var_r + eps)
        self.forward_normalized_inputs = (inputs - mean_r) * self.forward_inverse_std
        return self.gamma.reshape(param_shape) * self.forward_normalized_inputs + self.beta.reshape(param_shape)

    def backward_impl(self, grad_outputs):
        is_conv = grad_outputs.ndim == 4
        axes = (0, 2, 3) if is_conv else (0,)
        param_shape = (1, -1, 1, 1) if is_conv else (1, -1)
        N = np.prod(grad_outputs.shape) / grad_outputs.shape[1] if is_conv else grad_outputs.shape[0]
        self.beta_grad += np.sum(grad_outputs, axis=axes)
        self.gamma_grad += np.sum(grad_outputs * self.forward_normalized_inputs, axis=axes)
        grad_norm = grad_outputs * self.gamma.reshape(param_shape)
        sum_grad_norm = np.sum(grad_norm, axis=axes, keepdims=True)
        sum_grad_norm_x_norm = np.sum(grad_norm * self.forward_normalized_inputs, axis=axes, keepdims=True)
        term1 = N * grad_norm
        term2 = sum_grad_norm
        term3 = self.forward_normalized_inputs * sum_grad_norm_x_norm
        return self.forward_inverse_std * (term1 - term2 - term3) / N

# =============================== 4.1.4 Flatten ==============================
class Flatten(Layer):
    def build(self, *args, **kwargs):
        super().build(*args, **kwargs)
        self.output_shape = (int(np.prod(self.input_shape)),)
    def forward_impl(self, inputs):
        return inputs.reshape(inputs.shape[0], -1)
    def backward_impl(self, grad_outputs):
        return grad_outputs.reshape((grad_outputs.shape[0],) + self.input_shape)

# =============================== 4.1.5 Dropout ==============================
class Dropout(Layer):
    def __init__(self, p, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.p, self.forward_mask = p, None
    def forward_impl(self, inputs):
        if self.is_training:
            self.forward_mask = (np.random.uniform(size=inputs.shape) > self.p)
            return inputs * self.forward_mask
        return inputs * (1.0 - self.p)
    def backward_impl(self, grad_outputs):
        return grad_outputs * self.forward_mask

# ====================== 2.3 Train and Test on CIFAR-10 ======================
def train_cifar10_model(x_train, y_train, x_valid, y_valid):
    learning_rate = 0.01
    momentum = 0.9
    epochs = 3
    batch_size = 64

    optimizer = SGDMomentum(lr=learning_rate, momentum=momentum)
    loss = CategoricalCrossentropy()
    model = Model(loss=loss, optimizer=optimizer)

    input_shape = (3, 32, 32)

    model.add(Conv2D(output_channels=32, kernel_size=3, input_shape=input_shape))
    model.add(BatchNorm())
    model.add(ReLU())
    model.add(Pooling2D(pool_size=2, pool_mode='max'))

    model.add(Conv2D(output_channels=64, kernel_size=3))
    model.add(BatchNorm())
    model.add(ReLU())
    model.add(Pooling2D(pool_size=2, pool_mode='max'))

    model.add(Conv2D(output_channels=128, kernel_size=3))
    model.add(BatchNorm())
    model.add(ReLU())
    model.add(Pooling2D(pool_size=2, pool_mode='max'))

    model.add(Conv2D(output_channels=256, kernel_size=3))
    model.add(BatchNorm())
    model.add(ReLU())

    model.add(Pooling2D(pool_size=4, pool_mode='avg'))

    model.add(Flatten())

    model.add(Dense(units=10))
    model.add(Softmax())

    print(model)

    model.fit(
        x_train=x_train,
        y_train=y_train,
        x_valid=x_valid,
        y_valid=y_valid,
        epochs=epochs,
        batch_size=batch_size
    )

    return model

# ============================================================================