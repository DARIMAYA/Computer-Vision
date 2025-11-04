from interface import *


# ================================= 1.4.1 SGD ================================
class SGD(Optimizer):
    def __init__(self, lr):
        self.lr = lr

    def get_parameter_updater(self, parameter_shape):
        """
        :param parameter_shape: tuple, the shape of the associated parameter

        :return: the updater function for that parameter
        """


        def updater(parameter, parameter_grad):
            """
            :param parameter: np.array, current parameter values
            :param parameter_grad: np.array, current gradient, dLoss/dParam

            :return: np.array, new parameter values
            """
            # your code here \/
            return parameter - self.lr * parameter_grad
            # your code here /\

        return updater


# ============================= 1.4.2 SGDMomentum ============================
class SGDMomentum(Optimizer):
    def __init__(self, lr, momentum=0.0):
        self.lr = lr
        self.momentum = momentum

    def get_parameter_updater(self, parameter_shape):
        """
        :param parameter_shape: tuple, the shape of the associated parameter

        :return: the updater function for that parameter
        """

        def updater(parameter, parameter_grad):
            """
            :param parameter: np.array, current parameter values
            :param parameter_grad: np.array, current gradient, dLoss/dParam

            :return: np.array, new parameter values
            """
            # your code here \/
            # inertia = momentum * inertia + lr * grad
            updater.inertia = (self.momentum * updater.inertia +
                               self.lr * parameter_grad)
            # parameter_new = parameter - inertia
            return parameter - updater.inertia
            # your code here /\

        updater.inertia = np.zeros(parameter_shape)
        return updater


# ================================ 2.1.1 ReLU ================================
class ReLU(Layer):
    def forward_impl(self, inputs):
        """
        :param inputs: np.array((n, ...)), input values

        :return: np.array((n, ...)), output values

            n - batch size
            ... - arbitrary shape (the same for input and output)
        """
        # your code here \/
        # y = max(0, x)
        return np.maximum(0, inputs)
        # your code here /\

    def backward_impl(self, grad_outputs):
        """
        :param grad_outputs: np.array((n, ...)), dLoss/dOutputs

        :return: np.array((n, ...)), dLoss/dInputs

            n - batch size
            ... - arbitrary shape (the same for input and output)
        """
        # your code here \/
        # dL/dx = dL/dy * (x >= 0)
        return grad_outputs * (self.forward_inputs >= 0)
        # your code here /\


# =============================== 2.1.2 Softmax ==============================
class Softmax(Layer):
    def forward_impl(self, inputs):
        """
        :param inputs: np.array((n, d)), input values

        :return: np.array((n, d)), output values

            n - batch size
            d - number of units
        """
        # your code here \/
        # Стабильность: вычитаем максимум для избежания переполнения
        exp_inputs = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        return exp_inputs / np.sum(exp_inputs, axis=1, keepdims=True)
        # your code here /\

    def backward_impl(self, grad_outputs):
        """
        :param grad_outputs: np.array((n, d)), dLoss/dOutputs

        :return: np.array((n, d)), dLoss/dInputs

            n - batch size
            d - number of units
        """
        # your code here \/
        # Сложная производная: используем сохраненные выходы
        softmax_output = self.forward_outputs
        jacobian = softmax_output[:, :, np.newaxis] * (
                np.eye(softmax_output.shape[1])[np.newaxis, :, :] -
                softmax_output[:, np.newaxis, :]
        )
        return np.einsum('ijk,ik->ij', jacobian, grad_outputs)
        # your code here /\


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
        output_units = self.output_units

        # Register weights and biases as trainable parameters
        # Note, that the parameters and gradients *must* be stored in
        # self.<p> and self.<p>_grad, where <p> is the name specified in
        # self.add_parameter

        self.weights, self.weights_grad = self.add_parameter(
            name="weights",
            shape=(output_units, input_units),
            initializer=he_initializer(input_units),
        )

        self.biases, self.biases_grad = self.add_parameter(
            name="biases",
            shape=(output_units,),
            initializer=np.zeros,
        )

        self.output_shape = (output_units,)

    def forward_impl(self, inputs):
        """
        :param inputs: np.array((n, d)), input values

        :return: np.array((n, c)), output values

            n - batch size
            d - number of input units
            c - number of output units
        """
        # your code here \/
        # y = W @ x + b
        return inputs @ self.weights.T + self.biases
        # your code here /\

    def backward_impl(self, grad_outputs):
        """
        :param grad_outputs: np.array((n, c)), dLoss/dOutputs

        :return: np.array((n, d)), dLoss/dInputs

            n - batch size
            d - number of input units
            c - number of output units
        """
        # your code here \/
        # dL/dW = (dL/dy)^T @ x  (СУММА по batch)
        self.weights_grad[:] = grad_outputs.T @ self.forward_inputs

        # dL/db = sum(dL/dy, axis=0)  (СУММА по batch)
        self.biases_grad[:] = np.sum(grad_outputs, axis=0)

        # dL/dx = dL/dy @ W
        return grad_outputs @ self.weights
        # your code here /\


# ============================ 2.2.1 Crossentropy ============================
class CategoricalCrossentropy(Loss):
    def value_impl(self, y_gt, y_pred):
        """
        :param y_gt: np.array((n, d)), ground truth (correct) labels
        :param y_pred: np.array((n, d)), estimated target values

        :return: np.array((1,)), mean Loss scalar for batch

            n - batch size
            d - number of units
        """
        # your code here \/
        # Стабильность: ограничиваем снизу чтобы избежать log(0)
        y_pred = np.clip(y_pred, eps, 1.0)
        # L = -sum(y_gt * log(y_pred)) / batch_size
        loss = np.mean(-np.sum(y_gt * np.log(y_pred), axis=1))
        return np.array([loss])  # Обернуть в массив формы (1,)
        # your code here /\

    def gradient_impl(self, y_gt, y_pred):
        """
        :param y_gt: np.array((n, d)), ground truth (correct) labels
        :param y_pred: np.array((n, d)), estimated target values

        :return: np.array((n, d)), dLoss/dY_pred

            n - batch size
            d - number of units
        """
        # your code here \/
        # dL/dy_pred = -y_gt / y_pred / batch_size
        y_pred = np.clip(y_pred, eps, 1.0)
        return -y_gt / y_pred / y_gt.shape[0]
        # your code here /\


# ======================== 2.3 Train and Test on MNIST =======================
def train_mnist_model(x_train, y_train, x_valid, y_valid):
    # your code here \/
    # 1) Create a Model
    model = Model(
        loss=CategoricalCrossentropy(),
        optimizer=SGD(lr=0.001)
    )

    # 2) Add layers to the model
    #   (don't forget to specify the input shape for the first layer)
    model.add(Dense(128, input_shape=(784,)))
    model.add(ReLU())
    model.add(Dense(64))
    model.add(ReLU())
    model.add(Dense(10))
    model.add(Softmax())

    print(model)

    # 3) Train and validate the model using the provided data
    model.fit(
        x_train=x_train,
        y_train=y_train,
        batch_size=64,
        epochs=15,
        shuffle=True,
        verbose=True,
        x_valid=x_valid,
        y_valid=y_valid
    )
    # your code here /\
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
    c, d_k, kh, kw = kernels.shape
    assert d == d_k, "Input channels must match kernel channels"

    # Вычисляем выходные размеры
    oh = ih + 2 * padding - kh + 1
    ow = iw + 2 * padding - kw + 1

    # Добавляем паддинг
    if padding > 0:
        inputs_padded = np.pad(inputs,
                               ((0, 0), (0, 0), (padding, padding), (padding, padding)),
                               mode='constant', constant_values=0)
    else:
        inputs_padded = inputs

    outputs = np.zeros((n, c, oh, ow))

    # Математическая свертка требует переворота ядра на 180 градусов
    # Переворачиваем ядра по высоте и ширине
    flipped_kernels = np.flip(kernels, axis=(2, 3))

    # Выполняем свертку с перевернутыми ядрами
    for batch in range(n):
        for out_ch in range(c):
            for y in range(oh):
                for x in range(ow):
                    # Извлекаем патч
                    patch = inputs_padded[batch, :, y:y + kh, x:x + kw]
                    # Применяем свертку с перевернутым ядром
                    outputs[batch, out_ch, y, x] = np.sum(patch * flipped_kernels[out_ch])

    return outputs
    # your code here /\


# =============================== 4.1.1 Conv2D ===============================
class Conv2D(Layer):
    def __init__(self, output_channels, kernel_size=3, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert kernel_size % 2, "Kernel size should be odd"

        self.output_channels = output_channels
        self.kernel_size = kernel_size

        self.kernels, self.kernels_grad = None, None
        self.biases, self.biases_grad = None, None

    def build(self, *args, **kwargs):
        super().build(*args, **kwargs)

        input_channels, input_h, input_w = self.input_shape
        output_channels = self.output_channels
        kernel_size = self.kernel_size

        self.kernels, self.kernels_grad = self.add_parameter(
            name="kernels",
            shape=(output_channels, input_channels, kernel_size, kernel_size),
            initializer=he_initializer(input_h * input_w * input_channels),
        )

        self.biases, self.biases_grad = self.add_parameter(
            name="biases",
            shape=(output_channels,),
            initializer=np.zeros,
        )

        self.output_shape = (output_channels,) + self.input_shape[1:]

    def forward_impl(self, inputs):
        """
        :param inputs: np.array((n, d, h, w)), input values

        :return: np.array((n, c, h, w)), output values

            n - batch size
            d - number of input channels
            c - number of output channels
            (h, w) - image shape
        """
        # your code here \/
        """
               Прямой проход сверточного слоя
        """
        # Вычисляем паддинг для сохранения размеров (same convolution)
        padding = self.kernel_size // 2
        # Выполняем свертку
        conv_result = convolve(inputs, self.kernels, padding=padding)
        # Добавляем смещения (broadcast по батчу, высоте и ширине)
        return conv_result + self.biases[np.newaxis, :, np.newaxis, np.newaxis]
        # your code here /\

    def backward_impl(self, grad_outputs):
        """
        :param grad_outputs: np.array((n, c, h, w)), dLoss/dOutputs

        :return: np.array((n, d, h, w)), dLoss/dInputs

            n - batch size
            d - number of input channels
            c - number of output channels
            (h, w) - image shape
        """
        # your code here \/
        padding = self.kernel_size // 2

        # 1. Градиент по смещениям
        self.biases_grad[:] = np.sum(grad_outputs, axis=(0, 2, 3))

        # 2. Градиент по ядрам через корреляцию grad_outputs с inputs
        if padding > 0:
            inputs_padded = np.pad(self.forward_inputs,
                                   ((0, 0), (0, 0), (padding, padding), (padding, padding)),
                                   mode='constant', constant_values=0)
        else:
            inputs_padded = self.forward_inputs

        # Вычисляем градиенты ядер через транспонированную свертку
        # Меняем порядок осей для правильной свертки
        inputs_transposed = inputs_padded.transpose(1, 0, 2, 3)  # (d, n, h, w)
        grad_outputs_transposed = grad_outputs.transpose(1, 0, 2, 3)  # (c, n, h, w)

        # Вызываем convolve с перевернутыми входами для получения градиентов
        inputs_flipped = inputs_transposed[:, :, ::-1, ::-1]
        grad_kernels_raw = convolve(inputs_flipped, grad_outputs_transposed, padding=0)

        # Переставляем оси обратно
        self.kernels_grad[:] = grad_kernels_raw.transpose(1, 0, 2, 3)

        # 3. Градиент по входам - полная свертка с перевернутыми ядрами
        # Переворачиваем ядра для обратного распространения
        kernels_rotated = self.kernels[:, :, ::-1, ::-1]
        # Меняем местами входные и выходные каналы
        kernels_transposed = kernels_rotated.transpose(1, 0, 2, 3)

        # Полная свертка для градиента по входам
        grad_inputs = convolve(grad_outputs, kernels_transposed, padding=padding)

        return grad_inputs
        # your code here /\


# ============================== 4.1.2 Pooling2D =============================
class Pooling2D(Layer):
    def __init__(self, pool_size=2, pool_mode="max", *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert pool_mode in {"avg", "max"}

        self.pool_size = pool_size
        self.pool_mode = pool_mode
        self.forward_idxs = None

    def build(self, *args, **kwargs):
        super().build(*args, **kwargs)

        channels, input_h, input_w = self.input_shape
        output_h, rem_h = divmod(input_h, self.pool_size)
        output_w, rem_w = divmod(input_w, self.pool_size)
        assert not rem_h, "Input height should be divisible by the pool size"
        assert not rem_w, "Input width should be divisible by the pool size"

        self.output_shape = (channels, output_h, output_w)

    def forward_impl(self, inputs):
        """
        :param inputs: np.array((n, d, ih, iw)), input values

        :return: np.array((n, d, oh, ow)), output values

            n - batch size
            d - number of channels
            (ih, iw) - input image shape
            (oh, ow) - output image shape
        """
        # your code here \/
        n, d, ih, iw = inputs.shape
        pool_size = self.pool_size
        oh, ow = ih // pool_size, iw // pool_size

        # Переформатируем вход для векторизованных операций
        inputs_reshaped = inputs.reshape(n, d, oh, pool_size, ow, pool_size)
        inputs_reshaped = inputs_reshaped.transpose(0, 1, 2, 4, 3, 5)
        inputs_reshaped = inputs_reshaped.reshape(n, d, oh, ow, pool_size * pool_size)

        if self.pool_mode == "max":
            # Max pooling
            outputs = np.max(inputs_reshaped, axis=4)
            # Сохраняем индексы максимумов для обратного прохода
            max_indices = np.argmax(inputs_reshaped, axis=4)
            self.forward_idxs = max_indices
        else:
            # Average pooling
            outputs = np.mean(inputs_reshaped, axis=4)

        return outputs.reshape(n, d, oh, ow)
        # your code here /\

    def backward_impl(self, grad_outputs):
        """
        :param grad_outputs: np.array((n, d, oh, ow)), dLoss/dOutputs

        :return: np.array((n, d, ih, iw)), dLoss/dInputs

            n - batch size
            d - number of channels
            (ih, iw) - input image shape
            (oh, ow) - output image shape
        """
        # your code here \/
        n, d, oh, ow = grad_outputs.shape
        pool_size = self.pool_size
        ih, iw = oh * pool_size, ow * pool_size

        grad_inputs = np.zeros((n, d, ih, iw))

        if self.pool_mode == "max":
            # Для max pooling распределяем градиент только в позиции максимума
            for i in range(oh):
                for j in range(ow):
                    for batch in range(n):
                        for channel in range(d):
                            # Получаем индекс максимума в патче
                            max_idx = self.forward_idxs[batch, channel, i, j]
                            # Вычисляем координаты в исходном изображении
                            i_start = i * pool_size + max_idx // pool_size
                            j_start = j * pool_size + max_idx % pool_size
                            # Передаем градиент только в позицию максимума
                            grad_inputs[batch, channel, i_start, j_start] += \
                                grad_outputs[batch, channel, i, j]
        else:
            # Для average pooling равномерно распределяем градиент по патчу
            for i in range(oh):
                for j in range(ow):
                    i_start, i_end = i * pool_size, (i + 1) * pool_size
                    j_start, j_end = j * pool_size, (j + 1) * pool_size
                    # Равномерно распределяем градиент по всем элементам патча
                    patch_grad = grad_outputs[:, :, i:i + 1, j:j + 1] / (pool_size * pool_size)
                    grad_inputs[:, :, i_start:i_end, j_start:j_end] += \
                        patch_grad.repeat(pool_size, axis=2).repeat(pool_size, axis=3)

        return grad_inputs
        # your code here /\


# ============================== 4.1.3 BatchNorm =============================
class BatchNorm(Layer):
    def __init__(self, momentum=0.9, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.momentum = momentum

        self.running_mean = None
        self.running_var = None

        self.beta, self.beta_grad = None, None
        self.gamma, self.gamma_grad = None, None

        self.forward_inverse_std = None
        self.forward_normalized_inputs = None

    def build(self, *args, **kwargs):
        super().build(*args, **kwargs)

        # Универсальная обработка формы входа
        if len(self.input_shape) == 3:
            channels, _, _ = self.input_shape
        elif len(self.input_shape) == 1:
            (channels,) = self.input_shape
        else:
            raise ValueError(f"BatchNorm does not support input shape {self.input_shape}")

        self.running_mean = np.zeros(channels)
        self.running_var = np.ones(channels)

        self.beta, self.beta_grad = self.add_parameter(
            name="beta",
            shape=(channels,),
            initializer=np.zeros,
        )

        self.gamma, self.gamma_grad = self.add_parameter(
            name="gamma",
            shape=(channels,),
            initializer=np.ones,
        )

    def forward_impl(self, inputs):
        """
        :param inputs: np.array((n, d, h, w)), input values

        :return: np.array((n, d, h, w)), output values

            n - batch size
            d - number of channels
            (h, w) - image shape
        """
        # your code here \/
        is_conv = inputs.ndim == 4
        axes = (0, 2, 3) if is_conv else (0,)
        param_shape = (1, -1, 1, 1) if is_conv else (1, -1)

        if self.is_training:
            # Режим обучения: вычисляем статистики по батчу
            mean = np.mean(inputs, axis=axes)
            var = np.var(inputs, axis=axes)

            # Обновляем скользящие статистики
            self.running_mean = (self.momentum * self.running_mean +
                                 (1 - self.momentum) * mean)
            self.running_var = (self.momentum * self.running_var +
                                (1 - self.momentum) * var)

            mean_r, var_r = mean.reshape(param_shape), var.reshape(param_shape)
        else:
            # Режим инференса: используем скользящие статистики
            mean_r = self.running_mean.reshape(param_shape)
            var_r = self.running_var.reshape(param_shape)

        self.forward_inverse_std = 1.0 / np.sqrt(var_r + eps)
        self.forward_normalized_inputs = (inputs - mean_r) * self.forward_inverse_std

        # Масштабируем и сдвигаем
        gamma = self.gamma.reshape(param_shape)
        beta = self.beta.reshape(param_shape)
        return gamma * self.forward_normalized_inputs + beta
        # your code here /\

    def backward_impl(self, grad_outputs):
        """
        :param grad_outputs: np.array((n, d, h, w)), dLoss/dOutputs

        :return: np.array((n, d, h, w)), dLoss/dInputs

            n - batch size
            d - number of channels
            (h, w) - image shape
        """
        # your code here \/
        is_conv = grad_outputs.ndim == 4
        axes = (0, 2, 3) if is_conv else (0,)
        param_shape = (1, -1, 1, 1) if is_conv else (1, -1)

        # Количество элементов в нормализации
        N = np.prod(grad_outputs.shape) / grad_outputs.shape[1] if is_conv else grad_outputs.shape[0]

        # Градиенты по параметрам (накапливаем)
        self.beta_grad += np.sum(grad_outputs, axis=axes)
        self.gamma_grad += np.sum(grad_outputs * self.forward_normalized_inputs, axis=axes)

        # Градиент по входам (упрощенная формула как в работающем коде)
        grad_norm = grad_outputs * self.gamma.reshape(param_shape)
        sum_grad_norm = np.sum(grad_norm, axis=axes, keepdims=True)
        sum_grad_norm_x_norm = np.sum(grad_norm * self.forward_normalized_inputs, axis=axes, keepdims=True)

        term1 = N * grad_norm
        term2 = sum_grad_norm
        term3 = self.forward_normalized_inputs * sum_grad_norm_x_norm

        return self.forward_inverse_std * (term1 - term2 - term3) / N
        # your code here /\


# =============================== 4.1.4 Flatten ==============================
class Flatten(Layer):
    def build(self, *args, **kwargs):
        super().build(*args, **kwargs)

        self.output_shape = (int(np.prod(self.input_shape)),)

    def forward_impl(self, inputs):
        """
        :param inputs: np.array((n, d, h, w)), input values

        :return: np.array((n, (d * h * w))), output values

            n - batch size
            d - number of input channels
            (h, w) - image shape
        """
        # your code here \/
        return inputs.reshape(inputs.shape[0], -1)
        # your code here /\

    def backward_impl(self, grad_outputs):
        """
        :param grad_outputs: np.array((n, (d * h * w))), dLoss/dOutputs

        :return: np.array((n, d, h, w)), dLoss/dInputs

            n - batch size
            d - number of units
            (h, w) - input image shape
        """
        # your code here \/
        return grad_outputs.reshape(grad_outputs.shape[0], *self.input_shape)
        # your code here /\


# =============================== 4.1.5 Dropout ==============================
class Dropout(Layer):
    def __init__(self, p, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.p = p
        self.forward_mask = None

    def forward_impl(self, inputs):
        """
        :param inputs: np.array((n, ...)), input values

        :return: np.array((n, ...)), output values

            n - batch size
            ... - arbitrary shape (the same for input and output)
        """
        # your code here \/
        if self.is_training:
            # Создаем маску для dropout (True - оставить, False - отбросить)
            self.forward_mask = np.random.rand(*inputs.shape) > self.p
            # Применяем маску без масштабирования
            return inputs * self.forward_mask
        else:
            # Во время инференса масштабируем выходы
            return inputs * (1.0 - self.p)
        # your code here /\

    def backward_impl(self, grad_outputs):
        """
        :param grad_outputs: np.array((n, ...)), dLoss/dOutputs

        :return: np.array((n, ...)), dLoss/dInputs

            n - batch size
            ... - arbitrary shape (the same for input and output)
        """
        # your code here \/
        if self.is_training:
            # Применяем ту же маску к градиентам
            return grad_outputs * self.forward_mask
        else:
            # Во время инференса масштабируем градиенты
            return grad_outputs * (1.0 - self.p)
        # your code here /\


# ====================== 2.3 Train and Test on CIFAR-10 ======================
def train_cifar10_model(x_train, y_train, x_valid, y_valid):
    # Trial parameters
    learning_rate = 0.01
    momentum = 0.9
    epochs = 3
    batch_size = 64

    optimizer = SGDMomentum(lr=learning_rate, momentum=momentum)
    loss = CategoricalCrossentropy()
    model = Model(loss=loss, optimizer=optimizer)

    input_shape = (3, 32, 32)

    #Blocks
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

    # your code here /\
    return model


# ============================================================================
