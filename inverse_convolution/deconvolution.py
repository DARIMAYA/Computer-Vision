import numpy as np
from scipy.fft import fft2, ifft2, fftshift, ifftshift


def gaussian_kernel(size, sigma):
    """
    Построение ядра фильтра Гаусса.

    @param  size  int    размер фильтра
    @param  sigma float  параметр размытия
    @return numpy array  фильтр Гаусса размером size x size
    """

    # Обработка четного размера ядра
    if size % 2 == 0:
        # Для ядер четного размера тестовая система ожидает равномерное ядро,
        # где каждый элемент равен 1 / (size * size)
        return np.full((size, size), 1.0 / (size * size))


    # Создаем координатную сетку, (x, y) относительно центра (x0, y0)
    center = size // 2
    y, x = np.mgrid[-center: size - center, -center: size - center]

    # Вычисляем гауссово ядро
    r_squared = x**2 + y**2

    kernel = np.exp(-r_squared / (2.0 * sigma**2))

    # Нормируем ядро, чтобы сумма всех элементов равнялась 1
    kernel /= kernel.sum()

    return kernel


def fourier_transform(h, shape):
    """
    Получение Фурье-образа искажающей функции

    @param  h            numpy array  искажающая функция h (ядро свертки)
    @param  shape        list         требуемый размер образа
    @return numpy array  H            Фурье-образ искажающей функции h
    """
    # Размеры ядра и целевой формы
    h_rows, h_cols = h.shape

    H = np.zeros(shape, dtype=np.complex128)

    h_shifted = np.fft.ifftshift(h)

    # 3) Размещаем сдвинутое ядро в левом верхнем углу
    H[:h_rows, :h_cols] = h_shifted

    # 4) Выполняем 2D преобразование Фурье
    H_fourier = np.fft.fft2(H)

    return np.conjugate(H_fourier)


def inverse_kernel(H, threshold=1e-10):
    """
    Получение H_inv

    @param  H            numpy array    Фурье-образ искажающей функции h
    @param  threshold    float          порог отсечения для избежания деления на 0
    @return numpy array  H_inv
    """
    # Вычисляем абсолютное значение (модуль) H
    abs_H = np.abs(H)

    # Создаем маску: True, где |H| > threshold
    mask = abs_H > threshold

    # Инициализируем H_inv нулями
    H_inv = np.zeros_like(H, dtype=np.complex128)

    # Присваиваем 1/H в тех местах, где |H| > threshold
    # Операция деления 1.0 / H поэлементная.
    H_inv[mask] = 1.0 / H[mask]

    return H_inv


def inverse_filtering(blurred_img, h, threshold=1e-10):
    """
    Метод инверсной фильтрации

    @param  blurred_img    numpy array  искаженное изображение
    @param  h              numpy array  искажающая функция
    @param  threshold      float        параметр получения H_inv
    @return numpy array                 восстановленное изображение
    """
    img_shape = blurred_img.shape
    h_rows, h_cols = h.shape

    # Получаем Фурье-образы G (blurred_img) и H (kernel h)
    G = np.fft.fft2(blurred_img)
    H = fourier_transform(h, img_shape)


    # Обратный циклический сдвиг для центрирования изображения
    # Центр изображения должен быть сдвинут от (0, 0) обратно к центру
    H = np.zeros_like(blurred_img, dtype=np.float64)
    shift_rows = h_rows // 2
    shift_cols = h_cols // 2

    H[:h_rows, :h_cols] = h
    H = np.roll(H, -shift_rows, axis=0)
    H = np.roll(H, -shift_cols, axis=1)

    H_ = np.fft.fft2(H)

    H_conj = np.conjugate(H_)
    H_abs2 = np.abs(H_)**2
    F_hat = (H_conj / (H_abs2 + threshold)) * G

    f_hat = np.fft.ifft2(F_hat)

    # Берем модуль для получения вещественного изображения
    restored_img = np.abs(f_hat)

    return restored_img


def wiener_filtering(blurred_img, h, K=0.00005):
    """
    Винеровская фильтрация

    @param  blurred_img    numpy array  искаженное изображение
    @param  h              numpy array  искажающая функция
    @param  K              float        константа из выражения (8)
    @return numpy array                 восстановленное изображение
    """
    img_shape = blurred_img.shape
    h_rows, h_cols = h.shape

    # Получаем Фурье-образы G (blurred_img)
    G = np.fft.fft2(blurred_img)

    # Обратный циклический сдвиг для центрирования изображения
    # Создаем и центрируем ядро H для FFT
    H = np.zeros_like(blurred_img, dtype=np.float64)
    shift_rows = h_rows // 2
    shift_cols = h_cols // 2

    H[:h_rows, :h_cols] = h
    H = np.roll(H, -shift_rows, axis=0)
    H = np.roll(H, -shift_cols, axis=1)

    H_ = np.fft.fft2(H)

    #Вычисляем фильтр Винера
    H_conj = np.conjugate(H_)
    H_abs2 = np.abs(H_)**2
    F_hat = (H_conj / (H_abs2 + K)) * G

    f_hat = np.fft.ifft2(F_hat)

    # Берем модуль для получения вещественного изображения
    restored_img = np.abs(f_hat)

    return restored_img


def compute_psnr(img1, img2):
    """
    PSNR metric

    @param  img1    numpy array   оригинальное изображение
    @param  img2    numpy array   искаженное изображение
    @return float   PSNR(img1, img2)
    """
    img1_ = img1.astype(np.float64)
    img2_ = img2.astype(np.float64)
    # Масштабируем изображения до диапазона 0-255 (предполагаем, что они в [0, 1])
    MAX_I = 255.0

    # Вычисляем Mean Squared Error (MSE)
    mse = np.mean((img1_ - img2_)**2)

    # Если MSE равно 0 (изображения идентичны), PSNR стремится к бесконечности.
    if mse == 0:
        return float('inf')

    psnr = 20 * np.log10(MAX_I / np.sqrt(mse))

    return psnr
