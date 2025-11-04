import io
import pickle
import zipfile

import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter
from skimage.io import imread
from skimage.metrics import peak_signal_noise_ratio

# !Этих импортов достаточно для решения данного задания, нельзя использовать другие библиотеки!


def pca_compression(matrix, p):
    """Сжатие изображения с помощью PCA
    Вход: двумерная матрица (одна цветовая компонента картинки), количество компонент
    Выход: собственные векторы, проекция матрицы на новое пр-во и средние значения до центрирования
    """

    # Your code here

    # Отцентруем каждую строчку матрицы
    mean_val = np.mean(matrix, axis=1)
    centered_matrix = matrix - mean_val.reshape(-1, 1)

    # Найдем матрицу ковариации
    cov_matrix = np.cov(centered_matrix)

    # Ищем собственные значения и собственные векторы матрицы ковариации, используйте linalg.eigh из numpy
    eig_vals, eig_vecs = np.linalg.eigh(cov_matrix)

    # Посчитаем количество найденных собственных векторов
    num_eig_vecs = eig_vecs.shape[1]

    # Сортируем собственные значения в порядке убывания
    sorted_indices = np.argsort(eig_vals)[::-1]

    # Сортируем собственные векторы согласно отсортированным собственным значениям
    # !Это все для того, чтобы мы производили проекцию в направлении максимальной дисперсии!
    sorted_eig_vecs = eig_vecs[:, sorted_indices]

    # Оставляем только p собственных векторов
    selected_eig_vecs = sorted_eig_vecs[:, :p]

    # Проекция данных на новое пространство
    projection = selected_eig_vecs.T @ centered_matrix

    return selected_eig_vecs, projection, mean_val


def pca_decompression(compressed):
    """Разжатие изображения
    Вход: список кортежей из собственных векторов и проекций для каждой цветовой компоненты
    Выход: разжатое изображение
    """

    result_img = []
    for i, comp in enumerate(compressed):
        # Матрично умножаем собственные векторы на проекции и прибавляем среднее значение по строкам исходной матрицы
        # !Это следует из описанного в самом начале примера!

        # Your code here
        eig_vecs, projection, mean_val = comp

        # Восстанавливаем исходную матрицу: eigenvectors × projection + mean
        reconstructed = eig_vecs @ projection + mean_val.reshape(-1, 1)

        # Добавляем восстановленный канал в результат
        result_img.append(reconstructed)

        # Объединяем все каналы в одно изображение
        # Транспонируем, чтобы получить правильную форму (height, width, channels)
    result_img = np.stack(result_img, axis=-1)

    # Обеспечиваем, чтобы значения пикселей были в допустимом диапазоне [0, 255]
    result_img = np.clip(result_img, 0, 255).astype(np.uint8)

    return result_img


def pca_visualize():
    plt.clf()
    img = imread("cat.png")
    if len(img.shape) == 3:
        img = img[..., :3]
    fig, axes = plt.subplots(3, 3)
    fig.set_figwidth(12)
    fig.set_figheight(12)

    for i, p in enumerate([1, 5, 10, 20, 50, 100, 150, 200, 256]):
        compressed = []
        for j in range(0, 3):
            # Берем j-й цветовой канал и применяем PCA сжатие
            channel = img[:, :, j].astype(np.float64)
            compressed_channel = pca_compression(channel, p)
            compressed.append(compressed_channel)

        # Восстанавливаем изображение из сжатых данных
        reconstructed_img = pca_decompression(compressed)

        # Убеждаемся, что изображение правильного типа
        if reconstructed_img.dtype != np.uint8:
            reconstructed_img = reconstructed_img.astype(np.uint8)

        axes[i // 3, i % 3].imshow(reconstructed_img)
        axes[i // 3, i % 3].set_title("Компонент: {}".format(p))

    fig.savefig("pca_visualization.png")


def rgb2ycbcr(img):
    """Переход из пр-ва RGB в пр-во YCbCr
    Вход: RGB изображение
    Выход: YCbCr изображение
    """

    # Your code here
    # Преобразуем в float для точных вычислений
    img_float = img.astype(np.float64)

    # Разделяем каналы
    R = img_float[:, :, 0]
    G = img_float[:, :, 1]
    B = img_float[:, :, 2]

    # Формулы преобразования RGB -> YCbCr
    Y = 0.299 * R + 0.587 * G + 0.114 * B
    Cb = -0.168736 * R - 0.331264 * G + 0.5 * B + 128
    Cr = 0.5 * R - 0.418688 * G - 0.081312 * B + 128

    # Объединяем каналы обратно
    ycbcr_img = np.stack([Y, Cb, Cr], axis=-1)

    # Ограничиваем значения и преобразуем обратно в uint8
    ycbcr_img = np.clip(ycbcr_img, 0, 255).astype(np.uint8)

    return ycbcr_img


def ycbcr2rgb(img):
    """Переход из пр-ва YCbCr в пр-во RGB
    Вход: YCbCr изображение
    Выход: RGB изображение
    """

    # Your code here
    # Преобразуем в float для точных вычислений
    img_float = img.astype(np.float64)

    # Разделяем каналы
    Y = img_float[:, :, 0]
    Cb = img_float[:, :, 1]
    Cr = img_float[:, :, 2]

    # Вычитаем сдвиг 128 из цветовых компонент
    Cb_shifted = Cb - 128
    Cr_shifted = Cr - 128

    # Формулы обратного преобразования YCbCr -> RGB
    R = Y + 1.402 * Cr_shifted
    G = Y - 0.344136 * Cb_shifted - 0.714136 * Cr_shifted
    B = Y + 1.772 * Cb_shifted

    # Объединяем каналы обратно
    rgb_img = np.stack([R, G, B], axis=-1)

    # Ограничиваем значения и преобразуем обратно в uint8
    rgb_img = np.clip(rgb_img, 0, 255).astype(np.uint8)

    return rgb_img


def get_gauss_1():
    plt.clf()
    rgb_img = imread("Lenna.png")
    if len(rgb_img.shape) == 3:
        rgb_img = rgb_img[..., :3]

    # Your code here
    # Переходим в YCbCr пространство
    ycbcr_img = rgb2ycbcr(rgb_img)

    # Размываем только цветовые компоненты Cb и Cr
    ycbcr_img[:, :, 1] = gaussian_filter(ycbcr_img[:, :, 1].astype(np.float64), sigma=10)
    ycbcr_img[:, :, 2] = gaussian_filter(ycbcr_img[:, :, 2].astype(np.float64), sigma=10)

    # Преобразуем обратно в RGB
    result_img = ycbcr2rgb(ycbcr_img)

    # Отображаем результат
    plt.imshow(result_img)
    plt.axis('off')  # Убираем оси для лучшего отображения

    plt.savefig("gauss_1.png")


def get_gauss_2():
    plt.clf()
    rgb_img = imread("Lenna.png")
    if len(rgb_img.shape) == 3:
        rgb_img = rgb_img[..., :3]

    # Your code here
    # Переходим в YCbCr пространство
    ycbcr_img = rgb2ycbcr(rgb_img)

     # Размываем только компоненту яркости Y
    ycbcr_img[:, :, 0] = gaussian_filter(ycbcr_img[:, :, 0].astype(np.float64), sigma=10)

    # Преобразуем обратно в RGB
    result_img = ycbcr2rgb(ycbcr_img)

    # Отображаем результат
    plt.imshow(result_img)
    plt.axis('off')  # Убираем оси для лучшего отображения

    plt.savefig("gauss_2.png")


def downsampling(component):
    """Уменьшаем цветовые компоненты в 2 раза
    Вход: цветовая компонента размера [A, B]
    Выход: цветовая компонента размера [A // 2, B // 2]
    """

    # Your code here
    blurred_component = gaussian_filter(component, sigma=10)

    return blurred_component[::2, ::2]


def dct(block):
    """Дискретное косинусное преобразование
    Вход: блок размера 8x8
    Выход: блок размера 8x8 после ДКП
    """

    M, N = block.shape
    dct_block = np.zeros((M, N))

    for u in range(M):
        for v in range(N):
            # Коэффициенты alpha
            alpha_u = 1.0 / np.sqrt(2) if u == 0 else 1.0
            alpha_v = 1.0 / np.sqrt(2) if v == 0 else 1.0

            # Суммирование по всем пикселям блока
            total = 0.0
            for x in range(M):
                for y in range(N):
                    cos1 = np.cos((2 * x + 1) * u * np.pi / (2 * M))
                    cos2 = np.cos((2 * y + 1) * v * np.pi / (2 * N))
                    total += block[x, y] * cos1 * cos2

            # Нормализующий коэффициент
            dct_block[u, v] = (2.0 / np.sqrt(M * N)) * alpha_u * alpha_v * total

    return dct_block


# Матрица квантования яркости
y_quantization_matrix = np.array(
    [
        [16, 11, 10, 16, 24, 40, 51, 61],
        [12, 12, 14, 19, 26, 58, 60, 55],
        [14, 13, 16, 24, 40, 57, 69, 56],
        [14, 17, 22, 29, 51, 87, 80, 62],
        [18, 22, 37, 56, 68, 109, 103, 77],
        [24, 35, 55, 64, 81, 104, 113, 92],
        [49, 64, 78, 87, 103, 121, 120, 101],
        [72, 92, 95, 98, 112, 100, 103, 99],
    ]
)

# Матрица квантования цвета
color_quantization_matrix = np.array(
    [
        [17, 18, 24, 47, 99, 99, 99, 99],
        [18, 21, 26, 66, 99, 99, 99, 99],
        [24, 26, 56, 99, 99, 99, 99, 99],
        [47, 66, 99, 99, 99, 99, 99, 99],
        [99, 99, 99, 99, 99, 99, 99, 99],
        [99, 99, 99, 99, 99, 99, 99, 99],
        [99, 99, 99, 99, 99, 99, 99, 99],
        [99, 99, 99, 99, 99, 99, 99, 99],
    ]
)


def quantization(block, quantization_matrix):
    """Квантование
    Вход: блок размера 8x8 после применения ДКП; матрица квантования
    Выход: блок размера 8x8 после квантования. Округление осуществляем с помощью np.round
    """

    # Your code here
    # Делим каждый элемент блока на соответствующий элемент матрицы квантования
    quantized_block = block / quantization_matrix

    # Округляем до целых чисел
    quantized_block = np.round(quantized_block)

    return quantized_block


def own_quantization_matrix(default_quantization_matrix, q):
    """Генерация матрицы квантования по Quality Factor
    Вход: "стандартная" матрица квантования; Quality Factor
    Выход: новая матрица квантования
    Hint: если после проделанных операций какие-то элементы обнулились, то замените их единицами
    """

    assert 1 <= q <= 100

    # Your code here
    # Вычисляем Scale Factor S в зависимости от Quality Factor
    if q < 50:
        S = 5000.0 / q
    elif q <= 99:
        S = 200 - 2 * q
    else:  # q == 100
        S = 1

    # Вычисляем новую матрицу квантования
    new_matrix = np.floor((50 + S * default_quantization_matrix) / 100)

    # Заменяем нули единицами (чтобы избежать деления на ноль при квантовании)
    new_matrix[new_matrix == 0] = 1

    return new_matrix


def zigzag(block):
    """Зигзаг-сканирование
    Вход: блок размера 8x8
    Выход: список из элементов входного блока, получаемый после его обхода зигзаг-сканированием
    """

    # Your code here
    n = 8
    result = []

    # Проходим по всем диагоналям
    for diag in range(2 * n - 1):
        if diag % 2 == 0:  # Движение вверх-вправо
            i = min(diag, n - 1)
            j = diag - i
            while i >= 0 and j < n:
                result.append(block[i, j])
                i -= 1
                j += 1
        else:  # Движение вниз-влево
            j = min(diag, n - 1)
            i = diag - j
            while j >= 0 and i < n:
                result.append(block[i, j])
                i += 1
                j -= 1

    return result


def compression(zigzag_list):
    """Сжатие последовательности после зигзаг-сканирования
    Вход: список после зигзаг-сканирования
    Выход: сжатый список в формате, который был приведен в качестве примера в самом начале данного пункта
    """

    # Your code here
    compressed = []
    zero_count = 0

    for value in zigzag_list:
        if abs(value) < 1e-6:  # Проверяем, является ли значение нулем (с учетом погрешности)
            zero_count += 1
        else:
            # Если были нули до этого, добавляем пару (0, количество_нулей)
            if zero_count > 0:
                compressed.extend([0, zero_count])
                zero_count = 0
            # Добавляем ненулевое значение
            compressed.append(value)

    # Добавляем оставшиеся нули в конце последовательности
    if zero_count > 0:
        compressed.extend([0, zero_count])

    return compressed


def jpeg_compression(img, quantization_matrixes):
    """JPEG-сжатие
    Вход: цветная картинка, список из 2-ух матриц квантования
    Выход: список списков со сжатыми векторами: [[compressed_y1,...], [compressed_Cb1,...], [compressed_Cr1,...]]
    """

    # Your code here
    y_quant, color_quant = quantization_matrixes

    # Переходим из RGB в YCbCr
    ycbcr_img = rgb2ycbcr(img)

    # Уменьшаем цветовые компоненты
    Y = ycbcr_img[:, :, 0]  # Яркость - полное разрешение
    Cb = downsampling(ycbcr_img[:, :, 1])  # Цвет - уменьшенное разрешение
    Cr = downsampling(ycbcr_img[:, :, 2])  # Цвет - уменьшенное разрешение

    components = [Y, Cb, Cr]
    quant_matrices = [y_quant, color_quant, color_quant]

    compressed_components = []

    for comp_idx, (component, quant_matrix) in enumerate(zip(components, quant_matrices)):
        h, w = component.shape
        compressed_blocks = []

        # Делим на блоки 8x8 и обрабатываем
        for i in range(0, h, 8):
            for j in range(0, w, 8):
                # Извлекаем блок 8x8
                block = component[i:i + 8, j:j + 8]

                # Если блок меньше 8x8 (на границах), пропускаем
                if block.shape != (8, 8):
                    continue

                # Переводим значения из [0, 255] в [-128, 127]
                block_shifted = block.astype(np.float64) - 128

                # Применяем ДКП
                dct_block = dct(block_shifted)

                # Применяем квантование
                quant_block = quantization(dct_block, quant_matrix)

                # Применяем зигзаг-сканирование
                zigzag_list = zigzag(quant_block)

                # Применяем сжатие
                compressed_block = compression(zigzag_list)

                compressed_blocks.append(compressed_block)

        compressed_components.append(compressed_blocks)

    return compressed_components


def inverse_compression(compressed_list):
    """Разжатие последовательности
    Вход: сжатый список
    Выход: разжатый список
    """

    # Your code here
    result = []
    i = 0

    while i < len(compressed_list):
        if compressed_list[i] == 0 and i + 1 < len(compressed_list):
            # Нашли пару (0, N) - добавляем N нулей
            zero_count = compressed_list[i + 1]
            result.extend([0] * zero_count)
            i += 2  # Пропускаем два элемента (0 и счетчик)
        else:
            # Обычное ненулевое значение
            result.append(compressed_list[i])
            i += 1

    return result



def inverse_zigzag(input):
    """Обратное зигзаг-сканирование
    Вход: список элементов
    Выход: блок размера 8x8 из элементов входного списка, расставленных в матрице в порядке их следования в зигзаг-сканировании
    """

    # Your code here
    n = 8
    block = np.zeros((n, n))
    idx = 0

    # Проходим по всем диагоналям в том же порядке, что и при прямом зигзаг-сканировании
    for diag in range(2 * n - 1):
        if diag % 2 == 0:  # Движение вверх-вправо
            i = min(diag, n - 1)
            j = diag - i
            while i >= 0 and j < n and idx < len(input):
                block[i, j] = input[idx]
                idx += 1
                i -= 1
                j += 1
        else:  # Движение вниз-влево
            j = min(diag, n - 1)
            i = diag - j
            while j >= 0 and i < n and idx < len(input):
                block[i, j] = input[idx]
                idx += 1
                i += 1
                j -= 1

    return block


def inverse_quantization(block, quantization_matrix):
    """Обратное квантование
    Вход: блок размера 8x8 после применения обратного зигзаг-сканирования; матрица квантования
    Выход: блок размера 8x8 после квантования. Округление не производится
    """

    # Your code here
    # Умножаем каждый элемент блока на соответствующий элемент матрицы квантования
    dequantized_block = block * quantization_matrix

    return dequantized_block


def inverse_dct(block):
    """Обратное дискретное косинусное преобразование
    Вход: блок размера 8x8
    Выход: блок размера 8x8 после обратного ДКП. Округление осуществляем с помощью np.round
    """

    # Your code here
    M, N = block.shape
    idct_block = np.zeros((M, N))

    for x in range(M):
        for y in range(N):
            total = 0.0
            for u in range(M):
                alpha_u = 1.0 / np.sqrt(2) if u == 0 else 1.0
                for v in range(N):
                    alpha_v = 1.0 / np.sqrt(2) if v == 0 else 1.0

                    # Косинусные базисные функции
                    cos1 = np.cos((2 * x + 1) * u * np.pi / (2 * M))
                    cos2 = np.cos((2 * y + 1) * v * np.pi / (2 * N))

                    total += alpha_u * alpha_v * block[u, v] * cos1 * cos2

            # Нормализующий коэффициент
            idct_block[x, y] = (2.0 / np.sqrt(M * N)) * total

    # Округляем
    idct_block = np.round(idct_block)
    #idct_block = np.clip(idct_block, 0, 255)

    return idct_block


def upsampling(component):
    """Увеличиваем цветовые компоненты в 2 раза
    Вход: цветовая компонента размера [A, B]
    Выход: цветовая компонента размера [2 * A, 2 * B]
    """

    # Your code here
    h, w = component.shape
    upsampled = np.zeros((2 * h, 2 * w), dtype=component.dtype)

    # Дублируем каждый пиксель в блок 2x2
    for i in range(h):
        for j in range(w):
            value = component[i, j]
            upsampled[2 * i, 2 * j] = value
            upsampled[2 * i, 2 * j + 1] = value
            upsampled[2 * i + 1, 2 * j] = value
            upsampled[2 * i + 1, 2 * j + 1] = value

    return upsampled


def jpeg_decompression(result, result_shape, quantization_matrixes):
    """Разжатие изображения
    Вход: result список сжатых данных, размер ответа, список из 2-ух матриц квантования
    Выход: разжатое изображение
    """

    # Your code here
    y_quant, color_quant = quantization_matrixes
    quant_matrices = [y_quant, color_quant, color_quant]

    # Восстановление компонент
    recovered_components = []

    for comp_idx, compressed_blocks in enumerate(result):
        quant_matrix = quant_matrices[comp_idx]

        # Определяем размер компоненты
        if comp_idx == 0:  # Y компонента - полный размер
            comp_h, comp_w = result_shape[:2]
        else:  # Cb, Cr компоненты - уменьшенный размер
            comp_h, comp_w = result_shape[0] // 2, result_shape[1] // 2

        recovered_component = np.zeros((comp_h, comp_w))
        block_idx = 0

        # Восстановление блоков
        for i in range(0, comp_h - 7, 8):
            for j in range(0, comp_w - 7, 8):
                if block_idx < len(compressed_blocks):
                    # Получаем сжатый блок
                    compressed_block = compressed_blocks[block_idx]

                    # Обратное сжатие
                    decompressed_list = inverse_compression(compressed_block)

                    # Обратное зигзаг-сканирование
                    quant_block = inverse_zigzag(decompressed_list)

                    # Обратное квантование
                    dct_block = inverse_quantization(quant_block, quant_matrix)

                    # Обратное ДКП
                    recovered_block = inverse_dct(dct_block)

                    # Помещаем блок на свое место
                    recovered_component[i:i + 8, j:j + 8] = recovered_block
                    block_idx += 1

        recovered_components.append(recovered_component)

    # Увеличение цветовых компонент
    recovered_components[1] = upsampling(recovered_components[1])  # Cb
    recovered_components[2] = upsampling(recovered_components[2])  # Cr

    # Объединение компонент в YCbCr изображение
    ycbcr_result = np.stack(recovered_components, axis=-1)

    # Преобразование обратно в RGB
    rgb_result = ycbcr2rgb(ycbcr_result)

    return rgb_result


def jpeg_visualize():
    plt.clf()
    img = imread("Lenna.png")
    if len(img.shape) == 3:
        img = img[..., :3]
    fig, axes = plt.subplots(2, 3)
    fig.set_figwidth(12)
    fig.set_figheight(12)

    for i, p in enumerate([1, 10, 20, 50, 80, 100]):
        # Your code here
        # Генерируем матрицы квантования для данного Quality Factor
        y_quant = own_quantization_matrix(y_quantization_matrix, p)
        color_quant = own_quantization_matrix(color_quantization_matrix, p)
        quantization_matrixes = [y_quant, color_quant]

        # Выполняем JPEG сжатие и декомпрессию
        compressed = jpeg_compression(img, quantization_matrixes)
        reconstructed = jpeg_decompression(compressed, img.shape, quantization_matrixes)

        # Отображаем результат
        axes[i // 3, i % 3].imshow(reconstructed)
        axes[i // 3, i % 3].set_title("Quality Factor: {}".format(p))

    fig.savefig("jpeg_visualization.png")


def get_deflated_bytesize(data):
    raw_data = pickle.dumps(data)
    with io.BytesIO() as buf:
        with (
            zipfile.ZipFile(buf, mode="w", compression=zipfile.ZIP_DEFLATED) as zipf,
            zipf.open("data", mode="w") as handle,
        ):
            handle.write(raw_data)
            handle.flush()
            handle.close()
            zipf.close()
        buf.flush()
        return buf.getbuffer().nbytes


def compression_pipeline(img, c_type, param=1):
    """Pipeline для PCA и JPEG
    Вход: исходное изображение; название метода - 'pca', 'jpeg';
    param - кол-во компонент в случае PCA, и Quality Factor для JPEG
    Выход: изображение; количество бит на пиксель
    """

    assert c_type.lower() == "jpeg" or c_type.lower() == "pca"

    if c_type.lower() == "jpeg":
        y_quantization = own_quantization_matrix(y_quantization_matrix, param)
        color_quantization = own_quantization_matrix(color_quantization_matrix, param)
        matrixes = [y_quantization, color_quantization]

        compressed = jpeg_compression(img, matrixes)
        img = jpeg_decompression(compressed, img.shape, matrixes)
        compressed_size = get_deflated_bytesize(compressed)

    elif c_type.lower() == "pca":
        compressed = [
            pca_compression(c.copy(), param)
            for c in img.transpose(2, 0, 1).astype(np.float64)
        ]

        img = pca_decompression(compressed)
        compressed_size = sum(d.nbytes for c in compressed for d in c)

    raw_size = img.nbytes

    return img, compressed_size / raw_size


def calc_metrics(img_path, c_type, param_list):
    """Подсчет PSNR и Compression Ratio для PCA и JPEG. Построение графиков
    Вход: пусть до изображения; тип сжатия; список параметров: кол-во компонент в случае PCA, и Quality Factor для JPEG
    """

    assert c_type.lower() == "jpeg" or c_type.lower() == "pca"

    img = imread(img_path)
    if len(img.shape) == 3:
        img = img[..., :3]

    outputs = []
    for param in param_list:
        outputs.append(compression_pipeline(img.copy(), c_type, param))

    psnr = [peak_signal_noise_ratio(img, output[0]) for output in outputs]
    ratio = [output[1] for output in outputs]

    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2)
    fig.set_figwidth(20)
    fig.set_figheight(5)

    ax1.set_title("Quality Factor vs PSNR for {}".format(c_type.upper()))
    ax1.plot(param_list, psnr, "tab:orange")
    ax1.set_ylim(13, 64)
    ax1.set_xlabel("Quality Factor")
    ax1.set_ylabel("PSNR")

    ax2.set_title("PSNR vs Compression Ratio for {}".format(c_type.upper()))
    ax2.plot(psnr, ratio, "tab:red")
    ax2.set_xlim(13, 30)
    ax2.set_ylim(0, 1)
    ax2.set_xlabel("PSNR")
    ax2.set_ylabel("Compression Ratio")
    return fig


def get_pca_metrics_graph():
    plt.clf()
    fig = calc_metrics("Lenna.png", "pca", [1, 5, 10, 20, 50, 100, 150, 200, 256])
    fig.savefig("pca_metrics_graph.png")


def get_jpeg_metrics_graph():
    plt.clf()
    fig = calc_metrics("Lenna.png", "jpeg", [1, 10, 20, 50, 80, 100])
    fig.savefig("jpeg_metrics_graph.png")


if __name__ == "__main__":
    pca_visualize()
    get_gauss_1()
    get_gauss_2()
    jpeg_visualize()
    get_pca_metrics_graph()
    get_jpeg_metrics_graph()
