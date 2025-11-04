import numpy as np


def get_bayer_masks(n_rows, n_cols):
    """
    :param n_rows: `int`, number of rows
    :param n_cols: `int`, number of columns

    :return:
        `np.array` of shape `(n_rows, n_cols, 3)` and dtype `np.bool_`
        containing red, green and blue Bayer masks
    """
    red_mask = np.zeros((n_rows, n_cols), dtype=np.float64)
    green_mask = np.zeros((n_rows, n_cols), dtype=np.float64)
    blue_mask = np.zeros((n_rows, n_cols), dtype=np.float64)

    # RGGB-шаблон
    red_mask[0::2, 1::2] = 1.0
    green_mask[0::2, 0::2] = 1.0
    green_mask[1::2, 1::2] = 1.0
    blue_mask[1::2, 0::2] = 1.0

    masks = np.stack([red_mask, green_mask, blue_mask], axis=-1)
    return masks


def get_colored_img(raw_img):
    """
    :param raw_img:
        `np.array` of shape `(n_rows, n_cols)` and dtype `np.uint8`,
        raw image

    :return:
        `np.array` of shape `(n_rows, n_cols, 3)` and dtype `np.uint8`,
        each channel contains known color values or zeros
        depending on Bayer masks
    """
    n_rows, n_cols = raw_img.shape
    masks = get_bayer_masks(n_rows, n_cols)
    R = masks[..., 0]
    G = masks[..., 1]
    B = masks[..., 2]
    colored = np.zeros((n_rows, n_cols, 3), dtype=np.float64)
    colored[..., 0] = raw_img * R
    colored[..., 1] = raw_img * G
    colored[..., 2] = raw_img * B

    return colored.astype(np.uint8)


def get_raw_img(colored_img):
    """
    :param colored_img:
        `np.array` of shape `(n_rows, n_cols, 3)` and dtype `np.uint8`,
        colored image

    :return:
        `np.array` of shape `(n_rows, n_cols)` and dtype `np.uint8`,
        raw image as captured by camera
    """
    n_rows, n_cols, _ = colored_img.shape
    raw_img = np.zeros((n_rows, n_cols), dtype=np.uint8)

    raw_img[0::2, 0::2] = colored_img[0::2, 0::2, 1]  # Green
    raw_img[0::2, 1::2] = colored_img[0::2, 1::2, 0]  # Red
    raw_img[1::2, 0::2] = colored_img[1::2, 0::2, 2]  # Blue
    raw_img[1::2, 1::2] = colored_img[1::2, 1::2, 1]  # Green

    return raw_img


def bilinear_interpolation(raw_img):
    """
    :param raw_img:
        `np.array` of shape `(n_rows, n_cols)` and dtype `np.uint8`,
        raw image

    :return:
        `np.array` of shape `(n_rows, n_cols, 3)`, and dtype `np.uint8`,
        result of bilinear interpolation
    """
    def box_sum(x):
        P = np.pad(x, ((1, 1), (1, 1)), mode="edge")
        return (
            P[:-2, :-2] + P[:-2, 1:-1] + P[:-2, 2:] +
            P[1:-1, :-2] + P[1:-1, 1:-1] + P[1:-1, 2:] +
            P[2:, :-2] + P[2:, 1:-1] + P[2:, 2:]
        )

    n_rows, n_cols = raw_img.shape
    masks = get_bayer_masks(n_rows, n_cols)
    R = masks[..., 0]
    G = masks[..., 1]
    B = masks[..., 2]
    base = get_colored_img(raw_img).astype(np.uint16)
    result = np.zeros_like(base)

    for c, mask in enumerate([R, G, B]):
        values = base[..., c]
        s = box_sum(values)
        cnt = box_sum(mask)
        avg = s // np.maximum(cnt, 1)
        ch = values.copy()
        ch[mask == 0] = avg[mask == 0]
        result[..., c] = ch

    return np.clip(result, 0, 255).astype(np.uint8)


def improved_interpolation(raw_img: np.ndarray) -> np.ndarray:
    """
    Improved interpolation using edge-aware correction.
    """
    def conv2d(img, kernel):
        n_rows, n_cols = img.shape
        pad = np.pad(img, 2, mode="edge")
        res = np.zeros((n_rows, n_cols), dtype=np.float64)
        for i in range(5):
            for j in range(5):
                if kernel[i, j] != 0:
                    res += kernel[i, j] * pad[i:i + n_rows, j:j + n_cols]
        return res

    def round_half_up(x):
        return np.floor(x + 0.5)

    raw = raw_img.astype(np.float64)
    n_rows, n_cols = raw.shape
    masks = get_bayer_masks(n_rows, n_cols)
    Rm = masks[..., 0]
    Gm = masks[..., 1]
    Bm = masks[..., 2]
    base = get_colored_img(raw_img).astype(np.float64)

    # Свёрточные ядра
    k_G = np.array([
        [0, 0, -1, 0, 0],
        [0, 0, 2, 0, 0],
        [-1, 2, 4, 2, -1],
        [0, 0, 2, 0, 0],
        [0, 0, -1, 0, 0],
    ]) / 8.0

    k_RG_h = np.array([
        [0, 0, 0.5, 0, 0],
        [0, -1, 0, -1, 0],
        [-1, 4, 5, 4, -1],
        [0, -1, 0, -1, 0],
        [0, 0, 0.5, 0, 0],
    ]) / 8.0

    k_RG_v = k_RG_h.T
    k_BG_h, k_BG_v = k_RG_h.copy(), k_RG_v.copy()

    k_RB = np.array([
        [0, 0, -1.5, 0, 0],
        [0, 2, 0, 2, 0],
        [-1.5, 0, 6, 0, -1.5],
        [0, 2, 0, 2, 0],
        [0, 0, -1.5, 0, 0],
    ]) / 8.0
    k_BR = k_RB

    R = base[..., 0].copy()
    G = base[..., 1].copy()
    B = base[..., 2].copy()

    G_from_R = conv2d(raw, k_G)
    G_from_B = conv2d(raw, k_G)
    G[Rm == 1] = round_half_up(G_from_R[Rm == 1])
    G[Bm == 1] = round_half_up(G_from_B[Bm == 1])

    ii, jj = np.indices((n_rows, n_cols))
    G_rb = (Gm == 1) & (ii % 2 == 0) & (jj % 2 == 0)
    G_br = (Gm == 1) & (ii % 2 == 1) & (jj % 2 == 1)

    R_h, R_v = conv2d(raw, k_RG_h), conv2d(raw, k_RG_v)
    R[G_rb] = round_half_up(R_h[G_rb])
    R[G_br] = round_half_up(R_v[G_br])
    R[Bm == 1] = round_half_up(conv2d(raw, k_RB)[Bm == 1])

    B_h, B_v = conv2d(raw, k_BG_h), conv2d(raw, k_BG_v)
    B[G_rb] = round_half_up(B_v[G_rb])
    B[G_br] = round_half_up(B_h[G_br])
    B[Rm == 1] = round_half_up(conv2d(raw, k_BR)[Rm == 1])

    out = np.stack([R, G, B], axis=-1)
    return np.clip(out, 0, 255).astype(np.uint8)


def compute_psnr(img_pred, img_gt):
    if img_pred.shape != img_gt.shape:
        raise ValueError("Input images must have the same dimensions")

    img1 = img_pred.astype(np.float64, copy=False)
    img2 = img_gt.astype(np.float64, copy=False)

    error = np.mean((img1 - img2) ** 2)
    if error == 0:
        raise ValueError("MSE равен нулю")

    peak = img2.max()
    psnr_value = 10 * np.log10(peak ** 2 / error)

    return float(psnr_value)


if __name__ == "__main__":
    from PIL import Image

    raw_img_path = "tests/04_unittest_bilinear_img_input/02.png"
    raw_img = np.array(Image.open(raw_img_path))

    img_bilinear = bilinear_interpolation(raw_img)
    Image.fromarray(img_bilinear).save("bilinear.png")

    img_improved = improved_interpolation(raw_img)
    Image.fromarray(img_improved).save("improved.png")
