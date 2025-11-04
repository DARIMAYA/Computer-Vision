import numpy as np


def get_bayer_masks(n_rows, n_cols):
    """
    :param n_rows: `int`, number of rows
    :param n_cols: `int`, number of columns

    :return:
        `np.array` of shape `(n_rows, n_cols, 3)` and dtype `np.bool_`
        containing red, green and blue Bayer masks
    """
    r = np.zeros((n_rows, n_cols), dtype=np.bool_)
    g = np.zeros((n_rows, n_cols), dtype=np.bool_)
    b = np.zeros((n_rows, n_cols), dtype=np.bool_)

    g[0::2, 0::2] = True
    g[1::2, 1::2] = True
    r[0::2, 1::2] = True
    b[1::2, 0::2] = True

    return np.dstack((r, g, b))


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
    heigth, width = raw_img.shape
    masks = get_bayer_masks(heigth, width)
    colored = raw_img[..., None] * masks.astype(np.uint8)
    return colored


def get_raw_img(colored_img):
    """
    :param colored_img:
        `np.array` of shape `(n_rows, n_cols, 3)` and dtype `np.uint8`,
        colored image

    :return:
        `np.array` of shape `(n_rows, n_cols)` and dtype `np.uint8`,
        raw image as captured by camera
    """
    heigth, width, _ = colored_img.shape
    masks = get_bayer_masks(heigth, width)
    raw = np.zeros((heigth, width), dtype=np.uint8)

    r, g, b = colored_img[..., 0], colored_img[..., 1], colored_img[..., 2]
    raw[masks[..., 0]] = r[masks[..., 0]]
    raw[masks[..., 1]] = g[masks[..., 1]]
    raw[masks[..., 2]] = b[masks[..., 2]]

    return raw


def bilinear_interpolation(raw_img):
    """
    :param raw_img:
        `np.array` of shape `(n_rows, n_cols)` and dtype `np.uint8`,
        raw image

    :return:
        `np.array` of shape `(n_rows, n_cols, 3)`, and dtype `np.uint8`,
        result of bilinear interpolation
    """
    def box_sum(arr):
        P = np.pad(arr, ((1, 1), (1, 1)), mode='constant')
        return (
            P[0:-2, 0:-2] + P[0:-2, 1:-1] + P[0:-2, 2:] +
            P[1:-1, 0:-2] + P[1:-1, 1:-1] + P[1:-1, 2:] +
            P[2:,   0:-2] + P[2:,   1:-1] + P[2:,   2:]
        )
    
    heigth, width = raw_img.shape
    masks = get_bayer_masks(heigth, width)
    colored = get_colored_img(raw_img).astype(np.uint16)
    out = np.empty((heigth, width, 3), dtype=np.uint16)

    for c in range(3):
        known = masks[..., c]
        vals  = colored[..., c]

        s = box_sum(vals)
        cnt = box_sum(known.astype(np.uint16))

        avg = s // np.maximum(cnt, 1)

        ch = vals.copy()
        ch[~known] = avg[~known]
        out[..., c] = ch

    return out.astype(np.uint8)


def improved_interpolation(raw_img):
    """
    :param raw_img:
        `np.array` of shape `(n_rows, n_cols)` and dtype `np.uint8`, raw image

    :return:
        `np.array` of shape `(n_rows, n_cols, 3)` and dtype `np.uint8`,
        result of improved interpolation
    """
    def convolution(img, kernel):
    
        heigth, width = img.shape
        p = np.pad(img, 2, mode='edge')
        out = np.zeros((heigth, width), dtype=np.float64)

        for dy in range(5):
            row = p[dy:dy+heigth]
            for dx in range(5):
                w = kernel[dy, dx]
                if w != 0.0:
                    out += w * row[:, dx:dx+width]
        return out
    
    def half_up(x):
        return np.floor(x + 0.5)
    
    raw = raw_img.astype(np.float64, copy=False)
    heigth, width = raw.shape
    masks = get_bayer_masks(heigth, width)
    mR, mG, mB = masks[..., 0], masks[..., 1], masks[..., 2]
    base = get_colored_img(raw_img).astype(np.float64)

    k_G_at_R = (np.array([
    [0,  0, -1,  0, 0],
    [0,  0,  2,  0, 0],
    [-1, 2,  4,  2,-1],
    [0,  0,  2,  0, 0],
    [0,  0, -1,  0, 0],
    ], dtype=np.float64) / 8.0)
    k_G_at_B = k_G_at_R

    k_R_at_G_h = (np.array([
    [0,  0,  0.5, 0,  0],
    [0, -1,  0,  -1, 0],
    [-1, 4,  5,   4, -1],
    [0, -1,  0,  -1, 0],
    [0,  0,  0.5, 0,  0],
    ], dtype=np.float64) / 8.0)
    k_R_at_G_v = k_R_at_G_h.T
    k_B_at_G_h = k_R_at_G_h
    k_B_at_G_v = k_R_at_G_v

    k_R_at_B = (np.array([
    [ 0,   0, -1.5, 0,   0],
    [ 0,   2,  0,   2,   0],
    [-1.5, 0,  6,   0, -1.5],
    [ 0,   2,  0,   2,   0],
    [ 0,   0, -1.5, 0,   0],
    ], dtype=np.float64) / 8.0)
    k_B_at_R = k_R_at_B

    R = base[..., 0].copy()
    G = base[..., 1].copy()
    B = base[..., 2].copy()

    G_from_R = convolution(raw, k_G_at_R)
    G_from_B = convolution(raw, k_G_at_B)
    G[mR] = half_up(G_from_R[mR])
    G[mB] = half_up(G_from_B[mB])

    ii = np.arange(heigth)[:, None]
    jj = np.arange(width)[None, :]
    G_rb = mG & ((ii % 2 == 0) & (jj % 2 == 0))
    G_br = mG & ((ii % 2 == 1) & (jj % 2 == 1))

    R_from_G_h = convolution(raw, k_R_at_G_h)
    R_from_G_v = convolution(raw, k_R_at_G_v)
    R[G_rb] = half_up(R_from_G_h[G_rb])
    R[G_br] = half_up(R_from_G_v[G_br])

    R_from_B = convolution(raw, k_R_at_B)
    R[mB] = half_up(R_from_B[mB])

    B_from_G_h = convolution(raw, k_B_at_G_h)
    B_from_G_v = convolution(raw, k_B_at_G_v)
    B[G_rb] = half_up(B_from_G_v[G_rb])
    B[G_br] = half_up(B_from_G_h[G_br])

    B_from_R = convolution(raw, k_B_at_R)
    B[mR] = half_up(B_from_R[mR])

    out = np.stack([R, G, B], axis=-1)
    np.clip(out, 0, 255, out=out)
    return out.astype(np.uint8)


def compute_psnr(img_pred, img_gt):
    """
    :param img_pred:
        `np.array` of shape `(n_rows, n_cols, 3)` and dtype `np.uint8`,
        predicted image
    :param img_gt:
        `np.array` of shape `(n_rows, n_cols, 3)` and dtype `np.uint8`,
        ground truth image

    :return:
        `float`, PSNR metric
    """
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
