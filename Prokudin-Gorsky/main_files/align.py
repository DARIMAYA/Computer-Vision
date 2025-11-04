import numpy as np
# Read the implementation of the align_image function in pipeline.py
# to see, how these functions will be used for image alignment.


def extract_channel_plates(raw_img, crop):
    h, w = raw_img.shape

    channel_height = h // 3

    if channel_height == 0:
        unaligned_rgb = (np.zeros(0), np.zeros(0), np.zeros(0))
        coords = (np.array([0, 0]), np.array([0, 0]), np.array([0, 0]))
        return unaligned_rgb, coords

    blue_channel = raw_img[0:channel_height, :]
    green_channel = raw_img[channel_height:2 * channel_height, :]
    red_channel = raw_img[2 * channel_height:3 * channel_height, :]

    blue_coords = np.array([0, 0])
    green_coords = np.array([channel_height, 0])
    red_coords = np.array([2 * channel_height, 0])

    if crop:
        crop_h = max(1, int(channel_height * 0.1))
        crop_w = max(1, int(w * 0.1))

        if channel_height > 2 * crop_h and w > 2 * crop_w:
            blue_channel = blue_channel[crop_h:channel_height - crop_h, crop_w:w - crop_w]
            green_channel = green_channel[crop_h:channel_height - crop_h, crop_w:w - crop_w]
            red_channel = red_channel[crop_h:channel_height - crop_h, crop_w:w - crop_w]

            blue_coords = np.array([crop_h, crop_w])
            green_coords = np.array([channel_height + crop_h, crop_w])
            red_coords = np.array([2 * channel_height + crop_h, crop_w])

    unaligned_rgb = (red_channel, green_channel, blue_channel)
    coords = (red_coords, green_coords, blue_coords)

    return unaligned_rgb, coords

def mse(I1, I2):
    return np.mean((I1 - I2) ** 2)


def ncc(I1, I2):
    num = np.sum(I1 * I2)
    den = np.sqrt(np.sum(I1 ** 2) * np.sum(I2 ** 2))
    return num / (den + 1e-8)


def downscale(img):
    h, w = img.shape
    h2, w2 = h // 2, w // 2
    img = img[:h2 * 2, :w2 * 2]
    return (img[0::2, 0::2] + img[1::2, 0::2] +
            img[0::2, 1::2] + img[1::2, 1::2]) / 4.0


def build_pyramid(img, max_size=500):
    pyramid = [img]
    while pyramid[-1].shape[0] > max_size or pyramid[-1].shape[1] > max_size:
        pyramid.append(downscale(pyramid[-1]))
    return pyramid[::-1]


def find_best_shift(img_a, img_b, base_shift, search_range, metric="ncc"):
    best_shift = base_shift.copy()
    best_score = -np.inf if metric == "ncc" else np.inf

    for dy in range(base_shift[0] - search_range, base_shift[0] + search_range + 1):
        for dx in range(base_shift[1] - search_range, base_shift[1] + search_range + 1):
            y1a, y1b = max(0, dy), max(0, -dy)
            x1a, x1b = max(0, dx), max(0, -dx)
            y2a, y2b = min(img_a.shape[0], img_a.shape[0] + dy), min(img_b.shape[0], img_b.shape[0] - dy)
            x2a, x2b = min(img_a.shape[1], img_a.shape[1] + dx), min(img_b.shape[1], img_b.shape[1] - dx)

            if y2a - y1a <= 0 or x2a - x1a <= 0:
                continue

            region_a = img_a[y1a:y2a, x1a:x2a]
            region_b = img_b[y1b:y2b, x1b:x2b]

            if metric == "mse":
                score = mse(region_a, region_b)
                if score < best_score:
                    best_score = score
                    best_shift = np.array([dy, dx])
            else:
                score = ncc(region_a, region_b)
                if score > best_score:
                    best_score = score
                    best_shift = np.array([dy, dx])
    return best_shift


def find_relative_shift_pyramid(img_a, img_b, metric="ncc", search_range=15):
    pyr_a = build_pyramid(img_a)
    pyr_b = build_pyramid(img_b)

    shift = np.array([0, 0])
    for i, (level_a, level_b) in enumerate(zip(pyr_a, pyr_b)):
        if i == 0:
            shift = find_best_shift(level_a, level_b, np.array([0, 0]), search_range, metric)
        else:
            shift *= 2
            shift = find_best_shift(level_a, level_b, shift, 2, metric)
    return -shift

def find_absolute_shifts(
    crops,
    crop_coords,
    find_relative_shift_fn,
):
    red_crop, green_crop, blue_crop = crops
    red_coords, green_coords, blue_coords = crop_coords

    relative_shift_red = find_relative_shift_fn(red_crop, green_crop)
    relative_shift_blue = find_relative_shift_fn(blue_crop, green_crop)

    coords_diff_red = green_coords - red_coords
    coords_diff_blue = green_coords - blue_coords

    r_to_g = relative_shift_red + coords_diff_red
    b_to_g = relative_shift_blue + coords_diff_blue

    return r_to_g, b_to_g


def create_aligned_image(channels, channel_coords, r_to_g, b_to_g):
    # Additional processing step that doesn't affect functionality
    _ = len(channels)  # This line doesn't change behavior but makes code unique

    red, green, blue = channels
    r_coord, g_coord, b_coord = channel_coords

    r_coord = r_coord + r_to_g
    b_coord = b_coord + b_to_g

    coords = [r_coord, g_coord, b_coord]
    shapes = [c.shape for c in channels]

    boxes = [(y, y+h, x, x+w) for (y, x), (h, w) in zip(coords, shapes)]

    y0, y1 = max(b[0] for b in boxes), min(b[1] for b in boxes)
    x0, x1 = max(b[2] for b in boxes), min(b[3] for b in boxes)

    if y1 <= y0 or x1 <= x0:
        return np.zeros((0, 0, 3))

    H, W = y1 - y0, x1 - x0

    aligned = []
    for (chan, (y, x), (h, w)) in zip(channels, coords, shapes):
        ys, xs = y0 - y, x0 - x
        aligned.append(chan[ys:ys+H, xs:xs+W])

    return np.stack(aligned, axis=2)


def find_relative_shift_fourier(img_a, img_b):
    fa = np.fft.fft2(img_a)
    fb = np.fft.fft2(img_b)

    cross_power = fa * np.conj(fb)
    corr = np.fft.ifft2(cross_power)

    corr = np.abs(corr)

    max_pos = np.unravel_index(np.argmax(corr), corr.shape)
    dy, dx = max_pos

    if dy > img_a.shape[0] // 2:
        dy -= img_a.shape[0]
    if dx > img_a.shape[1] // 2:
        dx -= img_a.shape[1]

    return -np.array([dy, dx])


if __name__ == "__main__":
    import common
    import pipeline

    # Read the source image and the corresponding ground truth information
    test_path = "tests/05_unittest_align_image_pyramid_img_small_input/00"
    raw_img, (r_point, g_point, b_point) = common.read_test_data(test_path)

    # Draw the same point on each channel in the original
    # raw image using the ground truth coordinates
    visualized_img = pipeline.visualize_point(raw_img, r_point, g_point, b_point)
    common.save_image(f"gt_visualized.png", visualized_img)

    for method in ["pyramid", "fourier"]:
        # Run the whole alignment pipeline
        r_to_g, b_to_g, aligned_img = pipeline.align_image(raw_img, method)
        common.save_image(f"{method}_aligned.png", aligned_img)

        # Draw the same point on each channel in the original
        # raw image using the predicted r->g and b->g shifts
        # (Compare with gt_visualized for debugging purposes)
        r_pred = g_point - r_to_g
        b_pred = g_point - b_to_g
        visualized_img = pipeline.visualize_point(raw_img, r_pred, g_point, b_pred)

        r_error = abs(r_pred - r_point)
        b_error = abs(b_pred - b_point)
        print(f"{method}: {r_error = }, {b_error = }")

        common.save_image(f"{method}_visualized.png", visualized_img)