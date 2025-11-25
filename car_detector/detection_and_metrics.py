import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np


# ============================== 1 Classifier model ============================
def get_cls_model():
    """
    :return: nn model for classification
    """
    # your code here \/
    input_shape = (1, 40, 100)  # (n_channels, n_rows, n_cols)
    classification_model = nn.Sequential(
        nn.Conv2d(1, 32, kernel_size=5, stride=2, padding=2),
        nn.BatchNorm2d(32),
        nn.ReLU(),

        nn.Conv2d(32, 64, kernel_size=5, stride=2, padding=2),
        nn.BatchNorm2d(64),
        nn.ReLU(),

        nn.Conv2d(64, 128, kernel_size=5, stride=2, padding=2),
        nn.BatchNorm2d(128),
        nn.ReLU(),

        nn.Conv2d(128, 256, kernel_size=(5, 13), stride=1, padding=0),
        nn.BatchNorm2d(256),
        nn.ReLU(),

        nn.Flatten(),
        nn.Linear(256, 128),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(128, 2)
    )
    return classification_model
    # your code here /\


def fit_cls_model(X, y, fast_train=True):
    """
    :param X: 4-dim tensor with training images
    :param y: 1-dim tensor with labels for training
    :return: trained nn model
    """
    # your code here \/
    model = get_cls_model()
    # train model
    if not isinstance(X, torch.Tensor):
        X = torch.FloatTensor(X)
    if not isinstance(y, torch.Tensor):
        y = torch.LongTensor(y)

    if fast_train:
        n_epochs = 15
        batch_size = 32
        learning_rate = 0.001
    else:
        n_epochs = 100
        batch_size = 32
        learning_rate = 0.001

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)

    dataset = torch.utils.data.TensorDataset(X, y)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model.train()

    for epoch in range(n_epochs):
        epoch_loss = 0
        correct = 0
        total = 0

        for batch_X, batch_y in dataloader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += batch_y.size(0)
            correct += (predicted == batch_y).sum().item()

        scheduler.step()

    model.eval()
    return model
    # your code here /\


# ============================ 2 Classifier -> FCN =============================
def get_detection_model(cls_model):
    """
    :param cls_model: trained cls model
    :return: fully convolutional nn model with weights initialized from cls
             model
    """
    # your code here \/
    cls_model.eval()

    # Находим индекс Flatten слоя
    flatten_idx = None
    for i, layer in enumerate(cls_model):
        if isinstance(layer, nn.Flatten):
            flatten_idx = i
            break

    # Вычисляем размер фичей до Flatten
    with torch.no_grad():
        dummy_input = torch.zeros(1, 1, 40, 100)
        x = dummy_input
        for i in range(flatten_idx):
            x = cls_model[i](x)
        _, Cin, H, W = x.shape

    detection_layers = []

    # Копируем все слои до Flatten
    for i in range(flatten_idx):
        detection_layers.append(cls_model[i])

    # Заменяем Flatten + Linear на Conv2d
    first_linear = cls_model[flatten_idx + 1]
    Cout = first_linear.out_features

    conv_layer = nn.Conv2d(Cin, Cout, kernel_size=(H, W))
    conv_layer.weight.data = first_linear.weight.data.view(Cout, Cin, H, W)
    conv_layer.bias.data = first_linear.bias.data.clone()
    detection_layers.append(conv_layer)
    detection_layers.append(nn.ReLU())

    # Обрабатываем остальные слои после первого Linear
    for i in range(flatten_idx + 2, len(cls_model)):
        layer = cls_model[i]
        if isinstance(layer, nn.Linear):
            # Заменяем Linear на Conv2d 1x1
            conv = nn.Conv2d(layer.in_features, layer.out_features, kernel_size=1)
            conv.weight.data = layer.weight.data.view(layer.out_features, layer.in_features, 1, 1)
            conv.bias.data = layer.bias.data.clone()
            detection_layers.append(conv)
        elif isinstance(layer, nn.Dropout):
            # Пропускаем Dropout в детекторе
            continue
        else:
            detection_layers.append(layer)

    detection_model = nn.Sequential(*detection_layers)
    detection_model.eval()
    return detection_model
    # your code here /\


# ============================ 3 Simple detector ===============================
def get_detections(detection_model, dictionary_of_images):
    """
    :param detection_model: trained fully convolutional detector model
    :param dictionary_of_images: dictionary of images in format
        {filename: ndarray}
    :return: detections in format {filename: detections}. detections is a N x 5
        array, where N is number of detections. Each detection is described
        using 5 numbers: [row, col, n_rows, n_cols, confidence].
    """
    # your code here \/
    detections = {}
    detection_model.eval()

    det_h, det_w = 40, 100
    stride = 8

    with torch.no_grad():
        for filename, image in dictionary_of_images.items():
            if isinstance(image, np.ndarray):
                if image.ndim == 2:
                    orig_h, orig_w = image.shape
                    img_tensor = torch.FloatTensor(image).unsqueeze(0).unsqueeze(0)
                elif image.ndim == 3:
                    orig_h, orig_w = image.shape[:2]
                    if image.shape[2] == 3:
                        gray = 0.299 * image[:, :, 0] + 0.587 * image[:, :, 1] + 0.114 * image[:, :, 2]
                    else:
                        gray = image[:, :, 0]
                    img_tensor = torch.FloatTensor(gray).unsqueeze(0).unsqueeze(0)
                else:
                    raise ValueError(f"Unexpected image shape: {image.shape}")
            else:
                img_tensor = image
                if img_tensor.dim() == 2:
                    orig_h, orig_w = img_tensor.shape
                    img_tensor = img_tensor.unsqueeze(0).unsqueeze(0)
                else:
                    orig_h, orig_w = img_tensor.shape[-2:]
                    if img_tensor.dim() == 3:
                        img_tensor = img_tensor.unsqueeze(0)

            output = detection_model(img_tensor.float())

            logits = output[0]
            probs = torch.softmax(logits, dim=0)
            car_probs = probs[1]

            h_feat, w_feat = car_probs.shape

            det_list = []
            for i in range(h_feat):
                for j in range(w_feat):
                    conf = car_probs[i, j].item()
                    row = i * stride
                    col = j * stride

                    if row + det_h <= orig_h and col + det_w <= orig_w:
                        det_list.append([row, col, det_h, det_w, conf])

            detections[filename] = np.array(det_list) if det_list else np.zeros((0, 5))

    return detections
    # your code here /\


# =============================== 5 IoU ========================================
def calc_iou(first_bbox, second_bbox):
    """
    :param first bbox: bbox in format (row, col, n_rows, n_cols)
    :param second_bbox: bbox in format (row, col, n_rows, n_cols)
    :return: iou measure for two given bboxes
    """
    # your code here \/
    r1, c1, h1, w1 = first_bbox
    r2, c2, h2, w2 = second_bbox

    r_inter_start = max(r1, r2)
    c_inter_start = max(c1, c2)
    r_inter_end = min(r1 + h1, r2 + h2)
    c_inter_end = min(c1 + w1, c2 + w2)

    if r_inter_start < r_inter_end and c_inter_start < c_inter_end:
        intersection = (r_inter_end - r_inter_start) * (c_inter_end - c_inter_start)
    else:
        intersection = 0

    area1 = h1 * w1
    area2 = h2 * w2
    union = area1 + area2 - intersection

    if union == 0:
        return 0.0
    return intersection / union
    # your code here /\


# =============================== 6 AUC ========================================
def calc_auc(pred_bboxes, gt_bboxes):
    """
    :param pred_bboxes: dict of bboxes in format {filename: detections}
        detections is a N x 5 array, where N is number of detections. Each
        detection is described using 5 numbers: [row, col, n_rows, n_cols,
        confidence].
    :param gt_bboxes: dict of bboxes in format {filenames: bboxes}. bboxes is a
        list of tuples in format (row, col, n_rows, n_cols)
    :return: auc measure for given detections and gt
    """
    # your code here \/
    iou_thr = 0.5

    all_detections = []
    total_gt = 0

    for filename in pred_bboxes:
        preds = pred_bboxes[filename]
        gts = list(gt_bboxes.get(filename, []))

        total_gt += len(gts)

        if len(preds) == 0:
            continue

        sorted_indices = np.argsort(-preds[:, 4])
        preds_sorted = preds[sorted_indices]

        gt_matched = [False] * len(gts)

        for pred in preds_sorted:
            pred_bbox = pred[:4]
            conf = pred[4]

            best_iou = 0
            best_idx = -1

            for i, gt in enumerate(gts):
                if gt_matched[i]:
                    continue
                iou = calc_iou(pred_bbox, gt)
                if iou > best_iou:
                    best_iou = iou
                    best_idx = i

            if best_iou >= iou_thr:
                all_detections.append((conf, True))
                gt_matched[best_idx] = True
            else:
                all_detections.append((conf, False))

    if total_gt == 0 or len(all_detections) == 0:
        return 0.0

    all_detections.sort(key=lambda x: x[0], reverse=True)

    pr_curve = [(0, 1)]
    tp_count = 0
    fp_count = 0

    i = 0
    while i < len(all_detections):
        current_conf = all_detections[i][0]

        while i < len(all_detections) and all_detections[i][0] == current_conf:
            if all_detections[i][1]:
                tp_count += 1
            else:
                fp_count += 1
            i += 1

        recall = tp_count / total_gt
        precision = tp_count / (tp_count + fp_count)

        pr_curve.append((recall, precision))

    auc = 0.0
    for i in range(len(pr_curve) - 1):
        recall1, precision1 = pr_curve[i]
        recall2, precision2 = pr_curve[i + 1]

        width = recall2 - recall1
        height = (precision1 + precision2) / 2
        auc += width * height

    return auc
    # your code here /\


# =============================== 7 NMS ========================================
def nms(detections_dictionary, iou_thr=0.3):
    """
    :param detections_dictionary: dict of bboxes in format {filename: detections}
        detections is a N x 5 array, where N is number of detections. Each
        detection is described using 5 numbers: [row, col, n_rows, n_cols,
        confidence].
    :param iou_thr: IoU threshold for nearby detections
    :return: dict in same format as detections_dictionary where close detections
        are deleted
    """
    # your code here \/
    result = {}

    for filename, detections in detections_dictionary.items():
        if isinstance(detections, list):
            if len(detections) == 0:
                result[filename] = np.zeros((0, 5))
                continue
            detections = np.array(detections)

        if len(detections) == 0:
            result[filename] = detections
            continue

        sorted_indices = np.argsort(-detections[:, 4])
        detections_sorted = detections[sorted_indices]

        keep = []
        suppressed = [False] * len(detections_sorted)

        for i in range(len(detections_sorted)):
            if suppressed[i]:
                continue

            keep.append(detections_sorted[i])

            for j in range(i + 1, len(detections_sorted)):
                if suppressed[j]:
                    continue

                iou = calc_iou(detections_sorted[i][:4], detections_sorted[j][:4])
                if iou > iou_thr:
                    suppressed[j] = True

        result[filename] = np.array(keep) if keep else np.zeros((0, 5))

    return result
    # your code here /\