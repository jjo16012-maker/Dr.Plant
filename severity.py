import torch
import cv2
import numpy as np
from torchvision import transforms
from unet_model import ResNetUNet


# --------------------
# 고정 설정값
# --------------------
RESIZE = 256
THRESHOLD = 0.45
SHADOW_V_THRESHOLD = 40
LOWER_BROWN = np.array([10, 50, 50])
UPPER_BROWN = np.array([30, 255, 200])
BORDER_RATIO = 0.15


# --------------------
# 1) 모델 로드 (서버 구동 시 1번만)
# --------------------
device = torch.device("cpu")
seg_model = ResNetUNet(n_class=1)
seg_model.load_state_dict(torch.load("latest_weights.pth", map_location=device))
seg_model.eval()

to_tensor = transforms.ToTensor()


# --------------------
# 2) FastAPI에서 호출되는 함수
# --------------------
def calc_severity(img):
    """
    img: OpenCV BGR 이미지
    return: (progress %, level)
    """

    H, W = img.shape[:2]

    # ---------- PlantU-Net Segmentation ----------
    img_resized = cv2.resize(img, (RESIZE, RESIZE))
    tensor = to_tensor(img_resized).unsqueeze(0)

    with torch.no_grad():
        pred = seg_model(tensor.to(device))
        pred = torch.sigmoid(pred)

    mask_prob = pred.squeeze().cpu().numpy()
    mask_original = (mask_prob > THRESHOLD).astype(np.uint8)
    mask_original = cv2.resize(mask_original, (W, H), interpolation=cv2.INTER_NEAREST)

    mask_refined = mask_original.copy()

    # ---------- GrabCut Refine ----------
    grabcut_mask = np.zeros((H, W), np.uint8)
    grabcut_mask[mask_refined == 1] = cv2.GC_PR_FGD
    grabcut_mask[mask_refined == 0] = cv2.GC_PR_BGD

    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)

    cv2.grabCut(img, grabcut_mask, None, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_MASK)
    mask_refined = np.where(
        (grabcut_mask == 2) | (grabcut_mask == 0), 0, 1
    ).astype(np.uint8)

    # ---------- Shadow Handling ----------
    dist_transform = cv2.distanceTransform(mask_original, cv2.DIST_L2, 5)
    max_dist = dist_transform.max() if dist_transform.max() > 0 else 1.0

    border_mask = dist_transform < (BORDER_RATIO * max_dist)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    v_channel = hsv[:, :, 2]

    mask_refined[(v_channel < SHADOW_V_THRESHOLD) & border_mask] = 0

    internal_mask = dist_transform >= (BORDER_RATIO * max_dist)
    mask_refined[internal_mask] = 1

    # ---------- Disease Color Mask ----------
    hsv_mask = cv2.inRange(hsv, LOWER_BROWN, UPPER_BROWN)

    mask_refined[(hsv_mask > 0) & internal_mask] = 1
    mask_refined[(hsv_mask > 0) & border_mask] = 0

    # ---------- Neutral Background Removal ----------
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mask_refined[gray > 220] = 0

    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    L, A, B = cv2.split(lab)
    neutral_mask = ((A > 120) & (A < 135) & (B > 120) & (B < 135))
    mask_refined[neutral_mask] = 0

    # ---------- Morphology ----------
    kernel = np.ones((5, 5), np.uint8)
    mask_refined = cv2.morphologyEx(mask_refined, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask_refined = cv2.morphologyEx(mask_refined, cv2.MORPH_OPEN, kernel, iterations=1)

    # ---------- Largest Component ----------
    num_labels, labels = cv2.connectedComponents(mask_refined)
    if num_labels > 1:
        max_area = 0
        max_label = 1
        for label in range(1, num_labels):
            area = np.sum(labels == label)
            if area > max_area:
                max_area = area
                max_label = label
        mask_clean = (labels == max_label).astype(np.uint8)
    else:
        mask_clean = mask_refined.copy()

    # ---------- Disease Area ----------
    leaf_mask_bool = mask_clean.astype(bool)
    disease_mask = cv2.inRange(hsv, LOWER_BROWN, UPPER_BROWN)
    disease_mask_leaf = cv2.bitwise_and(
        disease_mask, disease_mask, mask=leaf_mask_bool.astype(np.uint8)
    )

    leaf_area = np.sum(leaf_mask_bool)
    disease_area = np.sum(disease_mask_leaf > 0)
    progress = (disease_area / leaf_area * 100) if leaf_area > 0 else 0.0

    # ---------- Level Rule ----------
    # 0~5%  → Level 0 (사실상 정상 / 처방 불필요)
    # 5~20% → Level 1
    # 20~40% → Level 2
    # 40~60% → Level 3
    # 60~80% → Level 4
    # 80~100% → Level 5
    if progress <= 5:
        level = 0
    elif progress <= 20:
        level = 1
    elif progress <= 40:
        level = 2
    elif progress <= 60:
        level = 3
    elif progress <= 80:
        level = 4
    else:
        level = 5

    return disease_area, progress, level
