"""视觉抄表 — LCD 七段数码管读数识别

流程：透视矫正 → 二值化 → 斜体扶正 → 连通域分割 → 逐位七段识别 → 小数点检测
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Tuple

import cv2
import numpy as np

# ──────────────────────────── 配置 ────────────────────────────


@dataclass(frozen=True)
class Config:
    """仪表标定参数，换表时仅需修改此处。"""

    # 透视矫正源点（左上、右上、右下、左下）
    pts_src: np.ndarray = field(default_factory=lambda: np.array(
        [[647, 306], [1168, 307], [1159, 518], [653, 511]],
        dtype=np.float32,
    ))

    # 矫正后目标尺寸 (width, height)
    warp_size: Tuple[int, int] = (520, 210)

    # 预处理
    blur_kernel: Tuple[int, int] = (5, 5)
    adapt_thresh_block: int = 15
    adapt_thresh_c: int = 4
    morph_kernel: Tuple[int, int] = (2, 2)

    # 斜体剪切因子
    global_shear: float = 0.06

    # 数字区域在二值图中的比例范围 (x0, x1, y0, y1)
    digit_region: Tuple[float, float, float, float] = (0.30, 0.93, 0.30, 0.70)

    # 连通域膨胀核 (w, h)
    dilate_kernel: Tuple[int, int] = (3, 5)

    # 七段识别阈值
    on_threshold: float = 0.6
    narrow_ratio: float = 0.22  # 宽高比小于此值判为 '1'


DEFAULT_CONFIG = Config()


# ──────────────────────────── 预处理 ────────────────────────────


def process_lcd_screen(
    frame: np.ndarray,
    cfg: Config = DEFAULT_CONFIG,
) -> Tuple[np.ndarray, np.ndarray]:
    """透视矫正 + 二值化，返回 (矫正原图, 二值图)。"""
    w, h = cfg.warp_size

    pts_dst = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32)
    matrix = cv2.getPerspectiveTransform(cfg.pts_src, pts_dst)
    warped = cv2.warpPerspective(frame, matrix, (w, h))

    gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, cfg.blur_kernel, 0)
    thresh = cv2.adaptiveThreshold(
        blurred, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        cfg.adapt_thresh_block, cfg.adapt_thresh_c,
    )
    kernel = np.ones(cfg.morph_kernel, np.uint8)
    processed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    return warped, processed


# ──────────────────────── 七段编码表 ────────────────────────

# 段顺序: (a上, b右上, c右下, d下, e左下, f左上, g中)
SEGMENT_PATTERNS: dict[tuple[int, ...], str] = {
    (1, 1, 1, 1, 1, 1, 0): '0',
    (0, 1, 1, 0, 0, 0, 0): '1',
    (1, 1, 0, 1, 1, 0, 1): '2',
    (1, 1, 1, 1, 0, 0, 1): '3',
    (0, 1, 1, 0, 0, 1, 1): '4',
    (1, 0, 1, 1, 0, 1, 1): '5',
    (1, 0, 1, 1, 1, 1, 1): '6',
    (1, 1, 1, 0, 0, 0, 0): '7',
    (1, 1, 1, 1, 1, 1, 1): '8',
    (1, 1, 1, 1, 0, 1, 1): '9',
}


# ──────────────────────── 工具函数 ────────────────────────


def _shear(img: np.ndarray, s: float) -> np.ndarray:
    """沿垂直方向施加剪切 x' = x + s*(y - h/2)，把斜体数字整体拉直。"""
    h, w = img.shape
    M = np.float32([[1, s, -s * h / 2.0], [0, 1, 0]])
    return cv2.warpAffine(img, M, (w, h))


def _cover(
    crop: np.ndarray,
    y1f: float, y2f: float,
    x1f: float, x2f: float,
    axis: int,
) -> float:
    """采样带覆盖率。

    axis=0 → 列覆盖率（用于横向段：真横段横向铺满，斜笔只经过少量列）
    axis=1 → 行覆盖率（用于纵向段：真竖段纵向铺满，横段只经过少量行）
    """
    ch, cw = crop.shape
    ya, yb = int(ch * y1f), max(int(ch * y1f) + 1, int(ch * y2f))
    xa, xb = int(cw * x1f), max(int(cw * x1f) + 1, int(cw * x2f))
    band = crop[ya:yb, xa:xb] > 0
    if band.size == 0:
        return 0.0
    return float(band.any(axis=axis).mean())


# ──────────────────── 单字符七段识别 ────────────────────


def match_seven_segment_digit(
    digit_crop: np.ndarray,
    on_threshold: float = DEFAULT_CONFIG.on_threshold,
    narrow_ratio: float = DEFAULT_CONFIG.narrow_ratio,
) -> str:
    """识别单个七段数码管数字（传入已 _shear 扶正的二值图）。

    1) 窄字（宽高比极小）直接判为 '1'；
    2) 按外接框采样 7 个段区域，用行/列覆盖率判断亮灭，映射到 0-9；
    3) 汉明距离容错（最多 1 段误差）。
    """
    ch, cw = digit_crop.shape
    if ch == 0 or cw == 0:
        return '?'

    if cw / float(ch) < narrow_ratio:
        return '1'

    # 水平段 (axis=0: 列覆盖率)
    a = _cover(digit_crop, 0.00, 0.12, 0.15, 0.85, axis=0) > on_threshold
    g = _cover(digit_crop, 0.45, 0.55, 0.15, 0.85, axis=0) > on_threshold
    d = _cover(digit_crop, 0.88, 1.00, 0.15, 0.85, axis=0) > on_threshold
    # 垂直段 (axis=1: 行覆盖率)
    f = _cover(digit_crop, 0.18, 0.42, 0.00, 0.30, axis=1) > on_threshold
    b = _cover(digit_crop, 0.18, 0.42, 0.70, 1.00, axis=1) > on_threshold
    e = _cover(digit_crop, 0.58, 0.82, 0.00, 0.30, axis=1) > on_threshold
    c = _cover(digit_crop, 0.58, 0.82, 0.70, 1.00, axis=1) > on_threshold

    pattern = (int(a), int(b), int(c), int(d), int(e), int(f), int(g))
    if pattern in SEGMENT_PATTERNS:
        return SEGMENT_PATTERNS[pattern]

    # 容错：取汉明距离最近的合法字形
    best_char, best_dist = '?', 99
    for p, val in SEGMENT_PATTERNS.items():
        dist = sum(1 for i in range(7) if p[i] != pattern[i])
        if dist < best_dist:
            best_dist, best_char = dist, val
    return best_char if best_dist <= 1 else '?'


# ──────────────────── 数字区域分割与识别 ────────────────────


def _find_digit_boxes(
    merged: np.ndarray,
    da_h: int, da_w: int,
) -> list[tuple[int, int, int, int]]:
    """从膨胀后的连通域中提取数字外接框 (x, y, w, h)。"""
    contours, _ = cv2.findContours(
        merged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE,
    )
    min_digit_h = da_h * 0.5
    boxes: list[tuple[int, int, int, int]] = []
    for c in contours:
        x, y, cw, ch = cv2.boundingRect(c)
        if cw * ch < 4:
            continue
        if ch >= min_digit_h and cw >= da_w * 0.02:
            boxes.append((x, y, cw, ch))
    boxes.sort(key=lambda b: b[0])
    return boxes


def _find_dot_boxes(
    digit_area: np.ndarray,
    da_h: int, da_w: int,
) -> list[tuple[int, int, int, int]]:
    """在未膨胀原图上检测小数点（紧凑小方块）。"""
    contours, _ = cv2.findContours(
        digit_area, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE,
    )
    dots: list[tuple[int, int, int, int]] = []
    for c in contours:
        x, y, cw, ch = cv2.boundingRect(c)
        area = int(cv2.countNonZero(digit_area[y:y + ch, x:x + cw]))
        if area < 10 or cw * ch == 0:
            continue
        fill = area / float(cw * ch)
        # 靠近基线、尺寸小、接近正方形且填充率高 → 小数点
        if (cw <= da_w * 0.06 and ch <= da_h * 0.25 and y > da_h * 0.5
                and 0.5 <= cw / float(ch) <= 2.0 and fill > 0.5):
            dots.append((x, y, cw, ch))
    return dots


def segment_and_recognize_digits(
    binary_img: np.ndarray,
    cfg: Config = DEFAULT_CONFIG,
) -> Tuple[np.ndarray, str]:
    """对 LCD 二值图执行数字分割 + 七段识别。

    返回 (调试标注图, 识别字符串)。
    """
    h, w = binary_img.shape
    rx0, rx1, ry0, ry1 = cfg.digit_region
    x0, x1 = int(w * rx0), int(w * rx1)
    y0, y1 = int(h * ry0), int(h * ry1)

    # 斜体扶正
    digit_area = _shear(binary_img[y0:y1, x0:x1], cfg.global_shear)
    da_h, da_w = digit_area.shape

    # 竖向膨胀连通笔画
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, cfg.dilate_kernel)
    merged = cv2.dilate(digit_area, kernel, iterations=1)

    # 提取数字框与小数点框
    digit_boxes = _find_digit_boxes(merged, da_h, da_w)
    dot_boxes = _find_dot_boxes(digit_area, da_h, da_w)

    # 小数点必须落在首/末数字之间，排除右侧 ¥ 等干扰
    if digit_boxes:
        span_left = digit_boxes[0][0]
        span_right = digit_boxes[-1][0] + digit_boxes[-1][2]
    else:
        span_left, span_right = 0, da_w

    # 逐位识别
    items: list[tuple[float, str]] = []
    debug_boxes: list[tuple[int, int, int, int, str, tuple]] = []

    for x, y, cw, ch in digit_boxes:
        crop = digit_area[y:y + ch, x:x + cw]
        char = match_seven_segment_digit(crop, cfg.on_threshold, cfg.narrow_ratio)
        items.append((x + cw / 2.0, char))
        debug_boxes.append((x, y, cw, ch, char, (0, 255, 0)))

    for x, y, cw, ch in dot_boxes:
        cx = x + cw / 2.0
        if span_left <= cx <= span_right:
            items.append((cx, '.'))
            debug_boxes.append((x, y, cw, ch, '.', (255, 0, 0)))

    # 按水平位置排序组装
    items.sort(key=lambda t: t[0])
    recognized = ''.join(ch for _, ch in items)

    # 调试标注图
    debug_img = cv2.cvtColor(digit_area, cv2.COLOR_GRAY2BGR)
    for bx, by, bw, bh, label, color in debug_boxes:
        cv2.rectangle(debug_img, (bx, by), (bx + bw, by + bh), color, 1)
        cv2.putText(debug_img, label, (bx, max(0, by - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    return debug_img, recognized


# ──────────────────────────── 入口 ────────────────────────────


def main() -> None:
    """读取图片、执行识别、展示结果。"""
    img = cv2.imread("image.jpg")
    if img is None:
        print("未找到 image.jpg，请确认测试图片存在于项目根目录。")
        return

    warped_img, binary_img = process_lcd_screen(img)
    debug_box_img, recognized_text = segment_and_recognize_digits(binary_img)

    # 清理首尾误识别的干扰字符，仅保留数字与小数点
    cleaned = ''.join(c for c in recognized_text if c in '0123456789.')
    print(f"原始识别字符串: {recognized_text}")
    print(f"当前识别出的完整数值为: {cleaned}")

    cv2.imwrite("dbg_warped.png", warped_img)
    cv2.imwrite("dbg_binary.png", binary_img)
    cv2.imwrite("dbg_result.png", debug_box_img)

    cv2.imshow("Segmentation & Recognition", debug_box_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
