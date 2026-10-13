"""辅助工具：通过鼠标点击标定仪表屏幕的 ROI 角点坐标。

用法：
    python get_roi_coordinates.py [image_path]

依次点击左上、右上、右下、左下四个角，终端打印坐标，按任意键退出。
"""

import sys

import cv2


def get_roi_coordinates(image_path: str) -> None:
    """打开图片窗口，鼠标点击打印坐标。"""
    img = cv2.imread(image_path)
    if img is None:
        print(f"未找到图片: {image_path}")
        return

    def click_event(event: int, x: int, y: int, _flags: int, _param: object) -> None:
        if event == cv2.EVENT_LBUTTONDOWN:
            print(f"  点击坐标: X={x}, Y={y}")
            cv2.circle(img, (x, y), 5, (0, 0, 255), -1)
            cv2.imshow("Get ROI", img)

    print("请在弹窗中依次点击屏幕的【左上 → 右上 → 右下 → 左下】四个角点。")
    print("记录终端打印的坐标，按任意键关闭窗口。\n")

    cv2.imshow("Get ROI", img)
    cv2.setMouseCallback("Get ROI", click_event)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "image.jpg"
    get_roi_coordinates(path)
