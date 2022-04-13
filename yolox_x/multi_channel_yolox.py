import cv2
import numpy as np

from .yolox.visualize import concat_tile

from .config import images, visualization

from pathlib import Path



def main():
    imgs = list(Path(images.source_dir).glob("*.*"))
    imgs = [cv2.imread(img.as_posix()) for img in imgs]
    imgs = [cv2.resize(img, (640, 480)) for img in imgs]


    tile = concat_tile([imgs[:2], imgs[2:]])
    cv2.imwrite(f"{visualization.output_dir}/demo.png", tile)
