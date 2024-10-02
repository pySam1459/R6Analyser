import cv2
import numpy as np
from typing import Sequence, cast


class TemplateMatcher:
    def __init__(self, template: np.ndarray) -> None:
        self.template = template

        self.__sift = cv2.SIFT.create()
        self.template_kp, self.template_desc = self.__sift_DaC(template)

        self.__flann = cv2.FlannBasedMatcher(indexParams=dict(algorithm=1, trees=5), ## KDTree
                                             searchParams=dict(checks=50))
    
    def set_template(self, new_template: np.ndarray) -> None:
        self.template = new_template
        self.template_kp, self.template_desc = self.__sift_DaC(new_template)
    
    def __sift_DaC(self, img: np.ndarray) -> tuple[Sequence[cv2.KeyPoint], cv2.typing.MatLike]:
        return self.__sift.detectAndCompute(img, None) # type: ignore  - mask = None not MatLike
    
    def match(self, img: np.ndarray, min_match_count = 10) -> bool:
        kp, desc = self.__sift_DaC(img)

        self.__flann.clear()
        matches = self.__flann.knnMatch(self.template_desc, desc, k=2)
        min_match_count = min(len(matches), min_match_count)

        good_matches = [m for m,n in matches if m.distance < 0.7 * n.distance]

        if len(good_matches) < min_match_count:
            return False

        template_points = np.asarray([self.template_kp[m.trainIdx].pt for m in good_matches], dtype=np.float32).reshape(-1, 1, 2)
        img_points = np.asarray([kp[m.queryIdx].pt for m in good_matches], dtype=np.float32).reshape(-1, 1, 2)

        # Find homography to check geometric consistency
        M, _ = cv2.findHomography(template_points, img_points, cv2.RANSAC, 5.0)
        return M is not None
