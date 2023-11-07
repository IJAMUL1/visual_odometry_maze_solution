import os
import numpy as np
import cv2 as cv2
from player import KeyboardPlayerPyGame


class TestPlayer(KeyboardPlayerPyGame):
    def __init__(self):
        super(TestPlayer, self).__init__()
        

    def get_target_images(self) -> list[np.ndarray]:
        test_target_path = "test_targets/set_1"

        if self.img_idx < 5:
            return None

        img_files = [f for f in os.listdir(test_target_path) if f.lower().endswith('.png')]
        img_files.sort()

        test_targets = []

        for img_file in img_files:
            img_path = os.path.join(test_target_path, img_file)
            img = cv2.imread(img_path)

            if img is not None:
                test_targets.append(img)

        return test_targets
    

if __name__ == "__main__":
    import vis_nav_game
    vis_nav_game.play(the_player=TestPlayer())