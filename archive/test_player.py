import os
import sys
import numpy as np
import cv2 as cv2
from player import KeyboardPlayerPyGame


class TestPlayer(KeyboardPlayerPyGame):
    def __init__(self, target_set=None):

        self.test_target_dir = "test_targets"

        if target_set is not None:
            if not os.path.exists(self.test_target_dir):
                print(f"Directory '{self.test_target_dir}' doesn't exist")
                exit()

            subdirs = [d for d in os.listdir(self.test_target_dir) if os.path.isdir(os.path.join(self.test_target_dir, d))]
            
            if target_set in subdirs:
                self.target_set = target_set
            else:
                print(f"Directory for set '{target_set}' doesn't exist")
                exit()
        else:
            self.target_set = "set_1"

        print("---- TEST PLAYER ----")
        print(f"Using test target set: {target_set}")
        super(TestPlayer, self).__init__()
        

    def get_target_images(self) -> list[np.ndarray]:
        test_target_path = self.test_target_dir + "/" + self.target_set
        

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

    if len(sys.argv) > 1:
        target_set = sys.argv[1]
    else:
        target_set = None

    vis_nav_game.play(the_player=TestPlayer(target_set=target_set))