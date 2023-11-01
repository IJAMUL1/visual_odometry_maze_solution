import pygame
import cv2
import os
import time
import numpy as np
import torch

import matplotlib.pyplot as plt

from vis_nav_game import Player, Action

from VisualSlam import SLAM
from plot_path import visualize_paths

from SuperGluePretrainedNetwork.models.matching import Matching
from SuperGluePretrainedNetwork.models.utils import frame2tensor, make_matching_plot_fast

START_STEP = 5
STEP_SIZE = 4




# SuperPoint / SuperGlue options
class SuperOpt():
    def __init__(self):
        self.nms_radius = 4
        self.keypoint_threshold = 0.005
        self.max_keypoints = 80

        self.superglue = 'indoor'
        self.sinkhorn_iterations = 20
        self.match_threshold = 0.3

        # self.show_keypoints = False


# Player class
class KeyboardPlayerPyGame(Player):
    def __init__(self):
        self.fpv = None
        self.last_act = Action.IDLE
        self.screen = None
        self.keymap = None
        self.relative_poses = [] 
        self.estimated_path = []
        self.cur_pose = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]])
        super(KeyboardPlayerPyGame, self).__init__()
                
        self.img_idx = 0  # Add a counter for image filenames
        self.last_save_time = time.time()  # Record the last time an image was saved
        self.all_fpv = []

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.super_opt = SuperOpt()

        self.super_config = {
            'superpoint': {
                'nms_radius': self.super_opt.nms_radius,
                'keypoint_threshold': self.super_opt.keypoint_threshold,
                'max_keypoints': self.super_opt.max_keypoints
            },
            'superglue': {
                'weights': self.super_opt.superglue,
                'sinkhorn_iterations': self.super_opt.sinkhorn_iterations,
                'match_threshold': self.super_opt.match_threshold,
            }
        }

        self.matching = Matching(self.super_config).eval().to(self.device)
        self.super_keys = ['keypoints', 'scores', 'descriptors']
        self.img_data_list = []
        self.img_tensor_list = []

        self.prev_img_tensor = None

        self.starting_step = START_STEP
        self.step_size = STEP_SIZE

        self.orb = cv2.ORB.create(1000)


    def reset(self):
        self.fpv = None
        self.last_act = Action.IDLE
        self.screen = None

        pygame.init()

        self.keymap = {
            pygame.K_LEFT: Action.LEFT,
            pygame.K_RIGHT: Action.RIGHT,
            pygame.K_UP: Action.FORWARD,
            pygame.K_DOWN: Action.BACKWARD,
            pygame.K_SPACE: Action.CHECKIN,
            pygame.K_ESCAPE: Action.QUIT
        }
        
    def pre_exploration(self) -> None:
        self.Cmat = self.get_camera_intrinsic_matrix()
        # print(self.Cmat)
        self.slam = SLAM(self.Cmat)

        # Save starting location
        self.estimated_path.append((self.cur_pose[0,3], self.cur_pose[2,3]))
       
    def act(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                self.last_act = Action.QUIT
                return Action.QUIT

            if event.type == pygame.KEYDOWN:
                if event.key in self.keymap:
                    self.last_act |= self.keymap[event.key]
                else:
                    self.show_target_images()
            if event.type == pygame.KEYUP:
                if event.key in self.keymap:
                    self.last_act ^= self.keymap[event.key]
        return self.last_act

    def show_target_images(self):
        targets = self.get_target_images()
        if targets is None or len(targets) <= 0:
            return
        hor1 = cv2.hconcat(targets[:2])
        hor2 = cv2.hconcat(targets[2:])
        concat_img = cv2.vconcat([hor1, hor2])

        w, h = concat_img.shape[:2]
        
        color = (0, 0, 0)

        concat_img = cv2.line(concat_img, (int(h/2), 0), (int(h/2), w), color, 2)
        concat_img = cv2.line(concat_img, (0, int(w/2)), (h, int(w/2)), color, 2)

        w_offset = 25
        h_offset = 10
        font = cv2.FONT_HERSHEY_SIMPLEX
        line = cv2.LINE_AA
        size = 0.75
        stroke = 1

        cv2.putText(concat_img, 'Front View', (h_offset, w_offset), font, size, color, stroke, line)
        cv2.putText(concat_img, 'Right View', (int(h/2) + h_offset, w_offset), font, size, color, stroke, line)
        cv2.putText(concat_img, 'Back View', (h_offset, int(w/2) + w_offset), font, size, color, stroke, line)
        cv2.putText(concat_img, 'Left View', (int(h/2) + h_offset, int(w/2) + w_offset), font, size, color, stroke, line)
        # data_dir = r"C:\Users\ifeda\ROB-GY-Computer-Vision\vis_nav_player"
        visualize_paths(self.estimated_path, "Visual Odometry",file_out="VO.html")
        
        cv2.imshow(f'KeyboardPlayer:target_images', concat_img)
        cv2.waitKey(1)

    def set_target_images(self, images):
        super(KeyboardPlayerPyGame, self).set_target_images(images)
        self.show_target_images()
        

    # Find feature points using SuperPoint
    def find_feature_points_superpoint(self, img):
        img_tensor = frame2tensor(img, self.device)
        img_data = self.matching.superpoint({'image': img_tensor})

        self.img_data_list.append(img_data)
        self.img_tensor_list.append(img_tensor)

        return img_data['keypoints'], img_data['descriptors']


    # Find feature points using ORB
    def find_feature_points_orb(self, img):
        pass


    # Find feature matches using SuperGlue
    def find_feature_matches_superglue(self, img1_index, img2_index):
        img1_data = {k+'0':self.img_data_list[img1_index][k] for k in self.super_keys}
        img1_data['image0'] = self.img_tensor_list[img1_index]

        img2_data = {k+'1': self.img_data_list[img2_index][k] for k in self.super_keys}
        img2_data['image1'] = self.img_tensor_list[img2_index]

        pred = self.matching({**img1_data, **img2_data})
        kpts0 = img1_data['keypoints0'][0].cpu().numpy()
        kpts1 = img2_data['keypoints1'][0].cpu().numpy()
        matches = pred['matches0'][0].cpu().numpy()
        # confidence = pred['matching_scores0'][0].cpu().detach().numpy()

        valid = matches > -1
        mkpts0 = kpts0[valid]
        mkpts1 = kpts1[matches[valid]]

        # Get matching points (q1 for img1, q2 for img2)
        q1 = np.array(mkpts0)
        q2 = np.array(mkpts1)

        return q1, q2


    # Find pose
    def find_pose(self, q1, q2):
        # Get pose from SLAM class
        relative_pose  = self.slam.get_pose(q1, q2)
        relative_pose = np.nan_to_num(relative_pose, neginf=0, posinf=0)

        # Save last x,z coordinates
        prev_xz = (self.cur_pose[0,3], self.cur_pose[2,3])

        # Calculate new pose from relative pose (transformation matrix)
        self.cur_pose = np.matmul(self.cur_pose, np.linalg.inv(relative_pose))
        # print("curr pose:\n{}".format(cur_pose))


        # If not moving forward or backward, ignore the translation vector
        # Translation vector seems to be normalized to 1 from decomposeEssentialMat()
        # See: https://answers.opencv.org/question/66839/units-of-rotation-and-translation-from-essential-matrix/
        if (self.last_act != Action.FORWARD) and (self.last_act != Action.BACKWARD):
            self.cur_pose[0,3] = prev_xz[0]
            self.cur_pose[2,3] = prev_xz[1]
            
        # Save current location
        self.estimated_path.append((self.cur_pose[0,3], self.cur_pose[2,3]))

        return self.cur_pose


    # Process image
    def process_image(self, fpv):
        state = self.get_state()
        if state is None:
            return None
        
        step = state[2]

        # If past starting step (to avoid static) and on a set interval (self.step_size)
        if (step > self.starting_step) and ((step % self.step_size) == 0):
            fpv_gray = cv2.cvtColor(fpv, cv2.COLOR_BGR2GRAY)
            
            # Find feature points
            keypts, desc = self.find_feature_points_superpoint(fpv_gray)

            # If more than one image processed (index >= 1)
            if self.img_idx >= 1:

                # Find feature matches between prev processed image and current image
                q1, q2 = self.find_feature_matches_superglue(self.img_idx-1, self.img_idx)

                pose = self.find_pose(q1, q2)

            # Increment index of processed images
            self.img_idx += 1

        return True


    # See (function used by game)
    def see(self, fpv):
        
        if fpv is None or len(fpv.shape) < 3:
            return

        self.fpv = fpv
                   
        if self.screen is None:
            h, w, _ = fpv.shape
            self.screen = pygame.display.set_mode((w, h))
       
        # Process image: find feature points, match feature points, get pose
        # ret = self.process_image(fpv)


        # ************************************
        #      SAVE THIS STUFF BELOW
        # ************************************
        
        # SuperGlue implementation
        # -----------------------------------------------------
        # BEGIN IF 0
        if 1:
            # Interval for capturing / processing images
            STEPSIZE = 4

            state = self.get_state()
            if state is not None:
                step = state[2]

                # if current_time - self.last_save_time >= 0.1:
                if (step > 5) and ((step % STEPSIZE) == 0):
                    fpv_gray = cv2.cvtColor(fpv, cv2.COLOR_BGR2GRAY)
                    img_tensor = frame2tensor(fpv_gray, self.device)
                    img_data = self.matching.superpoint({'image': img_tensor})

                    self.img_data_list.append(img_data)
                    self.img_tensor_list.append(img_tensor)

                    if self.img_idx >= 1:
                        img_prev_data = {k+'0':self.img_data_list[self.img_idx-1][k] for k in self.super_keys}
                        img_prev_data['image0'] = self.img_tensor_list[self.img_idx-1]

                        img_now_data = {k+'1': img_data[k] for k in self.super_keys}
                        img_now_data['image1'] = img_tensor

                        # d = {**img_prev_data, **img_now_data}

                        pred = self.matching({**img_prev_data, **img_now_data})
                        kpts0 = img_prev_data['keypoints0'][0].cpu().numpy()
                        kpts1 = img_now_data['keypoints1'][0].cpu().numpy()
                        matches = pred['matches0'][0].cpu().numpy()
                        # confidence = pred['matching_scores0'][0].cpu().detach().numpy()

                        valid = matches > -1
                        mkpts0 = kpts0[valid]
                        mkpts1 = kpts1[matches[valid]]

                        q1 = np.array(mkpts0)
                        q2 = np.array(mkpts1)

                        relative_pose  = self.slam.get_pose(q1, q2)
                        relative_pose = np.nan_to_num(relative_pose, neginf=0, posinf=0)

                        # Save last x,z coordinates
                        prev_xz = (self.cur_pose[0,3], self.cur_pose[2,3])

                        # Calculate new pose from relative pose (transformation matrix)
                        self.cur_pose = np.matmul(self.cur_pose, np.linalg.inv(relative_pose))
                        # print("curr pose:\n{}".format(cur_pose))


                        # If not moving forward or backward, ignore the translation vector
                        # Translation vector seems to be normalized to 1 from decomposeEssentialMat()
                        # See: https://answers.opencv.org/question/66839/units-of-rotation-and-translation-from-essential-matrix/
                        if (self.last_act != Action.FORWARD) and (self.last_act != Action.BACKWARD):
                            self.cur_pose[0,3] = prev_xz[0]
                            self.cur_pose[2,3] = prev_xz[1]
                            
                        # Save current location
                        self.estimated_path.append((self.cur_pose[0,3], self.cur_pose[2,3]))

                    # Update counters and previous data
                    self.img_idx += 1
        # END IF 0




    

        def convert_opencv_img_to_pygame(opencv_image):
            """
            Convert OpenCV images for Pygame.

            see https://blanktar.jp/blog/2016/01/pygame-draw-opencv-image.html
            """
            opencv_image = opencv_image[:, :, ::-1]  # BGR->RGB
            shape = opencv_image.shape[1::-1]  # (height,width,Number of colors) -> (width, height)
            pygame_image = pygame.image.frombuffer(opencv_image.tobytes(), shape, 'RGB')

            return pygame_image

        pygame.display.set_caption("KeyboardPlayer:fpv")
        rgb = convert_opencv_img_to_pygame(fpv)
        self.screen.blit(rgb, (0, 0))
        pygame.display.update()
       
    # def visualize_path(self):
    #     if len(self.relative_poses) > 0:
    #         initial_pose = np.eye(4)  # Identity matrix representing the initial pose
    #         camera_poses = [initial_pose]  # List to store camera poses
    #         for relative_pose in self.relative_poses:
    #             # Accumulate poses
    #             current_pose = np.dot(camera_poses[-1], relative_pose)
    #             camera_poses.append(current_pose)

    #         # Extract x and z positions from camera poses for a 2D path map
    #         x_positions = [pose[0, 3] for pose in camera_poses]
    #         z_positions = [pose[2, 3] for pose in camera_poses]

    #         # Plot the path map
    #         plt.plot(x_positions, z_positions, marker='o', linestyle='-')
    #         plt.xlabel('X Position')
    #         plt.ylabel('Z Position')
    #         plt.title('Camera Path Map')
    #         plt.grid()
    #         plt.show()


if __name__ == "__main__":
    import vis_nav_game
    vis_nav_game.play(the_player=KeyboardPlayerPyGame())
