import pygame
import cv2
import os
import time
import math
import numpy as np
import torch
import logging
import warnings

import matplotlib.pyplot as plt

from vis_nav_game import Player, Action, Phase

from tqdm import tqdm

from test_inputs import test_inputs

from VisualSlam import SLAM
from plot_path import visualize_paths, visualize_paths_with_target

from SuperGluePretrainedNetwork.models.matching import Matching
from SuperGluePretrainedNetwork.models.utils import frame2tensor, make_matching_plot_fast

from sklearn.cluster import KMeans
from sklearn.neighbors import BallTree



START_STEP = 5
STEP_SIZE = 3

logging.basicConfig(filename='vis_nav_player.log', filemode='w', level=logging.INFO)


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
        self.cur_pose = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]], dtype=np.float16)
        super(KeyboardPlayerPyGame, self).__init__()
                
        self.img_idx = 0  # Add a counter for image filenames
        self.last_save_time = time.time()  # Record the last time an image was saved
        self.all_fpv = []
        self.cum_turned_angle = 0
        self.prev_angle = 0

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
        self.save_raw_des = []
        self.all_database_des = []
        self.img_tensor_list = []

        self.prev_img_tensor = None

        self.starting_step = START_STEP
        self.step_size = STEP_SIZE

        # self.tick_turn_rad = 0.042454
        self.tick_turn_rad = 0.0426

        self.orb = cv2.ORB.create(3000)

        self.r = 0
        self.theta = 0

        self.prev_reck_act = None

        self.testing_with_inputs = False
        self.test_input_idx = 0
        self.test_action_idx = 0

        self.action_list = []

        self.last_act_set = Action.IDLE

        self.prev_transform = np.eye(4)

        self.exploration_status = True


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

    def pre_navigation(self):
        # self.find_target()
        self.vlad_find_target()
    
    # Method to compute the VLAD (Vector of Locally Aggregated Descriptors) feature
    def get_VLAD(self, descriptors, codebook):
        predicted_labels = codebook.predict(descriptors)
        centroids = codebook.cluster_centers_
        num_clusters = codebook.n_clusters

        m, d = descriptors.shape
        VLAD_feature = np.zeros([num_clusters, d])

        # Compute the differences for all clusters (visual words)
        for i in range(num_clusters):
            # Check if there is at least one descriptor in that cluster
            if np.sum(predicted_labels == i) > 0:
                # Add the differences
                VLAD_feature[i] = np.sum(descriptors[predicted_labels == i, :] - centroids[i], axis=0)

        VLAD_feature = VLAD_feature.flatten()

        # Power normalization, also called square-rooting normalization
        VLAD_feature = np.sign(VLAD_feature) * np.sqrt(np.abs(VLAD_feature))

        # L2 normalization
        VLAD_feature = VLAD_feature / np.linalg.norm(VLAD_feature)

        return VLAD_feature
    
    
    
    
    # Process image
    def process_image_orb(self, fpv):
        state = self.get_state()
        if state is None:
            return False
        
        step = state[2]

        if self.last_act_set == Action.IDLE:
            return False
        
        
        # If past starting step (to avoid static) and on a set interval (self.step_size)
        if (step > self.starting_step) and ((step % self.step_size) == 0):
            fpv_gray = cv2.cvtColor(fpv, cv2.COLOR_BGR2GRAY)
            #Apply Equalized Histogram
            fpv_gray = cv2.equalizeHist(fpv_gray)

            # Apply Gaussian blur
            fpv_gray = cv2.GaussianBlur(fpv_gray, (5, 5), 0)

            # Apply edge detection
            fpv_gray = cv2.Sobel(fpv_gray, cv2.CV_64F, 1, 1, ksize=5)

            alpha = 2.0
            beta = 0.0
            fpv_gray = cv2.convertScaleAbs(fpv_gray, alpha=alpha, beta=beta)
            
            kp, des = self.slam.find_feature_points_singe_img(fpv_gray)
            self.save_raw_des.append(des)
            self.extract_aggregate_feature(des)
            

            self.img_data_list.append({'image':fpv_gray, 'keypoints':kp, 'descriptors':des, 'image_raw':fpv})
                      
            # If more than one image processed (index >= 1)
            if self.img_idx >= 1:

                # Find feature matches between prev processed image and current image
                # Find feature points
                img_now = self.img_data_list[self.img_idx]['image']
                img_prev = self.img_data_list[self.img_idx-1]['image']
                kp_prev = self.img_data_list[self.img_idx-1]['keypoints']
                des_prev = self.img_data_list[self.img_idx-1]['descriptors']
                # q1, q2, good = self.slam.get_matches(kp_prev, kp, des_prev, des)
                # q1, q2, good = self.slam.get_matches(kp, kp_prev, des, des_prev)
                q1, q2, good = self.slam.get_matches(kp_prev, kp, des_prev, des)
                draw_params = dict(matchColor = -1, # draw matches in green color
                 singlePointColor = None,
                 matchesMask = None, # draw only inliers
                 flags = 2)

                # img3 = cv2.drawMatches(img_now, kp1, img_prev,kp2, good ,None,**draw_params)
                # img3 = cv2.drawMatches(img_now, kp_prev, img_prev, kp, good, None, **draw_params)
                # cv2.imshow("image", img3)
                # key = cv2.waitKey(2)
        
                # # Check if the 'q' key is pressed (you can change 'q' to any key you prefer)
                # if key & 0xFF == ord('q'):
                #     cv2.destroyAllWindows()  # Close the OpenCV window
                
                if(len(q1)>=8):
                    pose = self.find_pose(q1, q2)

            # Increment index of processed images
            self.img_idx += 1

            self.last_act_set = Action.IDLE
        else:
            return False


        # if step == 144:
        #     cv2.imshow('144', fpv)
        #     cv2.waitKey(0)

        return True

    def extract_aggregate_feature(self, des):
        try:
            # Read and process the image
            self.all_database_des.extend(des)
        except Exception as e:
            # Handle errors when processing images
            print(f"Error processing image")
        
  
    def vlad_find_target(self):
        target_list = self.get_target_images()
        if target_list is None or len(target_list) <= 0:
            return
        best_match_list = [None, None, None, None]
        print("Finding target image matches")
               
        kmeans_all_descriptors = np.asarray([self.all_database_des]).squeeze()
                
        # Perform k-means clustering on the entire bag of descriptors
        kmeans_codebook = KMeans(n_clusters=16, init='k-means++', n_init=1, verbose=1).fit(kmeans_all_descriptors)

        # Initialize lists to store VLAD representations and image names from the database
        database_VLAD = []
        # database_poses = []
        
        for image_id in range(len(self.img_data_list)):
            curr_des = self.save_raw_des[image_id]
            all_descriptors = np.asarray([curr_des]).squeeze()
            VLAD = self.get_VLAD(all_descriptors, kmeans_codebook)
            database_VLAD.append(VLAD)
            # database_poses.append(self.estimated_path[image_id])
            # print("we are at  :",database_poses)
            

        # Convert the lists to NumPy arrays
        database_VLAD = np.asarray(database_VLAD)
            
        # Build a BallTree for efficient nearest neighbor search
        tree = BallTree(database_VLAD, leaf_size=60)   
                
        # Set the number of closest images to retrieve
        num_of_imgs = 1
        final_pose_index = []
        
        target_list = self.get_target_images()
        if target_list is None or len(target_list) <= 0:
            return
        
        best_match_list = [None, None, None, None]

        print("Finding target image matches")
            
        for i, target in enumerate(target_list):
            num_good = 0

            target_gray = cv2.cvtColor(target, cv2.COLOR_BGR2GRAY)
            #Apply Equalized Histogram
            target_gray = cv2.equalizeHist(target_gray)

            # Apply Gaussian blur
            target_gray = cv2.GaussianBlur(target_gray, (5, 5), 0)

            # Apply edge detection
            target_gray = cv2.Sobel(target_gray, cv2.CV_64F, 1, 1, ksize=5)

            alpha = 2.0
            beta = 0.0
            target_gray = cv2.convertScaleAbs(target_gray, alpha=alpha, beta=beta)


            # target_kp, target_des = self.slam.find_feature_points_singe_img(target)
            _, target_des = self.slam.find_feature_points_singe_img(target_gray)

            final_target_descriptors = np.asarray([target_des]).squeeze()
            # Compute the VLAD representation for the query image
            query_VLAD = self.get_VLAD(target_des, kmeans_codebook).reshape(1, -1)
            
            # Retrieve the index of the closest image(s) in the database
            dist, index = tree.query(query_VLAD, num_of_imgs)
            
            # Index is an array of arrays of size 1
            # Get the name of the closest image and append it to the list
            # value_name = database_poses[index[0][0]]
            final_pose_index.append(index[0][0])
        
        self.show_best_matches(final_pose_index)   
        print(self.estimated_path[final_pose_index[0]])
        possible_targets = [self.estimated_path[idx] for idx in final_pose_index]
        print(possible_targets)
        
        visualize_paths_with_target(self.estimated_path, possible_targets, "Visual Odometry", file_out="VO.html")
        
       
               
            
            
            
            
            
            
            
            
            
            
        

    def find_target(self):
        target_list = self.get_target_images()
        if target_list is None or len(target_list) <= 0:
            return
        
        best_match_list = [None, None, None, None]

        print("Finding target image matches")

        for i, target in enumerate(target_list):
            num_good = 0

            target_gray = cv2.cvtColor(target, cv2.COLOR_BGR2GRAY)
            #Apply Equalized Histogram
            target_gray = cv2.equalizeHist(target_gray)

            # Apply Gaussian blur
            target_gray = cv2.GaussianBlur(target_gray, (5, 5), 0)

            # Apply edge detection
            target_gray = cv2.Sobel(target_gray, cv2.CV_64F, 1, 1, ksize=5)

            alpha = 2.0
            beta = 0.0
            target_gray = cv2.convertScaleAbs(target_gray, alpha=alpha, beta=beta)


            # target_kp, target_des = self.slam.find_feature_points_singe_img(target)
            target_kp, target_des = self.slam.find_feature_points_singe_img(target_gray)
            for j, img_data in enumerate(tqdm(self.img_data_list)):
                # img_kp, img_des = self.slam.find_feature_points_singe_img(img_data['image_raw'])
                img_kp, img_des = img_data['keypoints'], img_data['descriptors']
                _, _, good = self.slam.get_matches(target_kp, img_kp, target_des, img_des)

                if len(good) > num_good:
                    num_good = len(good)
                    best_match_list[i] = j

        self.show_best_matches(best_match_list)

        print("best matchs: {}".format(best_match_list))
        possible_targets = [self.estimated_path[idx] for idx in best_match_list]
        print("target loc:  {}".format(possible_targets))

        visualize_paths_with_target(self.estimated_path, possible_targets, "Visual Odometry", file_out="VO.html")

    def show_best_matches(self, match_idxs):
        img1 = self.img_data_list[match_idxs[0]]['image_raw']
        img2 = self.img_data_list[match_idxs[1]]['image_raw']
        img3 = self.img_data_list[match_idxs[2]]['image_raw']
        img4 = self.img_data_list[match_idxs[3]]['image_raw']

        # cv2.imshow('best match', img)
        # cv2.waitKey(0)

        hor1 = cv2.hconcat([img1, img2])
        hor2 = cv2.hconcat([img3, img4])
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

        cv2.putText(concat_img, 'match 1', (h_offset, w_offset), font, size, color, stroke, line)
        cv2.putText(concat_img, 'match 2', (int(h/2) + h_offset, w_offset), font, size, color, stroke, line)
        cv2.putText(concat_img, 'match 3', (h_offset, int(w/2) + w_offset), font, size, color, stroke, line)
        cv2.putText(concat_img, 'match 4', (int(h/2) + h_offset, int(w/2) + w_offset), font, size, color, stroke, line)
        # data_dir = r"C:\Users\ifeda\ROB-GY-Computer-Vision\vis_nav_player"
        # visualize_paths(self.estimated_path, "Visual Odometry",file_out="VO.html")
        
        cv2.imshow(f'matched images', concat_img)
        cv2.waitKey(1)


       
    def act(self):

        if not self.testing_with_inputs:
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

        else:
            if self.test_input_idx >= len(test_inputs):
                return Action.IDLE
                
            test_step = test_inputs[self.test_input_idx]

            self.last_act = Action.IDLE

            self.last_act |= test_step['actions']

            self.test_action_idx += 1
            if (self.test_action_idx >= test_step['steps']):
                self.test_action_idx = 0
                self.test_input_idx += 1
        
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
        cv2.putText(concat_img, 'Left View', (int(h/2) + h_offset, w_offset), font, size, color, stroke, line)
        cv2.putText(concat_img, 'Back View', (h_offset, int(w/2) + w_offset), font, size, color, stroke, line)
        cv2.putText(concat_img, 'Right View', (int(h/2) + h_offset, int(w/2) + w_offset), font, size, color, stroke, line)
        # data_dir = r"C:\Users\ifeda\ROB-GY-Computer-Vision\vis_nav_player"
        # visualize_paths(self.estimated_path, "Visual Odometry",file_out="VO.html")
        
        cv2.imshow(f'KeyboardPlayer:target_images', concat_img)
        cv2.waitKey(1)
        # print(self.estimated_path)
        


    def set_target_images(self, images):
        super(KeyboardPlayerPyGame, self).set_target_images(images)
        self.show_target_images()
        

    # Method to find SuperPoint feature descriptors in an image
    def find_feature_points_superpoint(self, img):
        img_tensor = frame2tensor(img, self.device)
        img_data = self.matching.superpoint({'image': img_tensor})
        self.img_data_list.append(img_data)
        self.img_tensor_list.append(img_tensor)
        return img_data['descriptors']
    
    # Find pose
    def find_pose(self, q1, q2):
        # Get pose from SLAM class
        relative_pose  = self.slam.get_pose(q1, q2)
        if Action.LEFT not in self.last_act_set and Action.RIGHT not in self.last_act_set:
            # relative_pose[:,:3] = 1
            relative_pose[:3,:3] = np.eye(3)
        elif Action.FORWARD not in self.last_act_set and Action.BACKWARD not in self.last_act_set:
            relative_pose[:,3] = [0,0,0,1]
        elif self.last_act == Action.IDLE:
            relative_pose[:,:] = np.eye(4)

        relative_pose = np.nan_to_num(relative_pose, neginf=0, posinf=0)
          
        # Save last x,z coordinates
        prev_xz = (self.cur_pose[0,3], self.cur_pose[2,3])
        prev_rot = self.cur_pose[:3,:3]


        if relative_pose[0,0] <= 0.9 and relative_pose[2,2] <= 0.9:
            relative_pose[:3,:3] = self.prev_transform[:3,:3]

        # Calculate new pose from relative pose (transformation matrix)
        self.cur_pose = np.matmul(self.cur_pose, np.linalg.inv(relative_pose))
        
        if (Action.FORWARD not in self.last_act_set) and (Action.BACKWARD not in self.last_act_set):
            self.cur_pose[0,3] = prev_xz[0]
            self.cur_pose[2,3] = prev_xz[1]
        if (Action.LEFT not in self.last_act_set) and (Action.RIGHT not in self.last_act_set):
            self.cur_pose[:3,:3] = prev_rot

        x_diff = self.cur_pose[0,3]-prev_xz[0]
        y_diff = self.cur_pose[2,3]-prev_xz[1]
        
        if abs(x_diff) > abs(y_diff):
        # If the difference on the x-axis is greater, only update the x-axis position
            self.cur_pose[2, 3] = prev_xz[1]
        else:
        # If the difference on the y-axis is greater or equal, only update the y-axis position
            self.cur_pose[0, 3] = prev_xz[0]
        # Save current location
        self.estimated_path.append((self.cur_pose[0,3], self.cur_pose[2,3]))
        # print(self.estimate)
        self.action_list.append(self.last_act_set)

        self.prev_transform = relative_pose

        return self.cur_pose


    # # Process image
    # def process_image_orb(self, fpv):
    #     state = self.get_state()
    #     if state is None:
    #         return False
        
    #     step = state[2]

    #     if self.last_act_set == Action.IDLE:
    #         return False
        
        
    #     # If past starting step (to avoid static) and on a set interval (self.step_size)
    #     if (step > self.starting_step) and ((step % self.step_size) == 0):
    #         fpv_gray = cv2.cvtColor(fpv, cv2.COLOR_BGR2GRAY)
    #         #Apply Equalized Histogram
    #         fpv_gray = cv2.equalizeHist(fpv_gray)

    #         # Apply Gaussian blur
    #         fpv_gray = cv2.GaussianBlur(fpv_gray, (5, 5), 0)

    #         # Apply edge detection
    #         fpv_gray = cv2.Sobel(fpv_gray, cv2.CV_64F, 1, 1, ksize=5)

    #         alpha = 2.0
    #         beta = 0.0
    #         fpv_gray = cv2.convertScaleAbs(fpv_gray, alpha=alpha, beta=beta)
            
    #         kp, des = self.slam.find_feature_points_singe_img(fpv_gray)
            
    #         self.all_database_des.extend(des)

    #         self.img_data_list.append({'image':fpv_gray, 'keypoints':kp, 'descriptors':des, 'image_raw':fpv})
                      
    #         # If more than one image processed (index >= 1)
    #         if self.img_idx >= 1:

    #             # Find feature matches between prev processed image and current image
    #             # Find feature points
    #             img_now = self.img_data_list[self.img_idx]['image']
    #             img_prev = self.img_data_list[self.img_idx-1]['image']
    #             kp_prev = self.img_data_list[self.img_idx-1]['keypoints']
    #             des_prev = self.img_data_list[self.img_idx-1]['descriptors']
    #             # q1, q2, good = self.slam.get_matches(kp_prev, kp, des_prev, des)
    #             # q1, q2, good = self.slam.get_matches(kp, kp_prev, des, des_prev)
    #             q1, q2, good = self.slam.get_matches(kp_prev, kp, des_prev, des)
    #             draw_params = dict(matchColor = -1, # draw matches in green color
    #              singlePointColor = None,
    #              matchesMask = None, # draw only inliers
    #              flags = 2)

    #             # img3 = cv2.drawMatches(img_now, kp1, img_prev,kp2, good ,None,**draw_params)
    #             # img3 = cv2.drawMatches(img_now, kp_prev, img_prev, kp, good, None, **draw_params)
    #             # cv2.imshow("image", img3)
    #             # key = cv2.waitKey(2)
        
    #             # # Check if the 'q' key is pressed (you can change 'q' to any key you prefer)
    #             # if key & 0xFF == ord('q'):
    #             #     cv2.destroyAllWindows()  # Close the OpenCV window
                
    #             if(len(q1)>=8):
    #                 pose = self.find_pose(q1, q2)

    #         # Increment index of processed images
    #         self.img_idx += 1

    #         self.last_act_set = Action.IDLE
    #     else:
    #         return False


    #     # if step == 144:
    #     #     cv2.imshow('144', fpv)
    #     #     cv2.waitKey(0)

    #     return True


    # See (function used by game)
    def see(self, fpv):
        
        if fpv is None or len(fpv.shape) < 3:
            return

        self.fpv = fpv
                   
        if self.screen is None:
            h, w, _ = fpv.shape
            self.screen = pygame.display.set_mode((w, h))
       
        state = self.get_state()
        
        if state is None:
            return None
        
        step = state[1]
                
        if self.exploration_status and step == Phase.EXPLORATION:
            ret = self.process_image_orb(fpv)
            # If image wasn't processed, add last action to set
            if not ret:
                self.last_act_set |= self.last_act


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


if __name__ == "__main__":
    import vis_nav_game
    vis_nav_game.play(the_player=KeyboardPlayerPyGame())
