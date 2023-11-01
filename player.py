from vis_nav_game import Player, Action
import pygame
import cv2
import os
import time
import numpy as np
from VisualSlam import SLAM
import matplotlib.pyplot as plt
from plot_path import visualize_paths


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
                
        self.image_counter = 0  # Add a counter for image filenames
        self.last_save_time = time.time()  # Record the last time an image was saved
        self.all_fpv = []

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
        
                

    def see(self, fpv):
        
        if fpv is None or len(fpv.shape) < 3:
            return

        self.fpv = fpv        

        # Check if 0.5 seconds have passed since the last image was saved
        current_time = time.time()
        
        if current_time - self.last_save_time >= 0.01:
            self.all_fpv.append(fpv) 
            self.image_counter += 1  # Increment the counter
            self.last_save_time = current_time  # Update the last save time
                   
        if self.screen is None:
            h, w, _ = fpv.shape
            self.screen = pygame.display.set_mode((w, h))
        
        
        if self.image_counter > 1:
            
            i = self.image_counter-1
            
            # Perform template matching on consecutive images
            img_now = self.all_fpv[i]
            img_prev = self.all_fpv[i-1]      
            
            q1, q2 = self.slam.get_matches(img_now, img_prev)
            
            if len(q1) >=5:
                # print("Q1 lenght =",len(q1))
                # print("Q2 length = ",len(q2))
                relative_pose = self.slam.get_pose(q1,q2)
                relative_pose = np.nan_to_num(relative_pose, neginf=0,posinf=0)
                self.estimated_path.append((self.cur_pose[0, 3], self.cur_pose[2, 3]))
                self.cur_pose = np.matmul(self.cur_pose, np.linalg.inv(relative_pose))
                print(self.estimated_path)            
                    
    

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
