import cv2
import os
import numpy as np
import matplotlib
import matplotlib.cm as cm
import torch

from SuperGluePretrainedNetwork.models.matching import Matching
from SuperGluePretrainedNetwork.models.utils import frame2tensor, make_matching_plot_fast

class SLAM():
    def __init__(self,Cmat):        
        # self.images = self._load_images(image_dir)
        # self.orb = cv2.ORB_create(50)
        self.orb = cv2.ORB.create(50, nlevels=3, edgeThreshold=10, patchSize=20)
        # self.sift = cv2.SIFT_create()
        FLANN_INDEX_LSH = 6
        index_params = dict(algorithm=FLANN_INDEX_LSH, table_number=6, key_size=12, multi_probe_level=1)
        search_params = dict(checks=50)
        self.flann = cv2.FlannBasedMatcher(indexParams=index_params, searchParams=search_params)
        self.K = Cmat
        # print("K: {}".format(self.K))
        
        
    def _form_transf(self, R, t):
        """
        Makes a transformation matrix from the given rotation matrix and translation vector

        Parameters
        ----------
        R (ndarray): The rotation matrix
        t (list): The translation vector

        Returns
        -------
        T (ndarray): The transformation matrix
        """
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = R
        T[:3, 3] = t
        return T
    
    def get_matches(self, img_now, img_prev):
        """
        This function detect and compute keypoints and descriptors from the i-1'th and i'th image using the class orb object

        Parameters
        ----------
        i (int): The current frame

        Returns
        -------
        q1 (ndarray): The good keypoints matches position in i-1'th image
        q2 (ndarray): The good keypoints matches position in i'th image
        """
        
        h = img_now.shape[0]
        w = img_now.shape[1]

        # Find the keypoints and descriptors with ORB
        # kp1, des1 = self.orb.detectAndCompute(self.images[i - 1], None)
        # kp2, des2 = self.orb.detectAndCompute(self.images[i], None)
        kp1, des1 = self.orb.detectAndCompute(img_prev, None)
        kp2, des2 = self.orb.detectAndCompute(img_now, None)

        # kp1, des1 = self.sift.detectAndCompute(img_prev, None)
        # kp2, des2 = self.sift.detectAndCompute(img_now, None)

        # Find matches
        bf = cv2.BFMatcher(cv2.NORM_HAMMING)
        # bf = cv2.BFMatcher()
        matches = bf.knnMatch(des1, des2, k=2)
        # matches = bf.match(des1, des2)

        # good = []
        # x1 = []
        # x2 = []

        # for m,n in matches:
        #     g = False
        #     if m.distance < 0.75*n.distance:
        #         pts1 = kp1[m.queryIdx]
        #         pts2 = kp2[m.queryIdx]
        #         g = True

        #     if g is True:
        #         if np.linalg.norm((pts1-pts2)) < 0.1 * np.linalg.norm([w, h]) and m.distance < 32:
        #             if m.queryIdx not in x1 and m.trainIdx not in x2:
        #                 x1.append(m.queryIdx)
        #                 x2.append(m.trainIdx)
        #                 good.append(m)


        # Find the matches there do not have a to high distance
        good = []
        try:
            for m, n in matches:
                if m.distance < 0.7 * n.distance:
                    good.append(m)
        except ValueError:
            pass

        draw_params = dict(matchColor = -1, # draw matches in green color
                 singlePointColor = None,
                 matchesMask = None, # draw only inliers
                 flags = 2)

        # for g in good:
        #     img3 = cv2.drawMatches(img_now, kp2, img_prev, kp1, [g], None, **draw_params)
        #     cv2.imshow("image", img3)
        #     key = cv2.waitKey(0)
        
        img3 = cv2.drawMatches(img_now, kp2, img_prev, kp1, good, None, **draw_params)
        cv2.imshow("image", img3)
        key = cv2.waitKey(0)

        # Check if the 'q' key is pressed (you can change 'q' to any key you prefer)
        # if key & 0xFF == ord('q'):
        #     cv2.destroyAllWindows()  # Close the OpenCV window

        # Get the image points form the good matches
        q1 = np.float32([kp1[m.queryIdx].pt for m in good])
        q2 = np.float32([kp2[m.trainIdx].pt for m in good])
        return q1, q2


    def olg_get_matches(self, img_now, img_prev):
        """
        This function detect and compute keypoints and descriptors from the i-1'th and i'th image using the class orb object

        Parameters
        ----------
        i (int): The current frame

        Returns
        -------
        q1 (ndarray): The good keypoints matches position in i-1'th image
        q2 (ndarray): The good keypoints matches position in i'th image
        """
        # Find the keypoints and descriptors with ORB
        # kp1, des1 = self.orb.detectAndCompute(self.images[i - 1], None)
        # kp2, des2 = self.orb.detectAndCompute(self.images[i], None)
        kp1, des1 = self.orb.detectAndCompute(img_prev, None)
        kp2, des2 = self.orb.detectAndCompute(img_now, None)

        # Find matches
        matches = self.flann.knnMatch(des1, des2, k=2)

        # Find the matches there do not have a to high distance
        good = []
        try:
            for m, n in matches:
                if m.distance < 0.9 * n.distance:
                    good.append(m)
        except ValueError:
            pass

        draw_params = dict(matchColor = -1, # draw matches in green color
                 singlePointColor = None,
                 matchesMask = None, # draw only inliers
                 flags = 2)

        img3 = cv2.drawMatches(img_now, kp2, img_prev, kp1, good, None, **draw_params)
        cv2.imshow("image", img3)
        key = cv2.waitKey(0)
        
        # Check if the 'q' key is pressed (you can change 'q' to any key you prefer)
        if key & 0xFF == ord('q'):
            cv2.destroyAllWindows()  # Close the OpenCV window

        # Get the image points form the good matches
        q1 = np.float32([kp1[m.queryIdx].pt for m in good])
        q2 = np.float32([kp2[m.trainIdx].pt for m in good])
        return q1, q2
    
    def get_pose(self, q1, q2):
        """
        Calculates the transformation matrix

        Parameters
        ----------
        q1 (ndarray): The good keypoints matches position in i-1'th image
        q2 (ndarray): The good keypoints matches position in i'th image

        Returns
        -------
        transformation_matrix (ndarray): The transformation matrix
        """
        # Essential matrix
        E, _ = cv2.findEssentialMat(q1, q2, self.K, threshold=1)      # Adjust the threshold

        # Decompose the Essential matrix into R and t
        self.P = np.column_stack((self.K, np.zeros((3, 1))))
        # print("P: {}".format(self.P))
        R, t = self.decomp_essential_mat(E, q1, q2)

        # Get transformation matrix
        transformation_matrix = self._form_transf(R, np.squeeze(t))
        return transformation_matrix
    
    def decomp_essential_mat(self, E, q1, q2):
        """
        Decompose the Essential matrix

        Parameters
        ----------
        E (ndarray): Essential matrix
        q1 (ndarray): The good keypoints matches position in i-1'th image
        q2 (ndarray): The good keypoints matches position in i'th image

        Returns
        -------
        right_pair (list): Contains the rotation matrix and translation vector
        """
        def sum_z_cal_relative_scale(R, t):
            # Get the transformation matrix
            T = self._form_transf(R, t)
            # Make the projection matrix
            P = np.matmul(np.concatenate((self.K, np.zeros((3, 1))), axis=1), T)

            # Triangulate the 3D points
            hom_Q1 = cv2.triangulatePoints(self.P, P, q1.T, q2.T)
            
            # Also seen from cam 2
            hom_Q2 = np.matmul(T, hom_Q1)

            # Un-homogenize
            uhom_Q1 = hom_Q1[:3, :] / hom_Q1[3, :]
            uhom_Q2 = hom_Q2[:3, :] / hom_Q2[3, :]

            # Find the number of points there has positive z coordinate in both cameras
            sum_of_pos_z_Q1 = sum(uhom_Q1[2, :] > 0)
            sum_of_pos_z_Q2 = sum(uhom_Q2[2, :] > 0)

            # Form point pairs and calculate the relative scale
            relative_scale = np.mean(np.linalg.norm(uhom_Q1.T[:-1] - uhom_Q1.T[1:], axis=-1)/
                                     np.linalg.norm(uhom_Q2.T[:-1] - uhom_Q2.T[1:], axis=-1))
            return sum_of_pos_z_Q1 + sum_of_pos_z_Q2, relative_scale

        # Decompose the essential matrix
        R1, R2, t = cv2.decomposeEssentialMat(E)
        t = np.squeeze(t)

        # Make a list of the different possible pairs
        pairs = [[R1, t], [R1, -t], [R2, t], [R2, -t]]

        # Check which solution there is the right one
        z_sums = []
        relative_scales = []
        for R, t in pairs:
            z_sum, scale = sum_z_cal_relative_scale(R, t)
            z_sums.append(z_sum)
            relative_scales.append(scale)

        # Select the pair there has the most points with positive z coordinate
        right_pair_idx = np.argmax(z_sums)
        right_pair = pairs[right_pair_idx]
        relative_scale = relative_scales[right_pair_idx]
        R1, t = right_pair
        t = t * relative_scale

        return [R1, t]
    
    

def extract_integer(filename):
    return int(filename.split('.')[0][5:])

if 0:
    K = np.array([[ 92.0, 0.0,  160.0],
                [  0.0, 92.0, 120.0],
                [  0.0,  0.0,   1.0]])

    slam = SLAM(K)

    filepath = "feature_test_images"
    image_paths = [os.path.join(filepath, file) for file in sorted(os.listdir(filepath), key=extract_integer)]
    imgs = [cv2.imread(path, cv2.IMREAD_GRAYSCALE) for path in image_paths]

    # print("len: {}".format(len(imgs)))
    # print("type: {}".format(type(imgs[0])))
    # print("shape: {}".format(imgs[0].shape))

    # cv2.imshow('img', imgs[0])
    # cv2.waitKey(0)


    imgs.pop(0)

    img_now = imgs[100]
    img_prev = imgs[103]

    # cv2.imshow('img_now', img_now)
    # cv2.waitKey(0)
    # cv2.imshow('img_prev', img_prev)
    # cv2.waitKey(0)

    slam.get_matches(img_now, img_prev)



if 1:
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    class Opt():
        def __init__(self):
            self.nms_radius = 4
            self.keypoint_threshold = 0.005
            self.max_keypoints = -1

            self.superglue = 'indoor'
            self.sinkhorn_iterations = 20
            self.match_threshold = 0.2

            self.show_keypoints = True

    opt = Opt()

    config = {
        'superpoint': {
            'nms_radius': opt.nms_radius,
            'keypoint_threshold': opt.keypoint_threshold,
            'max_keypoints': opt.max_keypoints
        },
        'superglue': {
            'weights': opt.superglue,
            'sinkhorn_iterations': opt.sinkhorn_iterations,
            'match_threshold': opt.match_threshold,
        }
    }

    matching = Matching(config).eval().to(device)
    keys = ['keypoints', 'scores', 'descriptors']

    filepath = "feature_test_images"
    image_paths = [os.path.join(filepath, file) for file in sorted(os.listdir(filepath), key=extract_integer)]
    imgs = [cv2.imread(path, cv2.IMREAD_GRAYSCALE) for path in image_paths]

    img1 = imgs[100]
    img2 = imgs[103]
    frame1_tensor = frame2tensor(img1, device)
    frame2_tensor = frame2tensor(img2, device)

    last_data = matching.superpoint({'image': frame1_tensor})
    last_data = {k+'0': last_data[k] for k in keys}
    last_data['image0'] = frame1_tensor
    last_frame = img1
    last_image_id = 0

    pred = matching({**last_data, 'image1': frame2_tensor})
    kpts0 = last_data['keypoints0'][0].cpu().numpy()
    kpts1 = pred['keypoints1'][0].cpu().numpy()
    matches = pred['matches0'][0].cpu().numpy()
    confidence = pred['matching_scores0'][0].cpu().detach().numpy()
    
    print("matches:\n{}".format(matches))
    print("kpts0:\n{}".format(matches))

    valid = matches > -1
    mkpts0 = kpts0[valid]
    mkpts1 = kpts1[matches[valid]]
    color = cm.jet(confidence[valid])
    text = [
        'SuperGlue',
        'Keypoints: {}:{}'.format(len(kpts0), len(kpts1)),
        'Matches: {}'.format(len(mkpts0))
    ]
    k_thresh = matching.superpoint.config['keypoint_threshold']
    m_thresh = matching.superglue.config['match_threshold']
    small_text = [
        'Keypoint Threshold: {:.4f}'.format(k_thresh),
        'Match Threshold: {:.2f}'.format(m_thresh),
        'Image Pair: {:06}:{:06}'.format(1, 2),
    ]

    # print("matches:\n{}".format(matches))
    # print("confidence:\n{}".format(confidence))
    # print("kpts0:\n{}".format(kpts0))
    # print("mkpts0:\n{}".format(mkpts0))


    # ----------------------------
    # for m0, m1 in zip(mkpts0, mkpts1):
    #     out = make_matching_plot_fast(
    #         img1, img2, kpts0, kpts1, [m0], [m1], color, text,
    #         path=None, show_keypoints=opt.show_keypoints, small_text=small_text)
        
    #     cv2.imshow("matches", out)
    #     cv2.waitKey(0)


    # ----------------------------
    idx = np.r_[0:1,15:16,25:26,30:31,40:41,50:51,60:61,70:71]

    m0 = mkpts0[idx]
    m1 = mkpts1[idx]

    out = make_matching_plot_fast(
        img1, img2, kpts0, kpts1, m0, m1, color, text,
        path=None, show_keypoints=opt.show_keypoints, small_text=small_text)
        
    cv2.imshow("matches", out)
    cv2.waitKey(0)



    # ----------------------------
    # out = make_matching_plot_fast(
    #     img1, img2, kpts0, kpts1, mkpts0, mkpts1, color, text,
    #     path=None, show_keypoints=opt.show_keypoints, small_text=small_text)
    
    # cv2.imshow("matches", out)
    # cv2.waitKey(0)