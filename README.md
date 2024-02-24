# **Maze-Conquering Solution: Visual Path Mapping and Target Search** 

*Enhancing mobile robots with autonomous path mapping and target image finding in a maze environment using Monocular Visual Odometry, VLAD, and Superglue Pretrain Network.*

![gif_visual odometry](https://github.com/IJAMUL1/visual_odometry_maze_solution/assets/60096099/436bca6c-7355-481d-854d-89233c8979fc)


## **Overview** 

* This project involves a multi-modal problem that requires multiple computer vision techniques with fundamental robotics and SLAM concepts. At a very high level, we need to explore a maze built in Pygame while saving information (in essence, remembering the maze), then in the navigation phase we reset to the beginning of the maze and must locate and navigate to a target as quickly as possible, based only on images of the target location.

## **Demo**

* **Demo Video:** *(([https://drive.google.com/file/d/1GZvM6IB3s34Q8Kr_OcSHLJ1U-ZD_yAf9/view?usp=drive_link](https://drive.google.com/file/d/10ZKu7E_6FD53SiSraNu6lZKTruy4-NvJ/view?usp=sharing)))*
   * See link to final presentation file *((https://docs.google.com/presentation/d/1q1fmdLSKATPFoNHQhgFF25TS9gMwMKQr/edit?usp=drive_link&ouid=100678161242482663381&rtpof=true&sd=true))*

## Installation Guide

Follow these steps to set up the project environment:

**Prerequisites**
    ```bash
    Jinja2 ==2.11.3
    bokeh==2.0.1
    markupsafe==2.0.1
    ```



1. **Clone this repository:**
    ```bash
    git clone https://github.com/IJAMUL1/visual_odometry_maze_solution.git
    cd visual_odometry_maze_solution
    ```

2. **Update Conda:**
    Ensure your Conda installation is up-to-date.
    ```bash
    conda update conda
    ```

3. **Create Conda Environment:**
    Create a new Conda environment using the provided YAML file.
    ```bash
    conda env create -f environment.yml
    ```

4. **Activate the Environment:**
    Activate the newly created Conda environment.
    ```bash
    conda activate game
    ```

5. **Clone Superglue Repo**
    Clone the SuperGluePreTrainedNetwork
   ```bash
    git clone https://github.com/magicleap/SuperGluePretrainedNetwork.git
    ```
6. **Installation Complete:**
    Your environment is now set up and ready to use.

