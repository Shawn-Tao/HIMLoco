# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

import numpy as np
from numpy.random import choice
import random
from scipy import interpolate

from isaacgym import terrain_utils
from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg

class Terrain:
    def __init__(self, cfg: LeggedRobotCfg.terrain, num_robots) -> None:

        self.cfg = cfg
        self.num_robots = num_robots
        self.type = cfg.mesh_type
        if self.type in ["none", 'plane']:
            return
        self.env_length = cfg.terrain_length
        self.env_width = cfg.terrain_width
        self.proportions = [np.sum(cfg.terrain_proportions[:i+1]) for i in range(len(cfg.terrain_proportions))]

        self.cfg.num_sub_terrains = cfg.num_rows * cfg.num_cols
        self.env_origins = np.zeros((cfg.num_rows, cfg.num_cols, 3))

        self.width_per_env_pixels = int(self.env_width / cfg.horizontal_scale)
        self.length_per_env_pixels = int(self.env_length / cfg.horizontal_scale)

        self.border = int(cfg.border_size/self.cfg.horizontal_scale)
        self.tot_cols = int(cfg.num_cols * self.width_per_env_pixels) + 2 * self.border
        self.tot_rows = int(cfg.num_rows * self.length_per_env_pixels) + 2 * self.border

        self.height_field_raw = np.zeros((self.tot_rows , self.tot_cols), dtype=np.int16)
        
        # 选择构造地形的方式
        if cfg.curriculum:
            self.curiculum()
        elif cfg.selected:
            self.selected_terrain()
        else:    
            self.randomized_terrain()   
        
        self.heightsamples = self.height_field_raw
        if self.type=="trimesh":
            self.vertices, self.triangles = terrain_utils.convert_heightfield_to_trimesh(   self.height_field_raw,
                                                                                            self.cfg.horizontal_scale,
                                                                                            self.cfg.vertical_scale,
                                                                                            self.cfg.slope_treshold)
    
    def randomized_terrain(self):
        for k in range(self.cfg.num_sub_terrains):
            # Env coordinates in the world
            (i, j) = np.unravel_index(k, (self.cfg.num_rows, self.cfg.num_cols))

            choice = np.random.uniform(0, 1)
            difficulty = np.random.choice([0.5, 0.75, 0.9])
            terrain = self.make_terrain(choice, difficulty)
            self.add_terrain_to_map(terrain, i, j)
        
    def curiculum(self):
        # 按照cols选择地形类型，按照rows选择难度
        for j in range(self.cfg.num_cols):
            for i in range(self.cfg.num_rows):
                # self.cfg.num_rows -> (self.cfg.num_rows - 1) 使得difficulty最小是0，最大是1
                difficulty = i / (self.cfg.num_rows - 1)
                choice = j / self.cfg.num_cols + 0.001
                
                # print(f"Terrain {i}, {j}: choice={choice}, difficulty={difficulty}")

                terrain = self.make_terrain(choice, difficulty)
                self.add_terrain_to_map(terrain, i, j)

    def selected_terrain(self):
        terrain_type = self.cfg.terrain_kwargs.pop('type')
        for k in range(self.cfg.num_sub_terrains):
            # Env coordinates in the world
            (i, j) = np.unravel_index(k, (self.cfg.num_rows, self.cfg.num_cols))

            terrain = terrain_utils.SubTerrain("terrain",
                              width=self.width_per_env_pixels,
                              length=self.width_per_env_pixels,
                              vertical_scale=self.vertical_scale,
                              horizontal_scale=self.horizontal_scale)

            eval(terrain_type)(terrain, **self.cfg.terrain_kwargs.terrain_kwargs)
            self.add_terrain_to_map(terrain, i, j)
    
    def make_terrain(self, choice, difficulty):
        terrain = terrain_utils.SubTerrain(   "terrain",
                                width=self.width_per_env_pixels,
                                length=self.width_per_env_pixels,
                                vertical_scale=self.cfg.vertical_scale,
                                horizontal_scale=self.cfg.horizontal_scale)
        
        # 实际最高高度是 x * 20 * 40 * difficulty / 20 == x * difficulty * 40
        # slope = difficulty * 0.4
        slope = difficulty * 1
        slope_amplitude = 0.01 + 0.07 * difficulty
        step_amplitude = 0.00 + 0.07 * difficulty
        step_height = 0.05 + 0.18 * difficulty
        discrete_obstacles_height = 0.05 + difficulty * 0.1
        stepping_stones_size = 1.5 * (1.05 - difficulty)
        stone_distance = 0.05 if difficulty==0 else 0.1
        gap_size = 1. * difficulty
        pit_depth = 1. * difficulty
        
        # max_slope_height = difficulty * 40
        
        # 金字塔斜面（smooth slope）
        if choice < self.proportions[0]:
            slope *= -1
            pyramid_sloped_terrain_redefine(terrain, slope=slope, platform_size=2.,pad_size=0.5)
            if choice < self.proportions[0]/ 2:
                random_uniform_terrain_redefine(terrain, min_height=-slope_amplitude, max_height=slope_amplitude, step=0.005, downsampled_scale=0.2)
            # terrain_utils.pyramid_sloped_terrain(terrain, slope=slope, platform_size=3.)
            
        # 金字塔斜面 + 复杂地形（rough slope）
        elif choice < self.proportions[1]:
            pyramid_sloped_terrain_redefine(terrain, slope=slope, platform_size=2.,pad_size=0.5)
            # terrain_utils.pyramid_sloped_terrain(terrain, slope=slope, platform_size=3.)
            if choice < (self.proportions[0] + (self.proportions[1]-self.proportions[0]) / 2) :
                random_uniform_terrain_redefine(terrain, min_height=-slope_amplitude, max_height=slope_amplitude, step=0.005, downsampled_scale=0.2)
        
        # 金字塔阶梯（pyramid stairs up），向上的和向下的
        elif choice < self.proportions[3]:
            if choice<self.proportions[2]:
                step_height *= -1
            pyramid_stairs_terrain_redefine(terrain, step_width=0.30, step_height=step_height, platform_size=3.,pad_size=0.5)
            random_uniform_terrain_redefine(terrain, min_height=-step_amplitude, max_height=step_amplitude, step=0.005, downsampled_scale=0.2)
        
        # discrete
        elif choice < self.proportions[4]:
            num_rectangles = 20
            rectangle_min_size = 1.
            rectangle_max_size = 2.
            discrete_obstacles_terrain_redefine(terrain, discrete_obstacles_height, rectangle_min_size, rectangle_max_size, num_rectangles, platform_size=3.)
        
        # test
        elif choice < self.proportions[5]:
            parkour_step_terrain(terrain,
                                   num_stones=8 - 2,
                                   step_height=0.1 + 0.35*1,
                                   x_range=[0.3,1.5],
                                   y_range=[-0.15, 0.15],
                                   half_valid_width=[0.5, 1],
                                   pad_height=0,
                                   )
            
        # 未使用
        elif choice < self.proportions[6]:
            terrain_utils.stepping_stones_terrain(terrain, stone_size=stepping_stones_size, stone_distance=stone_distance, max_height=0., platform_size=4.)
            
        elif choice < self.proportions[7]:
            gap_terrain(terrain, gap_size=gap_size, platform_size=3.)
        else:
            pit_terrain(terrain, depth=pit_depth, platform_size=4.)
        
        return terrain

    def add_terrain_to_map(self, terrain, row, col):
        i = row
        j = col
        # map coordinate system
        start_x = self.border + i * self.length_per_env_pixels
        end_x = self.border + (i + 1) * self.length_per_env_pixels
        start_y = self.border + j * self.width_per_env_pixels
        end_y = self.border + (j + 1) * self.width_per_env_pixels
        self.height_field_raw[start_x: end_x, start_y:end_y] = terrain.height_field_raw

        env_origin_x = (i + 0.5) * self.env_length
        env_origin_y = (j + 0.5) * self.env_width
        x1 = int((self.env_length/2. - 1) / terrain.horizontal_scale)
        x2 = int((self.env_length/2. + 1) / terrain.horizontal_scale)
        y1 = int((self.env_width/2. - 1) / terrain.horizontal_scale)
        y2 = int((self.env_width/2. + 1) / terrain.horizontal_scale)
        env_origin_z = np.max(terrain.height_field_raw[x1:x2, y1:y2])*terrain.vertical_scale
        self.env_origins[i, j] = [env_origin_x, env_origin_y, env_origin_z]
        
    # def add_roughness(self, terrain, difficulty=1):
    #     max_height = (self.cfg.height[1] - self.cfg.height[0]) * difficulty + self.cfg.height[0]
    #     height = random.uniform(self.cfg.height[0], max_height)
    #     terrain_utils.random_uniform_terrain(terrain, min_height=-height, max_height=height, step=0.005, downsampled_scale=self.cfg.downsampled_scale)

def pyramid_sloped_terrain_redefine(terrain, slope=2, platform_size=1., pad_size=1.):
    """
    Generate a sloped terrain

    Parameters:
        terrain (terrain): the terrain
        slope (int): positive or negative slope
        platform_size (float): size of the flat platform at the center of the terrain [meters]
    Returns:
        terrain (SubTerrain): update terrain
    """
    
    # 加一圈pad
    
    # terrain.width = int(terrain.width / terrain.horizontal_scale) -->80
    # terrain.length = int(terrain.length / terrain.horizontal_scale)
    
    # x = np.arange(0, terrain.width)
    # y = np.arange(0, terrain.length)
    
    # center_x = int(terrain.width / 2)
    # center_y = int(terrain.length / 2)
    
    # # 将 xx 和 yy 归一化为从 0 到 1，中心为 1，边缘为 0，最终构造一个“金字塔梯度面”：中心区域值最大，越远离中心越小
    # xx, yy = np.meshgrid(x, y, sparse=True)
    # xx = (center_x - np.abs(center_x-xx)) / center_x
    # yy = (center_y - np.abs(center_y-yy)) / center_y
    # xx = xx.reshape(terrain.width, 1)
    # yy = yy.reshape(1, terrain.length)
    
    pad_size_q = int(pad_size/ terrain.horizontal_scale)
    
    x = np.arange(0, terrain.width-2*pad_size_q)
    y = np.arange(0, terrain.length-2*pad_size_q)
    
    center_x = int((terrain.width-2*pad_size_q) / 2)
    center_y = int((terrain.length-2*pad_size_q) / 2)
    
    # 将 xx 和 yy 归一化为从 0 到 1，中心为 1，边缘为 0，最终构造一个“金字塔梯度面”：中心区域值最大，越远离中心越小
    xx, yy = np.meshgrid(x, y, sparse=True)
    xx = (center_x - np.abs(center_x-xx)) / center_x
    yy = (center_y - np.abs(center_y-yy)) / center_y
    xx = xx.reshape(terrain.width-2*pad_size_q, 1)
    yy = yy.reshape(1, terrain.length-2*pad_size_q)
    
    
    # slope * 0.1/0.005 * 40 == slope * 800 -- > 0.4 * 800 = 320
    max_height = int(slope * (terrain.horizontal_scale / terrain.vertical_scale) * ((terrain.width-2*pad_size_q) / 2))
    # print(max_height)
    
    terrain.height_field_raw[pad_size_q:terrain.width-pad_size_q,pad_size_q:terrain.length-pad_size_q] += (max_height * xx * yy).astype(terrain.height_field_raw.dtype)
    
    # 找一个点，截断平面
    platform_size = int(platform_size / terrain.horizontal_scale / 2)
    x1 = (terrain.width) // 2 - platform_size
    x2 = (terrain.width) // 2 + platform_size
    y1 = (terrain.length) // 2 - platform_size
    y2 = (terrain.length) // 2 + platform_size

    min_h = min(terrain.height_field_raw[x1, y1], 0)
    max_h = max(terrain.height_field_raw[x1, y1], 0)
    # print(f"pyramid sloped terrain: min_h={min_h}, max_h={max_h}")
    terrain.height_field_raw[pad_size_q:terrain.width-pad_size_q,pad_size_q:terrain.length-pad_size_q] = np.clip(terrain.height_field_raw[pad_size_q:terrain.width-pad_size_q,pad_size_q:terrain.length-pad_size_q], min_h, max_h)
    return terrain    

def pyramid_stairs_terrain_redefine(terrain, step_width, step_height, platform_size=1., pad_size=1.):
    """
    Generate stairs

    Parameters:
        terrain (terrain): the terrain
        step_width (float):  the width of the step [meters]
        step_height (float): the step_height [meters]
        platform_size (float): size of the flat platform at the center of the terrain [meters]
    Returns:
        terrain (SubTerrain): update terrain
    """
    pad_size_q = int(pad_size/ terrain.horizontal_scale)
    
    
    
    # switch parameters to discrete units
    step_width = int(step_width / terrain.horizontal_scale)
    step_height = int(step_height / terrain.vertical_scale)
    platform_size = int(platform_size / terrain.horizontal_scale)

    height = 0
    start_x = pad_size_q
    stop_x = terrain.width-pad_size_q
    start_y = pad_size_q
    stop_y = terrain.length-pad_size_q
    while (stop_x - start_x) > platform_size and (stop_y - start_y) > platform_size:
        start_x += step_width
        stop_x -= step_width
        start_y += step_width
        stop_y -= step_width
        height += step_height
        terrain.height_field_raw[start_x: stop_x, start_y: stop_y] = height
    return terrain    
    
def parkour_step_terrain(terrain,
                        platform_len=2.5, 
                        platform_height=0., 
                        num_stones=8,
                    #    x_range=[1.5, 2.4],
                        x_range=[0.2, 0.4],
                        y_range=[-0.15, 0.15],
                        half_valid_width=[0.45, 0.5],
                        step_height = 0.2,
                        pad_width=0.1,
                        pad_height=0.5):
    # goals = np.zeros((num_stones+2, 2))
    # terrain.height_field_raw[:] = -200
    mid_y = terrain.length // 2  # length is actually y width

    dis_x_min = round( (x_range[0] + step_height) / terrain.horizontal_scale)
    dis_x_max = round( (x_range[1] + step_height) / terrain.horizontal_scale)
    dis_y_min = round(y_range[0] / terrain.horizontal_scale)
    dis_y_max = round(y_range[1] / terrain.horizontal_scale)

    step_height = round(step_height / terrain.vertical_scale)

    half_valid_width = round(np.random.uniform(half_valid_width[0], half_valid_width[1]) / terrain.horizontal_scale)

    platform_len = round(platform_len / terrain.horizontal_scale)
    platform_height = round(platform_height / terrain.vertical_scale)
    terrain.height_field_raw[0:platform_len, :] = platform_height

    # stone_width = round(stone_width / terrain.horizontal_scale)
    
    # incline_height = round(incline_height / terrain.vertical_scale)
    # last_incline_height = round(last_incline_height / terrain.vertical_scale)

    dis_x = platform_len
    last_dis_x = dis_x
    stair_height = 0
    # goals[0] = [platform_len - round(1 / terrain.horizontal_scale), mid_y]
    for i in range(num_stones):
        rand_x = np.random.randint(dis_x_min, dis_x_max)
        rand_y = np.random.randint(dis_y_min, dis_y_max)
        if i < num_stones // 2:
            stair_height += step_height
        elif i > num_stones // 2:
            stair_height -= step_height
        terrain.height_field_raw[dis_x:dis_x+rand_x, ] = stair_height
        dis_x += rand_x
        terrain.height_field_raw[last_dis_x:dis_x, :mid_y+rand_y-half_valid_width] = 0
        terrain.height_field_raw[last_dis_x:dis_x, mid_y+rand_y+half_valid_width:] = 0
        
        last_dis_x = dis_x
        # goals[i+1] = [dis_x-rand_x//2, mid_y+rand_y]
    final_dis_x = dis_x + np.random.randint(dis_x_min, dis_x_max)
    # import ipdb; ipdb.set_trace()
    if final_dis_x > terrain.width:
        final_dis_x = terrain.width - 0.5 // terrain.horizontal_scale
    # goals[-1] = [final_dis_x, mid_y]
    
    # terrain.goals = goals * terrain.horizontal_scale
    
    # terrain.height_field_raw[:, :max(mid_y-half_valid_width, 0)] = 0
    # terrain.height_field_raw[:, min(mid_y+half_valid_width, terrain.height_field_raw.shape[1]):] = 0
    # terrain.height_field_raw[:, :] = 0
    # pad edges
    pad_width = int(pad_width // terrain.horizontal_scale)
    pad_height = int(pad_height // terrain.vertical_scale)
    terrain.height_field_raw[:, :pad_width] = pad_height
    terrain.height_field_raw[:, -pad_width:] = pad_height
    terrain.height_field_raw[:pad_width, :] = pad_height
    terrain.height_field_raw[-pad_width:, :] = pad_height

def discrete_obstacles_terrain_redefine(terrain, max_height, min_size, max_size, num_rects, platform_size=1.):
    """
    Generate a terrain with gaps

    Parameters:
        terrain (terrain): the terrain
        max_height (float): maximum height of the obstacles (range=[-max, -max/2, max/2, max]) [meters]
        min_size (float): minimum size of a rectangle obstacle [meters]
        max_size (float): maximum size of a rectangle obstacle [meters]
        num_rects (int): number of randomly generated obstacles
        platform_size (float): size of the flat platform at the center of the terrain [meters]
    Returns:
        terrain (SubTerrain): update terrain
    """
    # switch parameters to discrete units
    max_height = int(max_height / terrain.vertical_scale)
    min_size = int(min_size / terrain.horizontal_scale)
    max_size = int(max_size / terrain.horizontal_scale)
    platform_size = int(platform_size / terrain.horizontal_scale)

    (i, j) = terrain.height_field_raw.shape
    height_range = [-max_height, -max_height // 2, max_height // 2, max_height]
    width_range = range(min_size, max_size, 4)
    length_range = range(min_size, max_size, 4)

    for _ in range(num_rects):
        width = np.random.choice(width_range)
        length = np.random.choice(length_range)
        start_i = np.random.choice(range(0, i-width, 4))
        start_j = np.random.choice(range(0, j-length, 4))
        terrain.height_field_raw[start_i:start_i+width, start_j:start_j+length] = np.random.choice(height_range)

    x1 = (terrain.width - platform_size) // 2
    x2 = (terrain.width + platform_size) // 2
    y1 = (terrain.length - platform_size) // 2
    y2 = (terrain.length + platform_size) // 2
    terrain.height_field_raw[x1:x2, y1:y2] = 0
    return terrain

def random_uniform_terrain_redefine(terrain, min_height, max_height, step=1, downsampled_scale=None,):
    """
    Generate a uniform noise terrain

    Parameters
        terrain (SubTerrain): the terrain
        min_height (float): the minimum height of the terrain [meters]
        max_height (float): the maximum height of the terrain [meters]
        step (float): minimum height change between two points [meters]
        downsampled_scale (float): distance between two randomly sampled points ( musty be larger or equal to terrain.horizontal_scale)

    """
    if downsampled_scale is None:
        downsampled_scale = terrain.horizontal_scale

    # switch parameters to discrete units
    min_height = int(min_height / terrain.vertical_scale)
    max_height = int(max_height / terrain.vertical_scale)
    step = int(step / terrain.vertical_scale)

    heights_range = np.arange(min_height, max_height + step, step)
    height_field_downsampled = np.random.choice(heights_range, (int(terrain.width * terrain.horizontal_scale / downsampled_scale), int(
        terrain.length * terrain.horizontal_scale / downsampled_scale)))

    x = np.linspace(0, terrain.width * terrain.horizontal_scale, height_field_downsampled.shape[0])
    y = np.linspace(0, terrain.length * terrain.horizontal_scale, height_field_downsampled.shape[1])

    f = interpolate.interp2d(y, x, height_field_downsampled, kind='linear')

    x_upsampled = np.linspace(0, terrain.width * terrain.horizontal_scale, terrain.width)
    y_upsampled = np.linspace(0, terrain.length * terrain.horizontal_scale, terrain.length)
    z_upsampled = np.rint(f(y_upsampled, x_upsampled))

    terrain.height_field_raw += z_upsampled.astype(np.int16)
    return terrain


def gap_terrain(terrain, gap_size, platform_size=1.):
    gap_size = int(gap_size / terrain.horizontal_scale)
    platform_size = int(platform_size / terrain.horizontal_scale)

    center_x = terrain.length // 2
    center_y = terrain.width // 2
    x1 = (terrain.length - platform_size) // 2
    x2 = x1 + gap_size
    y1 = (terrain.width - platform_size) // 2
    y2 = y1 + gap_size
   
    terrain.height_field_raw[center_x-x2 : center_x + x2, center_y-y2 : center_y + y2] = -1000
    terrain.height_field_raw[center_x-x1 : center_x + x1, center_y-y1 : center_y + y1] = 0

def pit_terrain(terrain, depth, platform_size=1.):
    depth = int(depth / terrain.vertical_scale)
    platform_size = int(platform_size / terrain.horizontal_scale / 2)
    x1 = terrain.length // 2 - platform_size
    x2 = terrain.length // 2 + platform_size
    y1 = terrain.width // 2 - platform_size
    y2 = terrain.width // 2 + platform_size
    terrain.height_field_raw[x1:x2, y1:y2] = -depth
