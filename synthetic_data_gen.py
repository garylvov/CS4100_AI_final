import numpy as np
import matplotlib.pyplot as plt
import random
from scipy.spatial.transform import Rotation
import torch
from torch.utils.data import Dataset

import torch.nn as nn
import torch.optim as optim

from torchsummary import summary
from torch.utils.data import DataLoader

'''
Authored by Gary Lvov
'''

class SparsePointAssocDataset(Dataset):
    def __init__(self, num_samples=10000,
                 num_geom=5, num_points_per_geom=5, new_geos_per_sample=False, geos=None,
                 number_erronous=0, number_points_unobserved=0,
                 geom_x_max=.25, geom_y_max=.25, geom_z_max=.25,
                 obs_x_min=-3, obs_x_max=3,
                 obs_y_min=-3, obs_y_max=3,
                 obs_z_min=0, obs_z_max=3,
                 clustering=False):

        self.num_samples = num_samples
        self.num_geom = num_geom
        self.num_point_per_geom = num_points_per_geom
        self.unnoisy_input_dim = (self.num_geom * 3) * 2
        self.MAX_NUM_ERRONOUS = 7
        self.obs_input_size = (
            self.num_geom * self.num_point_per_geom * 3) + (self.MAX_NUM_ERRONOUS * 3)

        self.clustering = clustering

        self.input_size = 171
        self.output_size = 96

        self.new_geo_for_each_sample = new_geos_per_sample
        if not self.new_geo_for_each_sample:
            if geos is None:
                self.geom = []
                for _ in range(self.num_geom):
                    self.geom.append(self.generate_random_geometry(num_points=self.num_point_per_geom,
                                                                x_max=geom_x_max,
                                                                y_max=geom_y_max,
                                                                z_max=geom_z_max))

            else:
                self.geom = geos

        else:
            self.geom = None

        num_points_uno = lambda x : random.randint(0, number_points_unobserved) if x > 0 else 0
        num_points_error = lambda x : random.randint(0, number_erronous) if x > 0 else 0

        if not self.clustering:
            num_geom_uno = lambda x : random.randint(0, self.num_geom - 1) if x > 0 else 0
        else:
            num_geom_uno = lambda x : 0

        self.point_dicts = [self.obtain_points_dict(num_geom=self.num_geom,
                                                    num_point_per_geom=self.num_point_per_geom)
                                                    for _ in range(num_samples)]

        self.vectors = [self.construct_vectors(point_dict, number_erronous=num_points_error(number_erronous),
                                               number_points_unobserved=num_points_uno(number_points_unobserved),
                                               number_geometries_unobserved=num_geom_uno(self.num_geom -1),
                                               noise_x_min=obs_x_min, noise_x_max=obs_x_max + geom_x_max,
                                               noise_y_min=obs_y_min, noise_y_max=obs_y_max + geom_y_max,
                                               noise_z_min=obs_z_min, noise_z_max=obs_z_max + geom_z_max)
                                               for point_dict in self.point_dicts]
        self.vectors = np.array(self.vectors)

    @staticmethod
    def generate_random_geometry(num_points=5,
                                x_min=0, x_max=.25,
                                y_min=0, y_max=.25,
                                z_min=0, z_max=.25):
        coordinates = np.empty((num_points, 3))
        for i in range(num_points):
            x = random.uniform(x_min, x_max)
            y = random.uniform(y_min, y_max)
            z = random.uniform(z_min, z_max)
            coordinates[i] = (x, y, z)
        return coordinates

    def apply_random_rotation(self, points):
        r = random.uniform(0, 2 * np.pi)
        p = random.uniform(0, np.pi)
        y = random.uniform(0, 2 * np.pi)
        rotation = Rotation.from_euler('zyx', [r, p, y], degrees=False)
        return rotation.apply(points)

    def apply_random_translation(self, points,
                                x_min=-3, x_max=3,
                                y_min=-3, y_max=3,
                                z_min=0, z_max=3,
                                return_translation=False):
        x = random.uniform(x_min, x_max)
        y = random.uniform(y_min, y_max)
        z = random.uniform(z_min, z_max)
        if return_translation:
            return points + np.array([x, y, z]), np.array([x, y, z])
        else:
           return points + np.array([x, y, z])

    def apply_noise(self, points, noise_disp=.000001):
        points[:,] += random.uniform(-1 * noise_disp, noise_disp)
        return points

    def obtain_points_dict(self, num_geom=5, num_point_per_geom=5,
                           geom_x_max=.25, geom_y_max=.25, geom_z_max=.25,
                           obs_x_min=-3, obs_x_max=3,
                           obs_y_min=-3, obs_y_max=3,
                           obs_z_min=0, obs_z_max=3):
        point_dict = {}
        for idx in range(num_geom):
            t = None
            if not self.new_geo_for_each_sample:
                points = self.geom[idx]
            else:
                points = SparsePointAssocDataset.generate_random_geometry(num_points=num_point_per_geom,
                                                        x_max=geom_x_max,
                                                        y_max=geom_y_max,
                                                        z_max=geom_z_max)
            point_dict['gt_' + str(idx)] = points
            points = self.apply_random_rotation(points)
            if self.clustering:
                points, t= self.apply_random_translation(points,
                                                   x_min=obs_x_min, x_max=obs_x_max,
                                                   y_min=obs_y_min, y_max=obs_y_max,
                                                   z_min=obs_z_min, z_max=obs_z_max,
                                                   return_translation=True)
                point_dict['t_' + str(idx)] = t
            else:
                points = self.apply_random_translation(points,
                                                    x_min=obs_x_min, x_max=obs_x_max,
                                                    y_min=obs_y_min, y_max=obs_y_max,
                                                    z_min=obs_z_min, z_max=obs_z_max)
                
            points = self.apply_noise(points)
            point_dict['obs_' + str(idx)] = points
        return point_dict

    def construct_vectors(self, data, number_erronous=0,
                          number_points_unobserved=0,
                          number_geometries_unobserved=0,
                          noise_x_min=-3, noise_x_max=3.25,
                          noise_y_min=-3, noise_y_max=3.25,
                          noise_z_min=0, noise_z_max=3.25):
        gt_names = []
        for idx in range(self.num_geom):
            gt_names.append('gt_' + str(idx))

        obs_names = []

        unobserved_set = set()
        while len(unobserved_set) < number_geometries_unobserved:
            eliminated_geom_idx = random.randint(0, self.num_geom - 1)
            if eliminated_geom_idx not in unobserved_set:
                unobserved_set.add(eliminated_geom_idx)

        for idx in range(self.num_geom):
            if str(idx) not in unobserved_set:
                obs_names.append('obs_' + str(idx))

        gt_vec = []
        obs_w_gt_vec = []

        counter = 0
        coord_counter = 0
        for key, value in data.items():
            if key in gt_names:
                gt_vec.extend(value)

            if key in obs_names:
                for obs, gt in zip(value, data['gt_' + key[-1]]):
                    for i in range(min(len(obs), len(gt))):

                        obs_w_gt_vec.append(np.array([obs[i], counter]))

                        coord_counter += 1
                        if coord_counter == 3:
                            coord_counter = 0
                            counter += 1

        obs_w_gt_vec = np.array(obs_w_gt_vec).reshape(
            (len(obs_w_gt_vec)//3, 3, 2))

        while obs_w_gt_vec.shape[0] != (counter) - number_points_unobserved:
            random_idx = random.randint(0, obs_w_gt_vec.shape[0] - 1)
            obs_w_gt_vec = np.delete(obs_w_gt_vec, random_idx, axis=0)

        while obs_w_gt_vec.shape[0] != (counter) + number_erronous - number_points_unobserved:
            random_idx = random.randint(0, obs_w_gt_vec.shape[0] - 1)
            # insert number erronous
            obs_w_gt_vec = np.insert(obs_w_gt_vec, random_idx,
                                     np.array([[[random.uniform(noise_x_min, noise_x_max), -1],
                                                [random.uniform(
                                                    noise_y_min, noise_y_max), -1],
                                                [random.uniform(noise_z_min, noise_z_max), -1]]]), axis=0)

        np.random.shuffle(obs_w_gt_vec)

        obs_w_gt_vec = obs_w_gt_vec.reshape(
            (obs_w_gt_vec.shape[0] * obs_w_gt_vec.shape[1]), 2)

        flat_gt_vec = np.array(gt_vec).flatten()

        flat_obs_vec = np.array(obs_w_gt_vec)[:, 0].flatten()
        obs_padding = np.full(self.input_size - len(flat_obs_vec) - len(flat_gt_vec), 0)
        flat_obs_vec = np.concatenate((flat_obs_vec, obs_padding))

        flat_result_vec = np.array(obs_w_gt_vec)[:, 1].flatten()
        result_padding = np.full(self.output_size - len(flat_result_vec), 0)
        flat_result_vec = np.concatenate((flat_result_vec, result_padding))

        return np.concatenate((flat_gt_vec, flat_obs_vec), axis=0).flatten(), flat_result_vec

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        input, output = self.vectors[idx]
        input = torch.from_numpy(input)
        output = torch.from_numpy(output)
        return input, output

'''
The first tier will contain 3d features belonging to geometries,
where the geometries are clearly separated (far apart from each other),
with some slight noise in their positions
'''
class Tier1A(SparsePointAssocDataset):
    def __init__(self, num_samples=10000,
                 num_geom=5, num_points_per_geom=5, new_geos_per_sample=False, geos=None,
                 number_erronous=0, number_points_unobserved=0,
                 geom_x_max=.25, geom_y_max=.25, geom_z_max=.25,
                 obs_x_min=-3, obs_x_max=3,
                 obs_y_min=-3, obs_y_max=3,
                 obs_z_min=0, obs_z_max=3):
        super().__init__(num_samples,
                 num_geom, num_points_per_geom, new_geos_per_sample, geos,
                 number_erronous, number_points_unobserved,
                 geom_x_max, geom_y_max, geom_z_max,
                 obs_x_min, obs_x_max,
                 obs_y_min, obs_y_max,
                 obs_z_min, obs_z_max)


class Tier1B(SparsePointAssocDataset):
    def __init__(self, num_samples=10000,
                 num_geom=5, num_points_per_geom=5, new_geos_per_sample=False, geos=None,
                 number_erronous=0, number_points_unobserved=5,
                 geom_x_max=.25, geom_y_max=.25, geom_z_max=.25,
                 obs_x_min=-3, obs_x_max=3,
                 obs_y_min=-3, obs_y_max=3,
                 obs_z_min=0, obs_z_max=3):
        super().__init__(num_samples,
                 num_geom, num_points_per_geom, new_geos_per_sample, geos,
                 number_erronous, number_points_unobserved,
                 geom_x_max, geom_y_max, geom_z_max,
                 obs_x_min, obs_x_max,
                 obs_y_min, obs_y_max,
                 obs_z_min, obs_z_max)
'''
The second tier will contain 3d geometries which are far apart 
(with some slight positional noise), also containing false 3D features that don't belong to geometry.
'''       
class Tier2A(SparsePointAssocDataset):
    def __init__(self,num_samples=10000, 
                 num_geom=5, num_points_per_geom=5, new_geos_per_sample=False, geos=None,
                 number_erronous=7, number_points_unobserved=0, 
                 geom_x_max=.25, geom_y_max =.25, geom_z_max=.25,
                 obs_x_min=-3, obs_x_max=3, 
                 obs_y_min=-3, obs_y_max=3, 
                 obs_z_min= 0, obs_z_max=3):
        super().__init__(num_samples,
                 num_geom, num_points_per_geom, new_geos_per_sample, geos,
                 number_erronous, number_points_unobserved,
                 geom_x_max, geom_y_max, geom_z_max,
                 obs_x_min, obs_x_max,
                 obs_y_min, obs_y_max,
                 obs_z_min, obs_z_max)

class Tier2B(SparsePointAssocDataset):
    def __init__(self,num_samples=10000, 
                 num_geom=5, num_points_per_geom=5, new_geos_per_sample=False, geos=None,
                 number_erronous=7, number_points_unobserved=5, 
                 geom_x_max=.25, geom_y_max =.25, geom_z_max=.25,
                 obs_x_min=-3, obs_x_max=3, 
                 obs_y_min=-3, obs_y_max=3, 
                 obs_z_min= 0, obs_z_max=3):
        super().__init__(num_samples,
                 num_geom, num_points_per_geom, new_geos_per_sample, geos,
                 number_erronous, number_points_unobserved,
                 geom_x_max, geom_y_max, geom_z_max,
                 obs_x_min, obs_x_max,
                 obs_y_min, obs_y_max,
                 obs_z_min, obs_z_max)
'''
The third tier will have geometries that are less easily separable (close together)
'''
class Tier3A(SparsePointAssocDataset):
    def __init__(self,num_samples=10000, 
                 num_geom=5, num_points_per_geom=5, new_geos_per_sample=False, geos=None,
                 number_erronous=0, number_points_unobserved=0, 
                 geom_x_max=.25, geom_y_max =.25, geom_z_max=.25,
                 obs_x_min=-1, obs_x_max=1, 
                 obs_y_min=-1, obs_y_max=1, 
                 obs_z_min= 0, obs_z_max=1):
        super().__init__(num_samples,
                 num_geom, num_points_per_geom, new_geos_per_sample, geos,
                 number_erronous, number_points_unobserved,
                 geom_x_max, geom_y_max, geom_z_max,
                 obs_x_min, obs_x_max,
                 obs_y_min, obs_y_max,
                 obs_z_min, obs_z_max)

class Tier3B(SparsePointAssocDataset):
    def __init__(self,num_samples=10000, 
                 num_geom=5, num_points_per_geom=5, new_geos_per_sample=False, geos=None,
                 number_erronous=0, number_points_unobserved=5, 
                 geom_x_max=.25, geom_y_max =.25, geom_z_max=.25,
                 obs_x_min=-1, obs_x_max=1, 
                 obs_y_min=-1, obs_y_max=1, 
                 obs_z_min= 0, obs_z_max=1):
        super().__init__(num_samples,
                 num_geom, num_points_per_geom, new_geos_per_sample, geos,
                 number_erronous, number_points_unobserved,
                 geom_x_max, geom_y_max, geom_z_max,
                 obs_x_min, obs_x_max,
                 obs_y_min, obs_y_max,
                 obs_z_min, obs_z_max)

'''
The fourth tier will contain geometries that are not easily separable, with false 3D features
'''
class Tier4A(SparsePointAssocDataset):
    def __init__(self,num_samples=10000, 
                 num_geom=5, num_points_per_geom=5, new_geos_per_sample=False, geos=None,
                 number_erronous=7, number_points_unobserved=0, 
                 geom_x_max=.25, geom_y_max =.25, geom_z_max=.25,
                 obs_x_min=-1, obs_x_max=1, 
                 obs_y_min=-1, obs_y_max=1, 
                 obs_z_min= 0, obs_z_max=1):
        super().__init__(num_samples,
                 num_geom, num_points_per_geom, new_geos_per_sample, geos,
                 number_erronous, number_points_unobserved,
                 geom_x_max, geom_y_max, geom_z_max,
                 obs_x_min, obs_x_max,
                 obs_y_min, obs_y_max,
                 obs_z_min, obs_z_max)

class Tier4B(SparsePointAssocDataset):
    def __init__(self,num_samples=10000, 
                 num_geom=5, num_points_per_geom=5, new_geos_per_sample=False, geos=None,
                 number_erronous=0, number_points_unobserved=5, 
                 geom_x_max=.25, geom_y_max =.25, geom_z_max=.25,
                 obs_x_min=-1, obs_x_max=1, 
                 obs_y_min=-1, obs_y_max=1, 
                 obs_z_min= 0, obs_z_max=1):
        super().__init__(num_samples,
                 num_geom, num_points_per_geom, new_geos_per_sample, geos,
                 number_erronous, number_points_unobserved,
                 geom_x_max, geom_y_max, geom_z_max,
                 obs_x_min, obs_x_max,
                 obs_y_min, obs_y_max,
                 obs_z_min, obs_z_max)
        
if __name__ == '__main__':
    spad = SparsePointAssocDataset()