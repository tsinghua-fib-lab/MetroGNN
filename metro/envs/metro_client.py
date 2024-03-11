import logging
import math
import copy
import pickle
from pprint import pprint
from typing import Tuple, Dict, List, Text, Callable
from functools import partial

import numpy as np
import geopandas as gpd
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt

# from urban_planning.envs.plan_client import PlanClient
from metro.envs.metro import Metro
from metro.utils.config import Config

import time


class InfeasibleActionError(ValueError):
    """An infeasible action were passed to the env."""

    def __init__(self, action, mask):
        """Initialize an infeasible action error.

        Args:
          action: Infeasible action that was performed.
          mask: The mask associated with the current observation. mask[action] is
            `0` for infeasible actions.
        """
        super().__init__(self, action, mask)
        self.action = action
        self.mask = mask

    def __str__(self):
        return 'Infeasible action ({}) when the mask is ({})'.format(
            self.action, self.mask)


def reward_info_function(metro: Metro, stage) -> Tuple[float, Dict]:
    totoal_cost = metro.get_cost()
    # add_od = metro.get_reward()
    reward = metro.get_reward()
    add_od = 0
    # if stage == 'done':
    #     od_reward = metro.get_od()
    # else:  
    #     od_reward = 0
    # od_reward = add_od
    
    total_od = metro.get_od()
    # print(od_reward,stage)
    return reward, {'reward': reward, 'cost': totoal_cost, 'od': total_od,'add_od': add_od}


class MetroEnv:
    """ Environment for urban planning."""
    FAILURE_REWARD = -4.0
    INTERMEDIATE_REWARD = -4.0

    def __init__(self,
                 cfg: Config,
                 is_eval: bool = False,
                 reward_info_fn=reward_info_function):

        self.cfg = cfg
        self._is_eval = is_eval
        self._frozen = False
        self._action_history = []
        self._metro =  self.load_graph(cfg)
        self._copy_metro = copy.deepcopy(self._metro)
        self._reward_info_fn = partial(reward_info_fn)

        self._done = False
        # self._set_cached_reward_info()

    def load_graph(self,cfg):
        origin = cfg.build.get('origin')
        destination = cfg.build.get('destination')
        shape_param = cfg.build.get('shape')
        cost = cfg.build.get('cost_per')
        corridor = cfg.build.get('corridor')
        max_line = cfg.build.get('max_line')
        pre_line = cfg.build.get('pre_line')
        expand = cfg.build.get('expand')
        min_num = cfg.build.get('min_num')
        self.budget = cfg.build.get('budget')
        self.ax0 = None
        self.ax1 = None
        # self.ax0 = (gpd.read_file('data/beijing/all/edges.shp')).plot(figsize=(80,80),color='grey', zorder=1)
        # self.ax1 = (gpd.read_file('data/beijing/raw/beijing_city.geojson')).plot(figsize=(20,20), zorder=1)
        

        file_data = []
        if corridor != None:
            all_files = ['city_graph', 'region_processed', 'od_pair', 'line_info']
            for file_name in all_files:
                with open('data/{}/{}lines/{}.pickle'.format(cfg.city_name,pre_line,file_name), 'rb') as f:
                    file_data.append(pickle.load(f))

        m = Metro(file_data,self.budget,max_line,min_num,expand,cost,origin,destination,shape_param)

        return m

    def _set_cached_reward_info(self):
        """
        Set the cached reward.
        """
        if not self._frozen:
            self._cached_life_circle_reward = -1.0
            self._cached_greeness_reward = -1.0
            self._cached_concept_reward = -1.0

            self._cached_life_circle_info = dict()
            self._cached_concept_info = dict()

            self._cached_land_use_reward = -1.0
            self._cached_land_use_gdf = self.snapshot_land_use()

    def get_reward_info(self) -> Tuple[float, Dict]:
        return self._reward_info_fn(self._metro, self._stage)


    # def _get_all_reward_info(self) -> Tuple[float, Dict]:
    #     """
    #     Returns the entire reward and info. Used for loaded plans.
    #     """
    #     land_use_reward, land_use_info = self._reward_info_fn(self._metro, 'land_use')
    #     road_reward, road_info = self._reward_info_fn(self._metro, 'road')
    #     reward = land_use_reward + road_reward
    #     info = {
    #         'road_network': road_info['road_network'],
    #         'life_circle': land_use_info['life_circle'],
    #         'greeness': land_use_info['greeness'],
    #         'road_network_info': road_info['road_network_info'],
    #         'life_circle_info': land_use_info['life_circle_info']
    #     }
    #     return reward, info

    def eval(self):
        self._is_eval = True

    def train(self):
        self._is_eval = False

    def get_numerical_feature_size(self):
        return self._metro.get_numerical_dim()

    def get_node_dim(self):
        return self._metro.get_node_dim()
    
    def get_stage(self):
        if self._stage == 'build':
            return [1,0]
        elif self._stage == 'done':
            return [0,1]

    def _get_obs(self) -> List:
        numerical, node_feature, edge_distance, edge_od, node_mask = self._metro.get_obs()
        stage = self.get_stage()

        return [numerical, node_feature, edge_distance, edge_od, node_mask, stage]

    def add_station(self, action):
        self._metro.add_station_from_action(int(action))

    def snapshot_land_use(self):

        return self._metro.snapshot()
       
    def save_step_data(self):
        return
        self._metro.save_step_data()

    def failure_step(self, logging_str, logger):
        """
        Logging and reset after a failure step.
        """
        logger.info('{}: {}'.format(logging_str, self._action_history))
        info = {
            'road_network': -1.0,
            'life_circle': -1.0,
            'greeness': -1.0,
        }
        return self._get_obs(), self.FAILURE_REWARD, True, info

    def build(self,action):
        self._metro.add_station_from_action(int(action))

    def get_cost(self):
        return self._metro.get_cost()
    
    def fake_cost(self,action):
        return self._metro.fake_cost(int(action))

    def step(self, action,logger: logging.Logger) -> Tuple[List, float, bool, Dict]:
        
        if self._done:
            raise RuntimeError('Action taken after episode is done.')

        else:
            if self._stage == 'build':
                fake_cost = self.fake_cost(action)

                if self.get_cost() + fake_cost > self.budget:
                    self.transition_stage()
                else:
                    self.build(action)
                    self._action_history.append(int(action))

            if self._metro.get_done():
                self.transition_stage()

            reward, info = self.get_reward_info()
            if self._stage == 'done':
                self.save_step_data()

        return self._get_obs(), reward, self._done, info

    def reset(self,eval=False):
        # self._metro = copy.deepcopy(self._copy_metro)
        t1 = time.time()
        self._metro.reset(eval)
        self._action_history = []
        self._set_stage()
        self._done = False

        return self._get_obs()

    def _set_stage(self):
        self._stage = 'build'

    def transition_stage(self):
        if self._stage == 'build':
            self._stage = 'done'
            self._done = True
        else:
            raise RuntimeError('Error stage!')
        
    def plot_and_save(self,
                          save_fig: bool = False,
                          path: Text = None,
                          show=False, final = None) -> None:
        """
        Plot and save the gdf.
        """
        self._metro.plot_metro(self.ax0, self.ax1, final)
        if save_fig:
            assert path is not None
            plt.savefig(path, format='svg', transparent=True)
        if show:
            plt.show()

        plt.cla()
        plt.close('all')

    def visualize(self,
                  save_fig: bool = False,
                  path: Text = None,
                  show=False, final=None) -> None:
        """
        Visualize the city plan.
        """
        self.plot_and_save(save_fig, path, show, final)

    # def load_plan(self, gdf: GeoDataFrame) -> None:
    #     """
    #     Load a city plan.
    #     """
    #     self._plc.load_plan(gdf)

    # def score_plan(self, verbose=True) -> Tuple[float, Dict]:
    #     """
    #     Score the city plan.
    #     """
    #     reward, info = self._get_all_reward_info()
    #     if verbose:
    #         print(f'reward: {reward}')
    #         pprint(info, indent=4, sort_dicts=False)
    #     return reward, info

    # def get_init_plan(self) -> Dict:
    #     """
    #     Get the gdf of the city plan.
    #     """
    #     return self._plc.get_init_plan()
