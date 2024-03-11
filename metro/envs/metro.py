import networkx as nx
import numpy as np
import random
import copy

import geopandas as gpd
import pandas as pd
from shapely import geometry
from haversine import haversine, Unit
import matplotlib.pyplot as plt
import plotly.express as px

# metro line
class MLine(object):
    tot_seq = 0

    def __init__(self, name, stations, start, end, vstart=None, vend=None, vline=None) -> None:
        MLine.tot_seq += 1
        self.name = name
        self.seq = MLine.tot_seq
        self.stations = stations

        self.start_id = start[0]
        self.start = np.array(start[1])
        self.end_id = end[0]
        self.end = np.array(end[1])
        if vstart is None:
            self.vstart = self.start - self.end
        else:
            self.vstart = vstart
        if vend is None:
            self.vend = self.end - self.start
        else:
            self.vend = vend
        if vline is None:
            self.vline = self.start - self.end
        else:
            self.vline = vline

    def add_station(self, new_station, is_start):
        if is_start:
            tmp_pos = np.array(new_station[1])
            self.vstart = tmp_pos - self.start
            self.start = tmp_pos
            self.start_id = new_station[0]
            self.stations.insert(0, new_station[0])
        else:
            tmp_pos = np.array(new_station[1])
            self.vend = tmp_pos - self.end
            self.end = tmp_pos
            self.end_id = new_station[0]
            self.stations.append(new_station[0])

    def line_expand(self, old_station, new_station):
        if old_station[0] == self.start_id:
            self.add_station(new_station, True)
        elif old_station[0] == self.end_id:
            self.add_station(new_station, False)
        else:
            raise RuntimeError('[!]Expand failed')
        
    def get_start_and_end_pos(self):
        return self.start, self.end
    
    def get_start_and_end_id(self):
        return self.start_id, self.end_id
    
    def get_start_and_end_v(self):
        return self.vstart, self.vend
    
    def get_stations_num(self):
        return len(self.stations)
    
    def get_stations(self):
        return self.stations
    


class Metro(object):
    DUMMY_STATION = 0

    def __init__(self, data, budget, max_line, min_num, expand, cost, origin=None, destination=None, shape_param=None) -> None:

        self.G, self.region_processed, self.od_pair, self.line_info = data
        self.G = nx.Graph(self.G)

        self.cost_per_kmeter,self.cost_per_station,self.cost_per_trans_station = cost
        self.budget = budget
        self.max_line = max_line
        self.min_num = min_num
        self.build_line_num = 0
        self.expand = expand

        self.total_cost = 0
        self.total_od = 0

        self.node_list = [n for n in self.G.nodes()]
        self.node_list_idx = dict(zip(self.node_list, [i for i in range(len(self.node_list))]))
        self.node_num = len(self.node_list)
        self.node_pos = nx.get_node_attributes(self.G, 'pos')
        # self.sample_node_list = [n for n in self.G.nodes()] + [0] * (self.node_num // 3)

        self.edge_list = [e for e in self.G.edges()]
        # self.edge_num = len(self.edge_list)
        self.edge_build_idx = []
        attributes_idx = {}
        for idx in range(len(self.edge_list)):
            attributes_idx[self.edge_list[idx]] = {}
            attributes_idx[self.edge_list[idx]]['idx'] = idx
            attributes_idx[self.edge_list[idx]]['metro_dis'] = 1e6
        nx.set_edge_attributes(self.G, attributes_idx)

        self.done = 0
        self.build_new_line = True
        self.building_line = None
        
        

        # if self.origin != self.DUMMY_STATION and self.destination != self.DUMMY_STATION:
        #     poso = (self.node_pos[self.origin])
        #     pose = (self.node_pos[self.destination]) 
        #     self.line_v = pose-poso
        # else:
        #     self.line_v = np.array([0,0])

        self.build_node = []
        self.build_node_od = {}
        self.station = {node: 0 for node in self.node_list}
        self.seq = {node: [0,0,0,0] for node in self.node_list}
        self.build_edge = []
        self.metro_lines = []
        self.min_dis = {}

        self.env_init()

    def env_init(self):
        self.init_build_line()
        self.init_min_dis()
        self.init_mask()
        self.init_feature()

    def init_build_line(self):
        # print('init')
        self.metro_lines = []

        line_seq, lines = copy.deepcopy(self.line_info)
        self.init_lines = line_seq

        for line_name in self.init_lines:
            # print(line_name,len(lines[line_name]))
            start_id = lines[line_name][0]
            start_pos = self.node_pos[start_id]
            vstart = self.node_pos[start_id] - self.node_pos[lines[line_name][1]]
            end_id = lines[line_name][-1]
            end_pos = self.node_pos[end_id]
            vend = self.node_pos[end_id] - self.node_pos[lines[line_name][-2]]
            vline = start_pos - end_pos
            tmp_line = MLine(line_name, lines[line_name], [start_id,start_pos], [end_id,end_pos], vstart, vend, vline)
            self.metro_lines.append(tmp_line)

            for line_idx in range(len(lines[line_name])-1):
                pre = lines[line_name][line_idx]
                cur = lines[line_name][line_idx + 1]
                if not self.station[pre]:
                    self.build_node.append(pre)
                if not self.station[cur]:
                    self.build_node.append(cur)
                if pre != cur:
                    self.station[pre] += 1
                    self.station[cur] += 1


                    if line_idx == 0:
                        self.seq[pre][:2] = [tmp_line.seq, line_idx + 1]

                    if self.seq[cur][:2] == [0,0]:
                        self.seq[cur][:2] = [tmp_line.seq, line_idx + 2]
                    else:
                        self.seq[cur][2:] = [tmp_line.seq, line_idx + 2]

                    self.G[pre][cur]['metro_dis'] = self.G[pre][cur]['length']
                    idx = self.G[pre][cur]['idx']
                    self.edge_build_idx.append(idx)


        
        # self.pre_build_stations = []
        # for node in self.node_list:
        #     if self.station[node] > 0:
        #         self.pre_build_stations.append(node)
        self.pre_build_node = copy.deepcopy(self.build_node)
        for node in self.node_list:
            self.build_node_od[node] = 0
            for build_node in self.build_node:
                self.build_node_od[node] += (self.od_pair[node][build_node] + self.od_pair[build_node][node])

    def init_min_dis(self):
        metro_min_dis = dict(nx.shortest_path_length(self.G,weight='metro_dis'))
        self.min_dis_copy = copy.deepcopy(metro_min_dis)
        for n1 in self.build_node:
            self.min_dis[n1] = {}
            for n2 in self.build_node:
                self.min_dis[n1][n2] = metro_min_dis[n1][n2]

        for e in self.G.edges():
            self.G.add_edge(e[0],e[1],length=self.G[e[0]][e[1]]['length'],\
                            cost=self.G[e[0]][e[1]]['length'] * self.cost_per_kmeter + self.cost_per_station)

    def init_mask(self):
        self._cal_mask()

    def init_feature(self):
        self._cal_node_centrality()
        self._cal_node_od()
        self._cal_node_degree()
        self._cal_graph_node_feature()
        self._cal_edge_index()

    def reset(self,eval=False):
        # sample_origin = random.sample(self.sample_node_list,1)
        # if sample_origin == 0:
        #     self.origin = self.DUMMY_STATION
        # else:
        #     self.origin = sample_origin[0]

        # self.budget = random.randint(14,30)

        # if eval:
        #     self.origin = self.DUMMY_STATION
        #     self.budget = 30

        # if self.origin == self.DUMMY_STATION:
        #     self.build_node = []
        # else:
        #     self.build_node = [self.origin]
        self.build_new_line = True
        self.building_line = None
        self.done = 0
        self.build_line_num = 0

        self.build_node = []
        self.build_edge = []
        self.edge_build_idx = []
        self.station = {node: 0 for node in self.node_list}
        self.seq = {node: [0,0,0,0] for node in self.node_list}
        MLine.tot_seq = 0
        self.build_node_od = {}
        self.init_build_line()
        self.min_dis = copy.deepcopy(self.min_dis_copy)

        # if self.origin != self.DUMMY_STATION and self.destination != self.DUMMY_STATION:
        #     poso = (self.node_pos[self.origin])
        #     pose = (self.node_pos[self.destination]) 
        #     self.line_v = pose-poso
        # else:
        #     self.line_v = np.array([0,0])


        # self.line_v = np.array([0,0])
        self.reward = 0
        self.total_cost = 0
        self.total_od = 0
        self.init_mask()
        self._cal_node_endpoint()
        # print('[!]Reset')


    def _get_node_loc(self,n):
        longtitude_min = 116.161 - 0.01
        longtitude_max = 116.753 + 0.01
        latitude_min = 39.670 - 0.01
        latitude_max = 40.130 + 0.01
        x = (self.node_pos[n][0] - latitude_min) / (latitude_max - latitude_min)
        y = (self.node_pos[n][1] - longtitude_min) / (longtitude_max - longtitude_min)
        # return [0, 0]
        return [x, y]
    
    def _get_node_pre_build(self,n):
        if n in self.pre_build_stations:
            return 1
        else: 
            return 0
        
    def _cal_node_centrality(self):
        degree_cen = nx.degree_centrality(self.G)
        betweenness_cen = nx.betweenness_centrality(self.G,weight='length')
        eigenvector_cen = nx.eigenvector_centrality_numpy(self.G,weight='length')
        closeness_cen = nx.closeness_centrality(self.G, distance='length')
        self.node_centrality = {}
        for node in self.node_list:
            self.node_centrality[node] = [degree_cen[node], betweenness_cen[node],eigenvector_cen[node], closeness_cen[node]]
            # self.node_centrality[node] = [0,0,0,0]

    def _cal_node_od(self):
        self.node_od = {}
        for node in self.node_list:
            self.node_od[node] = [self.region_processed[node]['in']/1e5 , self.region_processed[node]['out']/1e5]
            near_node_od = 0
            for near in list(nx.neighbors(self.G,node)):
                try:
                    near_node_od += (self.od_pair[node][near] + self.od_pair[near][node])
                except:
                    raise RuntimeError(node,near)

            self.node_od[node].append(near_node_od/1e5)
            # self.node_od[node][0] = 0
            # self.node_od[node][1] = 0
            # self.node_od[node][2] = 0

    def _cal_graph_node_feature(self):
        self.graph_node_feature = {}
        for node in self.node_list:
            self.graph_node_feature[node] = self._get_node_loc(node) + self.node_centrality[node] + [self.node_degree_total[node]/1e1]\
                                            + [self.station[node]] + self.region_processed[node]['feature'] + self.node_od[node]

            # self.graph_node_feature[node] = self._get_node_loc(node) + self.node_centrality[node] + [self.node_degree_total[node]/1e1]\
            #                                 + [self._get_node_pre_build(node)] + self.region_processed[node]['feature'] + self.node_od[node]
            
            self.graph_node_feature[node][7] = self.graph_node_feature[node][7]/1e1
            self.graph_node_feature[node][12] = self.graph_node_feature[node][12]/1e1
            # self.graph_node_feature[node][7:17] = [0] * 10
            self.graph_node_feature[node][17:] = [0] * len(self.graph_node_feature[node][17:])

        
        
        self.node_centrality = None
        self.node_od = None
        self.region_processed = None
        self.node_degree_total = None

    def _cal_node_endpoint(self):
        is_endpoint = [[0,0,0]] * self.node_num
        self.node_is_endpoint = dict(zip(self.node_list, is_endpoint))

        for line in self.metro_lines:
            s,e = line.get_start_and_end_id()
            self.node_is_endpoint[s][0] = 1
            self.node_is_endpoint[e][0] = 1

        for node in self.pre_build_node:
                self.node_is_endpoint[node][1] = 1
       
        for node in self.build_node:
                self.node_is_endpoint[node][2] = 1

    def _cal_graph_node_feature_dim(self):
        return 26
    
    def get_numerical_dim(self):
        # if self.origin == self.DUMMY_STATION:
        #     return 7
        # else:
            return 7
    
    def get_node_dim(self):
        return self._cal_graph_node_feature_dim() + 5 + 4
        # return 14

    def _cal_edge_index(self):
        self.edge_index_dis = []
        self.edge_index_od = []

        for e in self.edge_list:
            idx1 = self.node_list_idx[e[0]]
            idx2 = self.node_list_idx[e[1]]
            self.edge_index_dis.append([idx1, idx2])

        for n1 in self.node_list:
            for n2 in self.od_pair[n1]:
                if n2 in self.node_list:
                    if self.od_pair[n1][n2] > 1e3 or self.od_pair[n2][n1] > 1e3:
                        idx1 = self.node_list_idx[n1]
                        idx2 = self.node_list_idx[n2]
                        if idx1 > idx2:
                            self.edge_index_od.append([idx1, idx2])
        

    def _cal_node_degree(self):
        self.node_degree_total = {}
        for n in self.node_list:
            self.node_degree_total[n] = len(list(self.G.neighbors(n)))
    
    def _get_node_feature(self,node):
        # return self.graph_node_feature[node] + [0] + [self.build_node_od[node]/5e5]
        # print(self.graph_node_feature[node])
        # print(self.station[node])
        # print(self.build_node_od[node])
        return self.graph_node_feature[node] + [self.build_node_od[node]/5e5] + self.node_is_endpoint[node] + [self.station[node]] + self.seq[node]

    def _get_numerical(self):

        numerical = [len(self.metro_lines),self.total_cost/self.budget,self.cost_per_kmeter/self.budget,self.cost_per_station/self.budget,self.cost_per_trans_station/self.budget]
        if self.build_new_line:
            numerical = numerical + [1,0]
        else:
            numerical = numerical + [0,1] 
                            
        return numerical

    def get_obs(self):
        numerical = self._get_numerical()
        node_feature = np.concatenate([[self._get_node_feature(n) for n in self.node_list]], axis=1)
        
        mask = self.get_mask()
        # print(len(self.edge_index_dis),len(self.edge_index_od))
        return numerical, node_feature, self.edge_index_dis, self.edge_index_od, mask


    # def find_nearest_station(self,new_station):
    #     neighbors = nx.neighbors(self.G,new_station)
        
    #     if not self.build_new_line: 
    #         for line in self.metro_lines:
    #             if if_expand:
    #                 nearest_station, if_expand = line.line_expand(new_station,neighbors)
    #                 return nearest_station, False
                
    #     else:
    #         nearest_station, if_expand = self.building_line.line_expand(new_station,neighbors)
    #         if if_expand:
    #             if self.building_line.get_station_num() >= self.MIN_STATION_NUM:
    #                 return nearest_station, False
    #             else:
    #                 return nearest_station, True
    #         else:
    #             raise RuntimeError('[!]Should build current line first, check mask')
            
    #     min_dis = 1e6
    #     nearest_station = self.DUMMY_STATION
    #     for s in neighbors:
    #         if self.station[s] > 0:
    #             if self.G[new_station][s]['length'] < min_dis and self.G[new_station][s]['idx'] not in self.edge_build_idx:
    #                 nearest_station = s
    #                 min_dis = self.G[new_station][s]['length']
    #     if nearest_station == self.DUMMY_STATION:
    #         print('len_neighbors:',len(neighbors))
    #         for s in neighbors:
    #             print(self.G[new_station][s]['length'])
    #         raise RuntimeError('[!]No valid neighbor')
            
    #     return nearest_station, True
    
    def update_dis_cost_reward(self,old_station,new_station,is_transfer,is_new_line):
        self.reward = 0
        cost = 0
        # if is_new_line:
        #     cost += 10 * self.cost_per_station
            
        if not is_transfer:
            self.min_dis[new_station] = {}
            for n in self.build_node:
                # try:
                self.min_dis[new_station][new_station] = 0
                self.min_dis[new_station][n] = self.min_dis[n][old_station] + self.G[new_station][old_station]['length']
                self.min_dis[n][new_station] = self.min_dis[new_station][n]
                # except:
                #     print(n,old_station,new_station)
                #     # print(self.min_dis[new_station][n])
                #     print(self.min_dis[n][old_station])
                #     print(self.G[old_station][new_station]['length'])

                linedis = haversine(self.node_pos[n], self.node_pos[new_station], unit=Unit.KILOMETERS)
                # linedis = np.linalg.norm((self.node_pos[n]) - (self.node_pos[new_station]))

                if linedis <= 3:
                    self.reward += (self.od_pair[n][new_station] + self.od_pair[new_station][n]) / 1e5
                else:
                    metrodis = self.min_dis[n][new_station]
                    self.reward += (self.od_pair[n][new_station] + self.od_pair[new_station][n]) / 1e5 * min(max(0.2, linedis/metrodis), 0.8)

                # print([n,new_station])    
                # print('l:',linedis)
                # print('m:',self.min_dis[n][new_station])
            self.build_node.append(new_station)
            cost += self.G[new_station][old_station]['cost']

        else:
            for n in self.build_node:
                linedis1 = haversine(self.node_pos[n], self.node_pos[new_station], unit=Unit.KILOMETERS)
                linedis2 = haversine(self.node_pos[n], self.node_pos[old_station], unit=Unit.KILOMETERS)
                # linedis1 = np.linalg.norm((self.node_pos[n]) - (self.node_pos[new_station]))
                # linedis2 = np.linalg.norm((self.node_pos[n]) - (self.node_pos[old_station]))
                # try:
                min_dis1 = min(self.min_dis[n][new_station], self.min_dis[n][old_station] + self.G[old_station][new_station]['length'])
                min_dis2 = min(self.min_dis[n][old_station], self.min_dis[n][new_station] + self.G[new_station][old_station]['length'])
                # except:
                #     print(self.min_dis[n][new_station])
                #     print(self.min_dis[n][old_station])
                #     print(self.G[old_station][new_station]['length'])

                if linedis1 > 3:
                    metrodis = self.min_dis[n][new_station]
                    r1 = min(max(0.2, linedis1/metrodis), 0.8)
                    r2 = min(max(0.2, linedis1/min_dis1), 0.8)
                    if min_dis1 < metrodis:
                        self.reward += (self.od_pair[n][new_station] + self.od_pair[new_station][n]) / 1e5 * (r2 - r1)

                if n != new_station:
                    if linedis2 > 3:
                        metrodis = self.min_dis[n][old_station]
                        r1 = min(max(0.2, linedis2/metrodis), 0.8)
                        r2 = min(max(0.2, linedis2/min_dis2), 0.8)
                        if min_dis2 < metrodis:
                            self.reward += (self.od_pair[n][old_station] + self.od_pair[old_station][n]) / 1e5 * (r2 - r1)
                
                # print('l1:',linedis1)
                # print('l2:',linedis2)
                # print('m:',metrodis)
                self.min_dis[n][new_station] = min_dis1
                self.min_dis[new_station][n] = min_dis1
                self.min_dis[n][old_station] = min_dis2
                self.min_dis[old_station][n] = min_dis2

            transfer_count = 0
            if self.station[new_station] <= 2:
                transfer_count += 1
            if self.station[old_station] <= 2:
                transfer_count += 1

            cost += self.G[new_station][old_station]['cost'] - self.cost_per_station + transfer_count * self.cost_per_trans_station

        self.total_od += self.reward
        # self.reward -= cost / 2e1
        self.total_cost += cost
        self.station[new_station] += 1
        self.station[old_station] += 1
        self.edge_build_idx.append(self.G[new_station][old_station]['idx'])


    def add_station_from_action(self,action):
        new_station = self.node_list[action]
        is_transfer = False
        is_new_line = False
        if new_station in self.build_node:
            is_transfer = True
        else:
            is_transfer = False

        # print(self.total_cost,self.total_cost + 12*self.cost_per_station,self.budget)
        if new_station in self.build_old_line_dict:
            is_new_line = False
            old_station = self.build_old_line_dict[new_station][0]
            cur_line = self.build_old_line_dict[new_station][1]
            cur_line.line_expand([old_station, self.node_pos[old_station]],  [new_station, self.node_pos[new_station]])
            self.node_is_endpoint[old_station][0] = 0
            self.node_is_endpoint[new_station][0] = 1
            self.node_is_endpoint[new_station][2] = 1

            if cur_line.get_stations_num() >= self.min_num and self.build_line_num < self.max_line:
                self.build_new_line = True
                # if self.total_cost + (self.min_num + 8) * self.cost_per_station + (self.min_num + 3) * self.cost_per_station > self.budget or self.build_line_num >= self.max_line:
                #     # print('[!]Stop')
                #     self.build_new_line = False
            else:
                self.build_new_line = False
            self.building_line = cur_line
            # print('[#IF]Old dict')

        elif new_station in self.build_new_line_dict:
            is_new_line = True
            old_station = self.build_new_line_dict[new_station]
            new_metro_line = MLine('new_{}'.format(len(self.metro_lines)+1), [old_station,new_station],\
                                   [old_station, self.node_pos[old_station]], [new_station, self.node_pos[new_station]])

            self.build_line_num += 1
            
            self.node_is_endpoint[old_station][0] = 1
            self.node_is_endpoint[new_station][0] = 1
            self.node_is_endpoint[new_station][0] = 1

            if self.seq[old_station][:2] == [0,0]:
                self.seq[old_station][:2] = [new_metro_line.seq, 1]
            else:
                self.seq[old_station][2:] = [new_metro_line.seq, 1]


            self.metro_lines.append(new_metro_line)
            self.build_new_line = False
            
            self.building_line = new_metro_line
            # print('[#IF]New dict')

        if self.seq[new_station][:2] == [0,0]:
            self.seq[new_station][:2] = [self.building_line.seq, self.building_line.get_stations_num()]
        else:
            self.seq[new_station][2:] = [self.building_line.seq, self.building_line.get_stations_num()]

        if old_station not in self.build_node:
            raise RuntimeError('[!]Invalid old station: ', old_station)
        self.build_edge.append([old_station,new_station])
        self.update_dis_cost_reward(old_station,new_station,is_transfer,is_new_line)

        if not is_transfer:
            for node in self.node_list:
                    self.build_node_od[node] += (self.od_pair[node][new_station] + self.od_pair[new_station][node])

    def _cal_mask(self):
        # print('[!]Cal_mask')
        mask = [0] * self.node_num

        self.build_old_line_dict = {}
        stations = []
        if not False:
            if self.build_line_num >= self.max_line and self.building_line.get_stations_num() >= self.min_num:
                lines = self.metro_lines
            elif not self.build_new_line or not self.expand:
                lines = [self.building_line]
                if lines[0] is None:
                    lines = self.metro_lines
            else:
                lines = self.metro_lines
            
            for line in lines:
                    stations += line.get_stations()

            for line in lines:
                idstart, idend = line.get_start_and_end_id()
                vstart, vend = line.get_start_and_end_v()
                
                ns = nx.neighbors(self.G, idstart)
                for n in ns:
                    if self.G[idstart][n]['idx'] not in self.edge_build_idx:
                        v1 = vstart
                        v2 = np.array(self.node_pos[n]) - np.array(self.node_pos[idstart])
                        # if np.linalg.norm(v1)*np.linalg.norm(v2) < 1e-6:
                        #     print(n,idstart)
                        #     print(v1,v2)
                        if np.dot(v1,v2)/(np.linalg.norm(v1)*np.linalg.norm(v2) + 1e-6) > 0.3 and np.dot(line.vline, v2)/(np.linalg.norm(line.vline)*np.linalg.norm(v2) + 1e-6) > 0.3 \
                            and self.total_cost + self.G[n][idstart]['cost'] <= self.budget:

                            self.build_old_line_dict[n] = [idstart,line]
                            idx = self.node_list_idx[n]
                            mask[idx] = 1

                ne = nx.neighbors(self.G, idend)
                for n in ne:
                    if self.G[idend][n]['idx'] not in self.edge_build_idx:
                        v1 = vend
                        v2 = np.array(self.node_pos[n]) - np.array(self.node_pos[idend])
                        if np.dot(v1,v2)/(np.linalg.norm(v1)*np.linalg.norm(v2) + 1e-6) > 0.3 and np.dot(line.vline, v2)/(np.linalg.norm(line.vline)*np.linalg.norm(v2) + 1e-6) < -0.3 \
                        and self.total_cost + self.G[n][idend]['cost'] <= self.budget:
                            self.build_old_line_dict[n] = [idend,line]
                            idx = self.node_list_idx[n]
                            mask[idx] = 1

        self.build_new_line_dict = {}
        if self.build_new_line:
            for bn in self.pre_build_node:
                ns = nx.neighbors(self.G, bn)
                for n in ns:
                    if self.G[bn][n]['idx'] not in self.edge_build_idx and n not in self.build_old_line_dict and self.total_cost + self.G[n][bn]['cost'] <= self.budget:
                        if n not in self.build_new_line_dict:
                            self.build_new_line_dict[n] = bn
                        else:
                            if self.G[bn][n]['length'] < self.G[self.build_new_line_dict[n]][n]['length']:
                                self.build_new_line_dict[n] = bn
                        idx = self.node_list_idx[n]
                        mask[idx] = 1

            for n in self.build_node:
                if n not in self.pre_build_node:
                    ns = nx.neighbors(self.G, n)
                    for nn in ns:
                        idx = self.node_list_idx[nn]
                        mask[idx] = 0

        # else:
        #     for n in self.build_node:
        #         if n not in self.pre_build_node and n not in stations:
        #             ns = nx.neighbors(self.G, n)
        #             for nn in ns:
        #                 idx = self.node_list_idx[nn]
        #                 mask[idx] = 0

        policy = 0
        if policy == 1:
            mask = [0] * self.node_num
            if self.build_new_line:
                for n in self.build_new_line_dict:
                    idx = self.node_list_idx[n]
                    mask[idx] = 2
            else:
                for n in self.build_old_line_dict:
                    idx = self.node_list_idx[n]
                    mask[idx] = 2

        elif policy == 2:
            mask = [0] * self.node_num
            od_max = 0
            id_max = 0

            if self.build_new_line:
                for n in self.build_new_line_dict:
                    idx = self.node_list_idxget_maskself.mask[n]
                    mask[idx] = 2
            else:
                for n in self.build_old_line_dict:
                    idx = self.node_list_idx[n]
                    if self.build_node_od[n] >= od_max:
                        od_max = self.build_node_od[n]
                        id_max = idx
                if id_max != 0:
                    mask[id_max] = 2


        if np.array(mask).sum() == 0:
            self.done = 1
        else:
            self.done = 0

        self.mask = mask

    def get_mask(self):
        # print(np.sum(self.mask))
        return self.mask

    def fake_cost(self,action):
        new_station = self.node_list[action]

        if new_station in self.build_old_line_dict:
            old_station = self.build_old_line_dict[new_station][0]
        elif new_station in self.build_new_line_dict:
            old_station = self.build_new_line_dict[new_station]
        else:
            # print(self.build_old_line_dict)
            # print(self.build_new_line_dict)
            raise RuntimeError('[!]Invalid action: ',new_station)

        transfer_count = 0
        if self.station[new_station] <= 2:
            transfer_count += 1
        if self.station[old_station] <= 2:
            transfer_count += 1

        return self.G[new_station][old_station]['cost'] - self.cost_per_station + transfer_count * self.cost_per_trans_station

    def get_reward(self):
        # self.total_od += self.reward
        if self.done and self.budget > 3:
            return self.reward + self.total_cost - self.budget
        else:
            return self.reward
    
    def get_cost(self):
        return self.total_cost
    
    def get_od(self):
        return self.total_od
    
    def get_done(self):
        self._cal_mask()
        # if self.done:
        #     print('[!]Done')
        return self.done
            
    # def get_mask(self):
    #     # print('cur:',self.cur_station)
    #     if self.cur_station == self.DUMMY_STATION:
    #         return [1] * self.node_num
           
    #     mask = [0] * self.node_num

    #     reward_max = 0
    #     reward_max_idx = 0

    #     candidate_station = nx.neighbors(self.G, self.cur_station)

    #     pos1 = (self.node_pos[self.pre_station])
    #     pos2 = (self.node_pos[self.cur_station])
    #     v1 = pos2 - pos1

    #     for s in candidate_station:
    #         if self.station[s] == 0 or s in self.pre_build_stations:
    #             if s != self.pre_station and s != self.cur_station:
    #                 pos3 = (self.node_pos[s])
    #                 v2 = pos3 - pos2

    #                 if self.pre_station != self.cur_station:
    #                     ang1 = np.arctan2(np.abs(np.cross(v2, self.line_v)), np.dot(v2, self.line_v))
    #                     ang2 = np.arctan2(np.abs(np.cross(v2, v1)), np.dot(v2, v1))
    #                     ang = max(ang1,ang2)
    #                 else:
    #                     ang = 0
    #                 # print(s,ang)
    #                 if ang <= np.pi/2:
    #                     # print(self.min_cost[s])
    #                     if self.destination != self.DUMMY_STATION:
    #                         fake_cost = self.min_cost[s]
    #                     else:
    #                         fake_cost = self.G[s][self.cur_station]['cost']
    #                     if fake_cost + self.total_cost <= self.budget:
    #                         idx_n = self.node_list_idx(s)
    #                         idx_e = self.G[self.cur_station][s]['idx']
    #                         if idx_e not in self.edge_build_idx:
    #                             idx = idx_n
    #                             mask[idx] = 1
    #                             # print(s)   
                            
    #                             f_reward = self.od_pair[self.cur_station][s] + self.od_pair[s][self.cur_station]
    #                             if f_reward >= reward_max:
    #                                 reward_max = f_reward
    #                                 reward_max_idx = idx_n

    #     if np.sum(mask) == 0:
    #         self.done = 1
    #     else:
    #         mask[reward_max_idx] = 2

    #     return mask

    def plot_metro(self,ax0,ax1,final):
        if final is not None:
            ax = copy.deepcopy(ax0)

            #resize ax to 20,20
            ax.set_xlim(0,20)
            ax.set_ylim(0,20)

            node_size1 = 0.3
            node_size2 = 0.8
            pre_line_width = 2
            new_line_width = 3

            # station_info = pd.read_csv('~/code/data/station.csv',encoding='gbk')
            # lines = ['line10','line13','line7']
            # for name in lines:
            #     stations = station_info[station_info['路线名称'] == name]
            #     lat = np.array(stations['gd纬度'])
            #     log = np.array(stations['gd经度'])
            #     pos = np.array([lat,log]).transpose()
            #     pre_node = pd.DataFrame({'lat':lat,'log':log})
            #     pre_node = gpd.GeoDataFrame(pre_node,geometry=gpd.points_from_xy(log,lat))
            #     pre_node.plot(ax=ax,color='grey',alpha=0.8)

            #     pre_line = []
            #     for i in range(len(pos)-1):
            #         pre_line.append(geometry.LineString([[pos[i,1],pos[i,0]],[pos[i+1,1],pos[i+1,0]]]))
            #     pre_line = gpd.GeoSeries(pre_line)
            #     ax = pre_line.plot(ax=ax,color='r',alpha=0.5)
        else:
            # ax = copy.deepcopy(ax1)
            ax = plt.figure(figsize=(20,20))
            ax = None
            node_size1 = 0.2
            node_size2 = 3
            pre_line_width = 0.3
            new_line_width = 3

        # plot old nodes with circle
        # pos_plot = []
        # for idr in self.node_list:
        #     if not self.station[idr] or idr in self.pre_build_node:
        #         pos_plot.append(self.node_pos[idr])
        # lat = np.array(pos_plot).transpose()[0]
        # log = np.array(pos_plot).transpose()[1]
        # df = pd.DataFrame({'lat':lat,'log':log})
        # node_build = gpd.GeoDataFrame(df,geometry=gpd.points_from_xy(df.log,df.lat))
        # ax = node_build.plot(ax=ax,color='black',markersize=node_size1,zorder=3)
        line_color = px.colors.qualitative.Plotly + ['maroon','navy','olive','purple','tan','aqua','azure','coral','crimson','fuchsia','gold','khaki','orchid','plum','salmon','sienna','violet','wheat']
        # print(line_color)
        # line_color = ['orange','green','blue','yellow','brown','cyan','magenta','lime','maroon','navy','olive','purple','tan','aqua','azure','coral','crimson','fuchsia','gold','khaki','orchid','plum','salmon','sienna','violet','wheat']
        for idx in range(len(self.init_lines)):
            name = self.init_lines[idx]
            pre_nodes = []
            for idr in self.line_info[-1][name]:
                if idr in self.node_pos:
                    n = np.flip(np.array(self.node_pos[idr]))
                    pre_nodes.append(n)

            lat = np.array(pre_nodes).transpose()[0]
            log = np.array(pre_nodes).transpose()[1]
            df = pd.DataFrame({'lat':lat,'log':log})
            node_build = gpd.GeoDataFrame(df,geometry=gpd.points_from_xy(df.lat,df.log))
            ax = node_build.plot(ax=ax,color='black',markersize=node_size1,zorder=3,marker='o')

            line = geometry.LineString(pre_nodes)
            line = gpd.GeoSeries(line)
            ax = line.plot(ax=ax,color=line_color[idx],linewidth=pre_line_width,zorder=2,alpha=0.5)

        # plot new nodes with square
        pos_plot = []
        for idr in self.build_node:
            if idr not in self.pre_build_node:
                pos_plot.append(self.node_pos[idr])
        if len(pos_plot) == 0:
            fid1 = [426, 389, 805,  88]
            # value1 = [0.6053, 0.1424, 0.0572, 0.0534, 0.0392]
            # value1 = [0.29841014,0.18783593,0.1724951 ,0.17184087,0.16941797]
            value1 = [0.45878029,0.15801734,0.12986878,0.12873741]
            for pid1 in fid1:
                idr = self.node_list[pid1]
                lat1 = self.node_pos[idr][0]
                log1 = self.node_pos[idr][1]
                # plot node
                df = pd.DataFrame({'lat':[lat1],'log':[log1]})
                node_build = gpd.GeoDataFrame(df,geometry=gpd.points_from_xy(df.log,df.lat))
                ax = node_build.plot(ax=ax,color='red',markersize=(80-value1[fid1.index(pid1)]*90)*3,zorder=3,marker='o',alpha=min(1,value1[fid1.index(pid1)]*1.5))

            fid2 = [ 88, 427]
            # value2 = [9.9647e-01, 2.0102e-03, 8.1702e-04, 7.0282e-04, 1.5425e-06]
            # value2 = [0.40354699,0.1492814,0.14910339,0.14908636,0.14898185]
            value2 = [0.71220724,0.07213509]
            for pid2 in fid2:
                idr = self.node_list[pid2]
                lat2 = self.node_pos[idr][0]
                log2 = self.node_pos[idr][1]
                # plot node
                df = pd.DataFrame({'lat':[lat2],'log':[log2]})
                node_build = gpd.GeoDataFrame(df,geometry=gpd.points_from_xy(df.log,df.lat))
                ax = node_build.plot(ax=ax,color='blue',markersize=(80-value2[fid2.index(pid2)]*90)*3,zorder=3,marker='o',alpha=min(1,value2[fid2.index(pid2)]*1.5))


        else:
            lat = np.array(pos_plot).transpose()[0]
            log = np.array(pos_plot).transpose()[1]
            df = pd.DataFrame({'lat':lat,'log':log})
            node_build = gpd.GeoDataFrame(df,geometry=gpd.points_from_xy(df.log,df.lat))
            ax = node_build.plot(ax=ax,color='black',markersize=node_size2,zorder=3,marker='o')

        for idx in range(len(self.metro_lines)):
            name = self.metro_lines[idx].name
            if name.startswith('new'):
                pos_plot = []
                for s in self.metro_lines[idx].get_stations():
                    pos_plot.append(self.node_pos[s])

                lat = np.array(pos_plot).transpose()[0]
                log = np.array(pos_plot).transpose()[1]
                df = pd.DataFrame({'lat':lat,'log':log})
                node_build = gpd.GeoDataFrame(df,geometry=gpd.points_from_xy(df.log,df.lat))
                ax = node_build.plot(ax=ax,color='red',markersize=node_size2,zorder=3,marker='o')
                
        
        # plot pre_build lines


        # plot build lines
        for idx in range(len(self.metro_lines)):
            name = self.metro_lines[idx].name
            if name.startswith('new'):
                plot_nodes = []
                for s in self.metro_lines[idx].get_stations():
                    n = np.flip(np.array(self.node_pos[s]))
                    plot_nodes.append(n)
                ax = gpd.GeoSeries(geometry.LineString(plot_nodes)).plot(ax=ax,color=line_color[idx],linewidth=new_line_width,zorder=2)
            
            else:
                plot_nodes = []
                pre_nodes = None
                for s in self.metro_lines[idx].get_stations():
                    n = np.flip(np.array(self.node_pos[s]))
                    if s not in self.pre_build_node:
                        if len(plot_nodes) == 0 and pre_nodes is not None:
                            plot_nodes.append(pre_nodes)
                        plot_nodes.append(n)
                    if s in self.pre_build_node and len(plot_nodes) > 0:
                        plot_nodes.append(n)
                        ax = gpd.GeoSeries(geometry.LineString(plot_nodes)).plot(ax=ax,color=line_color[idx],linewidth=new_line_width,zorder=2)
                        plot_nodes = []
                    if s in self.pre_build_node:
                        pre_nodes = n

                if len(plot_nodes) != 0:
                    ax = gpd.GeoSeries(geometry.LineString(plot_nodes)).plot(ax=ax,color=line_color[idx],linewidth=new_line_width,zorder=2)
        
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        plt.xticks([])
        plt.yticks([])

    # def plot_metro(self,ax0,ax1,final):
    #     if final is not None:
    #         ax = copy.deepcopy(ax0)
    #     else:
    #         ax = copy.deepcopy(ax1)
    #     # plot all nodes
    #     pos_all = []
    #     line_build = []
    #     for idr in self.node_list:
    #         pos_all.append(self.node_pos[idr])
    #     lat = np.array(pos_all).transpose()[0]
    #     log = np.array(pos_all).transpose()[1]
    #     df = pd.DataFrame({'lat':lat,'log':log})
    #     node_build = gpd.GeoDataFrame(df,geometry=gpd.points_from_xy(df.log,df.lat))
    #     ax = node_build.plot(ax=ax,color='purple',markersize=12,zorder=3)
        
    #     # plot pre_lines
    #     # pre_line = []
    #     line_color = ['orange','green','blue','yellow','pink','brown','black','grey','cyan','magenta','purple']
    #     for idx in range(len(self.init_lines)):
    #         name = self.init_lines[idx]
    #         color = line_color[idx]
    #         pre_nodes = []
    #         for idr in self.line_info[-1][name]:
    #             if idr in self.node_pos:
    #                 n = np.flip(np.array(self.node_pos[idr]))
    #                 pre_nodes.append(n)
    #         line = geometry.LineString(pre_nodes)
    #         line = gpd.GeoSeries(line)
    #         ax = line.plot(ax=ax,color=color,linewidth=3,zorder=2)
    #         # pre_line.append(geometry.LineString(pre_nodes))
    #     # pre_line = gpd.GeoSeries(pre_line)
    #     # ax = pre_line.plot(ax=ax,color='orange',linewidth=3,zorder=2)

    #     # plot build nodes
    #     pos_build = []
    #     for idr in self.build_node:
    #         pos_build.append(self.node_pos[idr])
    #     lat = np.array(pos_build).transpose()[0]
    #     log = np.array(pos_build).transpose()[1]
    #     df = pd.DataFrame({'lat':lat,'log':log})
    #     node_build = gpd.GeoDataFrame(df,geometry=gpd.points_from_xy(df.log,df.lat))
    #     ax = node_build.plot(ax=ax,color='black',markersize=25,zorder=3)

    #     # plot build lines
    #     build_edges = []
    #     for pair in self.build_edge:
    #         edge = []
    #         edge.append(np.flip(np.array(self.node_pos[pair[0]])))
    #         edge.append(np.flip(np.array(self.node_pos[pair[1]])))
    #         build_edges.append(geometry.LineString(edge))
    #     edge = gpd.GeoSeries(build_edges)
    #     ax = edge.plot(ax=ax,color='red',linewidth=3,zorder=2)

    #     # line_build_line = []
    #     # for line in self.metro_lines:
    #     #     # print(line.name)
    #     #     # if line.name.startswith('new_',0):
    #     #     line_build_nodes = []
    #     #     for s in line.get_stations():
    #     #         if s in self.node_pos:
    #     #             n = np.flip(np.array(self.node_pos[s]))
    #     #             line_build_nodes.append(n)
    #     #     line_build_line.append(geometry.LineString(line_build_nodes))
    #     # if len(line_build_line) > 0:
    #     #     line_build = gpd.GeoSeries(line_build_line)
    #     #     ax = line_build.plot(ax=ax,color='red',linewidth=3)


