import networkx as nx
import numpy as np

class Metro(object):
    DUMMY_STATION = 0

    def __init__(self, data, budget, cost, origin=None, destination=None, shape_param=None) -> None:
        self.G, self.region_processed, self.od_pair, self.line_info = data
        
        self.origin = origin
        if self.origin == None:
            self.origin = self.DUMMY_STATION

        self.destination = destination
        if self.destination == None:
            self.destination = self.DUMMY_STATION

        self.shape_param = shape_param
        self.cost_per_kmeter,self.cost_per_station,self.cost_per_trans_station = cost
        self.budget = budget
        self.done = 0

        self.total_cost = 0
        self.total_od = 0

        self.node_list = [n for n in self.G.nodes()]
        self.node_num = len(self.node_list)
        self.node_pos = nx.get_node_attributes(self.G, 'pos')

        self.edge_list = [e for e in self.G.edges()]
        self.edge_num = len(self.edge_list)

        attributes_idx = {}
        for idx in range(self.edge_num):
            attributes_idx[self.edge_list[idx]] = {}
            attributes_idx[self.edge_list[idx]]['idx'] = idx
        nx.set_edge_attributes(self.G, attributes_idx)

        self.edge_build_idx = []
        self.change_idx = -1

        self.station = dict(zip(self.node_list, [0] * len(self.node_list)))
        self.env_init()

    def __repr__(self) -> str:
        raise RuntimeError('TODO:')
    
    def reset(self):
        self.total_cost = 0
        self.total_od = 0
        self.done = 0
        self.change_idx = -1
        self.edge_build_idx = []
        self.station = dict(zip(self.node_list, [0] * len(self.node_list)))
        self.build_line()
        self.pre_station = [self.origin,self.origin]
        self.cur_station = [self.origin,self.origin]
        # print('reset')


    def _get_node_loc(self,n):
        longtitude_min = 116.161 - 0.01
        longtitude_max = 116.753 + 0.01
        latitude_min = 39.670 - 0.01
        latitude_max = 40.130 + 0.01
        x = (self.node_pos[n][0] - latitude_min) / (latitude_max - latitude_min)
        y = (self.node_pos[n][1] - longtitude_min) / (longtitude_max - longtitude_min)
        return [x, y]
    
    def env_init(self):
        self.feature_init()
        self.build_line()
        self.cal_min_dis()

        self.pre_station = [self.origin,self.origin]
        self.cur_station = [self.origin,self.origin]

    def build_line(self):
        self.pre_build_stations = []
        self.cross = 0
        self.cross_reward = 0
        # return
        # line_seq, lines = self.line_info
        # for line_name in list(lines.keys())[0:3]:
        #     line = lines[line_name]
        #     for line_idx in range(len(line)-1):
        #         pre = line[line_idx]
        #         cur = line[line_idx + 1]
        #         if pre != cur:
        #             self.station[pre] += 1
        #             self.station[cur] += 1
        #             self.pre_build_stations.append(pre)
        #             self.pre_build_stations.append(cur)
        #             idx = self.G[pre][cur]['idx']
        #             self.edge_build_idx.append(idx)
        
        self.build_node_od = {}
        for node in self.node_list:
            self.build_node_od[node] = 0
            for pre_node in self.pre_build_stations:
                self.build_node_od[node] += self.od_pair[node][pre_node] + self.od_pair[pre_node][node]

    def cal_min_dis(self):
        for e in self.G.edges():
            if_trans = 0
            # if self.station[e[0]] > 0:
            #     trans_cost += 1
            if self.station[e[1]] > 0:
                if_trans = 1

            self.G.add_edge(e[0],e[1],length=self.G[e[0]][e[1]]['length'],\
                            cost=self.G[e[0]][e[1]]['length'] * self.cost_per_kmeter\
                                + self.cost_per_station\
                                    + if_trans * self.cost_per_trans_station)
            
        self.min_cost = dict(zip(self.node_list, [0] * len(self.node_list)))
        if self.destination != self.DUMMY_STATION:
            print(self.destination)
            self.min_cost = nx.shortest_path_length(self.G, target=self.destination,weight='cost')

    def feature_init(self):
        self._cal_node_centrality()
        self._cal_node_od()
        self._cal_graph_node_feature()
        self._cal_edge_index()
        self._cal_node_degree()
        
    def _cal_node_centrality(self):
        degree_cen = nx.degree_centrality(self.G)
        betweenness_cen = nx.betweenness_centrality(self.G,weight='length')
        eigenvector_cen = nx.eigenvector_centrality_numpy(self.G,weight='length')
        closeness_cen = nx.closeness_centrality(self.G, distance='length')
        self.node_centrality = {}
        for node in self.node_list:
            self.node_centrality[node] = [degree_cen[node], betweenness_cen[node],eigenvector_cen[node], closeness_cen[node]]

    def _cal_node_od(self):
        self.node_od = {}
        for node in self.node_list:
            self.node_od[node] = [self.region_processed[node]['in']/1e4 , self.region_processed[node]['out']/1e4]
            near_node_od = 0
            for near in list(nx.neighbors(self.G,node)):
                near_node_od += self.od_pair[node][near] + self.od_pair[near][node]

            self.node_od[node].append(near_node_od/1e3)

    def _cal_graph_node_feature(self):
        self.graph_node_feature = {}
        for node in self.node_list:
            self.graph_node_feature[node] = self._get_node_loc(node) + self.node_centrality[node] + self.node_od[node]
                                            # + self.region_processed[node]['feature'].tolist() + self.node_od[node]
            
    def _cal_graph_node_feature_dim(self):
        return 9
    
    def get_numerical_dim(self):
        # print(1 + 3 + 2 * self.get_node_dim())
        # return 1 + 3 + 2 * (self.get_node_dim()-1)
        return 4
    
    def get_node_dim(self):
        return self._cal_graph_node_feature_dim() + 2

    def _cal_edge_index(self):
        self.edge_index_dis = []
        self.edge_index_od = []

        for e in self.edge_list:
            idx1 = self.node_list.index(e[0])
            idx2 = self.node_list.index(e[1])
            self.edge_index_dis.append([idx1, idx2])

        for n1 in self.node_list:
            for n2 in self.od_pair[n1]:
                if self.od_pair[n1][n2] > 1e4 or self.od_pair[n2][n1] > 1e4:
                    idx1 = self.node_list.index(n1)
                    idx2 = self.node_list.index(n2)
                    if idx1 > idx2:
                        self.edge_index_od.append([idx1, idx2])

    def _cal_node_degree(self):
        self.node_degree_total = {}
        self.node_degree_build = self.station
        for n in self.node_list:
            self.node_degree_total[n] = len(list(self.G.neighbors(n)))
    
    def _get_node_feature(self,node):
        return self.graph_node_feature[node] + [self.station[node]] + [self.build_node_od[node]]

    def _get_numerical(self):
        # if self.origin == self.DUMMY_STATION:
        #     o = [0] * self._cal_graph_node_feature_dim()
        # else:
        #     o = self.graph_node_feature[self.origin]

        # if self.destination == self.DUMMY_STATION:
        #     e = [0] * self._cal_graph_node_feature_dim()
        # else:
        #     e = self.graph_node_feature[self.destination]

        return [self.total_cost/self.budget]\
              + [self.cost_per_kmeter/self.budget,self.cost_per_station/self.budget,self.cost_per_trans_station/self.budget]

    def get_obs(self):
        numerical = self._get_numerical()
        node_feature = np.concatenate([[self._get_node_feature(n) for n in self.node_list]], axis=1)

        mask = self.get_mask()
        # return numerical, node_feature, edge_feature, self.edge_index_dis, self.edge_index_od, mask
        return numerical, node_feature, self.edge_index_dis, self.edge_index_od, mask
    
    def add_station_from_action(self,action):
        choose_station = self.node_list[action]

        if self.change_idx == 0:
            self.pre_station[0] = self.cur_station[0]
            self.cur_station[0] = self.node_list[action]
        elif self.change_idx == 1:
            self.pre_station[1] = self.cur_station[1]
            self.cur_station[1] = self.node_list[action]
        elif self.cur_station[0] == self.DUMMY_STATION and self.cur_station[1] == self.DUMMY_STATION:
            self.total_cost += self.cost_per_station
            self.station[choose_station] += 1
            self.cur_station = [choose_station] *2
            self.pre_station = [choose_station] *2
            return
        else:
            raise RuntimeError('Invalid stage')
        
        change_idx = self.change_idx

        if not self.G.has_edge(self.pre_station[change_idx], self.cur_station[change_idx]):
            print(self.pre_station[change_idx], self.cur_station[change_idx])
            for s in nx.neighbors(self.G,self.pre_station[change_idx]):
                print(s)
            raise RuntimeError('Error line!')
        else:
            # print(self.pre_station,self.cur_station)
            idx = self.G[self.pre_station[change_idx]][self.cur_station[change_idx]]['idx']

            if idx in self.edge_build_idx:
                print(self.edge_list[idx])
                print(change_idx,self.pre_station,self.cur_station)
                raise RuntimeError('Exist line!')
            else:
                self.total_cost += self.G[self.pre_station[change_idx]][self.cur_station[change_idx]]['cost']
                if self.pre_station[change_idx] == self.origin:
                    self.total_cost += self.cost_per_station

                self.station[self.pre_station[change_idx]] += 1
                self.station[self.cur_station[change_idx]] += 1
                self.edge_build_idx.append(idx)

                for node in self.node_list:
                    self.build_node_od[node] += self.od_pair[node][self.cur_station[change_idx]] + self.od_pair[self.cur_station[change_idx]][node]

    def fake_cost(self,action):
        
        choose_station = self.node_list[action]
        if choose_station in self.pre_build_stations:
            if self.cross == 0:
                self.cross = -1 
            # print('cross')
        
        if self.G.has_edge(self.cur_station[0],choose_station):
            if self.cur_station[0] != choose_station and self.G[self.cur_station[0]][choose_station]['idx'] not in self.edge_build_idx:
                self.change_idx = 0

        if self.G.has_edge(self.cur_station[1],choose_station):
            if self.cur_station[1] != choose_station and self.G[self.cur_station[1]][choose_station]['idx'] not in self.edge_build_idx:
                self.change_idx = 1

        change_idx = self.change_idx
        # print(change_idx,action,choose_station,self.pre_station,self.cur_station)

        if self.change_idx == -1:
            if self.station[choose_station] > 0:
                return self.cost_per_trans_station
            else:
                return self.cost_per_station

        elif self.cur_station[change_idx] == self.origin:
            return self.cost_per_station + self.G[self.cur_station[change_idx]][choose_station]['cost']
        else:
            return self.G[self.cur_station[change_idx]][choose_station]['cost']

    # def add_station_from_line(self,station_id):
    #     self.pre_station = self.cur_station
    #     self.cur_station = station_id
    #     edge = set(self.pre_station,self.cur_station)

    #     if self.G[self.pre_station][self.cur_station] == None:
    #         raise RuntimeError('Discontinuous line!')
    #     elif self.edge[edge][-1] != 0:
    #         raise RuntimeError('Exist line!')
    #     else:
    #         length = self.edge[edge][1]
    #         self.total_cost += self.cost_per_kmeter * length
    #         if self.station[self.cur_station] > 0:
    #             self.total_cost += self.cost_per_trans_station
    #         else:
    #             self.total_cost += self.cost_per_station

    #         if self.pre_station != self.origin:
    #             self.station[self.pre_station] += 1
    #         self.station[self.cur_station] += 1

    def get_reward(self):
        od_add = self.get_reward_node(self.cur_station[self.change_idx])
        self.total_od += od_add
        
        return od_add

    def get_reward_node(self,cur_node):
        od_add = 0
        if self.cross == 0:
            for id1 in self.node_list:
                if id1 not in self.pre_build_stations and self.station[id1] > 0:
                    od_add += self.od_pair[id1][cur_node]
                    od_add += self.od_pair[cur_node][id1]

            for pre_station in self.pre_build_stations:
                self.cross_reward += self.od_pair[id1][pre_station]
                self.cross_reward += self.od_pair[pre_station][id1]

        elif self.cross == -1:
            self.cross == 1
            od_add += self.cross_reward
            for id1 in self.node_list:
                if self.station[id1] > 0:
                    od_add += self.od_pair[id1][cur_node]
                    od_add += self.od_pair[cur_node][id1]
        else:
            for id1 in self.node_list:
                if self.station[id1] > 0:
                    od_add += self.od_pair[id1][cur_node]
                    od_add += self.od_pair[cur_node][id1]

        od_add = od_add / 1e5

        return od_add
    
    def get_cost(self):
        return self.total_cost
    
    def get_od(self):
        return self.total_od
    
    def get_done(self):
        self.get_mask()
        return self.done
            
    def get_mask(self):
        if self.cur_station[0] == self.DUMMY_STATION and self.cur_station[1] == self.DUMMY_STATION:
            return [1] * self.node_num
        
        mask = [0] * self.node_num

        reward_max = 0
        reward_max_idx = 0
        for i in range(2):
            candidate_station = nx.neighbors(self.G, self.cur_station[i])
            pos1 = np.array(self.node_pos[self.pre_station[i]])
            pos2 = np.array(self.node_pos[self.cur_station[i]])
            v1 = pos2 - pos1

            for s in candidate_station:
                if s not in self.pre_station and s not in self.cur_station:
                    pos3 = np.array(self.node_pos[s])
                    v2 = pos3 - pos2

                    if self.pre_station[i] != self.cur_station[i]:
                        ang = np.arctan2(np.abs(np.cross(v1, v2)), np.dot(v1, v2))
                    else:
                        ang = 0
                    
                    if ang < np.pi/2 + 0.1:
                        if self.min_cost[s] + self.total_cost <= self.budget:
                            idx_n = self.node_list.index(s)
                            idx_e = self.G[self.cur_station[i]][s]['idx']
                            if idx_e not in self.edge_build_idx:
                                idx = idx_n
                                mask[idx] = 1   
                            
                                f_reward = self.od_pair[self.cur_station[i]][s] + self.od_pair[s][self.cur_station[i]]
                                if f_reward >= reward_max:
                                    reward_max = f_reward
                                    reward_max_idx = idx_n

        if np.sum(mask) == 0:
            self.done = 1
        else:
            mask[reward_max_idx] = 2

        return mask



