from ipywidgets import interact
import ipywidgets as widgets

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

from sklearn.manifold import TSNE
import umap
from node2vec import Node2Vec

import networkx as nx
from shapely.geometry import LineString

class InteractiveLayout:

    def __init__(self, nodes, edges, method = "TSNE", nr_pos = 100, in_threshold = 0.4, out_threshold = 0.6, testing=True):
        """
        Starts the computation of the positions.

        :param str nodes: Dataframe containing the node information
        :param str edges: Dataframe containing the edge information
        :param str method: Sets the embedding method
        :param nr_pos: Sets the number of different layouts to be calculated
        :param in_threshold: Sets the threshold for the truncation step
        :param out_threshold: Sets the threshold for the truncation step
        :param testing: Enables the computation and return of metrics
        :return: InteractiveLayout object
        """

        self.nodes = nodes.sort_values(by=['id'])
        self.edges = edges
        self.edges["weight"] = 1
        # Order of nodes in graph could be different than in dataframe
        G = nx.from_pandas_edgelist(edges, 'source', 'target', 'weight')
        H = nx.Graph()
        H.add_nodes_from(sorted(G.nodes(data=True)))
        H.add_edges_from(G.edges(data=True))
        self.graph = H
        self.N = len(nodes)
        self.method = method
        self.embeddings = self.get_embeddings()
        self.sim_matrix = self.get_sim_matrix()
        self.adj_matrix = self.get_adj_matrix()
        self.group_matrix = self.get_group_matrix()
        self.nr_pos = nr_pos
        self.in_threshold = in_threshold
        self.out_threshold = out_threshold
        self.testing = testing
        self.positions, self.crossings_list, self.occlusions_list = self.get_positions()

    def get_embeddings(self):
        nodes = self.nodes
        edges = self.edges
        method = self.method

        if method == "TSNE":
            embeddings = TSNE(n_components=2, learning_rate='auto', init='random').fit_transform(nodes.drop('id', axis=1))
        elif method == "UMAP":
            embeddings = umap.UMAP().fit_transform(nodes.drop('id', axis=1))
        elif method == "node2vec":
            full_edges, virtual_nodes = self.get_real_and_virtual_edges(nodes, edges)

            embedding_graph = nx.from_pandas_edgelist(full_edges, 'source', 'target', 'weight')
            g2v = Node2Vec(embedding_graph, dimensions=2, walk_length=30, q=0.8, p=1, r=0.7, num_walks=150, workers=1, V=virtual_nodes[0])
        
            model = g2v.fit(window=10, min_count=1, batch_words=4)
            embedding_graph.remove_nodes_from(virtual_nodes[0])

            embeddings = np.zeros((len(nodes.id), 2))

            for i, name in enumerate(nodes['id']):
                embeddings[i] = model.wv[str(name)]

        return embeddings

    def get_real_and_virtual_edges(self, nodes, edges):
        '''
        Parameters:
        nodes: Pandas dataframe containing the node id in the first column ("id") and attributes in those following
        edges: Pandas dataframe containing source, target and edge weight
        
        Returns:
        full_edges: Dataframe identical to "edges" with added virtual edges between nodes and their attributes (weight 1)
        virtual_nodes: DataFrame containing all the virtual nodes added (for easier removal later on)
        '''
        virtual_nodes = []#[pd.DataFrame(columns = ['id'])]
        virtual_edges = pd.DataFrame(columns = ['source', 'target', 'weight'])
        for column in nodes.loc[:, nodes.columns != 'id'].columns:
            current_edges = pd.DataFrame({'source': nodes['id'], 'target': 'a'+str(column) + nodes[column].astype(str), 
                                        'weight': 1})
            
            virtual_nodes.append('a'+str(column) + nodes[column].astype(str).unique())# = pd.concat([virtual_nodes, 'a'+column + nodes[column].astype(str)], ignore_index = True)
            virtual_edges = pd.concat([virtual_edges, current_edges], ignore_index = True)
            
        full_edges = pd.concat([edges, virtual_edges], ignore_index = True)

        return full_edges, virtual_nodes
            

    def get_sim_matrix(self):
        N = self.N
        embeddings = self.embeddings

        distance_matrix = np.zeros((N, N))

        #Calc euclidean distances
        for i, p1 in enumerate(embeddings):
            for j, p2 in enumerate(embeddings):
                distance_matrix[i][j] = np.linalg.norm(p1 - p2)

        #Normalize
        distance_matrix = distance_matrix / distance_matrix.max()
        #Similarity
        similarity_matrix = 1 - distance_matrix

        np.fill_diagonal(similarity_matrix, 0)

        return similarity_matrix

    def get_adj_matrix(self):
        adj_matrix = nx.to_numpy_array(self.graph)
        
        #Normalize
        adj_matrix = adj_matrix / adj_matrix.max()

        return adj_matrix

    def get_group_matrix(self):
        N = self.N
        nodes = self.nodes

        if not ('label' in nodes.columns):
            return None

        group_matrix = np.zeros((N, N))

        for index, row in nodes.iterrows():
            group_matrix[index] = (nodes["label"] == row["label"])

        return group_matrix

    def get_positions(self):
        nr_pos = self.nr_pos
        nodes = self.nodes
        adj_matrix = self.adj_matrix
        sim_matrix = self.sim_matrix
        in_threshold = self.in_threshold
        out_threshold = self.out_threshold
        testing = self.testing
        group_matrix = self.group_matrix
        positions = []
        occlusions_list = []
        crossings_list = []
        edge_list = list(self.graph.edges)
        nr_nodes = self.N
        nr_edges = len(edge_list)

        for i, w in enumerate(np.arange(0,1,1/nr_pos)):

            emb_adj_matrix = w * adj_matrix + (1 - w) * sim_matrix

            # Normalization
            emb_adj_matrix = emb_adj_matrix / emb_adj_matrix.max()

            # Truncation
            if group_matrix is not None:
                emb_adj_matrix = np.where((emb_adj_matrix < in_threshold) & (group_matrix == 1), 0, emb_adj_matrix)
                emb_adj_matrix = np.where((emb_adj_matrix < out_threshold) & (group_matrix == 0), 0, emb_adj_matrix)
            else:
                emb_adj_matrix[emb_adj_matrix < 0.5] = 0

            # Fill diagonal with zero
            np.fill_diagonal(emb_adj_matrix, 0)
            
            emb_adj_dataframe = pd.DataFrame(data=emb_adj_matrix)
            emb_adj_dataframe.index = nodes['id']
            emb_adj_dataframe.columns = nodes['id']
            
            G = nx.from_pandas_adjacency(emb_adj_dataframe)
            np.random.seed(4040)
            position = nx.fruchterman_reingold_layout(G, iterations=50)
            positions.append(position)

            crossings = 0
            if testing:
                crossings = get_crossings(position, nr_edges, edge_list)

            crossings_list.append(crossings)

            occlusions = 0
            if testing:
                occlusions = get_occlusions(position, nr_nodes)
            occlusions_list.append(occlusions)

        return positions, crossings_list, occlusions_list

    def draw(self):
        positions = self.positions
        crossings_list = self.crossings_list
        occlusions_list = self.occlusions_list
        nodes = self.nodes
        graph = self.graph

        if 'label' in nodes.columns:
            node_color = nodes['label']
            labels = dict(zip(nodes.id, nodes.label))
            with_labels = True
        else:
            node_color = None
            labels = None
            with_labels = False
        
        def f(x):
            print("Crossings: ", crossings_list[x])
            print("Occlusions: ", occlusions_list[x])
            return nx.draw_networkx(graph, pos=positions[x], node_color=node_color, with_labels= with_labels, labels=labels,font_size=6, node_size=50, cmap='Set3')

        return interact(f, x=widgets.IntSlider(min=0, max=99, step=1, value=50))


def get_crossings(position, nr_edges, edge_list):
    crossings = 0
    for e1 in range(0, nr_edges-1):
        line = LineString([position[edge_list[e1][0]], position[edge_list[e1][1]]])
        for e2 in range(e1+1, nr_edges):
            if len({edge_list[e1][0], edge_list[e1][1], edge_list[e2][0], edge_list[e2][1]}) < 4:
                continue
            else:
                other = LineString([position[edge_list[e2][0]], position[edge_list[e2][1]]])
                crossings = crossings + line.intersects(other)
    return crossings

def get_occlusions(position, nr_nodes):
    occlusions = 0
    ordered_pos = sorted(position.items(), key=lambda e: e[1][0])
    for i in range(0, nr_nodes-1):
        for j in range(i+1, nr_nodes):
            if np.linalg.norm(ordered_pos[i][1] - ordered_pos[j][1]) < 0.05:
                occlusions += 1
            elif abs(ordered_pos[i][1][0] - ordered_pos[j][1][0]) >= 0.05:
                break
    return occlusions