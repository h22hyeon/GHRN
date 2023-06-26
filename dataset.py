from dgl.data import FraudDataset
from dgl.data.utils import load_graphs
import dgl
import torch
import warnings
import pickle as pkl
import torch
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
plt.rcParams['axes.unicode_minus']=False
import seaborn as sns
from dgl.nn.pytorch.conv import EdgeWeightNorm
import pickle as pkl
import dgl
import dgl.function as fn
from torch_geometric.data import Data, HeteroData, InMemoryDataset
from typing import Any, Optional, Union, Tuple
from typing import Optional, Callable

from utils import normalize

warnings.filterwarnings("ignore")

class AmazonFraudDataset(InMemoryDataset):
    
    """
    # Fraud Amazon Dataset

    The Amazon dataset includes product reviews under the Musical Instruments category.
    Users with more than 80% helpful votes are labelled as benign entities and users with less than 20% helpful votes are labelled as fraudulent entities.
    A fraudulent user detection task can be conducted on the Amazon dataset, which is a binary classification task.
    25 handcrafted features from <https://arxiv.org/pdf/2005.10150.pdf> are taken as the raw node features.
    Raw data is downoaded from DGL.

    Users are nodes in the graph, and three relations are: 
        - U-P-U : it connects users reviewing at least one same product
        - U-S-U : it connects users having at least one same star rating within one week
        - U-V-U : it connects users with top 5% mutual review text similarities (measured by TF-IDF) among all users.

    ## Statistics

    - Nodes: 11,944
        - Node feature size: 25
        - Classes:
            - Positive (fraudulent): 821
            - Negative (benign): 7,818
            - Unlabeled: 3,305
            - Positive-Negative ratio: 1 : 10.5
    - Edges:
        - U-P-U: 175,608
        - U-S-U: 3,566,479
        - U-V-U: 1,036,737
        - Homogeneous: 4,417,576
        
    ## Parameters
    
    - root (str) – Root directory where the dataset should be saved.
    - transform (callable, optional) – A function/transform that takes in an Data object and returns a transformed version. The data object will be transformed before every access. (default: None)
    - pre_transform (callable, optional) – A function/transform that takes in an Data object and returns a transformed version. The data object will be transformed before being saved to disk. (default: None)
    """    
    
    url = 'https://data.dgl.ai/dataset/FraudAmazon.zip'

    def __init__(self, root: str, transform: Optional[Callable] = None, pre_transform: Optional[Callable] = None, pre_filter: Optional[Callable] = None):        
        root += '/AmazonFraud'
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])
    
    @property
    def raw_file_names(self):
        return ['Amazon.mat']

    @property
    def processed_file_names(self):
        return ['AmazonFraud_data.pt']

    def download(self):
        # Download to `self.raw_dir`.
        download_url(self.url, self.raw_dir)
        zip_path = self.raw_dir+'/FraudAmazon.zip'
        extract_zip(zip_path, self.raw_dir)
        os.remove(zip_path)

    def process(self):
        # Read data into huge `Data` list.
        dic = loadmat(os.path.join(self.raw_dir, self.raw_file_names[0]))
        data = HeteroData()
        data['user'].x = torch.tensor(np.asarray(dic['features'].todense()), dtype=torch.float32)
        data['user'].y = torch.tensor(dic['label'].reshape(-1))
        data['user', 'net_upu', 'user'].edge_index = from_scipy_sparse_matrix(dic['net_upu'])[0]
        data['user', 'net_usu', 'user'].edge_index = from_scipy_sparse_matrix(dic['net_usu'])[0]
        data['user', 'net_uvu', 'user'].edge_index = from_scipy_sparse_matrix(dic['net_uvu'])[0]

        if self.pre_filter is not None:
            data = self.pre_filter(data)
            
        if self.pre_transform is not None:
            data = self.pre_transform(data)

        torch.save(self.collate([data]), self.processed_paths[0])

class Dataset:
    def __init__(self, args, load_epoch, name='tfinance', del_ratio=0., homo=True, data_path='', adj_type='sym'):
        PYG_DIR_PATH = "./data/pyg"
        self.name = name
        graph = None
        prefix = data_path
        model = 'BWGNN(homo)' if args.homo == 1 else 'BWGNN(hetero)'
        pkl_path = f"./{model}/{args.dataset}/probs_best({str(args.seed).zfill(2)}_{args.train_ratio}).pkl"
        
        if name == 'tfinance':
            prefix = './data/tfinance/raw'
            graph, label_dict = load_graphs(f'{prefix}/tfinance')
            graph = graph[0]
            graph.ndata['label'] = graph.ndata['label'].argmax(1)
            if del_ratio != 0.:
                graph = graph.add_self_loop()
                with open(pkl_path, 'rb') as f:
                    pred_y = pkl.load(f).cpu()
                    graph.ndata['pred_y'] = pred_y
                graph = random_walk_update(graph, del_ratio, adj_type)
                graph = dgl.remove_self_loop(graph)

        elif name == 'tsocial':
            prefix = './data/tsocial/raw'
            graph, label_dict = load_graphs(f'{prefix}/tsocial')
            graph = graph[0]
            if del_ratio != 0.:
                graph = graph.add_self_loop()
                with open(pkl_path, 'rb') as f:
                    pred_y = pkl.load(f).cpu()
                    graph.ndata['pred_y'] = pred_y
                graph = random_walk_update(graph, del_ratio, adj_type)
                graph = dgl.remove_self_loop(graph)

        elif name == 'yelp':
            dataset = FraudDataset(name, train_size=0.4, val_size=0.2)
            graph = dataset[0]
            if homo:
                graph = dgl.to_homogeneous(dataset[0], ndata=['feature', 'label', 'train_mask', 'val_mask', 'test_mask'])
                graph = dgl.add_self_loop(graph)
                if del_ratio != 0.:
                    with open(pkl_path, 'rb') as f:
                        graph.ndata['pred_y'] = pkl.load(f).cpu()
                    graph = random_walk_update(graph, del_ratio, adj_type)
                    graph = dgl.add_self_loop(dgl.remove_self_loop(graph))
            else:
                if del_ratio != 0.:
                    with open(pkl_path, 'rb') as f:
                        pred_y = pkl.load(f).cpu()
                    data_dict = {}
                    flag = 1
                    for relation in graph.canonical_etypes:
                        graph_r = dgl.to_homogeneous(graph[relation], ndata=['feature', 'label', 'train_mask', 'val_mask', 'test_mask'])
                        graph_r = dgl.add_self_loop(graph_r)
                        graph_r.ndata['pred_y'] = pred_y.cpu()
                        graph_r = random_walk_update(graph_r, del_ratio, adj_type)
                        graph_r = dgl.remove_self_loop(graph_r)
                        data_dict[('review', str(flag), 'review')] = graph_r.edges()
                        flag += 1
                    graph_new = dgl.heterograph(data_dict) 
                    graph_new.ndata['label'] = graph.ndata['label']
                    graph_new.ndata['feature'] = graph.ndata['feature']
                    graph_new.ndata['train_mask'] = graph.ndata['train_mask']
                    graph_new.ndata['val_mask'] = graph.ndata['val_mask']
                    graph_new.ndata['test_mask'] = graph.ndata['test_mask']
                    graph = graph_new

        elif name.startswith("amazon"):
            amazon = AmazonFraudDataset(PYG_DIR_PATH).data
            if name == 'amazon_new':
                amazon['user'].y[:3305] = 2
                features = amazon['user'].x.numpy()
                mask_dup = torch.BoolTensor(pd.DataFrame(features).duplicated(keep=False).values)
                amazon = amazon.subgraph({'user': ~mask_dup})
            
            if homo:
                homo = amazon.to_homogeneous(node_attrs=['x', 'y'], add_edge_type=False, add_node_type=False).coalesce()["edge_index"]
                graph = dgl.heterograph({("user","net_homo","user"): (homo[0], homo[1])})
                graph.ndata["feature"] = amazon["user"].x
                graph.ndata["label"] = amazon["user"].y
                graph = dgl.add_self_loop(graph)
                if del_ratio != 0.:
                    with open(pkl_path, 'rb') as f:
                        graph.ndata['pred_y'] = pkl.load(f).cpu()
                    graph = random_walk_update(graph, del_ratio, adj_type)
                    graph = dgl.add_self_loop(dgl.remove_self_loop(graph))
                    
            else:
                graph = dgl.heterograph({("user","net_upu","user"): (amazon["user","net_upu","user"]["edge_index"][0], amazon["user","net_upu","user"]["edge_index"][1]),
                    ("user","net_usu","user"): (amazon["user","net_usu","user"]["edge_index"][0], amazon["user","net_usu","user"]["edge_index"][1]),
                    ("user","net_uvu","user"): (amazon["user","net_uvu","user"]["edge_index"][0], amazon["user","net_uvu","user"]["edge_index"][1])})
                graph.ndata['label'] = amazon['user'].y
                graph.ndata['feature'] = amazon['user'].x
                if del_ratio != 0.:
                    with open(pkl_path, 'rb') as f:
                        pred_y = pkl.load(f).cpu()
                    data_dict = {}
                    flag = 1
                    for relation in graph.canonical_etypes:
                        graph[relation].ndata['pred_y'] = pred_y
                        graph_r = dgl.add_self_loop(graph[relation])
                        graph_r = random_walk_update(graph_r, del_ratio, adj_type)
                        graph_r = dgl.remove_self_loop(graph_r)
                        data_dict[('review', str(flag), 'review')] = graph_r.edges()
                        flag += 1
                    graph_new = dgl.heterograph(data_dict) 
                    graph_new.ndata['label'] = amazon['user'].y
                    graph_new.ndata['feature'] = torch.FloatTensor(normalize(amazon['user'].x.numpy()))
                    graph = graph_new
        else:
            print('no such dataset')
            exit(1)

        graph.ndata['label'] = graph.ndata['label'].long().squeeze(-1)
        graph.ndata['feature'] = graph.ndata['feature'].float()
        print(graph)

        self.graph = graph

def random_walk_update(graph, delete_ratio, adj_type):
    edge_weight = torch.ones(graph.num_edges())
    if adj_type == 'sym':
        norm = EdgeWeightNorm(norm='both')
    else:
        norm = EdgeWeightNorm(norm='left')
    graph.edata['w'] = norm(graph, edge_weight)
    # functions
    aggregate_fn = fn.u_mul_e('h', 'w', 'm')
    reduce_fn = fn.sum(msg='m', out='ay')

    graph.ndata['h'] = graph.ndata['pred_y']
    graph.update_all(aggregate_fn, reduce_fn)
    graph.ndata['ly'] = graph.ndata['pred_y'] - graph.ndata['ay']
    # graph.ndata['lyyl'] = torch.matmul(graph.ndata['ly'], graph.ndata['ly'].T)
    graph.apply_edges(inner_product_black)
    # graph.apply_edges(inner_product_white)
    black = graph.edata['inner_black']
    # white = graph.edata['inner_white']
    # delete
    threshold = int(delete_ratio * graph.num_edges())
    edge_to_move = set(black.sort()[1][:threshold].tolist())
    # edge_to_protect = set(white.sort()[1][-threshold:].tolist())
    edge_to_protect = set()
    graph_new = dgl.remove_edges(graph, list(edge_to_move.difference(edge_to_protect)))
    return graph_new

def inner_product_black(edges):
    return {'inner_black': (edges.src['ly'] * edges.dst['ly']).sum(axis=1)}

def inner_product_white(edges):
    return {'inner_white': (edges.src['ay'] * edges.dst['ay']).sum(axis=1)}

def find_inter(edges):
    return edges.src['label'] != edges.dst['label'] 

def cal_hetero(edges):
    return {'same': edges.src['label'] != edges.dst['label']}

def cal_hetero_normal(edges):
    return {'same_normal': (edges.src['label'] != edges.dst['label']) & (edges.src['label'] == 0)}

def cal_normal(edges):
    return {'normal': edges.src['label'] == 0}

def cal_hetero_anomal(edges):
    return {'same_anomal': (edges.src['label'] != edges.dst['label']) & (edges.src['label'] == 1)}

def cal_anomal(edges):
    return {'anomal': edges.src['label'] == 1}