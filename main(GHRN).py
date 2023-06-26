import torch
import torch.nn.functional as F
import argparse
import time
from dataset import Dataset
from sklearn.metrics import f1_score, recall_score, roc_auc_score, precision_score
from BWGNN import *
from sklearn.model_selection import train_test_split
import pickle as pkl


from utils import *

def train(model, g, args):
    device = torch.device(args.cuda_id)
    torch.cuda.set_device(device)
    
    log = log_handler(args)
    log.write_valid_log(write_configurations(args))
    log.write_test_log(write_configurations(args), False)
    path_saver = os.path.join(log.save_dir_path, f"{log.model}_best({str(args.seed).zfill(2)}_{args.train_ratio}).pkl")
    prob_saver = os.path.join(log.save_dir_path, f"probs_best({str(args.seed).zfill(2)}_{args.train_ratio}).pkl")
    
    
    # [STEP-1-1] Load the node features and labels.
    labels_index = g.ndata['label'].clone().cpu().numpy()
    # [STEP-1-2] Split the train/valid/test dataset.
    index = list(range(len(labels_index)))
    if dataset_name == 'amazon':
        index = list(range(3305, len(labels_index)))
    idx_train, idx_rest, y_train, y_rest = train_test_split(index, labels_index[index], stratify=labels_index[index],
                                                            train_size=args.train_ratio,
                                                            random_state=args.seed, shuffle=True)
    idx_valid, idx_test, y_valid, y_test = train_test_split(idx_rest, y_rest, stratify=y_rest,
                                                            test_size=0.67,
                                                            random_state=args.seed, shuffle=True)
    
    features = g.ndata['feature']
    labels = g.ndata['label']
    
    # Set the mask of the graph
    train_mask = torch.zeros([len(labels_index)]).bool()
    val_mask = torch.zeros([len(labels_index)]).bool()
    test_mask = torch.zeros([len(labels_index)]).bool()
    train_mask[idx_train] = 1
    val_mask[idx_valid] = 1
    test_mask[idx_test] = 1
    line = f"train/dev/test samples: {train_mask.sum().item()}, {val_mask.sum().item()}, {test_mask.sum().item()}"
    log.write_valid_log(line, False)
    log.write_test_log(line, False)
    
    # [STEP-2-1] Initialize the optimizer and variables.
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    best_f1, final_tf1, final_trec, final_tpre, final_tmf1, final_tauc = 0., 0., 0., 0., 0., 0.
    auc_best, epoch_best = 0.0, 0.0

    # [STEP-2-2] Set the frequency weight for the cross entorpy loss.
    weight = (1-labels[train_mask]).sum().item() / labels[train_mask].sum().item()
    print('cross entropy weight: ', weight)
    
    # [STEP-3] Train
    time_start = time.time()
    for epoch in range(1, args.epoch+1):
        
        # [STEP-3-1] Feed the node features and update the model based on the weighted cross entropy loss.
        model.train()
        logits = model(features)
        loss = F.cross_entropy(logits[train_mask], labels[train_mask], weight=torch.tensor([1., weight]).cuda())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # [STEP-3-2] Performance validation with updated model.
        if (epoch+1) % args.valid_epochs == 0:
            probs = logits.softmax(1)
            log.write_valid_log(f"[Epoch - {str(epoch+1).zfill(4)}] - Validation results")
            auc_val, recall_val, f1_val, precision_val = test(labels[val_mask], probs[val_mask], log, flag='validation')
            if auc_val > auc_best:
                auc_best, recall_best, f1_best, precision_best, epoch_best = auc_val, recall_val, f1_val, precision_val, epoch
                torch.save(model.state_dict(), path_saver)
                if args.del_ratio == 0:
                    with open(prob_saver, 'wb') as f:
                        pkl.dump(probs, f)
                
        if (epoch - epoch_best) > args.patience:
                log.write_valid_log(write_single_configurations("Eearly_stop", epoch, int), False)
                break

    time_end = time.time()
    log.write_valid_log(f'time cost: {time_end - time_start}s')
    print("Model path: {}".format(path_saver))   
    model.load_state_dict(torch.load(path_saver))
    logits = model(features)
    probs = logits.softmax(1)
    auc_test, recall_test, f1_test, precision_test = test(labels[test_mask], probs[test_mask], log, flag='test')
        
    return auc_test, recall_test, f1_test, precision_test


# threshold adjusting for best macro f1
def get_best_f1(labels, probs):
    best_f1, best_thre = 0, 0
    for thres in np.linspace(0.05, 0.95, 19):
        preds = np.zeros_like(labels)
        preds[probs[:,1] > thres] = 1
        mf1 = f1_score(labels, preds, average='macro')
        if mf1 > best_f1:
            best_f1 = mf1
            best_thre = thres
    return best_f1, best_thre

def set_seeds(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='BWGNN')
    parser.add_argument("--dataset", type=str, default="amazon",
                        help="Dataset for this model (yelp/amazon/tfinance/tsocial)")
    parser.add_argument("--seed", type=int, default=41, help="set the seed")
    parser.add_argument("--train_ratio", type=float, default=0.4, help="Training ratio")
    parser.add_argument("--hid_dim", type=int, default=64, help="Hidden layer dimension")
    parser.add_argument("--order", type=int, default=2, help="Order C in Beta Wavelet")
    parser.add_argument("--homo", type=int, default=0, help="1 for BWGNN(Homo) and 0 for BWGNN(Hetero)")
    parser.add_argument("--epoch", type=int, default=500, help="The max number of epochs")
    parser.add_argument("--valid_epochs", type=int, default=5, help="The number of validation epochs")
    parser.add_argument("--run", type=int, default=5, help="Running times")
    parser.add_argument("--lr", type=float, default=0.01, help="delete ratios")
    parser.add_argument("--weight_decay", type=float, default=0.001, help="delete ratios")
    parser.add_argument("--del_ratio", type=float, default=0., help="delete ratios")
    parser.add_argument("--adj_type", type=str, default='sym', help="sym or rw")
    parser.add_argument("--load_epoch", type=int, default=100, help="load epoch prediction")
    parser.add_argument("--patience", type=int, default=100, help="load epoch prediction")
    parser.add_argument("--cuda_id", type=int, default=0, help="load epoch prediction")
    parser.add_argument("--data_path", type=str, default='./data', help="data path")

    args = parser.parse_args()
    device = torch.device(args.cuda_id)
    torch.cuda.set_device(device)
    # with open('result.txt', 'a+') as f:
    #     f.write(f'{args}\n')
    dataset_name = args.dataset
    del_ratio = args.del_ratio
    homo = args.homo
    order = args.order
    h_feats = args.hid_dim
    adj_type = args.adj_type
    load_epoch = args.load_epoch
    data_path = args.data_path
    graph = Dataset(args, load_epoch, dataset_name, del_ratio, homo, data_path, adj_type=adj_type).graph
    
    graph = graph.to(device)
    in_feats = graph.ndata['feature'].shape[1]
    num_classes = 2
    
    # official seed
    if args.run == 1:
        set_seeds(args.seed)
        if homo:
            model = BWGNN(in_feats, h_feats, num_classes, graph, d=order)
            model = model.to(device)
        else:
            model = BWGNN_Hetero(in_feats, h_feats, num_classes, graph, d=order)
            model = model.to(device)
        train(model, graph, args)

    else:
        del_ratio_l = None
        if args.dataset == 'amazon':
            del_ratio_l = list(np.arange(0.01, 0.026, 0.005))
        elif args.dataset == 'yelp':
            del_ratio_l = list(np.arange(0.05, 0.21, 0.05))
        final_mf1s, final_aucs = [], []
        for tt in range(args.run):
            args.del_ratio = del_ratio_l[tt]
            set_seeds(args.seed)
            if homo:
                model = BWGNN(in_feats, h_feats, num_classes, graph, d=order)
                model = model.to(device)
            else:
                model = BWGNN_Hetero(in_feats, h_feats, num_classes, graph, d=order)
                model = model.to(device)
            auc_test, recall_test, f1_test, precision_test = train(model, graph, args)

