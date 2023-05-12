import os
from typing import Tuple
from typing import Optional
from collections import defaultdict
import torch
import random
import numpy as np
from torch_geometric.data import Data

from sklearn.utils import shuffle
from sklearn.metrics import f1_score, accuracy_score, recall_score, roc_auc_score, precision_score

#################### [Functions for TRAIN] ####################

#################### [Functions for TEST/VALIDATION] ####################

def test(labels, probs, log, flag=None):
        labels = labels.data.cpu().numpy()
        probs = probs.data.cpu().numpy()
        
        predictions = probs.argmax(axis=1)
        
        f1 = f1_score(labels, predictions)
        f1_macro = f1_score(labels, predictions, average='macro')
        precision = precision_score(labels, predictions, zero_division=0)
        precision_macro = precision_score(labels, predictions, zero_division=0, average='macro')
        accuracy = accuracy_score(labels, predictions)
        recall = recall_score(labels, predictions)
        recall_macro = recall_score(labels, predictions, average='macro')
        auc = roc_auc_score(labels, probs[:, 1])

        line= f"- F1: {f1:.4f}\t\t- Recall: {recall:.4f}\t\t- Precision: {precision:.4f}\t- Accuracy: {accuracy:.4f}\t- AUC-ROC: {auc:.4f}\n- F1-macro: {f1_macro:.4f}\t- Recall-macro: {recall_macro:.4f}\t\t- AP: {precision_macro:.4f}\t\n"   
        
        if flag == 'validation':
                log.write_valid_log(line)
        elif flag == 'test':
                log.write_test_log(line)
        
        return auc, recall, f1, precision

#################### [Functions for WRITE LOGS] ####################

class log_handler():
        def __init__(self, args) -> None:
                if args.del_ratio != 0:
                        self.model = 'BHomo-GHRN' if args.homo == 1 else 'BHetero-GHRN'
                else:
                        self.model = 'BWGNN(homo)' if args.homo == 1 else 'BWGNN(hetero)'
                self.save_dir_path = f"./{self.model}/{args.dataset}"
                self.validation_log = os.path.join(self.save_dir_path, f"validation({str(args.seed).zfill(2)}).log")
                self.test_log = os.path.join(self.save_dir_path, f"test({str(args.seed).zfill(2)}).log")
                os.makedirs(self.save_dir_path, exist_ok=True)
                
        def write_valid_log(self, line: str, print_line: Optional[bool]=True) -> None:
                if print_line:
                        print(line)
                log_file = open(self.validation_log, 'a')
                log_file.write(line + "\n")
                log_file.close()

        def write_test_log(self, line: str, print_line: Optional[bool]=True) -> None:
                if print_line:
                        print(line)
                log_file = open(self.test_log, 'a')
                log_file.write(line + "\n")
                log_file.close()
        
def write_configurations(configuration: str) -> str:
    """
    Write the logs for the Train/Valid/Test log files.
    """
    
    # Initialize the empty line
    configuration = vars(configuration)
    logs = ""
    for key in sorted(configuration.keys()): # Iteratively write the key and value pairs of the configuration.
        val = configuration[key]
        keystr = "{}".format(key) + (" " * (24 - len(key)))
        line = "{} -->   {}\n".format(keystr, val)
        logs += line

    return logs

def write_single_configurations(key: str, val: float, type: type) -> str:
    logs = ""
    keystr = "{}".format(key) + (" " * (24 - len(key)))
    line = "{} -->   {}\n".format(keystr, type(val))

    return line

#################### [Functions for SET SEEDS] ####################
def set_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)