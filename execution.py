from synrfp import synrfp
from synkit.Chem.Reaction.standardize import Standardize
from tqdm import tqdm
from pathlib import Path
from typing import Optional, List
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, matthews_corrcoef
import pandas as pd
import xgboost as xgb
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import copy
import os
import csv
import time
from datetime import timedelta
from utils import setup_logging, configure_warnings_and_logs
from torch.utils.data import DataLoader, TensorDataset
configure_warnings_and_logs(ignore_warnings=True, disable_rdkit_logs=True)


def fps(
    rxn: str,
    tokenizer: Optional[str] = "wl",
    radius: Optional[int] = 2,
    sketch: Optional[str] = "parity",
    mode: Optional[str] = "delta",
    bits: Optional[int] = 2048,
    seed: Optional[int] = 42,
    node_attrs: Optional[List[str]] = ['element', 'hcount', 'aromatic', "in_ring", "isomer", "hybridization"],
    edge_attrs: Optional[List[str]] = ['order', "ez_isomer", "conjugated", "in_ring"],
):
    """
    Compute a SynRFP fingerprint for a reaction SMILES.

    This helper is a thin wrapper around your `synrfp(...)` API.  Any argument
    left as `None` will fall back to the library/default behaviour.

    -----------------------------------------------------------------------
    Common (recommended) choices you can pass for the configurable arguments
    -----------------------------------------------------------------------
    tokenizer (str)
        - "wl"     : Weisfeiler–Lehman subtree labels (graph-native, default)
        - "nauty"  : Nauty/Traces canonical ego-subgraphs (canonical graphs)
        - "morgan" : Morgan / circular (ECFP-like) tokeniser (RDKit)
        - "path"   : Path-based tokenizer
        - other tokenizer class-names supported by your synrfp installation

    radius (int)
        - Typical values: 0, 1, 2, 3, 4
        - Recommended: 2 (good trade-off), 3 (more expressive, may overfit)
        - Must be a non-negative integer.

    sketch (str)
        - "parity"  : ParityFold / XOR-fold binary sketch (fast, good for ML)
        - "minhash" : MinHash sketch (Jaccard similarity friendly)
        - "cws"     : CWSketch (weighted minhash / weighted Jaccard)
        - "srp"     : SRPSketch / signed random projection (for cosine-like)
        - other sketch names supported by your synrfp installation

    mode (str)
        - "delta" : signed difference (P - R). Direction-aware; default for many tasks.
        - "union" : union/presence mode (R + P counts treated as positive weights).

    bits (int)
        - Common sizes: 64, 128, 256, 512, 1024, 2048, 4096
        - Choose based on collision tolerance / model capacity (1024 is a common sweet spot).

    seed (int)
        - Determinism seed for hashing/sketching. Typical default: 42.

    node_attrs (list[str] | None)
        - Typical node attributes: ["element", "hcount", "aromatic", "in_ring",
                                   "isomer", "hybridization", "formal_charge"]
        - If None, the tokenizer's defaults are used.

    edge_attrs (list[str] | None)
        - Typical edge attributes: ["order", "ez_isomer", "conjugated", "in_ring", "stereo"]
        - If None, the tokenizer's defaults are used.

    -----------------------------------------------------------------------
    Behaviour
    -----------------------------------------------------------------------
    - Any argument explicitly set to `None` will use library defaults.
    - The function returns the fingerprint object exactly as produced by
      `synrfp(...)` (commonly a list of 0/1 bits for parity sketches).
    - Use `mode="delta"` for direction-sensitive tasks (mechanistic classification);
      use `mode="union"` when you want presence-only features.
    - Increasing `radius` or `bits` increases expressiveness and memory / collision
      resistance respectively — but may reduce generalization if set too high.

    Example
    -------
    >>> # use defaults for everything except rxn string
    >>> fp = fps("CCO.O>>CC(=O)O")
    >>> # specify a few options
    >>> fp = fps("CCO>>CCO", tokenizer="morgan", radius=2, sketch="parity", bits=2048)

    -----------------------------------------------------------------------
    Notes
    -----------------------------------------------------------------------
    - For a programmatic list of available tokenizers/sketchers (on your system)
      you can probe the synrfp package or consult your environment. Typical
      installations include at least "wl" and "parity".
    - If you want metadata (delta counts, raw tokens) returned alongside the
      fingerprint, call the lower-level `SynRFP`/`SynRFPResult` APIs directly.
    """
    return synrfp(
        rxn,
        tokenizer=tokenizer,
        radius=radius if radius is not None else 2,
        sketch=sketch,
        mode=mode,
        bits=bits if bits is not None else 1024,
        seed=seed if seed is not None else 42,
        node_attrs=node_attrs,
        edge_attrs=edge_attrs,
    )

def standard_rxn(df):
    df = df.to_dict('records')
    for value in tqdm(df,desc="Standardize" ):
        try:
            value['rxn'] = Standardize().fit(value['rxn'])
        except:
            value['rxn'] = None

    df = [value for value in df if value['rxn']]
    df = pd.DataFrame(df)
    return df

def read_data(path):
    df = pd.read_csv(path, compression='gzip')
    # df = standard_rxn(df)
    df_train = df[df['split']=='train']
    df_test = df[df['split']=='test']
    df_train = df_train.reset_index()
    df_test = df_test.reset_index()
    return df_train, df_test 

def train_knn(x_train, x_test, y_train, y_test):
    classifier = KNeighborsClassifier(n_neighbors=5,n_jobs=14)
    # x_train = np.array(x_train)
    # x_test = np.array(x_test)
    # y_train = np.array(y_train)
    # y_test = np.array(y_test)
    classifier.fit(x_train, y_train)
    y_pred = classifier.predict(x_test)
    return accuracy_score(y_test, y_pred), matthews_corrcoef(y_test, y_pred)

class MLP(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(MLP, self).__init__()
        self.layer_1 = nn.Linear(input_dim, output_dim)
    
    def forward(self, x):
        return self.layer_1(x)
    
def train_mlp(x_train, x_test, y_train, y_test, device, epochs =100, batch_size=32, lr=0.0001):
    start = time.perf_counter()
    x_train = torch.tensor(x_train, dtype=torch.float32).to(device)
    x_test = torch.tensor(x_test, dtype = torch.float32).to(device)
    y_train = torch.tensor(y_train, dtype = torch.long).to(device)
    y_test = torch.tensor(y_test,dtype = torch.long).to(device)
    model = MLP(x_train.shape[1], len(torch.unique(y_train)))
    
    train_dataset = TensorDataset(x_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle = True)

    test_dataset = TensorDataset(x_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size = batch_size, shuffle = False)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(),lr=lr)

    print(f'Training on {x_train.shape[0]} samples')
    loss_history = []
    best_loss = float('inf')
    best_weight = None

    for epoch in range(epochs):
        epoch_loss =0
        for batch_x, batch_y in train_loader:
            logits = model(batch_x)

            loss = criterion(logits, batch_y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss+=loss.item()
        avg_loss = epoch_loss/len(train_loader)
        loss_history.append(avg_loss)

        if (epoch+1) % 10 == 0:
            print(f'At epoch: {(epoch+1)}/{epochs}, loss: {avg_loss:.4f}')

        if avg_loss < best_loss:
            best_loss = avg_loss
            best_weight = copy.deepcopy(model.state_dict())
    
    model.load_state_dict(best_weight)
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            test_logits = model(batch_x)

            _, preds = torch.max(logits, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch_y.cpu().numpy())
    
    predict_values = np.array(all_preds)
    true_values = np.array(all_labels)

    acc = accuracy_score(true_values, predict_values)
    mcc = matthews_corrcoef(true_values, predict_values)

    end = time.perf_counter()
    time_process = end - start
    train_formatted = str(timedelta(seconds=time_process))
    print(f"MLP time (Formatted): {train_formatted}")

    return acc, mcc

def train_xgboost(x_train, x_test, y_train, y_test):
    # x_train = np.array(x_train)
    # x_test = np.array(x_test)
    # y_train = np.array(y_train)
    # y_test = np.array(y_test)
    xgb_model = xgb.XGBClassifier(
        n_estimators=100, 
        learning_rate=0.0001, 
        use_label_encoder=False, 
        eval_metric='mlogloss',
        n_jobs=14
    )
    start = time.perf_counter()
    xgb_model.fit(x_train, y_train)

    preds = xgb_model.predict(x_test)

    acc = accuracy_score(y_test, preds)
    mcc = matthews_corrcoef(y_test, preds)
    end = time.perf_counter()
    time_process = end - start
    train_formatted = str(timedelta(seconds=time_process))
    print(f"XGBoost time (Formatted): {train_formatted}")

    return acc, mcc


def main(training_model,folder_data='data/raw_trial', log_file='experiment_results.csv'):
    logger = setup_logging(log_filename="monitor/monitor.log")
    radius_lst = [1,2,3]
    bits_lst = [512,1024, 2048]
    rb_lst = [[radius_lst[i], bits_lst[i]] for i in range(len(radius_lst))]
    tokenizer_lst = ['wl', 'path', 'morgan']
    sketch_lst = ['parity', 'minhash', 'cw']
    mode_lst = ['delta','union']
    for radius,bits in rb_lst:
        for tokenizer in tokenizer_lst: 
            for sketch in sketch_lst:
                for mode in mode_lst:

                    folder_path = Path(folder_data)
                    
                    # Get all files (ignoring folders inside)
                    files = [f.name for f in folder_path.iterdir() if f.is_file()]
                    for file in files:
                        df_train, df_test = read_data(folder_data+'/'+file)
                        # x_train = np.array([synrfp(df_train.loc[i,'rxn'],tokenizer=tokenizer,radius=radius,sketch=sketch,mode=mode,bits=bits) for i in range(df_train.shape[0])])
                        file_name = file.split('.')[0]
                        path_fps = f'data/fps/{file_name}/ra_{radius}/bi_{bits}/to_{tokenizer}/sk_{sketch}/mo_{mode}'
                        path_folder_fps = Path(path_fps)
                        path_folder_fps.mkdir(parents=True, exist_ok=True)
                        if not any(path_folder_fps.iterdir()): 
                            compute_fps = True
                        else: 
                            compute_fps = False
                        if compute_fps:
                            x_train_lst = []
                            for i in tqdm(range(df_train.shape[0]),desc=f"fps for training {file_name}"):
                                try:
                                    x_train_lst.append(fps(df_train.loc[i,'rxn'],tokenizer=tokenizer,radius=radius,sketch=sketch,mode=mode,bits=bits))
                                except Exception as e:
                                    logger.info(f'Failed at index {i} in {file}. Reason: {e}')
                                    break 
                            x_train = np.array(x_train_lst)
                            np.savez_compressed(f"{path_fps}/train.npz", fps=x_train)
                        else:
                            x_train = np.load(f"{path_fps}/train.npz",allow_pickle=True)['fps']
                        y_train = df_train['y'].values
                        if x_train.shape[0] != y_train.shape[0]:
                            continue
                        # x_test = np.array([fps(df_test.loc[i,'rxn'],tokenizer=tokenizer,radius=radius,sketch=sketch,mode=mode,bits=bits) for i in range(df_test.shape[0])])
                        if compute_fps:
                            x_test_lst = []
                            for i in tqdm(range(df_test.shape[0]), desc=f"fps for test {file_name}"):
                                try:
                                    x_test_lst.append(fps(df_test.loc[i,'rxn'],tokenizer=tokenizer,radius=radius,sketch=sketch,mode=mode,bits=bits))
                                except Exception as e:
                                    logger.info(f'failed idx test is {i} in {file} dataset. Reason: {e}')
                                    break
                                
                            x_test = np.array(x_test_lst)
                            np.savez_compressed(f"{path_fps}/test.npz", fps=x_test)
                        else:
                            x_test = np.load(f"{path_fps}/test.npz",allow_pickle=True)['fps']
                        y_test = df_test['y'].values
                        if  x_test.shape[0] != y_test.shape[0]:
                            continue
                        # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                        device = torch.device("cuda:0")
                        if training_model == 'knn':
                            acc, mcc = train_knn(x_train, x_test, y_train, y_test)
                        elif training_model == 'mlp':
                            acc, mcc = train_mlp(x_train, x_test, y_train, y_test, device)
                        elif training_model == 'xgb':
                            acc, mcc = train_xgboost(x_train, x_test, y_train, y_test)
                        else:
                            raise ValueError("This is not this option for training_model")

                        # 4. SAVE RESULTS TO CSV
                        # Check if file exists to determine if we need to write headers
                        file_exists = os.path.isfile(log_file)
                        
                        try:
                            with open(log_file, mode='a', newline='') as f:
                                writer = csv.writer(f)
                                
                                # Write Header only if file is new
                                if not file_exists:
                                    writer.writerow(['Dataset','Model', 'Radius', 'Bits', 'Tokenizer', 'Sketch', 'Mode' ,'Accuracy', 'MCC'])
                                
                                # Write the Result Row
                                writer.writerow([file_name,training_model, radius, bits, tokenizer,sketch, mode ,acc, mcc])
                                
                            print(f"Saved results to {log_file} (Acc: {acc:.4f}, MCC: {mcc:.4f})")
                            
                        except Exception as e:
                            print(f"Error saving to CSV: {e}")

if __name__ == "__main__":
    # You can now iterate through models or folders easily
    # models_to_run = ['knn', 'mlp', 'xgb']
    
    # for model in models_to_run:
    #     # This will append 3 rows to 'experiment_results.csv'
    #     main(training_model=model, folder_data='data/raw_trial')
    main('mlp', folder_data='data/raw_std')
    # main('mlp', folder_data='data/raw_new')
    # main('xgb')




    
    