

### some code are from https://github.com/ahxt/g-mixup

from time import time
import logging
import os
import os.path as osp
import numpy as np
import time

import torch
import torch.nn.functional as F
from torch_geometric.datasets import TUDataset
from torch_geometric.data import DataLoader
from torch_geometric.utils import degree
from torch.autograd import Variable
from torch_geometric.utils import to_dense_adj


import random
from torch.optim.lr_scheduler import StepLR


from utils import stat_graph, split_class_graphs, align_graphs
from utils import two_graphons_mixup
from graphon_estimator_transfer import (
        neighborhood_smoothing,
        compute_ot_matrix,
        estimate_graphon,
        residual_smoothing,
        aux_nbdsmooth,
        debias_with_structural_kernel
    )
from models import GIN

import argparse

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(levelname)s: - %(message)s', datefmt='%Y-%m-%d')


def set_all_seeds(seed: int = 42):
    random.seed(seed)                      
    np.random.seed(seed)                   
    torch.manual_seed(seed)               
    torch.cuda.manual_seed_all(seed)      
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_all_seeds(2025)


def prepare_dataset_x(dataset):
    if dataset[0].x is None:
        max_degree = 0
        degs = []
        for data in dataset:
            degs += [degree(data.edge_index[0], dtype=torch.long)]
            max_degree = max( max_degree, degs[-1].max().item() )
            data.num_nodes = int( torch.max(data.edge_index) ) + 1

        if max_degree < 2000:
            # dataset.transform = T.OneHotDegree(max_degree)

            for data in dataset:
                degs = degree(data.edge_index[0], dtype=torch.long)
                data.x = F.one_hot(degs, num_classes=max_degree+1).to(torch.float)
        else:
            deg = torch.cat(degs, dim=0).to(torch.float)
            mean, std = deg.mean().item(), deg.std().item()
            for data in dataset:
                degs = degree(data.edge_index[0], dtype=torch.long)
                data.x = ( (degs - mean) / std ).view( -1, 1 )
    return dataset



def prepare_dataset_onehot_y(dataset):

    y_set = set()
    for data in dataset:
        y_set.add(int(data.y))
    num_classes = len(y_set)

    for data in dataset:
        data.y = F.one_hot(data.y, num_classes=num_classes).to(torch.float)[0]
    return dataset


def mixup_cross_entropy_loss(input, target, size_average=True):
    """Origin: https://github.com/moskomule/mixup.pytorch
    in PyTorch's cross entropy, targets are expected to be labels
    so to predict probabilities this loss is needed
    suppose q is the target and p is the input
    loss(p, q) = -\sum_i q_i \log p_i
    """
    assert input.size() == target.size()
    assert isinstance(input, Variable) and isinstance(target, Variable)
    loss = - torch.sum(input * target)
    return loss / input.size()[0] if size_average else loss




def train(model, train_loader):
    model.train()
    loss_all = 0
    graph_all = 0
    for data in train_loader:
        # print( "data.y", data.y )
        data = data.to(device)
        optimizer.zero_grad()
        output = model(data.x, data.edge_index, data.batch)
        y = data.y.view(-1, num_classes)
        loss = mixup_cross_entropy_loss(output, y)
        loss.backward()
        loss_all += loss.item() * data.num_graphs
        graph_all += data.num_graphs
        optimizer.step()
    loss = loss_all / graph_all
    return model, loss


def test(model, loader):
    model.eval()
    correct = 0
    total = 0
    loss = 0
    for data in loader:
        data = data.to(device)
        output = model(data.x, data.edge_index, data.batch)
        pred = output.max(dim=1)[1]
        y = data.y.view(-1, num_classes)
        loss += mixup_cross_entropy_loss(output, y).item() * data.num_graphs
        y = y.max(dim=1)[1]
        correct += pred.eq(y).sum().item()
        total += data.num_graphs
    acc = correct / total
    loss = loss / total
    return acc, loss

def str2bool(v):
    return v.lower() in ('true', '1', 'yes')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default="./")
    ### parser.add_argument('--data_path', type=str, default=None)
    parser.add_argument('--dataset', type=str, default="REDDIT-BINARY")
    parser.add_argument('--model', type=str, default="GIN")
    parser.add_argument('--epoch', type=int, default=800)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--num_hidden', type=int, default=64)
    parser.add_argument('--gmixup', type=str, default="False")    
    parser.add_argument('--lam_range', type=str, default="[0.005, 0.01]")
    parser.add_argument('--aug_ratio', type=float, default=0.15)
    parser.add_argument('--aug_num', type=int, default=10)
    parser.add_argument('--gnn', type=str, default="gin")
    parser.add_argument('--seed', type=int, default=1314)
    parser.add_argument('--log_screen', type=str, default="False")
    ### parser.add_argument('--log_screen', type=str2bool, default=False)
    parser.add_argument('--ge', type=str, default="MC")
    parser.add_argument('--use_transfer', type=str, default="False")      
    ## parser.add_argument('--use_transfer', type=str2bool, default=False)
    parser.add_argument('--source_label', type=int, default=0)       
    parser.add_argument('--target_label', type=int, default=0) 

    parser.add_argument('--resolution', type=int, default=None)
   
    args = parser.parse_args()

    data_path = args.data_path
    dataset_name = args.dataset
    seed = args.seed
    lam_range = eval(args.lam_range)
    log_screen = eval(args.log_screen)
    gmixup = eval(args.gmixup)
        
    use_transfer = eval(args.use_transfer) 
    if use_transfer:
        source_label = args.source_label #
        target_label = args.target_label #
    else:
        source_label = None
        target_label = None

    num_epochs = args.epoch

    num_hidden = args.num_hidden
    batch_size = args.batch_size
    learning_rate = args.lr
    ge = args.ge
    aug_ratio = args.aug_ratio
    aug_num = args.aug_num
    model = args.model
    resolution = args.resolution if args.resolution else int(median_num_nodes)


    if log_screen is True:
        ch = logging.StreamHandler()
        ch.setLevel(logging.DEBUG)
        ch.setFormatter(formatter)
        logger.addHandler(ch)


    logger.info('parser.prog: {}'.format(parser.prog))
    logger.info("args:{}".format(args))

    torch.manual_seed(seed)



    #### device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #### logger.info(f"runing device: {device}")

    #### path = osp.join(data_path, dataset_name)
    #### dataset = TUDataset(path, name=dataset_name)
    #### dataset = list(dataset)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"runing device: {device}")

 
    custom_root = args.data_path if args.data_path is not None else "./"

    dataset = TUDataset(root=osp.join(custom_root, args.dataset), name=args.dataset)
    dataset = list(dataset)


    for graph in dataset:
        graph.y = graph.y.view(-1)

    dataset = prepare_dataset_onehot_y(dataset)


    random.seed(seed)
    random.shuffle( dataset )
    num_classes = dataset[0].y.shape[0] 

    train_nums = int(len(dataset) * 0.7)
    train_val_nums = int(len(dataset) * 0.8)
    
    avg_num_nodes, avg_num_edges, avg_density, median_num_nodes, median_num_edges, median_density = stat_graph(dataset[: train_nums])
    logger.info(f"avg num nodes of training graphs: { avg_num_nodes }")
    logger.info(f"avg num edges of training graphs: { avg_num_edges }")
    logger.info(f"avg density of training graphs: { avg_density }")
    logger.info(f"median num nodes of training graphs: { median_num_nodes }")
    logger.info(f"median num edges of training graphs: { median_num_edges }")
    logger.info(f"median density of training graphs: { median_density }")

    resolution = int(median_num_nodes)
    use_transfer = eval(args.use_transfer)
    if gmixup == True:
        if use_transfer:
    
            ### 1. Load the Source dataset

            ### source_dataset = TUDataset(root=data_path, name="COLLAB")
            ## source_dataset = TUDataset(root=data_path, name="DD")

            source_dataset = TUDataset(root=data_path, name="REDDIT-BINARY")
            
            source_dataset = list(source_dataset)
            source_dataset = prepare_dataset_onehot_y(source_dataset)
            source_dataset = prepare_dataset_x(source_dataset)

              
            logger.info(f"[🔁] Loaded source dataset: REDDIT-BINARY, total graphs: {len(source_dataset)}")


            ### 2. Estimate graphon for each source label
            
            source_class_graphs = split_class_graphs(source_dataset)
            source_graphon_dict = {}
            for s_label, s_graphs in source_class_graphs:
                aligned_s, _, _, _ = align_graphs(
                    s_graphs, 
                    padding=True,
                    N=resolution
                    )
                s_tensor_list = [torch.tensor(A, dtype=torch.float32).to(device) for A in aligned_s]
                ### P_s = neighborhood_smoothing(s_tensor_list, h=3, device=device)
                P_s = neighborhood_smoothing(s_tensor_list, device=device)
                label_id = int(np.argmax(s_label))  
                source_graphon_dict[label_id] = P_s
              


                import matplotlib.pyplot as plt
                import seaborn as sns

                n_labels = len(source_graphon_dict)
                fig, axes = plt.subplots(1, n_labels, figsize=(4 * n_labels, 4))

                for idx, label_id in enumerate(sorted(source_graphon_dict.keys())):
                    ax = axes[idx] if n_labels > 1 else axes
                    sns.heatmap(source_graphon_dict[label_id].cpu().numpy(), cmap="plasma", vmin=0, vmax=1, ax=ax, cbar=False)
                    ax.set_title(f"Label {label_id}")
                    ax.axis("off")

                # Add global caption
                plt.subplots_adjust(bottom=0.12)       
                ##fig.text(0.5, 0.08, "COLLAB Dataset", ha='center', va='center', fontsize=14, fontweight='bold')
                ## plt.savefig("graphon_COLLAB_all_labels.png", dpi=300)
                ##fig.text(0.5, 0.08, "Reddit-Binary Dataset", ha='center', va='center', fontsize=14, fontweight='bold')
                ###plt.savefig("graphon_Reddit-Binary_all_labels.png", dpi=300)
                
                plt.show()



            ### 3. Match the best source label for each target label
            import ot
            graphons = []

            for target_label, target_graphs in split_class_graphs(dataset[:train_nums]):
                aligned_t, _, _, _ = align_graphs(
                     target_graphs,
                    padding=True,
                    N=resolution
                )
                A_t = np.mean(aligned_t, axis=0)
                A_t_tensor = torch.tensor(A_t, dtype=torch.float32).to(device)
                ## P_t_ini = neighborhood_smoothing([A_t_tensor],  h=3, device=device)
                P_t_ini = neighborhood_smoothing([A_t_tensor], device=device)
                                
                logger.info(f"[🎯] Processing target label: {target_label}, num_graphs: {len(target_graphs)}")
                logger.info(f"[📊] P_t_ini mean: {P_t_ini.mean().item():.6f}, shape: {P_t_ini.shape}")

              
                best_dist = float('inf')
                best_source_label = None
                best_pi = None

                for s_label, P_s in source_graphon_dict.items():
                    p_s = np.ones(P_s.shape[0]) / P_s.shape[0]  # shape: (n_s,)
                    p_t = np.ones(P_t_ini.shape[0]) / P_t_ini.shape[0]  # shape: (n_t,)
                    C1 = P_s.cpu().numpy()
                    C2 = P_t_ini.cpu().numpy()
                    print("Check input matrix stats:")
                    print("P_s: min =", np.min(C1), "max =", np.max(C1), "nan =", np.isnan(C1).any())
                    print("P_t: min =", np.min(C2), "max =", np.max(C2), "nan =", np.isnan(C2).any())
                    print("Marginals: p_s sum =", p_s.sum(), ", p_t sum =", p_t.sum())

                    import ot
                  

                    ### epsilon = 1e-5
                    ## epsilon = 0.01
                    epsilon = 0.01

                    dist_gw, log = ot.gromov.gromov_wasserstein2(
                        C1, C2, p_s, p_t,
                        loss_fun='square_loss',
                        epsilon=epsilon,
                        log=True
                    )
                    gw_distance = np.sqrt(dist_gw)


                    # Step 4: Extract the optimal coupling π
                    P_s_np = P_s.cpu().numpy()
                    P_t_np = A_t_tensor.cpu().numpy()
                    P_t_ini_np = P_t_ini.cpu().numpy()

                    pi = log['T']
             
                   
                
                    
                    pi_norm = pi / pi.sum(axis=1, keepdims=True)
                    # Normalize column-wise
                    pi_norm = pi_norm / pi_norm.sum(axis=0, keepdims=True)
                    # Now use normalized pi
                    pi = pi_norm
                    
                    
                    
                    dist_ini = np.linalg.norm(P_t_ini_np - pi.T @ P_s_np @ pi)

                    # Extra: GW distance (regularized) for P_t_ini and P_t (ground truth)
                    dist_gw_Ptini = np.sqrt(ot.gromov.gromov_wasserstein2(
                        C1=P_s_np, C2=P_t_ini_np, p=p_s, q=p_t,
                        loss_fun="square_loss",  epsilon=0.01
                    ))

                    dist_gw_Pt = np.sqrt(ot.gromov.gromov_wasserstein2(
                        C1=P_s_np, C2=P_t_np, p=p_s, q=p_t,
                        loss_fun="square_loss", epsilon=0.01
                    ))

                    logger.info(f"[🔍] Source label {s_label} → Target label {target_label}")
                    logger.info(f"     ℓ₂ distance (P_t_ini): {dist_ini:.6f}")
                    logger.info(f"     GW(P_s, P_t_ini): {dist_gw_Ptini:.6f}")
                    logger.info(f"     GW(P_s, P_t): {dist_gw_Pt:.6f}")

 
    

                    ### pi = compute_ot_matrix(P_s.cpu().numpy(), P_t_ini.cpu().numpy())
                   
                    if dist_gw < best_dist:
                        best_dist = dist_gw
                        best_source_label = s_label
                        best_pi = pi
        
                   

                logger.info(f"✔️ Best match for target label {target_label}: source label {best_source_label}, dist={best_dist:.4f}")
            

                P_s_best = source_graphon_dict[best_source_label]

              
                P_s_best_np = P_s_best.cpu().numpy()
                P_t_ini_np = P_t_ini.cpu().numpy()

                gw_distance = np.sqrt(ot.gromov.gromov_wasserstein2(
                    C1=P_s_best_np, C2=P_t_ini_np, p=p_s, q=p_t,
                    loss_fun="square_loss", epsilon=0.01
                ))
               
               




                P_t_trans = torch.tensor(best_pi.T @ P_s_best.cpu().numpy() @ best_pi, dtype=torch.float32).to(device)
                ### R_t = A_t_tensor - P_t_trans
                ########################   R_t = P_t_ini - P_t_trans 
                #######################   R_t = torch.clamp(R_t, 0, 1)
                ########################  P_t_final = torch.clamp(P_t_trans + R_t, 0, 1)

                 
                P_t_trans_smoothed = neighborhood_smoothing([P_t_trans], device=device)
                

                threshold = 0.3
                ## threshold = 0.15
        

                logger.info(f"     gw_distance(P_s_best, P_t): {gw_distance:.6f}")
                # ⬇️ Replace this with structure-aware smoothing!
                ## delta_hat = debias_with_structural_kernel(R_t_numpy, h=np.sqrt(np.log(R_t_numpy.shape[0]) / R_t_numpy.shape[0]))
                ##  P_t_final = torch.tensor(delta_hat + P_t_trans.cpu().numpy(), dtype=torch.float32).to(device)
                ### P_t_final = torch.clamp(P_t_final, 0, 1)
                if gw_distance > threshold:

                    # Step 5: Residual smoothing (bias correction) using structure kernel
                    P_t_1 = P_t_trans_smoothed
                    ## np.fill_diagonal(P_t_1, 0)
          
                    P_t_1_np = P_t_1.cpu().numpy()  

                   
                    np.fill_diagonal(P_t_1_np, 0)

                  
                    P_t_1 = torch.tensor(P_t_1_np, dtype=torch.float32).to(device)

                    R_t = P_t_ini - P_t_trans_smoothed
                    R_t_smoothed = neighborhood_smoothing([R_t], device=device)
                    ##R_t = torch.clamp(R_t, 0, 1)
                    R_t_smoothed_numpy = R_t_smoothed.cpu().numpy()

                    P_t_final = P_t_1 + R_t_smoothed
                else:
                    P_t_final = P_t_trans_smoothed


                ### P_t_final =  P_t_1 + R_t_smoothed
                ## np.fill_diagonal(P_t_final, 0)
       
                P_t_final.fill_diagonal_(0)
                graphons.append((target_label, P_t_final.cpu().numpy()))




                logger.info(f"[🧩] Final estimated graphon for target label {target_label}: mean={P_t_final.mean().item():.6f}, shape={P_t_final.shape}")


            # Step 3: Mixup
            num_sample = int(train_nums * aug_ratio / aug_num)
            lam_list = np.random.uniform(low=lam_range[0], high=lam_range[1], size=(aug_num,))

            random.seed(seed)
            new_graph = []
            for lam in lam_list:
                logger.info(f"lam: {lam}")
                logger.info(f"num_sample: {num_sample}")
                two_graphons = random.sample(graphons, 2)
                new_graph += two_graphons_mixup(two_graphons, la=lam, num_sample=num_sample)
                logger.info(f"label: {new_graph[-1].y}")

            avg_num_nodes, avg_num_edges, avg_density, median_num_nodes, median_num_edges, median_density = stat_graph(new_graph)
            logger.info(f"avg num nodes of new graphs: {avg_num_nodes}")
            logger.info(f"avg num edges of new graphs: {avg_num_edges}")
            logger.info(f"avg density of new graphs: {avg_density}")
            logger.info(f"median num nodes of new graphs: {median_num_nodes}")
            logger.info(f"median num edges of new graphs: {median_num_edges}")
            logger.info(f"median density of new graphs: {median_density}")

            dataset = new_graph + dataset
            logger.info(f"real aug ratio: {len(new_graph) / train_nums}")
            train_nums = train_nums + len(new_graph)
            train_val_nums = train_val_nums + len(new_graph)

        else:
    
            
            class_graphs = split_class_graphs(dataset[:train_nums])
            graphons = []

            for label, graphs in class_graphs:
                logger.info(f"Target-only setting: label {label}, {len(graphs)} graphs")



                
                aligned_graphs, _, _, _ = align_graphs(
                    graphs,   
                    padding=True,
                    N=resolution
                )

                # Apply neighborhood smoothing
                ## graphon_tensor = neighborhood_smoothing(
                    ## [torch.tensor(adj, dtype=torch.float32).to(device) for adj in aligned_graphs],
                    ## h=3, device=device)
                graphon_tensor = neighborhood_smoothing(
                                    [torch.tensor(adj, dtype=torch.float32).to(device) for adj in aligned_graphs],
                                     device=device)
                
                onehot_label = np.eye(num_classes)[int(label[0])]
                graphons.append((onehot_label, graphon_tensor.cpu().numpy()))


            random.seed(seed)
            new_graph = []
            num_sample = int(train_nums * aug_ratio / aug_num)
            lam_list = np.random.uniform(low=lam_range[0], high=lam_range[1], size=(aug_num,))

            for lam in lam_list:
                logger.info(f"lam: {lam}")
                logger.info(f"num_sample: {num_sample}")
                
                if len(graphons) >= 2:
                    two_graphons = random.sample(graphons, 2)
                else:
                    two_graphons = graphons * 2  # fallback：

                new_graph += two_graphons_mixup(two_graphons, la=lam, num_sample=num_sample)
                logger.info(f"label: {new_graph[-1].y}")


            avg_num_nodes, avg_num_edges, avg_density, median_num_nodes, median_num_edges, median_density = stat_graph(new_graph)
            logger.info(f"avg num nodes of new graphs: {avg_num_nodes}")
            logger.info(f"avg num edges of new graphs: {avg_num_edges}")
            logger.info(f"avg density of new graphs: {avg_density}")
            logger.info(f"median num nodes of new graphs: {median_num_nodes}")
            logger.info(f"median num edges of new graphs: {median_num_edges}")
            logger.info(f"median density of new graphs: {median_density}")

            dataset = new_graph + dataset
            logger.info(f"real aug ratio: {len(new_graph) / train_nums}")
            train_nums += len(new_graph)
            train_val_nums += len(new_graph)


            logger.info("🔍 [Mixup Summary]")
            logger.info(f"📌 Number of labels: {num_classes}")
            logger.info(f"📌 Number of graphons created: {len(graphons)}")
            logger.info(f"📌 Number of mixup iterations: {len(lam_list)}")
            logger.info(f"📌 Each iteration sample size: {num_sample}")
            logger.info(f"📌 Total new graphs (augmented): {len(new_graph)}")



    dataset = prepare_dataset_x( dataset )

    logger.info(f"num_features: {dataset[0].x.shape}" )
    logger.info(f"num_classes: {dataset[0].y.shape}"  )

    num_features = dataset[0].x.shape[1]
    num_classes = dataset[0].y.shape[0]

    train_dataset = dataset[:train_nums]
    random.shuffle(train_dataset)
    val_dataset = dataset[train_nums:train_val_nums]
    test_dataset = dataset[train_val_nums:]

    logger.info(f"train_dataset size: {len(train_dataset)}")
    logger.info(f"val_dataset size: {len(val_dataset)}")
    logger.info(f"test_dataset size: {len(test_dataset)}" )


    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)


    if model == "GIN":
        model = GIN(num_features=num_features, num_classes=num_classes, num_hidden=num_hidden).to(device)
    else:
        logger.info(f"No model."  )


    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=5e-4)
    scheduler = StepLR(optimizer, step_size=100, gamma=0.5)


    for epoch in range(1, num_epochs):
        model, train_loss = train(model, train_loader)
        train_acc = 0
        val_acc, val_loss = test(model, val_loader)
        test_acc, test_loss = test(model, test_loader)
        scheduler.step()

        logger.info('Epoch: {:03d}, Train Loss: {:.6f}, Val Loss: {:.6f}, Test Loss: {:.6f},  Val Acc: {: .6f}, Test Acc: {: .6f}'.format(
            epoch, train_loss, val_loss, test_loss, val_acc, test_acc))
