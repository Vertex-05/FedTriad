import random
import models as model_utils
from utils import *
from client import Device
import hdbscan
from utils import kd_loss, DiffAugment

#此处为DYN修改
from sklearn.cluster import AgglomerativeClustering, DBSCAN
VOTE_FOR_BENIGN = 1
VOTE_FOR_POISONED = 0

def create_cluster_map_from_labels(expected_number_of_labels, clustering_labels):
    """
    Converts a list of labels into a dictionary where each label is the key and
    the values are lists/np arrays of the indices from the samples that received
    the respective label
    :param expected_number_of_labels number of samples whose labels are contained in
    clustering_labels
    :param clustering_labels list containing the labels of each sample
    :return dictionary of clusters
    """
    assert len(clustering_labels) == expected_number_of_labels

    clusters = {}
    for i, cluster in enumerate(clustering_labels):
        if cluster not in clusters:
            clusters[cluster] = []
        clusters[cluster].append(i)
    return {index: np.array(cluster) for index, cluster in clusters.items()}

def determine_biggest_cluster(clustering):
    """
    Given a clustering, given as dictionary of the form {cluster_id: [items in cluster]}, the
    function returns the id of the biggest cluster
    """
    biggest_cluster_id = None
    biggest_cluster_size = None
    for cluster_id, cluster in clustering.items():
        size_of_current_cluster = np.array(cluster).shape[0]
        if biggest_cluster_id is None or size_of_current_cluster > biggest_cluster_size:
            biggest_cluster_id = cluster_id
            biggest_cluster_size = size_of_current_cluster
    return biggest_cluster_id




device = 'cuda' if torch.cuda.is_available() else 'cpu'

def cos_sim_nd(tensor1, tensor2):
    # return 1 - (p * q / (p.norm() * q.norm())).sum()
    dot_product = torch.sum(tensor1 * tensor2)
    norm1 = torch.norm(tensor1)
    norm2 = torch.norm(tensor2)
    similarity = (dot_product+1e-8)/ (norm1 * norm2 + 1e-8)
    return 1-similarity
def cos(a, b):
    res = np.sum(a*b.T)/((np.sqrt(np.sum(a * a.T)) + 1e-9)
                         * (np.sqrt(np.sum(b * b.T))) + 1e-9)
    '''relu'''
    if res < 0:
        res = 0
    return res

def model2vector(model):
    nparr = np.array([])
    for key, var in model.items():
        nplist = var.cpu().numpy()
        nplist = nplist.ravel()
        nparr = np.append(nparr, nplist)
    return nparr

def cosScoreAndClipValue(net1, net2):
    '''net1 -> centre, net2 -> local, net3 -> early model'''
    vector1 = model2vector(net1)
    vector2 = model2vector(net2)

    return cos(vector1, vector2), norm_clip(vector1, vector2)


def norm_clip(nparr1, nparr2):
    '''v -> nparr1, v_clipped -> nparr2'''
    vnum = np.linalg.norm(nparr1, ord=None, axis=None, keepdims=False) + 1e-9
    # import pdb; pdb.set_trace()
    return vnum / (np.linalg.norm(nparr2, ord=None, axis=None, keepdims=False) + 1e-9)


def get_update(update, model):
    '''get the update weight'''
    output = OrderedDict()
    for key, var in update.items():
        output[key] = update[key].detach()-model[key].detach()
    return output

def epoch(mode, dataloader, net, optimizer, criterion, aug=True, args=None):
    loss_avg, acc_avg, num_exp = 0, 0, 0
    net = net.cuda()
    if mode == 'train':
        net.train()
    else:
        net.eval()

    for i_batch, datum in enumerate(dataloader):
        img = datum[0].float().cuda()
        lab = datum[1].cuda()
        if aug and mode == "train":
            img = DiffAugment(img, args.dsa_strategy, param=args.dsa_param)
        n_b = lab.shape[0]
        output = net(img)
        loss = criterion(output, lab)
        if mode == 'train':
            acc = np.sum(np.equal(np.argmax(output.cpu().data.numpy(), axis=-1), np.argmax(lab.cpu().data.numpy(), axis=-1)))
        else:
            acc = np.sum(np.equal(np.argmax(output.cpu().data.numpy(), axis=-1), lab.cpu().data.numpy()))
        loss_avg += loss.item()*n_b
        acc_avg += acc
        num_exp += n_b
        if mode == 'train':
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    loss_avg /= num_exp
    acc_avg /= num_exp

    return loss_avg, acc_avg



class Server(Device):
  def __init__(self, model_names, loader, num_classes=10, dataset = 'cifar10', val_loader=None):
    super().__init__(loader)
    # import pdb; pdb.set_trace()
    print(f"dataset server {dataset}")
    self.model_dict = {model_name : partial(model_utils.get_model(model_name)[0], num_classes=num_classes, dataset = dataset)().to(device) for model_name in model_names}
    self.parameter_dict = {model_name : {key : value for key, value in model.named_parameters()} for model_name, model in self.model_dict.items()}
    self.val_loader = val_loader
    self.my_client = {model_name: partial(model_utils.get_model(model_name)[0], num_classes=num_classes, dataset = dataset)().to(device) for model_name in model_names}

    
    self.models = list(self.model_dict.values())

    # === FedREDefense + CrowdGuard + SPRT: 初始化区域 ===
    # ==========================================================
    self.sprt_state = {}
    self.sprt_params = {}
    print("[Server Init] SPRT module not configured yet. Call server.init_sprt() in run_ours.py.")
    # ==========================================================


  def evaluate_ensemble(self):
    return eval_op_ensemble(self.models, self.loader)

  def evaluate_ensemble_with_preds(self):
    return eval_op_ensemble_with_preds(self.models, self.loader)

  def evaluate_attack(self, loader=None):
    return eval_op_ensemble_attack(self.models, self.loader if not loader else loader)


  def evaluate_tr_lf_attack(self, loader=None):
    return eval_op_ensemble_tr_lf_attack(self.models, self.loader if not loader else loader)

  def evaluate_attack_with_preds(self, loader=None):
    return eval_op_ensemble_attack_with_preds(self.models, self.loader if not loader else loader)


  def centralized_training(self,syn_data, syn_label,args):
    # import pdb; pdb.set_trace()
    syn_data = torch.cat(syn_data[0:72],dim = 0)
    syn_label = torch.cat(syn_label[0:72],dim = 0)
    for model_name in self.my_client:
      evaluate_synset(0, self.my_client[model_name],0.1,syn_data, syn_label, self.loader, args)
    exit()

  def select_clients(self, clients, frac=1.0):
    return random.sample(clients, int(len(clients)*frac))

  def select_clients_masked(self, clients, frac=1.0, mask = None):
    # return [clients[0]]
    available_clients = [item for i, item in enumerate(clients) if mask[i]]
    k=int(len(clients)*frac)
    if k > len(available_clients):
        return available_clients
        raise ValueError("Sample larger than population or not enough masked values.")
    return random.sample(available_clients, k)


  def fedavg(self, clients):
    unique_client_model_names = np.unique([client.model_name for client in clients])
    self.weights = torch.Tensor([1. / len(clients)] * len(clients))
    for model_name in unique_client_model_names:
      reduce_average(target=self.parameter_dict[model_name], sources=[client.W for client in clients if client.model_name == model_name])

  def median(self, clients):
        # import pdb; pdb.set_trace()
    unique_client_model_names = np.unique(
        [client.model_name for client in clients])
    for model_name in unique_client_model_names:
      reduce_median(target=self.parameter_dict[model_name], sources=[
                    client.W for client in clients if client.model_name == model_name])

  def TrimmedMean(self, clients, mali_ratio):
    unique_client_model_names = np.unique(
        [client.model_name for client in clients])
    for model_name in unique_client_model_names:
      reduce_trimmed_mean(target=self.parameter_dict[model_name], sources=[
                          client.W for client in clients if client.model_name == model_name], mali_ratio=mali_ratio)

  def krum(self, clients, mali_ratio):
    unique_client_model_names = np.unique([client.model_name for client in clients])
    for model_name in unique_client_model_names:
      reduce_krum(target=self.parameter_dict[model_name], sources=[client.W for client in clients if client.model_name == model_name], mali_ratio = mali_ratio)

  def normbound(self, clients, mali_ratio):
    unique_client_model_names = np.unique([client.model_name for client in clients])
    self.weights = torch.Tensor([1. / len(clients)] * len(clients))
    user_num = len(clients)
    weight = []
    for name in  self.parameter_dict[unique_client_model_names[0]]:
        weight.append(torch.flatten( self.parameter_dict[unique_client_model_names[0]][name].detach()))
    weight = torch.cat(weight)
    new_model = []
    updates = []
    for client in clients:
        source = client.W
        new_model_i = []
        for name in client.W:
            new_model_i.append(torch.flatten(source[name].detach()))
        new_model_i = torch.cat(new_model_i)
        updates_i = new_model_i - weight
        new_model.append(new_model_i)
        updates.append(updates_i)
    new_model = torch.stack(new_model)
    # updates = torch.stack(updates)
    norm_list = [update.norm().unsqueeze(dim=0) for update in updates]
    # import pdb; pdb.set_trace()
    benign_norm_list = []
    for client, norm in zip(clients,norm_list):
      if client.id < (1 - mali_ratio)* user_num:
        benign_norm_list.append(norm)
    if len(benign_norm_list) != 0:
      median_tensor = sum(benign_norm_list)/len(benign_norm_list)
    else:
      median_tensor = sum(norm_list)/len(norm_list)
    # import pdb; pdb.set_trace()
    clipped_models = [update * min(1, (median_tensor+1e-8) / (update.norm()+1e-8)) for update in updates]
    clipped_models = torch.mean(torch.stack(clipped_models), dim=0)
    for model_name in unique_client_model_names:
      idx = 0
      for name in self.parameter_dict[model_name]:
        self.parameter_dict[model_name][name].data = self.parameter_dict[model_name][name].data + clipped_models[idx:(idx+self.parameter_dict[model_name][name].data.numel())].reshape(self.parameter_dict[model_name][name].data.shape)
        idx += self.parameter_dict[model_name][name].data.numel()

  def init_sprt(self, clients, use_table10=True):
      """
      初始化 SPRT 参数与状态。
      - clients: 当前参与的客户端列表
      - use_table10: 是否根据 FedREDefense 的 Table 10 初始化 P(G|H)
      """
      import numpy as np

      # ------------------- 常规参数 -------------------
      W = 2
      M_min = 3
      min_hard_count = 2
      alpha = 0.01
      beta = 0.05
      logA = np.log((1 - beta) / alpha)
      logB = np.log(beta / (1 - alpha))

      # ------------------- P(G|H) 概率参数 -------------------
      if use_table10:
          # 示例参数，可根据 Table 10 μ/σ 重新拟合
          P_G_b = {'soft': 0.90, 'defer': 0.09, 'hard': 0.01}
          P_G_m = {'soft': 0.02, 'defer': 0.08, 'hard': 0.90}
      else:
          # 默认经验参数
          P_G_b = {'soft': 0.8, 'defer': 0.15, 'hard': 0.05}
          P_G_m = {'soft': 0.05, 'defer': 0.15, 'hard': 0.8}

      # ------------------- CrowdGuard 投票模型参数 -------------------
      p_vote_b = 0.05
      p_vote_m = 0.90

      # ------------------- 初始化 SPRT 状态 -------------------
      sp_state = {}
      for c in clients:
          cid = c.id if hasattr(c, 'id') else c
          sp_state[cid] = {
              'LLR': 0.0,
              'obs': 0,
              're_count': {'soft': 0, 'defer': 0, 'hard': 0},
              'n_votes': 0,
              'k_votes': 0,
              'first_hard_round': None
          }

      # ------------------- 写入 server 属性 -------------------
      self.sprt_state = sp_state
      self.sprt_params = {
          'W': W, 'M_min': M_min, 'min_hard_count': min_hard_count,
          'alpha': alpha, 'beta': beta, 'logA': logA, 'logB': logB,
          'P_G_b': P_G_b, 'P_G_m': P_G_m, 'p_vote_b': p_vote_b, 'p_vote_m': p_vote_m
      }

      print("[Server] SPRT initialized successfully.")
      print(f"  Warm-up = {W}, min_obs = {M_min}, logA={logA:.2f}, logB={logB:.2f}")
      print(f"  P_G_b: {P_G_b}, P_G_m: {P_G_m}")
      print(f"  p_vote_b={p_vote_b}, p_vote_m={p_vote_m}")



  def crowdguard_aggregate(self, clients, votes_matrix, all_client_names):
      # Following the CrowdGuard paper, this should be executed within SGX

      # votes_matrix: List[List[int]]，每行是一个客户端对所有模型的投票
      all_names = [client.id for client in clients]
      all_models = [client.model for client in clients]
      binary_votes = votes_matrix  # 直接用传入的二维列表

      ac_e = AgglomerativeClustering(n_clusters=2, distance_threshold=None,
                                      compute_full_tree=True,
                                      metric="euclidean", memory=None, connectivity=None,
                                      linkage='single',
                                      compute_distances=True).fit(binary_votes)
      ac_e_labels: list = ac_e.labels_.tolist()
      agglomerative_result = create_cluster_map_from_labels(len(all_names), ac_e_labels)
      print(f'Agglomerative Clustering: {[all_client_names[i] for i in agglomerative_result]}')
      agglomerative_negative_cluster = agglomerative_result[
          determine_biggest_cluster(agglomerative_result)]

      db_scan_input_idx_list = agglomerative_negative_cluster
      db_scan_input_name_list = [all_client_names[i] for i in db_scan_input_idx_list]
      print(f'DBScan Input: {[all_client_names[i] for i in db_scan_input_idx_list]}')
      db_scan_input_list = [binary_votes[vote_id] for vote_id in db_scan_input_idx_list]

      db = DBSCAN(eps=0.5, min_samples=1).fit(db_scan_input_list)
      dbscan_clusters = create_cluster_map_from_labels(len(agglomerative_negative_cluster),
                                                        db.labels_.tolist())
      biggest_dbscan_cluster = dbscan_clusters[determine_biggest_cluster(dbscan_clusters)]
      print(f'DBScan Clustering: {[db_scan_input_name_list[i] for i in biggest_dbscan_cluster]}')

      single_sample_of_biggest_cluster = biggest_dbscan_cluster[0]
      final_voting = db_scan_input_list[single_sample_of_biggest_cluster]
      negatives = [i for i, vote in enumerate(final_voting) if vote == VOTE_FOR_BENIGN]
      # recognized_benign_models = [all_models[n] for n in negatives]

      print(f'Negatives: {[all_client_names[i] for i in negatives]}')

      return negatives

      # self.fedavg([client for client in clients if client.model in recognized_benign_models])
