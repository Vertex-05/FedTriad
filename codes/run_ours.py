"""
FedTriad Server Logic
Core architecture: Self-Others-Global Triadic Trust.

Portions of the aggregation and reconstruction logic are adapted from FedREDefense 
(https://github.com/xyq7/FedREDefense).
See README.md for full citations.
"""

from copy import deepcopy
import random
from client import *
from utils import *
from server import Server
from image_synthesizer import Synthesizer
import resource
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from scipy.spatial.distance import cdist, pdist
from ExogenousDetectionClientValidation import ExogenousDetectionClientValidation


rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (2048, rlimit[1]))
np.set_printoptions(precision=4, suppress=True)
def reduce_average(target, sources):
  for name in target:
      target[name].data = torch.mean(torch.stack([source[name].detach() for source in sources]), dim=0).clone()

channel_dict =  {
  "cifar10": 3,
  "cinic10": 3,
  "fmnist": 1,
}
imsize_dict =  {
  "cifar10": (32, 32),
  "cinic10": (32, 32),
  "fmnist": (28, 28),
}
import os

parser = argparse.ArgumentParser()
parser.add_argument("--start", default=0, type=int)
parser.add_argument("--end", default=None, type=int)
parser.add_argument("--hp", default=None, type=str)
parser.add_argument("--DATA_PATH", default=None, type=str)
parser.add_argument("--RESULTS_PATH", default=None, type=str)
parser.add_argument("--CHECKPOINT_PATH", default=None, type=str)

parser.add_argument('--ipc', type=int, default=1, help='image(s) per class')
parser.add_argument('--label_init', type=float, default=0, help='how to init label')
parser.add_argument('--batch_syn', type=int, default=None, help='should only use this if you run out of VRAM')



args = parser.parse_args()

args.RESULTS_PATH = os.path.join(args.RESULTS_PATH, str(random.randint(0,1000)))
if not os.path.exists(args.RESULTS_PATH):
  os.makedirs(args.RESULTS_PATH)



def detection_metric_per_round(real_label, label_pred):
  nobyz = sum(real_label)
  real_label = np.array(real_label)
  label_pred = np.array(label_pred)
  acc = len(label_pred[label_pred == real_label])/label_pred.shape[0]
  recall = np.sum(label_pred[real_label==1]==1)/nobyz
  fpr = np.sum(label_pred[real_label==0]==1)/(label_pred.shape[0]-nobyz)
  fnr = np.sum(label_pred[real_label==1]==0)/nobyz
  return acc, recall, fpr, fnr, label_pred

def threshold_detection(loss, real_label, threshold=0.8):
  loss = np.array(loss)
  # import pdb; pdb.set_trace()
  if np.isnan(loss).any() == True:
    label_pred =np.where(np.isnan(loss), 1, 0).squeeze()
  else:
    label_pred = loss > threshold
  # import pdb; pdb.set_trace()
  real_label = np.array(real_label)
  if np.mean(loss[label_pred == 0]) > np.mean(loss[label_pred == 1]):
      #1 is the label of malicious clients
      label_pred = 1 - label_pred
      
  # import pdb; pdb.set_trace()
  nobyz = sum(real_label)
  acc = len(label_pred[label_pred == real_label])/loss.shape[0]
  recall = np.sum(label_pred[real_label==1]==1)/nobyz
  fpr = np.sum(label_pred[real_label==0]==1)/(loss.shape[0]-nobyz)
  fnr = np.sum(label_pred[real_label==1]==0)/nobyz
  return acc, recall, fpr, fnr, label_pred

def ExogenousDetection_validate(participating_clients ,all_models, train_loaders, global_model, all_client_names):
      VOTE_FOR_BENIGN = 1
      VOTE_FOR_POISONED = 0

      device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

      # 收集所有客户端的投票
      votes_matrix = []
      for own_client_index, client in enumerate(participating_clients):
          client_name = all_client_names[own_client_index]
          detected_suspicious_models = ExogenousDetectionClientValidation.validate_models(
              global_model,
              all_models,
              own_client_index,
              train_loaders[client_name],
              device
          )
          detected_suspicious_models = sorted(detected_suspicious_models)
          suspicious_names = [all_client_names[idx] for idx in detected_suspicious_models]
          print(f'Suspicious Models detected by {client_name}: {suspicious_names}')


          votes_of_this_client = []
          for c in range(len(all_models)):
              if c == own_client_index:
                  votes_of_this_client.append(VOTE_FOR_BENIGN)
              elif c in detected_suspicious_models:
                  votes_of_this_client.append(VOTE_FOR_POISONED)
              else:
                  votes_of_this_client.append(VOTE_FOR_BENIGN)
          votes_matrix.append(votes_of_this_client)
      return votes_matrix



def run_experiment(xp, xp_count, n_experiments):
  print(xp)
  hp = xp.hyperparameters
  args.attack_method = hp["attack_method"] 
  num_classes = {"mnist" : 10, "fmnist" : 10,"femnist" : 62, "cifar10" : 10,"cinic10" : 10, "cifar100" : 100, "nlp" : 4, 'news20': 20}[hp["dataset"]]
  if hp.get("loader_mode", "normal") != "normal":
    num_classes = 3

  args.num_classes = num_classes
  args.channel = channel_dict[hp['dataset']]
  args.imsize = imsize_dict[hp['dataset']]
  args.dataset = hp['dataset']

  args.syn_steps = hp["syn_steps"]
  args.lr_img = hp["lr_img"]
  args.lr_teacher= hp["lr_teacher"]
  args.lr_label=hp["lr_label"]
  args.lr_lr=hp["lr_lr"]
  args.img_optim=hp["img_optim"]
  args.lr_optim=hp["lr_optim"]
  args.Iteration= hp["Iteration"]
  args.Max_Iter = hp["Max_Iter"]

  # --- Endogenous Audit thresholds ---
  args.re_thresh_hard = hp.get("re_thresh_hard", 0.973)
  args.re_thresh_defer = hp.get("re_thresh_defer", 0.75)

  # --- Global Temporal Arbitration Parameters ---
  args.sprt_W = hp.get("sprt_W", 2)
  args.sprt_M_min = hp.get("sprt_M_min", 3)
  args.sprt_min_hard_count = hp.get("sprt_min_hard_count", 2)
  args.sprt_alpha = hp.get("sprt_alpha", 0.01)
  args.sprt_beta = hp.get("sprt_beta", 0.05)
  args.sprt_P_G_b = hp.get("sprt_P_G_b", {'soft': 0.9, 'defer': 0.09, 'hard': 0.01})
  args.sprt_P_G_m = hp.get("sprt_P_G_m", {'soft': 0.02, 'defer': 0.08, 'hard': 0.9})
  args.sprt_p_vote_b = hp.get("sprt_p_vote_b", 0.05)
  args.sprt_p_vote_m = hp.get("sprt_p_vote_m", 0.9)
  
  if args.batch_syn is None:
    args.batch_syn = num_classes * args.ipc
  print(f"num classes {num_classes}, dsa mode {hp.get('dsa', True)}")
  model_names = [model_name for model_name, k in hp["models"].items() for _ in range(k)]
  optimizer, optimizer_hp = getattr(torch.optim, hp["local_optimizer"][0]), hp["local_optimizer"][1]
  optimizer_fn = lambda x : optimizer(x, **{k : hp[k] if k in hp else v for k, v in optimizer_hp.items()})
  print(f"dataset : {hp['dataset']}")
  train_data_all, test_data = data.get_data(hp["dataset"], args.DATA_PATH)
  
  # Creating data indices for training and validation splits:
  np.random.seed(hp["random_seed"])
  torch.manual_seed(hp["random_seed"])
  train_data = train_data_all
  client_loaders, test_loader = data.get_loaders(train_data, test_data, n_clients=len(model_names),
        alpha=hp["alpha"], batch_size=hp["batch_size"], n_data=None, num_workers=4, seed=hp["random_seed"])
  
  server = Server(np.unique(model_names), test_loader,num_classes=num_classes, dataset = hp['dataset'])
  
  # 1. (attack_rate == 0)
  if hp["attack_rate"] == 0:
      clients = [Client(model_name, optimizer_fn, loader, idnum=i, num_classes=num_classes, dataset = hp['dataset']) 
                 for i, (loader, model_name) in enumerate(zip(client_loaders, model_names))]
  
  # 2. 
  else:
    clients = []
    total_n = len(client_loaders)
    
    n_malicious = int(hp["attack_rate"] * total_n) 
    n_benign = total_n - n_malicious               
    
    n_backdoor = int(0.10 * total_n)
    
    n_model_poison = n_malicious - n_backdoor
    
    print(f"\n[Attack Configuration] Total Clients: {total_n}")
    print(f"  - Benign: {n_benign} (IDs: 0 ~ {n_benign-1})")
    print(f"  - Model Poisoning ({hp['attack_method']}): {n_model_poison} (IDs: {n_benign} ~ {n_benign + n_model_poison - 1})")
    print(f"  - Data Poisoning (Backdoor): {n_backdoor} (IDs: {n_benign + n_model_poison} ~ {total_n - 1})")
    print("----------------------------------------------------\n")

    for i, (loader, model_name) in enumerate(zip(client_loaders, model_names)):
        # --- A. 0 ~ n_benign-1 ---
        if i < n_benign:
            clients.append(Client(model_name, optimizer_fn, loader, idnum=i, num_classes=num_classes, dataset = hp['dataset']))
        
        # --- B. modelpoisoning(n_benign ~ n_benign + n_model_poison - 1) ---
        elif i < n_benign + n_model_poison:
            method = hp["attack_method"]

            if method == "label_flip":
                clients.append(Client_flip(model_name, optimizer_fn, loader, idnum=i, num_classes=num_classes, dataset = hp['dataset']))
            elif method == "Fang":
                clients.append(Client_Fang(model_name, optimizer_fn, loader, idnum=i, num_classes=num_classes, dataset = hp['dataset']))
            elif method == "MPAF":
                clients.append(Client_MPAF(model_name, optimizer_fn, loader, idnum=i, num_classes=num_classes, dataset = hp['dataset']))
                if hasattr(clients[-1], 'init_model'): 
                    clients[-1].init_model = initial_model_state
            elif method == "Min-Max":
                clients.append(Client_MinMax(model_name, optimizer_fn, loader, idnum=i, num_classes=num_classes, dataset = hp['dataset']))
            elif method == "Min-Sum":
                clients.append(Client_MinSum(model_name, optimizer_fn, loader, idnum=i, num_classes=num_classes, dataset = hp['dataset']))
            elif method == "Scaling":
                clients.append(Client_Scaling(model_name, optimizer_fn, loader, idnum=i, num_classes=num_classes, dataset = hp['dataset']))
            elif method == "DBA":
                clients.append(Client_DBA(model_name, optimizer_fn, loader, idnum=i, num_classes=num_classes, dataset = hp['dataset']))
            else:
                # default method Scaling
                print(f"[Warning] Unknown attack method '{method}', fallback to Scaling.")
                clients.append(Client_Scaling(model_name, optimizer_fn, loader, idnum=i, num_classes=num_classes, dataset = hp['dataset']))

        # --- C. datapoisoning (last 10%) ---
        else:
            pdr_val = hp.get("pdr", 0.7)  

            clients.append(Client_Backdoor(
                model_name, optimizer_fn, loader, idnum=i, 
                num_classes=num_classes, dataset = hp['dataset'], 
                pdr=pdr_val, 
                scale=3.0,            
                target_label=2,       
                prepare_poisoned=True 
            ))


  print(clients[0].model)
  # initialize data synthesizer
  synthesizer = Synthesizer(deepcopy(clients[0].model), args)
  server.number_client_all = len(client_loaders)
  
  models.print_model(clients[0].model)
  
  clients_flags = [True] * len(clients)
  overall_label = [True] * int((1-hp["attack_rate"])* len(client_loaders)) + [False] * int(hp["attack_rate"]* len(client_loaders))
  # Start Distributed Training Process
  print("Start Distributed Training..\n")
  maximum_acc_test, maximum_acc_val = 0, 0
  xp.log({"server_val_{}".format(key) : value for key, value in server.evaluate_ensemble().items()})
  test_accs = []
  start_trajectories = []
  end_trajectories = []
  syn_data = []
  syn_label = []
  syn_lr = []
  for i in range(len(clients)):
      start_trajectories.append([])
      end_trajectories.append([])
      syn_data.append([])
      syn_label.append([])
      syn_lr.append([])
  
  print(f"model key {list(server.model_dict.keys())[0]}")


  #  SPRT Initialization on Server 
  server.init_sprt(clients, hp={
    "sprt_W": args.sprt_W,
    "sprt_M_min": args.sprt_M_min,
    "sprt_min_hard_count": args.sprt_min_hard_count,
    "sprt_alpha": args.sprt_alpha,
    "sprt_beta": args.sprt_beta,
    "sprt_P_G_b": args.sprt_P_G_b,
    "sprt_P_G_m": args.sprt_P_G_m,
    "sprt_p_vote_b": args.sprt_p_vote_b,
    "sprt_p_vote_m": args.sprt_p_vote_m
  })
  xp.results["ExogenousDetection_round_groups"] = []  

  for c_round in range(1, hp["communication_rounds"]+1):

    if sum(clients_flags) == 0:
      print("[Warning] All clients have been removed by SPRT — training halted.")
      break

    participating_clients = server.select_clients_masked(clients, hp["participation_rate"],clients_flags)
    print({"Remaining Client Count": sum(clients_flags )})
    xp.log({"participating_clients" : np.array([c.id for c in participating_clients])})
    if hp["attack_method"] in ["Fang", "Min-Max", "Min-Sum"]:
      mali_clients = []
      flag = False
      for client in participating_clients:
        if client.id >= (1 - hp["attack_rate"])* len(client_loaders):
          client.synchronize_with_server(server)
          benign_stats = client.compute_weight_benign_update(hp["local_epochs"])
          mali_clients.append(client)
          flag = True
      if flag == True:
        mal_user_grad_mean2, mal_user_grad_std2, all_updates = get_benign_updates(mali_clients, server)
      for client in participating_clients:
        if client.id >= (1 - hp["attack_rate"])* len(client_loaders):
          client.mal_user_grad_mean2 = mal_user_grad_mean2
          client.mal_user_grad_std2 = mal_user_grad_std2
          client.all_updates = all_updates
    
    for client in participating_clients:
      client.synchronize_with_server(server)
      train_stats = client.compute_weight_update(hp["local_epochs"])

    if "FedTriad" in hp["aggregation_mode"]:

      loss = []
      labels = []
      group_soft = set() 
      group_defer = set()
      group_hard = set() 
      round_iter = 0

      # format: list of [client_id, re_value]
      current_round_re_data = []

    # ============ 1. Endogenous Audit ============
      for client in participating_clients:
        if client.id >= (1 - hp["attack_rate"])* len(client_loaders):
          labels.append(1)
        else:
          labels.append(0)
        if len(syn_data[client.id]) == 0:
          first = True
        else:
          first = False
        start_trajectories[client.id].append([server.models[0].state_dict().copy()[name].cpu().clone() for name in server.models[0].state_dict()])
        end_trajectories[client.id].append([client.model.state_dict().copy()[name].cpu().clone() for name in client.model.state_dict()])

        otme_score, syn_data_client, syn_label_client, syn_lr_client, cur_iter = synthesizer.synthesize_single(start_trajectories, end_trajectories, syn_data, syn_label, syn_lr,  client.id, args, c_round)
        loss.append(otme_score)

        current_round_re_data.append([client.id, otme_score])

        # --- Client Triage based on OTME ---
        if otme_score > args.re_thresh_hard:
            group_hard.add(client.id)
        elif otme_score >= args.re_thresh_defer:
            group_defer.add(client.id)
        else:
            group_soft.add(client.id)
        

        # --- save syn_data ---
        syn_data[client.id] = syn_data_client
        syn_label[client.id] = syn_label_client
        syn_lr[client.id] = syn_lr_client
        round_iter += cur_iter

      # ===Write this round's RE data to log  ===
      # xp.log will automatically append this list to the history
      # 're_raw_history' will be an array of shape (Rounds, Clients_per_round, 2)
      xp.log({'re_raw_history': current_round_re_data}, printout=False)

      # ---------- Average OTME per group ----------
      # build as {client.id: otme_score} 
      loss_dict = {client.id: loss[i] for i, client in enumerate(participating_clients)}

      def avg_loss(group):
          if len(group) == 0:
              return 0.0
          return sum(loss_dict[i] for i in group) / len(group)

      avg_soft = avg_loss(group_soft)
      avg_defer = avg_loss(group_defer)
      avg_hard = avg_loss(group_hard)

      print(f"soft_group: {group_soft} | avg RE: {avg_soft:.4f}")
      print(f"defer_group: {group_defer} | avg RE: {avg_defer:.4f}")
      print(f"hard_group: {group_hard} | avg RE: {avg_hard:.4f}")

      # ------------------ Exogenous Detection ------------------
      soft_clients = [c for c in participating_clients if c.id in group_soft]
      defer_clients = [c for c in participating_clients if c.id in group_defer]
      soft_defer_clients = [c for c in participating_clients if c.id in group_soft.union(group_defer)]

      soft_benign_ids, defer_benign_ids = set(), set()
      soft_final_malicious, defer_final_malicious,malicious_clients = set(), set(), set()

      # ---- Step 2: Validation on Set_S ----
      if len(soft_clients) > 2:
          print(f"[Exogenous] Validating Set_S (size= {len(soft_clients)})")

          all_models = [client.model for client in soft_clients]
          global_model = server.models[0]
          all_client_names = [client.id for client in soft_clients]

          soft_votes_matrix = ExogenousDetection_validate(soft_clients, all_models, client_loaders, global_model, all_client_names)
          soft_benign_ids = set(server.ExogenousDetection_aggregate(soft_clients, soft_votes_matrix, all_client_names))
      else:
          soft_benign_ids = set()
          print("[Exogenous] Set_S too small (<3), skipping Step 1.")

      # ---- Step 2: Validation on Set_C (Set_S U Set_D) ----
      if len(group_defer) == 0:
          defer_benign_ids = soft_benign_ids
          print("Defer group empty — skip soft+defer joint validation.")
      elif len(soft_defer_clients) > 2:
          print(f"[Exogenous] Validating Candidate Set_C (Set_S U Set_D, size= {len(soft_defer_clients)})")
          print(f"soft_defer_clients ids: {[c.id for c in soft_defer_clients]}")

          all_models = [client.model for client in soft_defer_clients]
          global_model = server.models[0]
          all_client_names = [client.id for client in soft_defer_clients]

          defer_votes_matrix = ExogenousDetection_validate(soft_defer_clients, all_models, client_loaders, global_model, all_client_names)
          # defer_votes_matrix = ExogenousDetection_validate(soft_defer_clients, server)
          defer_benign_ids = set(server.ExogenousDetection_aggregate(soft_defer_clients, defer_votes_matrix, all_client_names))
      else:
          print("[Exogenous] Set_D empty or Set_C too small, skipping Step 2.")

      # ---- Malicious Client Identification Logic ----
      if len(soft_clients) == 0 and len(defer_clients) == 0:
          print("[Exogenous] No clients to validate (all hard or none).")
          malicious_clients = group_hard

      else:
          # Check Set_S members
          for idx, client in enumerate(soft_clients):
              in_soft_benign = (idx in soft_benign_ids)  # Passed Step 1?
              # Check if passed Step 2 (Mapping global ID to local index in soft_defer_clients)
              if client.id in [c.id for c in soft_defer_clients]:
                  j = [c.id for c in soft_defer_clients].index(client.id)
                  in_defer_benign = (j in defer_benign_ids)
              else:
                  in_defer_benign = True  

              # Logic: "A client in this group (Set_S) is marked as malicious only if it fails... both"
              if (not in_soft_benign) and (not in_defer_benign):
                  soft_final_malicious.add(client.id)

          # Check Set_D members
          for idx, client in enumerate(defer_clients):
              if client.id in [c.id for c in soft_defer_clients]:
                  j = [c.id for c in soft_defer_clients].index(client.id)
                  in_defer_benign = (j in defer_benign_ids)
              else:
                  in_defer_benign = True  
              if not in_defer_benign:
                  defer_final_malicious.add(client.id)

          # Logic: "Clients in Set_D are immediately removed if they fail... the combined set"
          malicious_clients = group_hard.union(soft_final_malicious).union(defer_final_malicious)

          # Debug 
          print(f"[ExogenousDetection] soft_final_malicious: {soft_final_malicious}")
          print(f"[ExogenousDetection] defer_final_malicious: {defer_final_malicious}")
          print(f"[ExogenousDetection] hard_malicious: {group_hard}")
          print(f"[ExogenousDetection] malicious_clients (all): {malicious_clients}")

      # ============ Global Temporal Arbitration ============
      params = server.sprt_params
      for client in participating_clients:
          cid = client.id
          g = ("hard" if cid in group_hard else "defer" if cid in group_defer else "soft")

          st = server.sprt_state[cid]
          # --- (1) Endogenous Evidence Accumulation ---
          ΔL_RE = np.log(params['P_G_m'][g]) - np.log(params['P_G_b'][g])
          st['SMI'] += ΔL_RE
          st['re_count'][g] += 1
          st['obs'] += 1

          # --- (2) Exogenous Evidence Accumulation ---
          if g in ['soft', 'defer']:
              n_votes = 0
              k_votes = 0
              if cid in (group_soft.union(group_defer)):
                  k_votes = 1 if cid in malicious_clients else 0
                  n_votes = 1
              if n_votes > 0:
                  ΔL_vote = k_votes * np.log(params['p_vote_m']/params['p_vote_b']) \
                            + (1 - k_votes) * np.log((1 - params['p_vote_m'])/(1 - params['p_vote_b']))
                  st['SMI'] += ΔL_vote
                  st['n_votes'] += n_votes
                  st['k_votes'] += k_votes
                  st['obs'] += n_votes

          server.sprt_state[cid] = st

      # ============ SPRT Decision Logic ============
      malicious_sp_clients = set()
      removed_clients_this_round = []
      for cid, st in server.sprt_state.items():
          # Apply Warm-up (W) and Minimum Observations (M_min) constraints
          if c_round < params['W']:
              continue

          if st['obs'] < params['M_min']:
              continue

          # Decision: Accept H_1 (Malicious) -> Permanent Exclusion
          if st['SMI'] >= params['logA']:
              malicious_sp_clients.add(cid)
              clients_flags[cid] = False  
              removed_clients_this_round.append(cid)
              print(f"[SPRT] Client {cid} removed from participation (SMI={st['SMI']:.3f})")
          #  Decision: Accept H_0 (Benign) -> Reset SMI
          elif st['SMI'] <= params['logB']:
              st['SMI'] = 0.0
              server.sprt_state[cid]['SMI'] = 0.0

      # Final Aggregation List
      malicious_all = malicious_clients.union(malicious_sp_clients)
      benign_clients = [c for c in participating_clients if c.id not in malicious_all]

      print(f"[Round {c_round}] hard={len(group_hard)}, defer={len(group_defer)}, soft={len(group_soft)}, "
            f"malicious={len(malicious_all)}, benign={len(benign_clients)}")

      active_count = sum(clients_flags)
      removed_count = len(clients) - active_count
      print(f"[Round {c_round}] Active clients: {active_count}, Removed: {removed_count}")
      # ========== record ==========
      xp.results["ExogenousDetection_round_groups"].append({
          "round": c_round,
          "soft_group": sorted(list(group_soft)),
          "defer_group": sorted(list(group_defer)),
          "hard_group": sorted(list(group_hard)),
          "soft_final_malicious": sorted(list(soft_final_malicious)),
          "defer_final_malicious": sorted(list(defer_final_malicious)),
          "hard_malicious": sorted(list(group_hard)),  
          "removed_clients": sorted(list(removed_clients_this_round)),
      })


      total_n_clients = len(client_loaders) 
      current_SMIs = []
      
      for cid in range(total_n_clients):
          # 检查 server.sprt_state 是否有该客户端记录
          if cid in server.sprt_state:
              val = server.sprt_state[cid]['SMI']
          else:
              val = 0.0
          current_SMIs.append(val)
      

      xp.log({'sprt_SMI_history': current_SMIs}, printout=False)


      # ============ aggregation ============
      if len(benign_clients) == 0:
          print("[Warning] No benign clients found this round — skipping aggregation.")
      else:
          server.fedavg(benign_clients)





    else:
      raise NotImplementedError
    
    if xp.is_log_round(c_round):
      xp.log({'communication_round' : c_round, 'epochs' : c_round*hp['local_epochs']})
      xp.log({key : clients[0].optimizer.__dict__['param_groups'][0][key] for key in optimizer_hp})
      print({"server_{}_a_{}".format(key, hp["alpha"]) : value for key, value in server.evaluate_ensemble().items()})
      if hp["attack_method"] in ["DBA", "Scaling"]:
        xp.log({"server_att_{}_a_{}".format(key, hp["alpha"]) : value for key, value in server.evaluate_attack().items()})
        print({"server_att_{}_a_{}".format(key, hp["alpha"]) : value for key, value in server.evaluate_attack().items()})

      stats = server.evaluate_ensemble()
      test_accs.append(stats['test_accuracy'])
      xp.save_to_disc(path=args.RESULTS_PATH, name="logfiles")

  # Save model to disk
  server.save_model(path=args.CHECKPOINT_PATH, name=hp["save_model"])
  # Delete objects to free up GPU memory
  del server; clients.clear()
  torch.cuda.empty_cache()


def run():
  experiments_raw = json.loads(args.hp)
  hp_dicts = [hp for x in experiments_raw for hp in xpm.get_all_hp_combinations(x)][args.start:args.end]
  experiments = [xpm.Experiment(hyperparameters=hp) for hp in hp_dicts]

  print("Running {} Experiments..\n".format(len(experiments)))
  for xp_count, experiment in enumerate(experiments):
    run_experiment(experiment, xp_count, len(experiments))
 
  
if __name__ == "__main__":

  
  run()
   