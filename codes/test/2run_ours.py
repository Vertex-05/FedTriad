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
from CrowdGuardClientValidation import CrowdGuardClientValidation
from auror_defense import *
from krum_defense import *
from median_defense import *
from DP_defense import *

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

def CrowdGuard_validate(participating_clients ,all_models, train_loaders, global_model, all_client_names):
      VOTE_FOR_BENIGN = 1
      VOTE_FOR_POISONED = 0

      device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
      # all_models = [client.model for client in participating_clients]
      # train_loaders = [client.train_loader for client in participating_clients]
      # global_model = server.models[0]
      # all_client_names = [client.id for client in participating_clients]

      # 收集所有客户端的投票
      votes_matrix = []
      for own_client_index, client in enumerate(participating_clients):
          client_name = all_client_names[own_client_index]
          detected_suspicious_models = CrowdGuardClientValidation.validate_models(
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
  # --- FedREDefense thresholds ---
  args.re_thresh_hard = hp.get("re_thresh_hard", 0.973)
  args.re_thresh_defer = hp.get("re_thresh_defer", 0.75)

  # --- SPRT 参数 ---
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


  if hp["attack_rate"] == 0:
        clients = [Client(model_name, optimizer_fn, loader, idnum=i, num_classes=num_classes, dataset = hp['dataset']) for i, (loader, model_name) in enumerate(zip(client_loaders, model_names))]
  else:
    clients = []
    for i, (loader, model_name) in enumerate(zip(client_loaders, model_names)):
        if i < (1 - hp["attack_rate"])* len(client_loaders):
          clients.append(Client(model_name, optimizer_fn, loader, idnum=i, num_classes=num_classes, dataset = hp['dataset']) )
        else:
          print(i)
          if hp["attack_method"] == "label_flip":
            clients.append(Client_flip(model_name, optimizer_fn, loader, idnum=i, num_classes=num_classes, dataset = hp['dataset']))
          elif hp["attack_method"] == "Fang":
            clients.append(Client_Fang(model_name, optimizer_fn, loader, idnum=i, num_classes=num_classes, dataset = hp['dataset']) )
          elif hp["attack_method"] == "MPAF":
            clients.append(Client_MPAF(model_name, optimizer_fn, loader, idnum=i, num_classes=num_classes, dataset = hp['dataset']) )
          elif hp["attack_method"] == "Min-Max":
            clients.append(Client_MinMax(model_name, optimizer_fn, loader, idnum=i, num_classes=num_classes, dataset = hp['dataset']) )
          elif hp["attack_method"] == "Min-Sum":
            clients.append(Client_MinSum(model_name, optimizer_fn, loader, idnum=i, num_classes=num_classes, dataset = hp['dataset']) )
          elif hp["attack_method"] == "Scaling":
            clients.append(Client_Scaling(model_name, optimizer_fn, loader, idnum=i, num_classes=num_classes, dataset = hp['dataset']) )
          elif hp["attack_method"] == "DBA":
            clients.append(Client_DBA(model_name, optimizer_fn, loader, idnum=i, num_classes=num_classes, dataset = hp['dataset']) )
          else:
            import pdb; pdb.set_trace()  
        
  # initialize server and clients
  server = Server(np.unique(model_names), test_loader,num_classes=num_classes, dataset = hp['dataset'])
  initial_model_state = server.models[0].state_dict().copy()


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
  # ========= 初始化 CrowdGuard + SPRT 日志容器 =========
  xp.results["crowdguard_round_groups"] = []  # ✅ 初始化列表，避免 log 嵌套

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

    if "CrowdGuard" in hp["aggregation_mode"]:
      VOTE_FOR_BENIGN = 1
      VOTE_FOR_POISONED = 0

      device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
      all_models = [client.model for client in participating_clients]
      train_loaders = client_loaders
      global_model = server.models[0]
      all_client_names = [client.id for client in participating_clients]

      # 收集所有客户端的投票
      votes_matrix = []
      for own_client_index, client in enumerate(participating_clients):
          client_name = all_client_names[own_client_index]
          detected_suspicious_models = CrowdGuardClientValidation.validate_models(
              global_model,
              all_models,
              own_client_index,
              train_loaders[client_name],
              device
          )
          detected_suspicious_models = sorted(detected_suspicious_models)
          print(f'Suspicious Models detected by {client_name}: {detected_suspicious_models}')

          votes_of_this_client = []
          for c in range(len(all_models)):
              if c == own_client_index:
                  votes_of_this_client.append(VOTE_FOR_BENIGN)
              elif c in detected_suspicious_models:
                  votes_of_this_client.append(VOTE_FOR_POISONED)
              else:
                  votes_of_this_client.append(VOTE_FOR_BENIGN)
          votes_matrix.append(votes_of_this_client)

      # 聚合：调用 server 的 crowdguard_aggregate 方法
      server.crowdguard_aggregate(participating_clients, votes_matrix,all_client_names)

    elif "Auror" in hp["aggregation_mode"]:
      print("\n[Auror Defense] Start detecting malicious clients...")

      # 1) 收集参与客户端的模型（注意：这里用的是 client.model 而不是 raw updates）
      participating_models = [client.model for client in participating_clients]

      # 2) 初始化 Auror（可调参数从 hp 读取）
      auror = AurorDefense(
          alpha=hp.get("alpha", 0.02),
          tau=hp.get("tau", 0.5),
          epochs_to_analyze=hp.get("epochs_to_analyze", 10),
          verbose=True,
          pca_dim=hp.get("auror_pca_dim", 8)
      )

      # 3) 执行检测（返回 malicious_users 列表）
      malicious_users, indicative_features = auror.defend(participating_models, train_global_model_func=None)

      # 4) 过滤掉被识别的恶意客户端，使用剩下的模型做 FedAvg 聚合
      benign_indices = [i for i in range(len(participating_clients)) if i not in malicious_users]
      if len(benign_indices) == 0:
          print("[Auror] WARNING: all participating clients marked malicious -> skip aggregation this round")
      else:
          # 将 benign client 的 state_dict 聚合为新的 global model (按 equal weighting)
          with torch.no_grad():
              # 取第一个 benign 的 state_dict 作为模板
              base_state = deepcopy(participating_clients[benign_indices[0]].model.state_dict())
              # stack tensors per key
              for k in base_state.keys():
                  stacked = torch.stack([participating_clients[idx].model.state_dict()[k].to(base_state[k].device) for idx in benign_indices], dim=0)
                  base_state[k] = torch.mean(stacked, dim=0)
              server.models[0].load_state_dict(base_state)  # 或 server.global_model 视你的实现
          print(f"[Auror] Aggregated with {len(benign_indices)} benign clients; removed {len(malicious_users)} malicious clients.")
      xp.log({"Auror_detected_clients": malicious_users, "Auror_indicative_features": indicative_features})


    elif "Krum" in hp["aggregation_mode"]:

      n_part = len(participating_clients)
      attack_rate = hp.get("attack_rate", 0.0)
      f = int(np.floor(n_part * attack_rate))
      if n_part <= 2 * f + 2:
          f = max(0, (n_part - 3) // 2)
          print(f"[Krum Warning] adjusted f -> {f}")

      new_state_dict, selected_indices, metrics = aggregate_clients_with_krum(
          participating_clients=participating_clients,
          server_model=server.models[0],
          f=f,
          select_ratio=2/3,  # ✅ 每轮选中 2/3 客户端
          device=next(server.models[0].parameters()).device,
          overall_label=overall_label,  # ✅ 传入全局30客户端标签
          total_clients=len(client_loaders)
      )

      server.models[0].load_state_dict(new_state_dict)

      xp.log({
          "krum_round": c_round,
          "krum_selected_clients": np.array(selected_indices),
          "krum_TPR": metrics["TPR"],
          "krum_TNR": metrics["TNR"],
          "krum_PRC": metrics["PRC"]
      })

    elif "Median" in hp["aggregation_mode"]:
        # Median 聚合：不检测恶意，仅更新模型
        new_state_dict = aggregate_clients_with_median(participating_clients, server)

        # 将新的全局模型状态载入服务器
        server.models[0].load_state_dict(new_state_dict)

        # 记录日志
        xp.log({
            "median_round": c_round,
            "aggregation_mode": "Median"
        })

    elif "DP" in hp["aggregation_mode"]:

      print(f"\n[Round {c_round}] Using Differential Privacy Aggregation (DP Mode)\n")

      # 参数写死，不再从 hp 读取
      clip_norm        = 1.0
      noise_multiplier = 0.5
      delta            = 1e-5
      sample_rate      = 0.1
      clip_frequency   = 1

      server.models[0] = dp_aggregate(
          participating_clients,
          server,
          {
              "clip_norm": clip_norm,
              "noise_multiplier": noise_multiplier,
              "delta": delta,
              "sample_rate": sample_rate,
              "clip_frequency": clip_frequency,
          }
      )

      # # === 可选：测试聚合后全局模型性能 ===
      # eval_results = server.evaluate_ensemble()
      # xp.log({
      #     f"round_{c_round}_DP_MA": eval_results["acc"],
      #     f"round_{c_round}_DP_loss": eval_results["loss"]
      # })

      # print(f"[DP] Round {c_round} | Test Accuracy: {eval_results['acc']:.4f}")
    # else:
    #   raise NotImplementedError
    
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
   