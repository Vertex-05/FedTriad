import random
from tqdm import tqdm
from functools import partial
from collections import OrderedDict
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import numpy as np 
from utils import *
import models as model_utils
from sklearn.linear_model import LogisticRegression
import os
# import mxnet.ndarray as nd

from math import sqrt

device = 'cuda' if torch.cuda.is_available() else 'cpu'
MEAN = torch.tensor([0.4914, 0.4822, 0.4465])
STD  = torch.tensor([0.2023, 0.1994, 0.2010])

class Device(object):
  def __init__(self, loader):
    
    self.loader = loader

  def evaluate(self, loader=None):
    return eval_op(self.model, self.loader if not loader else loader)

  def save_model(self, path=None, name=None, verbose=True):
    if name:
      torch.save(self.model.state_dict(), path+name)
      if verbose: print("Saved model to", path+name)

  def load_model(self, path=None, name=None, verbose=True):
    if name:
      self.model.load_state_dict(torch.load(path+name))
      if verbose: print("Loaded model from", path+name)
  
class Client(Device):
  def __init__(self, model_name, optimizer_fn, loader, idnum=0, num_classes=10, dataset = 'cifar10'):
    super().__init__(loader)
    self.id = idnum
    print(f"dataset client {dataset}")
    self.model_name = model_name
    self.model_fn = partial(model_utils.get_model(self.model_name)[0], num_classes=num_classes , dataset = dataset)
    self.model = self.model_fn().to(device)

    self.W = {key : value for key, value in self.model.named_parameters()}

    self.optimizer_fn = optimizer_fn
    self.optimizer = self.optimizer_fn(self.model.parameters())

  def synchronize_with_server(self, server):
    server_state = server.model_dict[self.model_name].state_dict()
    self.model.load_state_dict(server_state, strict=False)



  def compute_weight_update(self, epochs=1, loader=None, print_train_loss=False,  hp=None):
    clip_bound, privacy_sigma = None, None
    train_stats = train_op(self.model, self.loader if not loader else loader, self.optimizer, epochs, print_train_loss=print_train_loss)
    return train_stats

  def predict_logit(self, x):
    """Softmax prediction on input"""
    self.model.train()

    with torch.no_grad():
      y_ = self.model(x)

    return y_
  
  def predict_logit_eval(self, x):
    """Softmax prediction on input"""
    self.model.eval()
    with torch.no_grad():
      y_ = self.model(x)

    return y_

class Client_flip(Device):
  def __init__(self, model_name, optimizer_fn, loader, idnum=0, num_classes=10, dataset = 'cifar10'):
    super().__init__(loader)
    self.id = idnum
    print(f"dataset client {dataset}")
    self.model_name = model_name
    self.model_fn = partial(model_utils.get_model(self.model_name)[0], num_classes=num_classes , dataset = dataset)
    self.model = self.model_fn().to(device)

    self.W = {key : value for key, value in self.model.named_parameters()}

    self.optimizer_fn = optimizer_fn
    self.optimizer = self.optimizer_fn(self.model.parameters())
    self.num_classes = num_classes

    
  def synchronize_with_server(self, server):
    server_state = server.model_dict[self.model_name].state_dict()
    self.model.load_state_dict(server_state, strict=False)

    
  def compute_weight_update(self, epochs=1, loader=None):
    train_stats = train_op_flip(self.model, self.loader if not loader else loader, self.optimizer, epochs, class_num=self.num_classes)
    return train_stats


  def predict_logit(self, x):
    """Softmax prediction on input"""
    self.model.train()

    with torch.no_grad():
      y_ = self.model(x)

    return y_
  def predict_logit_eval(self, x):
    """Softmax prediction on input"""
    self.model.eval()
    with torch.no_grad():
      y_ = self.model(x)

    return y_

class Client_tr_flip(Device):
  def __init__(self, model_name, optimizer_fn, loader, idnum=0, num_classes=10, dataset = 'cifar10'):
    super().__init__(loader)
    self.id = idnum
    print(f"dataset client {dataset}")
    self.model_name = model_name
    self.model_fn = partial(model_utils.get_model(self.model_name)[0], num_classes=num_classes , dataset = dataset)
    self.model = self.model_fn().to(device)

    self.W = {key : value for key, value in self.model.named_parameters()}

    self.optimizer_fn = optimizer_fn
    self.optimizer = self.optimizer_fn(self.model.parameters())
    self.num_classes = num_classes

    
  def synchronize_with_server(self, server):
    self.server_state = server.model_dict[self.model_name].state_dict()
    self.model.load_state_dict(self.server_state, strict=False)

    
  def compute_weight_update(self, epochs=1, loader=None):
    train_stats = train_op_tr_flip(self.model, self.loader if not loader else loader, self.optimizer, epochs, class_num=self.num_classes)
    return train_stats


  def predict_logit(self, x):
    """Softmax prediction on input"""
    self.model.train()

    with torch.no_grad():
      y_ = self.model(x)

    return y_
  def predict_logit_eval(self, x):
    """Softmax prediction on input"""
    self.model.eval()
    with torch.no_grad():
      y_ = self.model(x)

    return y_


class Client_MinMax(Device):
  def __init__(self, model_name, optimizer_fn, loader, idnum=0, num_classes=10, dataset = 'cifar10'):
    super().__init__(loader)
    self.id = idnum

    self.model_name = model_name
    self.model_fn = partial(model_utils.get_model(self.model_name)[0], num_classes=num_classes , dataset = dataset)
    self.model = self.model_fn().to(device)

    self.W = {key : value for key, value in self.model.named_parameters()}

    self.optimizer_fn = optimizer_fn
    self.optimizer = self.optimizer_fn(self.model.parameters())  
    self.scale = 1
    self.mal_user_grad_mean2 = None
    self.mal_user_grad_std2 = None
    self.all_updates = None
    
  def synchronize_with_server(self, server):
    server_state = server.model_dict[self.model_name].state_dict()
    self.server_state = server_state
    self.model.load_state_dict(server_state, strict=False)
    
  def compute_weight_benign_update(self, epochs=1, loader=None):
    train_stats = train_op(self.model, self.loader if not loader else loader, self.optimizer, epochs)
    return train_stats
  
  def compute_weight_update(self, epochs=1, loader=None, dev_type='std', threshold=30):
    all_updates = torch.Tensor(np.array(self.all_updates)).cuda()
    model_re = torch.mean(all_updates, dim = 0)
    if dev_type == 'unit_vec':
        deviation = model_re / torch.norm(model_re)  # unit vector, dir opp to good dir
    elif dev_type == 'sign':
        deviation = torch.sign(model_re)
    elif dev_type == 'std':
        deviation = torch.std(all_updates, 0)
    lamda = torch.Tensor([threshold]).float().cuda()

    threshold_diff = 1e-5
    lamda_fail = lamda
    lamda_succ = 0
    if len(all_updates) != 1:
      distances = []
      for update in all_updates:
          distance = torch.norm((all_updates - update), dim=1) ** 2
          distances = distance[None, :] if not len(distances) else torch.cat((distances, distance[None, :]), 0)
      max_distance = torch.max(distances)
      del distances

      while torch.abs(lamda_succ - lamda) > threshold_diff:
          mal_update = (model_re - lamda * deviation)
          distance = torch.norm((all_updates - mal_update), dim=1) ** 2
          max_d = torch.max(distance)
          
          if max_d <= max_distance:
              # print('successful lamda is ', lamda)
              lamda_succ = lamda
              lamda = lamda + lamda_fail / 2
          else:
              lamda = lamda - lamda_fail / 2

          lamda_fail = lamda_fail / 2
      mal_update = (model_re - lamda_succ * deviation)
    else:
      mal_update = (model_re - model_re)

    idx = 0
    user_grad = OrderedDict()
    for name in self.W:
      user_grad[name] = mal_update[idx:(idx+self.W[name].numel())].reshape(self.W[name].shape)
      self.W[name].data = self.server_state[name] + user_grad[name]
      idx += self.W[name].numel()

  def predict_logit(self, x):
    """Softmax prediction on input"""
    self.model.train()

    with torch.no_grad():
      y_ = self.model(x)

    return y_

  def predict_logit_eval(self, x):
    """Softmax prediction on input"""
    self.model.eval()
    print(self.W['classification_layer.bias'])
    print(self.model.state_dict()['classification_layer.bias'])
    with torch.no_grad():
      y_ = self.model(x)

    return y_

class Client_MinSum(Device):
  def __init__(self, model_name, optimizer_fn, loader, idnum=0, num_classes=10, dataset = 'cifar10'):
    super().__init__(loader)
    self.id = idnum

    self.model_name = model_name
    self.model_fn = partial(model_utils.get_model(self.model_name)[0], num_classes=num_classes , dataset = dataset)
    self.model = self.model_fn().to(device)

    self.W = {key : value for key, value in self.model.named_parameters()}

    self.optimizer_fn = optimizer_fn
    self.optimizer = self.optimizer_fn(self.model.parameters())  
    self.scale = 1
    self.mal_user_grad_mean2 = None
    self.mal_user_grad_std2 = None
    self.all_updates = None
    
  def synchronize_with_server(self, server):
    server_state = server.model_dict[self.model_name].state_dict()
    self.server_state = server_state
    self.model.load_state_dict(server_state, strict=False)
    
  def compute_weight_benign_update(self, epochs=1, loader=None):
    train_stats = train_op(self.model, self.loader if not loader else loader, self.optimizer, epochs)
    return train_stats
  
  def compute_weight_update(self, epochs=1, loader=None, dev_type='std', threshold=30):
    # import pdb; pdb.set_trace()
    all_updates = torch.Tensor(np.array(self.all_updates)).cuda()
    model_re = torch.mean(all_updates, dim = 0)
    if dev_type == 'unit_vec':
        deviation = model_re / torch.norm(model_re)  # unit vector, dir opp to good dir
    elif dev_type == 'sign':
        deviation = torch.sign(model_re)
    elif dev_type == 'std':
        deviation = torch.std(all_updates, 0)
    
    lamda = torch.Tensor([threshold]).float().cuda()
    # print(lamda)
    threshold_diff = 1e-5
    lamda_fail = lamda
    lamda_succ = 0
    if len(all_updates) != 1:
      distances = []
      for update in all_updates:
          distance = torch.norm((all_updates - update), dim=1) ** 2
          distances = distance[None, :] if not len(distances) else torch.cat((distances, distance[None, :]), 0)
      
      scores = torch.sum(distances, dim=1)
      min_score = torch.min(scores)
      del distances

      while torch.abs(lamda_succ - lamda) > threshold_diff:
          mal_update = (model_re - lamda * deviation)
          distance = torch.norm((all_updates - mal_update), dim=1) ** 2
          score = torch.sum(distance)
          
          if score <= min_score:
              # print('successful lamda is ', lamda)
              lamda_succ = lamda
              lamda = lamda + lamda_fail / 2
          else:
              lamda = lamda - lamda_fail / 2

          lamda_fail = lamda_fail / 2
      mal_update = (model_re - lamda_succ * deviation)
    # print(lamda_succ)
    else:
      mal_update = (model_re - model_re)
   
    
    
    idx = 0
    user_grad = OrderedDict()
    for name in self.W:
      user_grad[name] = mal_update[idx:(idx+self.W[name].numel())].reshape(self.W[name].shape)
      self.W[name].data = self.server_state[name] + user_grad[name]
      idx += self.W[name].numel()
    # import pdb; pdb.set_trace()
    # return train_stats
    # print(self.W['classification_layer.bias'])
  
  def predict_logit(self, x):
    """Softmax prediction on input"""
    self.model.train()

    with torch.no_grad():
      y_ = self.model(x)

    return y_

  def predict_logit_eval(self, x):
    """Softmax prediction on input"""
    self.model.eval()
    # import pdb; pdb.set_trace()
    print(self.W['classification_layer.bias'])
    print(self.model.state_dict()['classification_layer.bias'])
    with torch.no_grad():
      y_ = self.model(x)

    return y_
  
def compute_lambda(all_updates, model_re, n_attackers):
    # import pdb; pdb.set_trace()
    distances = []
    n_benign, d = all_updates.shape
    for update in all_updates:
        distance = nd.norm(all_updates - update, axis=1)
        distances.append(distance)
    distances = nd.stack(*distances)

    distances = nd.sort(distances, axis=1)
    scores = nd.sum(distances[:, :n_benign - 1 - n_attackers], axis=1)
    min_score = nd.min(scores)
    term_1 = min_score / ((n_benign - n_attackers - 1)
                          * nd.sqrt(nd.array([d]))[0])
    max_wre_dist = nd.max(nd.norm(all_updates - model_re,
                          axis=1)) / (nd.sqrt(nd.array([d]))[0])

    return (term_1 + max_wre_dist)

def score(gradient, v, nbyz):
    num_neighbours = v.shape[0] - 2 - nbyz
    sorted_distance = torch.sort(torch.sum((v - gradient) ** 2, axis=1))[0]
    return torch.sum(sorted_distance[1:(1+num_neighbours)]).item()

def multi_krum(all_updates, n_attackers, multi_k=False):
    nusers = all_updates.shape[0]
    candidates = []
    candidate_indices = []
    remaining_updates = all_updates.clone()
    all_indices = torch.arange(len(all_updates))
    candidates = None

    while len(remaining_updates) > 2 * n_attackers + 2:
        scores = torch.tensor([score(gradient, remaining_updates, n_attackers) for gradient in remaining_updates])
        min_idx = int(scores.argmin(axis=0).item())
        candidate_indices.append(min_idx)
        candidates = torch.reshape(remaining_updates[min_idx].clone(), shape=(1, -1)) if not isinstance(
            candidates, torch.Tensor) else torch.cat((candidates, torch.reshape(remaining_updates[min_idx].clone(), shape=(1, -1))), dim=0)
        if min_idx == remaining_updates.shape[0] - 1:
            remaining_updates = remaining_updates[:min_idx, :]
        elif min_idx == 0:
            remaining_updates = remaining_updates[min_idx + 1:, :]
        else:
            remaining_updates = torch.cat((remaining_updates[:min_idx, :], remaining_updates[min_idx + 1:, :]), dim=0)
        if not multi_k:
            break
    aggregate = torch.mean(candidates, axis=0)
    if multi_k == False:
        return aggregate, candidate_indices[0]
    else:
        return aggregate, candidate_indices

class Client_Krum(Device):
  def __init__(self, model_name, optimizer_fn, loader, idnum=0, num_classes=10, dataset = 'cifar10'):
    super().__init__(loader)
    self.id = idnum

    self.model_name = model_name
    self.model_fn = partial(model_utils.get_model(self.model_name)[0], num_classes=num_classes , dataset = dataset)
    self.model = self.model_fn().to(device)

    self.W = {key : value for key, value in self.model.named_parameters()}

    self.optimizer_fn = optimizer_fn
    self.optimizer = self.optimizer_fn(self.model.parameters())  
    self.scale = 1
    self.mal_user_grad_mean2 = None
    self.mal_user_grad_std2 = None
    self.all_updates = None
    
  def compute_weight_benign_update(self, epochs=1, loader=None):
    train_stats = train_op(self.model, self.loader if not loader else loader, self.optimizer, epochs)
    return train_stats
  
  def synchronize_with_server(self, server):
    server_state = server.model_dict[self.model_name].state_dict()
    self.server_state = server_state
    self.model.load_state_dict(server_state, strict=False)
    
  def compute_weight_update(self, epochs=1, loader=None):
    all_updates = torch.Tensor(np.array(self.all_updates)).cuda()
    model_re = torch.mean(all_updates, dim = 0)
    if len(all_updates) != 1:
      user_grad = OrderedDict()
      
      deviation = torch.sign(model_re)/torch.norm(torch.sign(model_re))
      lamda = compute_lambda(all_updates, model_re, len(all_updates))
      threshold = 1e-5
      mal_update = []
      while lamda > threshold:
          mal_update = (-lamda * deviation)
          agg_grads, krum_candidate = multi_krum(all_updates,len(all_updates), multi_k=False)
          if krum_candidate <len(all_updates):
              break
          else:
              mal_update = []
          lamda *= 0.5

      mal_update = (model_re - lamda * deviation)
    else:
      mal_update =  model_re  -  model_re 
    idx = 0
    user_grad = OrderedDict()
    for name in self.W:
      user_grad[name] = mal_update[idx:(idx+self.W[name].numel())].reshape(self.W[name].shape)
      self.W[name].data = self.server_state[name] + user_grad[name]
      idx += self.W[name].numel()  
  
  def predict_logit(self, x):
    """Softmax prediction on input"""
    self.model.train()

    with torch.no_grad():
      y_ = self.model(x)

    return y_

  def predict_logit_eval(self, x):
    """Softmax prediction on input"""
    self.model.eval()
    # import pdb; pdb.set_trace()
    print(self.W['classification_layer.bias'])
    print(self.model.state_dict()['classification_layer.bias'])
    with torch.no_grad():
      y_ = self.model(x)

    return y_
class Client_Fang(Device):
  def __init__(self, model_name, optimizer_fn, loader, idnum=0, num_classes=10, dataset = 'cifar10'):
    super().__init__(loader)
    self.id = idnum

    self.model_name = model_name
    self.model_fn = partial(model_utils.get_model(self.model_name)[0], num_classes=num_classes , dataset = dataset)
    self.model = self.model_fn().to(device)

    self.W = {key : value for key, value in self.model.named_parameters()}

    self.optimizer_fn = optimizer_fn
    self.optimizer = self.optimizer_fn(self.model.parameters())  
    self.scale = 1
    self.mal_user_grad_mean2 = None
    self.mal_user_grad_std2 = None
    self.all_updates = None
    
  def compute_weight_benign_update(self, epochs=1, loader=None):
    train_stats = train_op(self.model, self.loader if not loader else loader, self.optimizer, epochs)
    return train_stats
  
  def synchronize_with_server(self, server):
    server_state = server.model_dict[self.model_name].state_dict()
    self.server_state = server_state
    self.model.load_state_dict(server_state, strict=False)
    
  def compute_weight_update(self, epochs=1, loader=None):
    all_updates = torch.Tensor(np.array(self.all_updates)).cuda()
    model_re = torch.mean(all_updates, dim = 0)
    if len(all_updates) != 1:
      model_std = torch.std(all_updates, 0)
      user_grad = OrderedDict()
      
      deviation = torch.sign(model_re)

      max_vector_low = model_re + 3 * model_std 
      max_vector_hig = model_re + 4 * model_std
      min_vector_low = model_re - 4 * model_std
      min_vector_hig = model_re - 3 * model_std
      max_range = torch.cat((max_vector_low[:,None], max_vector_hig[:,None]), dim=1)
      min_range = torch.cat((min_vector_low[:,None], min_vector_hig[:,None]), dim=1)
    
      rand = torch.from_numpy(np.random.uniform(0, 1, [len(deviation)])).type(torch.FloatTensor).cuda()
      max_rand = max_range[:, 0].T + rand * (max_range[:, 1] - max_range[:, 0]).T
      min_rand = min_range[:, 0].T + rand * (min_range[:, 1] - min_range[:, 0]).T

      mal_update = (deviation < 0) * max_rand.T + (deviation > 0) * min_rand.T
    else:
      mal_update =  model_re  -  model_re 
    # import pdb; pdb.set_trace()
    idx = 0
    user_grad = OrderedDict()
    for name in self.W:
      user_grad[name] = mal_update[idx:(idx+self.W[name].numel())].reshape(self.W[name].shape)
      self.W[name].data = self.server_state[name] + user_grad[name]
      idx += self.W[name].numel()
    # print(self.W['classification_layer.bias'])
  
  def predict_logit(self, x):
    """Softmax prediction on input"""
    self.model.train()

    with torch.no_grad():
      y_ = self.model(x)

    return y_

  def predict_logit_eval(self, x):
    """Softmax prediction on input"""
    self.model.eval()
    # import pdb; pdb.set_trace()
    print(self.W['classification_layer.bias'])
    print(self.model.state_dict()['classification_layer.bias'])
    with torch.no_grad():
      y_ = self.model(x)

    return y_

class Client_MPAF(Device):
  def __init__(self, model_name, optimizer_fn, loader, idnum=0, num_classes=10, dataset = 'cifar10'):
    super().__init__(loader)
    self.id = idnum
    print(f"dataset client {dataset}")
    self.model_name = model_name
    self.model_fn = partial(model_utils.get_model(self.model_name)[0], num_classes=num_classes , dataset = dataset)
    self.model = self.model_fn().to(device)

    self.W = {key : value for key, value in self.model.named_parameters()}
    self.init_model = None
    self.optimizer_fn = optimizer_fn
    self.optimizer = self.optimizer_fn(self.model.parameters())
    self.scale = 3
    
  def synchronize_with_server(self, server):
    self.server_state = server.model_dict[self.model_name].state_dict()
    self.model.load_state_dict(self.server_state, strict=False)

  def compute_weight_update(self, epochs=1, loader=None):
    # import pdb; pdb.set_trace()
    user_grad = OrderedDict()
    # import pdb; pdb.set_trace()
    for name in self.W:
      user_grad[name] = self.init_model[name] - self.W[name].detach()
      self.W[name].data = self.server_state[name] + self.scale*user_grad[name]

  def predict_logit(self, x):
    """Softmax prediction on input"""
    self.model.train()

    with torch.no_grad():
      y_ = self.model(x)

    return y_
  
  def predict_logit_eval(self, x):
    """Softmax prediction on input"""
    self.model.eval()
    with torch.no_grad():
      y_ = self.model(x)

    return y_

class Client_Scaling(Device):
  def __init__(self, model_name, optimizer_fn, loader, idnum=0, num_classes=10, dataset = 'cifar10'):
    super().__init__(loader)
    self.id = idnum
    print(f"dataset client {dataset}")
    self.model_name = model_name
    self.model_fn = partial(model_utils.get_model(self.model_name)[0], num_classes=num_classes , dataset = dataset)
    self.model = self.model_fn().to(device)

    self.W = {key : value for key, value in self.model.named_parameters()}
    self.init_model = None
    self.optimizer_fn = optimizer_fn
    self.optimizer = self.optimizer_fn(self.model.parameters())
    self.scale = 3
    
  def synchronize_with_server(self, server):
    self.server_state = server.model_dict[self.model_name].state_dict()
    self.model.load_state_dict(self.server_state, strict=False)

    
  def compute_weight_update(self, epochs=1, loader=None):
    # print(self.scale)
    train_stats = train_op_backdoor(self.model, self.loader if not loader else loader, self.optimizer, epochs)
    
    user_grad = OrderedDict()
    # import pdb; pdb.set_trace()
    for name in self.W:
      user_grad[name] = self.W[name].detach() - self.server_state[name]
      self.W[name].data = self.server_state[name] + self.scale*user_grad[name]

    return train_stats


  def predict_logit(self, x):
    """Softmax prediction on input"""
    self.model.train()

    with torch.no_grad():
      y_ = self.model(x)

    return y_
  def predict_logit_eval(self, x):
    """Softmax prediction on input"""
    self.model.eval()
    with torch.no_grad():
      y_ = self.model(x)

    return y_
  
  
class Client_DBA(Device):
  def __init__(self, model_name, optimizer_fn, loader, idnum=0, num_classes=10, dataset = 'cifar10'):
    super().__init__(loader)
    self.id = idnum
    print(f"dataset client {dataset}")
    self.model_name = model_name
    self.model_fn = partial(model_utils.get_model(self.model_name)[0], num_classes=num_classes , dataset = dataset)
    self.model = self.model_fn().to(device)

    self.W = {key : value for key, value in self.model.named_parameters()}
    self.init_model = None
    self.optimizer_fn = optimizer_fn
    self.optimizer = self.optimizer_fn(self.model.parameters())
    self.scale = 3
  def synchronize_with_server(self, server):
    self.server_state = server.model_dict[self.model_name].state_dict()
    self.model.load_state_dict(self.server_state, strict=False)

    
  def compute_weight_update(self, epochs=1, loader=None):
    train_stats = train_op_dba(self.model, self.loader, self.optimizer, epochs, cid = self.id)
    
    user_grad = OrderedDict()
    # import pdb; pdb.set_trace()
    for name in self.W:
      user_grad[name] = self.W[name].detach() - self.server_state[name]
      self.W[name].data = self.server_state[name] + self.scale*user_grad[name]

    return train_stats


  def predict_logit(self, x):
    """Softmax prediction on input"""
    self.model.train()

    with torch.no_grad():
      y_ = self.model(x)

    return y_
  def predict_logit_eval(self, x):
    """Softmax prediction on input"""
    self.model.eval()
    with torch.no_grad():
      y_ = self.model(x)

    return y_
  


# ---- trigger_single_image: 与 CrowdGuard 官方一致 ----
def trigger_single_image(image):
    """
    Add 6x6 red square to (C,H,W) normalized image — identical to CrowdGuard official.
    image: torch.Tensor (C,H,W) in normalized space (same normalization as training)
    """
    # official: color = (torch.Tensor((1, 0, 0)) - MEAN) / STD
    color = (torch.tensor((1.0, 0.0, 0.0)) - MEAN) / STD
    color = color.to(image.device).type(image.dtype)
    # replicate to (6,6,3) then permute to (3,6,6) for channel-first image
    image[:, 0:6, 0:6] = color.repeat((6, 6, 1)).permute(2, 1, 0)
    return image

# ---- poison_data: 与 CrowdGuard 官方一致 ----
def poison_data(samples_to_poison, labels_to_poison, pdr=0.5, target_label=2):
    """
    Poison a batch of samples (NxCxHxW) by injecting the trigger into pdr fraction,
    and set their label to target_label. Implementation follows CrowdGuard official.
    Returns (poisoned_samples, poisoned_labels) as tensors (labels = long).
    """
    if pdr == 0:
        return samples_to_poison, labels_to_poison

    assert 0 < pdr <= 1.0
    samples = samples_to_poison.clone()
    labels = labels_to_poison.clone().long()

    dataset_size = samples.shape[0]
    num_to_poison = int(dataset_size * pdr)
    # corner case: ensure at least 1 if pdr > 0 and dataset_size > 1
    if num_to_poison == 0 and dataset_size > 1:
        num_to_poison = 1

    if num_to_poison == 0:
        return samples, labels

    indices = np.random.choice(dataset_size, size=num_to_poison, replace=False)
    for idx in indices:
        img = trigger_single_image(samples[idx])
        samples[idx] = img
    labels[indices] = int(target_label)

    return samples, labels

# ---- dataloader_to_tensor_dataset: 保留 / 兼容的 helper ----
def dataloader_to_tensor_dataset(loader, max_samples=None, device='cpu'):
    """
    Convert a DataLoader --> TensorDataset(images, labels).
    Returns None on failure. (We keep this robust helper.)
    """
    imgs = []
    labs = []
    cnt = 0
    for batch in loader:
        # support (img, label) or ((img,label), idx) etc.
        if isinstance(batch, (list, tuple)):
            X = batch[0]
            y = batch[1]
        else:
            continue
        imgs.append(X.cpu())
        labs.append(y.cpu())
        cnt += X.size(0)
        if max_samples is not None and cnt >= max_samples:
            break
    if len(imgs) == 0:
        return None
    imgs = torch.cat(imgs, dim=0)
    labs = torch.cat(labs, dim=0).long()
    return TensorDataset(imgs, labs)

# ---- Client_Backdoor: 完整实现，行为与官方一致并会更新 client_loaders ----
class Client_Backdoor(Device):
    """
    Data-poisoning client compatible with Client_Scaling interface.
    - prepare_poisoned_loader(): 将原 loader 转为内存并注入 trigger，生成被投毒的 DataLoader 并写回 self.loader
    - 在 run_ours.py 中创建 client 后调用 cl.prepare_poisoned_loader()，
      然后用 client_loaders[i] = cl.loader 让 CrowdGuard 使用投毒数据。
    """
    def __init__(self, model_name, optimizer_fn, loader, idnum=0, num_classes=10, dataset='cifar10',
                 pdr=0.5, target_label=2, scale=1.0, prepare_poisoned=True, max_poison_samples=None):
        super().__init__(loader)
        self.id = idnum
        self.model_name = model_name
        self.model_fn = partial(model_utils.get_model(self.model_name)[0], num_classes=num_classes, dataset=dataset)
        self.model = self.model_fn().to(device)
        self.W = {key: value for key, value in self.model.named_parameters()}
        self.init_model = None
        self.optimizer_fn = optimizer_fn
        self.optimizer = self.optimizer_fn(self.model.parameters())
        self.pdr = pdr
        self.target_label = target_label
        self.scale = scale
        self.server_state = None

        self._original_loader = loader

        # 如果设置为 True，尝试立即构造并写回被投毒的 loader
        if prepare_poisoned:
            try:
                ok = self.prepare_poisoned_loader(max_samples=max_poison_samples)
                if not ok:
                    print(f"[Client_Backdoor] client {self.id}: prepare_poisoned_loader returned False — keeping original loader.")
            except Exception as e:
                # 不让初始化失败（避免整个训练流程被中断），只打印警告
                print(f"[Client_Backdoor] client {self.id}: prepare_poisoned_loader() raised exception: {e}. Keeping original loader.")
    
    def synchronize_with_server(self, server):
        self.server_state = server.model_dict[self.model_name].state_dict()
        self.model.load_state_dict(self.server_state, strict=False)

    def prepare_poisoned_loader(self, max_samples=None, keep_on_device='cpu'):
        """
        将 self.loader 转为内存 TensorDataset，按 self.pdr 注入 trigger，并把被投毒的 DataLoader 写回 self.loader。
        - max_samples: 若为 int，则只读取前 max_samples 个样本（可节省内存）
        - keep_on_device: 'cpu'（推荐）或 'cuda' —— 被写回的 TensorDataset 会在该 device (但 DataLoader 通常用 cpu tensors)
        返回: True 若成功生成被投毒 loader，否则 False（保留原 loader）
        """
        try:
            tensor_ds = dataloader_to_tensor_dataset(self.loader, max_samples=max_samples, device='cpu')
        except Exception as e:
            # 无法读取 -> 失败
            print(f"[Client_Backdoor] prepare_poisoned_loader: failed to build tensor dataset for client {self.id}: {e}")
            return False

        if tensor_ds is None:
            # 空或无法转换
            print(f"[Client_Backdoor] prepare_poisoned_loader: tensor_ds is None for client {self.id}")
            return False

        # 优先用标准 .tensors 属性（TensorDataset）
        imgs, labs = None, None
        try:
            tensors_attr = getattr(tensor_ds, "tensors", None)
            if tensors_attr is not None:
                imgs, labs = tensors_attr
        except Exception:
            imgs, labs = None, None

        # 回退：遍历 dataset 抽取
        if imgs is None or labs is None:
            try:
                imgs_list, labs_list = [], []
                for item in tensor_ds:
                    if isinstance(item, (list, tuple)):
                        x, y = item[0], item[1]
                    else:
                        continue
                    imgs_list.append(x)
                    labs_list.append(y)
                if len(imgs_list) > 0:
                    imgs = torch.stack(imgs_list, dim=0)
                    labs = torch.tensor([int(x) for x in labs_list], dtype=torch.long)
            except Exception as e:
                print(f"[Client_Backdoor] prepare_poisoned_loader: fallback extraction failed for client {self.id}: {e}")
                imgs, labs = None, None

        if imgs is None or labs is None or imgs.shape[0] == 0:
            print(f"[Client_Backdoor] prepare_poisoned_loader: no data found for client {self.id}")
            return False

        # 将 tensors 保持在 cpu（poison_data 期望 tensor）
        imgs_cpu = imgs.to('cpu')
        labs_cpu = labs.to('cpu').long()

        # 注入后门（poison_data 在你的文件中定义）
        poisoned_imgs, poisoned_labs = poison_data(imgs_cpu, labs_cpu, pdr=self.pdr, target_label=self.target_label)

        # 构建被投毒的 TensorDataset / DataLoader（使用和原 loader 相同的 batch_size/num_workers）
        batch_size = getattr(self.loader, 'batch_size', 32)
        num_workers = getattr(self.loader, 'num_workers', 0)
        poisoned_ds = TensorDataset(poisoned_imgs, poisoned_labs)
        poisoned_loader = DataLoader(poisoned_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)

        # 写回 self.loader（以及保留 self._poisoned_ds 以备后用）
        self.loader = poisoned_loader
        self._poisoned_ds = poisoned_ds
        print(f"[Client_Backdoor] client {self.id} prepared poisoned loader: {poisoned_imgs.shape[0]} samples, pdr={self.pdr}")
        return True

    def compute_weight_update(self, epochs=1, loader=None):
        """
        训练时使用 self.loader（若 prepare_poisoned_loader 成功则是被投毒的 loader）。
        保持与 Scaling 客户端一致的更新格式。
        """
        orig_loader = loader if loader is not None else self.loader

        # 使用 train_op_backdoor（你已有），保持其行为一致
        train_stats = train_op_backdoor(self.model, orig_loader, self.optimizer, epochs)

        # 计算更新（与 Scaling 保持一致）
        user_grad = OrderedDict()
        if self.server_state is None:
            raise RuntimeError("server_state is None: call synchronize_with_server() before compute_weight_update().")

        for name in self.W:
            base = self.server_state[name]
            user_grad[name] = self.W[name].detach() - base
            self.W[name].data = base + self.scale * user_grad[name]

        return train_stats

    def predict_logit(self, x):
        self.model.train()
        with torch.no_grad():
            y_ = self.model(x)
        return y_

    def predict_logit_eval(self, x):
        self.model.eval()
        with torch.no_grad():
            y_ = self.model(x)
        return y_