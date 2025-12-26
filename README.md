FedTriad
The official code repository for the paper "FedTriad: A 'Self-Others-Global' Trust Triangle for Federated Learning Backdoor Defense".

Overview
Federated Learning faces severe threats from backdoor attacks. Existing defenses often rely on data similarity across clients or assume that benign clients form the majority. These methods frequently fail when dealing with Non IID data or when malicious clients dominate the network. To address these challenges, we introduce FedTriad. Unlike single dimension defenses, FedTriad establishes a triadic trust architecture based on a Self Others Global perspective.
Self (Endogenous Auditing) This module uses the Optimization Trajectory Matching Error (OTME) to reconstruct model updates. It filters out malicious gradients generated from abnormal training processes right at the source.
Others (Exogenous Detection) This module employs the Deep Representation Consistency Metric (DRCM) for client peer reviews. It helps identify stealthy backdoors hidden within deep neural layers.
Global (Temporal Arbitration) This module utilizes the Sequential Malicious Index (SMI) and the SPRT algorithm. By accumulating evidence over time, it dynamically identifies and removes adaptive attackers.
Experiments show that FedTriad effectively defends against hybrid backdoor attacks.It remains robust even when malicious clients exceed 50 percent and data is highly Non IID.

Quick Start
You can evaluate FedTriad on three benchmark datasets.
Bash
# CIFAR-10 Dataset
bash ./scripts/cifar10/ours.sh

Acknowledgement
We appreciate the inspiration and code support from the following open source projects.
FedREDefense: Defending against Model Poisoning Attacks for Federated Learning using Model Update Reconstruction Error 
CrowdGuard: Federated Backdoor Detection in Federated Learning Dataset Distillation by Matching Training Trajectories