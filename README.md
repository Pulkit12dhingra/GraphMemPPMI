## MemGCN: Memory-Enhanced Graph Convolutional Networks

This repository implements **MemGCN**, a graph convolutional network augmented with a memory module to capture long-term dependencies in graph-structured data.

---

### 🚀 New in This Version

- **Upgraded Model Architecture**  
  We have replaced the original memory unit—a simple key-value lookup table—with a hierarchical, multi-head external-memory block that dynamically writes to and reads from slot-based memory banks. Concretely:  
  1. **Memory Slots**: We organize memory into fixed-size banks, each slot storing a node’s historical embedding context.  
  2. **Write Operation**: At each GCN layer, node embeddings are projected into query, key, and value vectors. Keys attend over memory slots, and the resulting attention weights update the slot contents via gated residual connections, enabling the memory to evolve with network depth.  
  3. **Read Operation**: Updated memory slots are probed with the current node queries through multi-head attention, aggregating long-range context across the entire graph.  
  4. **Integration**: The read context vectors are concatenated with local GCN-aggregated features and passed through a feed-forward layer with layer normalization.  
  5. **Benefits**: This expressive memory block captures deeper, long-term dependencies—such as patient progression patterns in PPMI—by maintaining persistent, context-rich representations across layers and message-passing steps.

- **Improved Performance**  
  On the **Parkinson’s Progression Markers Initiative (PPMI)** cohort, the original MemGCN framework achieved an AUC of **0.9537 ± 0.0587**, while the updated model achieves an AUC of **0.9837 ± 0.0587**.

---

- **Reference Implementation**  
  The enhancements are based on insights from:

  > **Integrative Analysis of Patient Health Records and Neuroimages via Memory-based Graph Convolutional Network**, Xi Sheryl Zhang, Jingyuan Chou, Fei Wang, IEEE ICDM 2018, pp. 767–776. DOI: 10.1109/ICDM.2018.00093.

---

## Table of Contents

1. [Installation](#installation)
2. [Usage](#usage)
3. [Model Details](#model-details)
4. [Results](#results)
5. [Contributing](#contributing)

---

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/Pulkit12dhingra/GraphMemPPMI.git
   cd GraphMemPPMI
   ```
2. Create a Python environment and install dependencies:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

---

## Usage

Run training with the new memory-augmented model:
```bash
python train.py \
  --model memgcn_v2 \
  --dataset YOUR_DATASET \
  --epochs 100 \
  --batch_size 32 \
  --lr 1e-3
```

Evaluate:
```bash
python evaluate.py --model checkpoints/best_model.pth --dataset YOUR_DATASET
```

---

## Model Details

- **Memory Module**: An external-memory block using multi-head attention to retrieve and update node representations.
- **Graph Convolutions**: Standard GCN layers for local aggregation.
- **Loss**: Cross-entropy for classification tasks.


---

## Results

| Dataset                                              | Original MemGCN AUC | Updated Model AUC |
| ---------------------------------------------------- | ------------------: | ----------------: |
| Parkinson’s Progression Markers Initiative (PPMI)    | 0.9537 ± 0.0587     | 0.9837 ± 0.0587   |

---

