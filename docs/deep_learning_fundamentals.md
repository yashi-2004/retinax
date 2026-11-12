# Deep Learning Foundations in RetinaX

## 1. Custom Convolutional Neural Network (`CustomFundusCNN`)

`src/models/custom_cnn.py` implements a configurable multi-block CNN from scratch in PyTorch to expose every core building block transparently:

- **2D Convolution & Receptive Fields:** A kernel of size $k \times k$, stride $s$, and padding $p$ transforms spatial resolution $H$ to $\lfloor (H + 2p - k)/s \rfloor + 1$. Across blocks $l = 1 \dots L$, the theoretical receptive field grows recursively with effective jump $j_l$:
  $$RF_l = RF_{l-1} + (k_l - 1)\, j_{l-1}, \qquad j_l = j_{l-1} \cdot s_l$$
  This allows verifying whether final feature maps span retinal lesions ($15\text{--}60\text{ px}$ at $224\text{px}$ scale).
- **Batch Normalization & Dropout:** BatchNorm stabilizes feature activations across mini-batches; Dropout regularizes classifier heads and remains controllable during stochastic Monte Carlo inference.
- **Activation & Softmax:** Non-linear activations (`ReLU`, `GELU`, `LeakyReLU`) feed logits $z \in \mathbb{R}^5$, mapped to class probabilities via $\hat{p}_c = \frac{\exp(z_c)}{\sum_j \exp(z_j)}$.

## 2. Imbalance-Aware Objectives (`src/training/losses.py`)

Because diabetic retinopathy datasets skew strongly toward grade `0 (No DR)`:
- **Class-Weighted Cross-Entropy:** $\mathcal{L}_{\text{WCE}}(z, y) = -w_y \log \hat{p}_y$, where $w_c \propto \sqrt{N / (C N_c)}$ boosts minority grades (`Mild`, `Severe`, `Proliferative DR`).
- **Focal Loss:** $\mathcal{L}_{\text{Focal}}(z, y) = -\alpha_y (1 - \hat{p}_y)^\gamma \log \hat{p}_y$ down-weights easy well-classified majority samples.
- **Label Smoothing:** Softens one-hot targets toward $(1 - \epsilon)e_y + \epsilon/C$ to curb extreme logit overconfidence.

## 3. Backpropagation, Optimizers & Transfer Learning

- Gradient updates follow $\theta_{t+1} = \theta_t - \eta\,\hat{m}_t / (\sqrt{\hat{v}_t} + \epsilon) - \eta\lambda\theta_t$ (AdamW), with gradient norm clipping ($\|\nabla \theta\|_2 \le 1.0$) and differential learning rates ($0.25\times$ backbone LR when fine-tuning pretrained CNNs).
- Transfer regimes (`scratch`, `frozen`, `partial_finetune`, `full_finetune`) isolate representation reuse versus domain adaptation risk.

## 4. Vision Transformers & Self-Supervised Contrastive Learning

- **ViT (`src/models/vision_transformer.py`):** Splits a $224 \times 224$ image into $14 \times 14 = 196$ non-overlapping $16 \times 16$ patches, linearly projects them with positional embeddings and a `[CLS]` token, and processes global relationships via Multi-Head Self-Attention: $\text{Attention}(Q,K,V) = \text{softmax}(QK^\top / \sqrt{d_k})V$.
- **SimCLR (`src/foundation/ssl_simclr.py`):** Learns invariant retinal features without labels by maximizing agreement between two stochastic clinical views $(x_i^{(1)}, x_i^{(2)})$ via NT-Xent loss:
  $$\ell_{i,j} = -\log \frac{\exp(\text{sim}(z_i, z_j)/\tau)}{\sum_{k=1}^{2B}\mathbb{1}_{[k \ne i]}\exp(\text{sim}(z_i, z_k)/\tau)}$$
