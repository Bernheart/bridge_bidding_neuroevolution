import numpy as np

EPS = 1e-5  # finite‐difference step
GAMMA = 0.99


def compute_policy_loss(model, trajectories):
    """
    trajectories: list of per‐episode data, each entry is a dict
      {
        'states':   [state_vec_t],
        'masks':    [avail_mask_t],      # same length as states
        'actions':  [action_idx_t],
        'rewards':  [reward_t]           # reward received *after* t
      }
    Returns float loss = -E[ sum_t γ^t log π(a_t|s_t) * G_t ]
    """
    total_loss = 0.0
    for traj in trajectories:
        # 1) compute returns-to-go
        returns = []
        G = 0.0
        for r in reversed(traj['rewards']):
            G = r + GAMMA * G
            returns.append(G)
        returns = returns[::-1]

        # 2) accumulate policy loss
        for s, m, a, Gt in zip(traj['states'],
                               traj['masks'],
                               traj['actions'],
                               returns):
            # forward to get π(.|s)
            probs = model.forward(s, m)  # returns one‐hot; tweak forward to give probs if needed
            # if your forward only returns one_hot, you’ll need a small wrapper:
            #    def get_probs(self, x, mask): …
            p = probs[a] + 1e-12
            total_loss -= np.log(p) * Gt
    return total_loss


def apply_gradients(model, trajectories, lr=1e-3):
    """
    Approximate ∂L/∂W, ∂L/∂b by finite differences, then do:
      W ← W − lr * ∂L/∂W
      b ← b − lr * ∂L/∂b
    """
    # 1) baseline loss
    base_loss = compute_policy_loss(model, trajectories)

    # 2) prepare gradient containers
    grad_W = [np.zeros_like(W) for W in model.wn]
    grad_b = [np.zeros_like(b) for b in model.bn]

    # 3) for each weight matrix
    for idx, W in enumerate(model.wn):
        it = np.nditer(W, flags=['multi_index'], op_flags=['readwrite'])
        while not it.finished:
            i, j = it.multi_index
            orig = W[i, j]
            W[i, j] = orig + EPS
            l_plus = compute_policy_loss(model, trajectories)
            W[i, j] = orig - EPS
            l_minus = compute_policy_loss(model, trajectories)
            W[i, j] = orig
            grad_W[idx][i, j] = (l_plus - l_minus) / (2 * EPS)
            it.iternext()

    # 4) for each bias vector
    for idx, b in enumerate(model.bn):
        for i in range(b.shape[0]):
            orig = b[i]
            b[i] = orig + EPS
            l_plus = compute_policy_loss(model, trajectories)
            b[i] = orig - EPS
            l_minus = compute_policy_loss(model, trajectories)
            b[i] = orig
            grad_b[idx][i] = (l_plus - l_minus) / (2 * EPS)

    # 5) gradient descent update
    for i in range(len(model.wn)):
        model.wn[i] -= lr * grad_W[i]
        model.bn[i] -= lr * grad_b[i]
