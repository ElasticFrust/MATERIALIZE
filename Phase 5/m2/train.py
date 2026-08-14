r"""Phase 5 / M2 — training scaffold for the GNN forward surrogate (plain torch).

Loads the consolidated dataset (Phase 5/m2/data/dataset.npz), batches graphs with a
plain batch-of-graphs index scheme (no torch_geometric), trains ForwardGNN to predict
C6 (MSE on C6 components normalised by their TRAIN std), and evaluates the DERIVED
nu(theta),E(theta) MAE on a held-out validation split.

This is a SCAFFOLD / smoke check — a short run to PROVE the pipeline learns end-to-end,
NOT a converged model.  It saves:
  - Phase 5/m2/results/loss_curve.png     (train/val C6 loss)
  - Phase 5/m2/results/pred_vs_true.png   (val nu & E, predicted vs true from C6)
  - Phase 5/m2/checkpoint.pt              (weights + C6 normalisation stats + config)

Run:  "C:\Users\doron\anaconda3\python.exe" "Phase 5\m2\train.py"
      "C:\Users\doron\anaconda3\python.exe" "Phase 5\m2\train.py" --epochs 400
"""
import os, sys, json, argparse, time
import numpy as np
import torch

HERE = os.path.dirname(__file__)
sys.path.insert(0, HERE)
from model import (ForwardGNN, build_edge_features, node_degree_feature,   # noqa: E402
                   c6_pred_to_nuE, c6_pred_to_nuE_theta)
from model import ANG                                                       # noqa: E402
torch.set_default_dtype(torch.float64)

DATA = os.path.join(HERE, 'data', 'dataset.npz')
RESULTS = os.path.join(HERE, 'results')
CKPT = os.path.join(HERE, 'checkpoint.pt')


# ---- dataset in memory (list of per-graph tensors) -------------------------------------------
def load_samples(path):
    """Return a list of per-sample dicts (torch tensors) + label matrix C6 (S,6)."""
    d = np.load(path, allow_pickle=True)
    npr, epr = d['node_ptr'], d['edge_ptr']
    pts, bu, bv, bR, k = d['pts'], d['bond_u'], d['bond_v'], d['bond_R'], d['k']
    C6 = torch.as_tensor(d['C6'])
    samples = []
    for i in range(len(npr) - 1):
        ns, ne = slice(npr[i], npr[i + 1]), slice(epr[i], epr[i + 1])
        n_nodes = int(npr[i + 1] - npr[i])
        edge_attr, _ = build_edge_features(bR[ne], k[ne])            # (2M,5)
        bu_i = torch.as_tensor(bu[ne], dtype=torch.long)
        bv_i = torch.as_tensor(bv[ne], dtype=torch.long)
        # directed edges both ways (matches build_edge_features fwd|bwd order)
        src = torch.cat([bu_i, bv_i]); dst = torch.cat([bv_i, bu_i])
        edge_index = torch.stack([src, dst], 0)
        node_feat = node_degree_feature(bu[ne], bv[ne], n_nodes)
        samples.append(dict(node_feat=node_feat, edge_index=edge_index,
                            edge_attr=edge_attr, n_nodes=n_nodes, C6=C6[i]))
    return samples, C6


# ---- collate a batch of graphs into one big graph --------------------------------------------
def collate(batch):
    node_feats, edge_attrs, edge_indices, batch_ids, C6s = [], [], [], [], []
    off = 0
    for gi, s in enumerate(batch):
        node_feats.append(s['node_feat'])
        edge_attrs.append(s['edge_attr'])
        edge_indices.append(s['edge_index'] + off)
        batch_ids.append(torch.full((s['n_nodes'],), gi, dtype=torch.long))
        C6s.append(s['C6'])
        off += s['n_nodes']
    return dict(node_feat=torch.cat(node_feats, 0),
                edge_index=torch.cat(edge_indices, 1),
                edge_attr=torch.cat(edge_attrs, 0),
                batch=torch.cat(batch_ids, 0),
                num_graphs=len(batch),
                C6=torch.stack(C6s, 0))


def iterate(samples, idx, batch_size, shuffle, rng):
    order = idx.copy()
    if shuffle:
        rng.shuffle(order)
    for i in range(0, len(order), batch_size):
        yield collate([samples[j] for j in order[i:i + batch_size]])


# ---- training --------------------------------------------------------------------------------
def run(epochs=250, batch_size=16, hidden=64, n_layers=4, lr=2e-3, val_frac=0.2, seed=0):
    assert os.path.exists(DATA), f"dataset not found: {DATA} (run build_dataset.py first)"
    os.makedirs(RESULTS, exist_ok=True)
    rng = np.random.default_rng(seed)
    samples, C6_all = load_samples(DATA)
    S = len(samples)
    print(f"loaded {S} samples from {DATA}")

    # train/val split
    perm = rng.permutation(S)
    n_val = max(1, int(round(val_frac * S)))
    val_idx, train_idx = perm[:n_val], perm[n_val:]
    print(f"train={len(train_idx)}  val={len(val_idx)}")

    # C6 normalisation from TRAIN split
    C6_train = C6_all[train_idx]
    c6_mean = C6_train.mean(0)
    c6_std = C6_train.std(0).clamp_min(1e-6)

    model = ForwardGNN(node_in=1, edge_in=5, hidden=hidden, n_layers=n_layers)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=max(1, epochs // 3), gamma=0.5)

    def batch_loss(b):
        pred = model(b['node_feat'], b['edge_index'], b['edge_attr'], b['batch'], b['num_graphs'])
        tgt = (b['C6'] - c6_mean) / c6_std
        predn = (pred - c6_mean) / c6_std
        return torch.mean((predn - tgt) ** 2), pred

    hist = {'train': [], 'val': []}
    t0 = time.time()
    for ep in range(epochs):
        model.train()
        tl, ntl = 0.0, 0
        for b in iterate(samples, train_idx, batch_size, True, rng):
            opt.zero_grad()
            loss, _ = batch_loss(b)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            tl += loss.item() * b['num_graphs']; ntl += b['num_graphs']
        sched.step()
        model.eval()
        vl, nvl = 0.0, 0
        with torch.no_grad():
            for b in iterate(samples, val_idx, batch_size, False, rng):
                loss, _ = batch_loss(b)
                vl += loss.item() * b['num_graphs']; nvl += b['num_graphs']
        hist['train'].append(tl / ntl); hist['val'].append(vl / max(nvl, 1))
        if ep % max(1, epochs // 20) == 0 or ep == epochs - 1:
            print(f"  epoch {ep:4d}  train {hist['train'][-1]:.4e}  val {hist['val'][-1]:.4e}")
    print(f"training done in {time.time() - t0:.1f}s")

    # ---- validation metrics on DERIVED nu,E ----
    model.eval()
    nu_t, nu_p, E_t, E_p = [], [], [], []
    with torch.no_grad():
        for j in val_idx:
            b = collate([samples[j]])
            pred = model(b['node_feat'], b['edge_index'], b['edge_attr'], b['batch'], b['num_graphs'])[0]
            try:
                nu_pr, E_pr = c6_pred_to_nuE(pred.numpy())
                nu_tr, E_tr = c6_pred_to_nuE(samples[j]['C6'].numpy())
            except Exception:
                continue
            if np.all(np.isfinite([nu_pr, E_pr, nu_tr, E_tr])):
                nu_p.append(nu_pr); nu_t.append(nu_tr); E_p.append(E_pr); E_t.append(E_tr)
    nu_t, nu_p, E_t, E_p = map(np.array, (nu_t, nu_p, E_t, E_p))
    mae_nu = float(np.mean(np.abs(nu_p - nu_t))) if len(nu_t) else float('nan')
    mae_E = float(np.mean(np.abs(E_p - E_t))) if len(E_t) else float('nan')
    print(f"\nVAL MAE(nu) = {mae_nu:.4f}    VAL MAE(E) = {mae_E:.4f}   (n_val_valid={len(nu_t)})")

    _plot_curves(hist, os.path.join(RESULTS, 'loss_curve.png'))
    _plot_scatter(nu_t, nu_p, E_t, E_p, mae_nu, mae_E,
                  os.path.join(RESULTS, 'pred_vs_true.png'))

    torch.save(dict(state_dict=model.state_dict(),
                    c6_mean=c6_mean, c6_std=c6_std,
                    config=dict(hidden=hidden, n_layers=n_layers, node_in=1, edge_in=5),
                    val_mae_nu=mae_nu, val_mae_E=mae_E), CKPT)
    print(f"saved checkpoint -> {CKPT}")
    print(f"saved figures    -> {RESULTS}")
    return dict(mae_nu=mae_nu, mae_E=mae_E, hist=hist, n=S)


# ---- prediction helper (for README / downstream use) -----------------------------------------
@torch.no_grad()
def predict_c6(model, geo, k):
    """Predict C6 (numpy (6,)) for a single (geo, k) with a loaded ForwardGNN."""
    edge_attr, _ = build_edge_features(geo['bond_R'], k)
    bu = torch.as_tensor(geo['bond_u'], dtype=torch.long)
    bv = torch.as_tensor(geo['bond_v'], dtype=torch.long)
    edge_index = torch.stack([torch.cat([bu, bv]), torch.cat([bv, bu])], 0)
    node_feat = node_degree_feature(geo['bond_u'], geo['bond_v'], len(geo['pts']))
    batch = torch.zeros(len(geo['pts']), dtype=torch.long)
    pred = model(node_feat, edge_index, edge_attr, batch, 1)[0]
    return pred.numpy()


def load_model(ckpt=CKPT):
    d = torch.load(ckpt, weights_only=False)
    m = ForwardGNN(**d['config'])
    m.load_state_dict(d['state_dict']); m.eval()
    return m, d


# ---- figures ---------------------------------------------------------------------------------
def _plot_curves(hist, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 4.2))
    ax.plot(hist['train'], label='train', color='#4C78A8')
    ax.plot(hist['val'], label='val', color='#E45756')
    ax.set_yscale('log'); ax.set_xlabel('epoch'); ax.set_ylabel('normalised C6 MSE')
    ax.set_title('M2 smoke-train loss (scaffold)'); ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(path, dpi=170); plt.close(fig)


def _plot_scatter(nu_t, nu_p, E_t, E_p, mae_nu, mae_E, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    for ax, t, p, name, mae in [(axes[0], nu_t, nu_p, 'ν', mae_nu),
                                (axes[1], E_t, E_p, 'E', mae_E)]:
        if len(t):
            lo, hi = float(min(t.min(), p.min())), float(max(t.max(), p.max()))
            ax.plot([lo, hi], [lo, hi], '--', color='0.5', lw=1)
            ax.scatter(t, p, s=22, alpha=0.7, c='#4C78A8', edgecolors='none')
            ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
        ax.set_aspect('equal', 'box')
        ax.set_xlabel(f'true {name}'); ax.set_ylabel(f'predicted {name}')
        ax.set_title(f'val {name}: MAE={mae:.4f}'); ax.grid(alpha=0.3)
    fig.suptitle('M2 forward surrogate — val predicted vs true (derived from C6)')
    fig.tight_layout(); fig.savefig(path, dpi=170); plt.close(fig)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--epochs', type=int, default=250)
    ap.add_argument('--batch_size', type=int, default=16)
    ap.add_argument('--hidden', type=int, default=64)
    ap.add_argument('--n_layers', type=int, default=4)
    ap.add_argument('--lr', type=float, default=2e-3)
    args = ap.parse_args()
    run(epochs=args.epochs, batch_size=args.batch_size, hidden=args.hidden,
        n_layers=args.n_layers, lr=args.lr)
