# run_improved.py
"""
Improved VSF inference script.

This file is a drop-in replacement for train_multi_step.py.  It re-uses the
*exact same* model (MTGNN / gtnet), trainer, data-loading, and evaluation
code from the original repository.  The only part that changes is the
inference wrapper (our three improvements in improved_wrapper.py).

Usage – train (identical to original):
    python run_improved.py --data ./data/SOLAR --model_name mtgnn \
        --device cuda:0 --expid 1 --epochs 100 --batch_size 64 \
        --runs 10 --step_size1 2500

Usage – improved VSF inference (our method):
    python run_improved.py --data ./data/SOLAR --model_name mtgnn \
        --device cuda:0 --expid 1 --epochs 0 --batch_size 64 \
        --runs 10 --random_node_idx_split_runs 100 \
        --lower_limit_random_node_selections 15 \
        --upper_limit_random_node_selections 15 \
        --use_improved_wrapper True \
        --num_neighbors_borrow 5 \
        --joint_alpha 0.3

Baselines still accessible via the original flags:
    --borrow_from_train_data True --use_ewp False   → UW  (uniform)
    --borrow_from_train_data True --use_ewp True    → FDW (original)
    --mask_remaining True                            → Partial (zero-fill)
"""

import torch
import numpy as np
import argparse
import time
import math
import os
from copy import deepcopy

from util import (load_dataset, load_adj, metric,
                  get_node_random_idx_split, zero_out_remaining_input,
                  obtain_instance_prototypes)
from trainer import Trainer
from net import gtnet
from improved_wrapper import (
    cosine_retrieval,
    build_neighbour_inputs,
    compute_forecast_discrepancy,
    joint_weights,
    build_prototype_bank,
)


# ─────────────────────────────────────────────────────────────────────────────
# Argument parsing  (superset of original args + our new flags)
# ─────────────────────────────────────────────────────────────────────────────

def str_to_bool(value):
    if isinstance(value, bool):
        return value
    if value.lower() in {'false', 'f', '0', 'no', 'n'}:
        return False
    elif value.lower() in {'true', 't', '1', 'yes', 'y'}:
        return True
    raise ValueError(f'{value} is not a valid boolean value')


parser = argparse.ArgumentParser()

# ── hardware / paths ──
parser.add_argument('--device',         type=str,          default='cuda:0')
parser.add_argument('--data',           type=str,          default='data/SOLAR')
parser.add_argument('--adj_data',       type=str,          default='data/sensor_graph/adj_mx.pkl')
parser.add_argument('--path_model_save',type=str,          default=None)
parser.add_argument('--expid',          type=int,          default=1)

# ── model architecture ──
parser.add_argument('--model_name',          type=str,          default='mtgnn')
parser.add_argument('--gcn_true',            type=str_to_bool,  default=True)
parser.add_argument('--buildA_true',         type=str_to_bool,  default=True)
parser.add_argument('--load_static_feature', type=str_to_bool,  default=False)
parser.add_argument('--cl',                  type=str_to_bool,  default=True)
parser.add_argument('--gcn_depth',           type=int,          default=2)
parser.add_argument('--num_nodes',           type=int,          default=207)
parser.add_argument('--dropout',             type=float,        default=0.3)
parser.add_argument('--subgraph_size',       type=int,          default=20)
parser.add_argument('--node_dim',            type=int,          default=40)
parser.add_argument('--dilation_exponential',type=int,          default=1)
parser.add_argument('--conv_channels',       type=int,          default=32)
parser.add_argument('--residual_channels',   type=int,          default=32)
parser.add_argument('--skip_channels',       type=int,          default=64)
parser.add_argument('--end_channels',        type=int,          default=128)
parser.add_argument('--in_dim',              type=int,          default=2)
parser.add_argument('--seq_in_len',          type=int,          default=12)
parser.add_argument('--seq_out_len',         type=int,          default=12)
parser.add_argument('--layers',              type=int,          default=3)
parser.add_argument('--propalpha',           type=float,        default=0.05)
parser.add_argument('--tanhalpha',           type=float,        default=3)
parser.add_argument('--num_split',           type=int,          default=1)

# ── training ──
parser.add_argument('--epochs',         type=int,          default=100)
parser.add_argument('--batch_size',     type=int,          default=64)
parser.add_argument('--learning_rate',  type=float,        default=0.001)
parser.add_argument('--weight_decay',   type=float,        default=0.0001)
parser.add_argument('--clip',           type=float,        default=5.0)
parser.add_argument('--step_size1',     type=int,          default=2500)
parser.add_argument('--step_size2',     type=int,          default=100)
parser.add_argument('--print_every',    type=int,          default=50)
parser.add_argument('--seed',           type=int,          default=101)
parser.add_argument('--runs',           type=int,          default=10)

# ── VSF experiment settings ──
parser.add_argument('--random_node_idx_split_runs',         type=int,         default=100)
parser.add_argument('--lower_limit_random_node_selections', type=int,         default=15)
parser.add_argument('--upper_limit_random_node_selections', type=int,         default=15)
parser.add_argument('--mask_remaining',        type=str_to_bool, default=False)
parser.add_argument('--predefined_S',          type=str_to_bool, default=False)
parser.add_argument('--predefined_S_frac',     type=int,         default=15)
parser.add_argument('--adj_identity_train_test',type=str_to_bool,default=False)
parser.add_argument('--do_full_set_oracle',    type=str_to_bool, default=False)
parser.add_argument('--full_set_oracle_lower_limit', type=int,   default=15)
parser.add_argument('--full_set_oracle_upper_limit', type=int,   default=15)

# ── original FDW flags (kept for baseline comparison) ──
parser.add_argument('--borrow_from_train_data', type=str_to_bool, default=False)
parser.add_argument('--num_neighbors_borrow',   type=int,          default=5)
parser.add_argument('--dist_exp_value',          type=float,        default=0.5)
parser.add_argument('--neighbor_temp',           type=float,        default=0.1)
parser.add_argument('--use_ewp',                 type=str_to_bool,  default=False)
parser.add_argument('--fraction_prots',          type=float,        default=1.0)

# ── NEW: our improved wrapper flags ──
parser.add_argument('--use_improved_wrapper', type=str_to_bool, default=False,
                    help='Enable all three improvements (cosine retrieval + '
                         'adaptive temp + joint weighting)')
parser.add_argument('--joint_alpha', type=float, default=0.3,
                    help='Mixing coefficient α for joint weighting: '
                         '0 = pure FDW, 1 = pure cosine DDW')

args = parser.parse_args()
torch.set_num_threads(3)


# ─────────────────────────────────────────────────────────────────────────────
# Training loop  (unchanged from original)
# ─────────────────────────────────────────────────────────────────────────────

def train_model(args, engine, dataloader, device):
    his_loss   = []
    val_time   = []
    train_time = []
    minl       = 1e5

    for i in range(1, args.epochs + 1):
        train_loss, train_rmse = [], []
        t1 = time.time()
        dataloader['train_loader'].shuffle()

        for iter, (x, y) in enumerate(dataloader['train_loader'].get_iterator()):
            trainx = torch.Tensor(x).to(device).transpose(1, 3)
            trainy = torch.Tensor(y).to(device).transpose(1, 3)
            if iter % args.step_size2 == 0:
                perm = np.random.permutation(range(args.num_nodes))
            num_sub = int(args.num_nodes / args.num_split)
            for j in range(args.num_split):
                id_ = perm[j * num_sub:(j + 1) * num_sub] if j != args.num_split - 1 \
                      else perm[j * num_sub:]
                id_ = torch.tensor(id_).to(device)
                tx, ty = trainx[:, :, id_, :], trainy[:, :, id_, :]
                metrics = engine.train(args, tx, ty[:, 0, :, :], i,
                                       dataloader['train_loader'].num_batch, iter, id_)
                train_loss.append(metrics[0])
                train_rmse.append(metrics[1])
            if iter % args.print_every == 0:
                print(f'Iter: {iter:03d}, Train Loss: {train_loss[-1]:.4f}, '
                      f'Train RMSE: {train_rmse[-1]:.4f}', flush=True)
        t2 = time.time()
        train_time.append(t2 - t1)

        valid_loss, valid_rmse = [], []
        s1 = time.time()
        for _, (x, y) in enumerate(dataloader['val_loader'].get_iterator()):
            testx = torch.Tensor(x).to(device).transpose(1, 3)
            testy = torch.Tensor(y).to(device).transpose(1, 3)
            m = engine.eval(args, testx, testy[:, 0, :, :])
            valid_loss.append(m[0])
            valid_rmse.append(m[1])
        s2 = time.time()
        val_time.append(s2 - s1)

        mvalid_loss = np.mean(valid_loss)
        his_loss.append(mvalid_loss)
        print(f'Epoch: {i:03d}, '
              f'Train Loss: {np.mean(train_loss):.4f}, '
              f'Train RMSE: {np.mean(train_rmse):.4f}, '
              f'Valid Loss: {mvalid_loss:.4f}, '
              f'Valid RMSE: {np.mean(valid_rmse):.4f}, '
              f'Time: {t2-t1:.2f}s', flush=True)

        if mvalid_loss < minl:
            save_path = (args.path_model_save +
                         f'exp{args.expid}_{args.runid}.pth')
            torch.save(engine.model.state_dict(), save_path)
            minl = mvalid_loss

    if args.epochs > 0:
        print(f'Avg train time: {np.mean(train_time):.4f}s/epoch')
        print(f'Avg val time:   {np.mean(val_time):.4f}s')
        bestid = np.argmin(his_loss)
        print(f'Best valid loss: {his_loss[bestid]:.4f}')


# ─────────────────────────────────────────────────────────────────────────────
# Inference helpers
# ─────────────────────────────────────────────────────────────────────────────

def _split_preds_by_neighbor(preds_flat, b_size, k):
    """(B*k, |S|, T) → (B, k, |S|, T)"""
    chunks = []
    for j in range(k):
        chunks.append(preds_flat[j * b_size:(j + 1) * b_size].unsqueeze(1))
    return torch.cat(chunks, dim=1)   # (B, k, |S|, T)


def run_improved_inference(args, engine, dataloader, instance_prototypes,
                           device, scaler):
    """
    Our improved VSF inference using all three improvements.
    Returns per-horizon MAE and RMSE lists (length = seq_out_len).
    """
    outputs = []
    realy   = torch.Tensor(dataloader['y_test']).to(device)
    realy   = realy.transpose(1, 3)[:, 0, :, :]

    idx_current_nodes = get_node_random_idx_split(
        args, args.num_nodes,
        args.lower_limit_random_node_selections,
        args.upper_limit_random_node_selections
    )
    realy = realy[:, idx_current_nodes, :]

    for _, (x, y) in enumerate(dataloader['test_loader'].get_iterator()):
        testx  = torch.Tensor(x).to(device).transpose(1, 3)  # (B, C, N, T)
        b_size = testx.shape[0]
        k      = args.num_neighbors_borrow

        # ── Improvement 1: cosine retrieval ──────────────────────────────────
        cos_dist, topk_idxs = cosine_retrieval(
            testx, instance_prototypes, idx_current_nodes, k, device
        )

        # fill missing variables from each neighbour
        testx_filled, orig_neighs = build_neighbour_inputs(
            testx, instance_prototypes, topk_idxs,
            idx_current_nodes, k, device
        )

        with torch.no_grad():
            # forward pass for neighbour-filled inputs  →  (B*k, C, N, T)
            preds_flat = engine.model(
                testx_filled, args=args,
                mask_remaining=False,
                test_idx_subset=idx_current_nodes
            )
            preds_flat = preds_flat.transpose(1, 3)[:, 0, :, :]   # (B*k, N, T)
            preds_flat = preds_flat[:, idx_current_nodes, :]       # (B*k, |S|, T)

            # reshape to (B, k, |S|, T)
            preds_split = _split_preds_by_neighbor(preds_flat, b_size, k)

            # forward pass for raw neighbour windows (needed for FDW discrepancy)
            orig_neighs_preds = engine.model(
                orig_neighs, args=args,
                mask_remaining=False,
                test_idx_subset=idx_current_nodes
            )
            orig_neighs_preds = orig_neighs_preds.transpose(1, 3)[:, 0, :, :]
            orig_neighs_preds = orig_neighs_preds[:, idx_current_nodes, :]
            orig_neighs_split = _split_preds_by_neighbor(
                orig_neighs_preds, b_size, k
            )

            # ── Improvement 2 is inside joint_weights (adaptive τ) ───────────
            # ── Improvement 3: joint weighting (cosine DDW + FDW) ────────────
            forecast_disc = compute_forecast_discrepancy(
                preds_split, orig_neighs_split, k, idx_current_nodes, device
            )                                                      # (B, k)

            w = joint_weights(cos_dist, forecast_disc,
                              alpha=args.joint_alpha)              # (B, k, 1, 1)

            # weighted aggregation  →  (B, |S|, T)
            final_pred = (w * preds_split).sum(dim=1)

        outputs.append(final_pred)

    yhat  = torch.cat(outputs, dim=0)[:realy.size(0)]

    mae_list, rmse_list = [], []
    for i in range(args.seq_out_len):
        pred   = scaler.inverse_transform(yhat[:, :, i])
        real   = realy[:, :, i]
        m      = metric(pred, real)
        mae_list.append(m[0])
        rmse_list.append(m[1])

    return mae_list, rmse_list, idx_current_nodes


# ─────────────────────────────────────────────────────────────────────────────
# Original inference  (UW / FDW / Partial / Oracle  – unchanged logic)
# ─────────────────────────────────────────────────────────────────────────────

def run_original_inference(args, engine, dataloader, instance_prototypes,
                           device, scaler):
    """Replicates the original train_multi_step.py inference block exactly."""
    from util import (obtain_relevant_data_from_prototypes,
                      obtain_discrepancy_from_neighs,
                      zero_out_remaining_input)

    outputs = []
    realy   = torch.Tensor(dataloader['y_test']).to(device)
    realy   = realy.transpose(1, 3)[:, 0, :, :]

    idx_current_nodes = get_node_random_idx_split(
        args, args.num_nodes,
        args.lower_limit_random_node_selections,
        args.upper_limit_random_node_selections
    )
    realy = realy[:, idx_current_nodes, :]

    for _, (x, y) in enumerate(dataloader['test_loader'].get_iterator()):
        testx = torch.Tensor(x).to(device).transpose(1, 3)

        if args.borrow_from_train_data:
            testx, dist_prot, orig_neighs, neighbs_idxs, _ = \
                obtain_relevant_data_from_prototypes(
                    args, testx, instance_prototypes, idx_current_nodes
                )
        else:
            testx = zero_out_remaining_input(testx, idx_current_nodes, args.device)

        with torch.no_grad():
            preds = engine.model(
                testx, args=args,
                mask_remaining=args.mask_remaining,
                test_idx_subset=idx_current_nodes
            )
            preds = preds.transpose(1, 3)[:, 0, :, :]
            preds = preds[:, idx_current_nodes, :]

            if args.borrow_from_train_data:
                b_size  = preds.shape[0] // args.num_neighbors_borrow
                k       = args.num_neighbors_borrow
                _chunks = [preds[j*b_size:(j+1)*b_size].unsqueeze(1)
                           for j in range(k)]
                preds   = torch.cat(_chunks, dim=1)   # (B, k, |S|, T)

                if args.use_ewp:
                    orig_neighs_fc = engine.model(
                        orig_neighs, args=args,
                        mask_remaining=args.mask_remaining,
                        test_idx_subset=idx_current_nodes
                    )
                    dist_prot, _ = obtain_discrepancy_from_neighs(
                        preds, orig_neighs_fc, args, idx_current_nodes
                    )
                    dist_prot = torch.nn.functional.softmax(
                        -dist_prot / args.neighbor_temp, dim=-1
                    ).view(b_size, k, 1, 1)
                else:
                    uniform = (torch.ones(k) / k).to(device)
                    dist_prot = uniform.unsqueeze(0).repeat(b_size, 1).view(
                        b_size, k, 1, 1
                    )

                preds = (dist_prot * preds).sum(dim=1)

        outputs.append(preds)

    yhat  = torch.cat(outputs, dim=0)[:realy.size(0)]

    mae_list, rmse_list = [], []
    for i in range(args.seq_out_len):
        pred = scaler.inverse_transform(yhat[:, :, i])
        real = realy[:, :, i]
        m    = metric(pred, real)
        mae_list.append(m[0])
        rmse_list.append(m[1])

    return mae_list, rmse_list, idx_current_nodes


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main(runid):
    device       = torch.device(args.device)
    dataloader   = load_dataset(args, args.data, args.batch_size,
                                args.batch_size, args.batch_size)
    scaler       = dataloader['scaler']
    args.num_nodes = dataloader['train_loader'].num_nodes
    print(f'Number of variables/nodes = {args.num_nodes}')

    dataset_name = args.data.strip().split('/')[-1].strip()
    args.in_dim  = 2 if dataset_name == 'METR-LA' else 1
    args.runid   = runid

    predefined_A = None
    if dataset_name == 'METR-LA':
        args.adj_data = 'data/sensor_graph/adj_mx.pkl'
        if os.path.exists(args.adj_data):
            predefined_A  = load_adj(args.adj_data)
            predefined_A  = (torch.tensor(predefined_A) -
                             torch.eye(dataloader['total_num_nodes'])).to(device)
        else:
            print(f'[WARN] adj_mx.pkl not found, using adaptive graph (buildA_true) only.')

    if args.adj_identity_train_test and predefined_A is not None:
        predefined_A = torch.eye(predefined_A.shape[0]).to(device)

    args.path_model_save = f'./saved_models/{args.model_name}/{dataset_name}/'
    os.makedirs(args.path_model_save, exist_ok=True)

    model = gtnet(
        args.gcn_true, args.buildA_true, args.gcn_depth, args.num_nodes,
        device, predefined_A=predefined_A,
        dropout=args.dropout, subgraph_size=args.subgraph_size,
        node_dim=args.node_dim,
        dilation_exponential=args.dilation_exponential,
        conv_channels=args.conv_channels,
        residual_channels=args.residual_channels,
        skip_channels=args.skip_channels,
        end_channels=args.end_channels,
        seq_length=args.seq_in_len, in_dim=args.in_dim,
        out_dim=args.seq_out_len,
        layers=args.layers, propalpha=args.propalpha,
        tanhalpha=args.tanhalpha, layer_norm_affline=True
    )
    print(f'Receptive field: {model.receptive_field}')
    print(f'Parameters: {sum(p.nelement() for p in model.parameters())}')

    engine = Trainer(args, model, args.model_name,
                     args.learning_rate, args.weight_decay,
                     args.clip, args.step_size1,
                     args.seq_out_len, scaler, device, args.cl)

    # ── Training phase ────────────────────────────────────────────────────────
    if args.epochs > 0:
        train_model(args, engine, dataloader, device)

    # ── Load best checkpoint ──────────────────────────────────────────────────
    ckpt = args.path_model_save + f'exp{args.expid}_{runid}.pth'
    engine.model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
    engine.model.eval()
    print(f'\nModel loaded from {ckpt}\n')

    # ── Build prototype bank (needed for retrieval-based methods) ─────────────
    instance_prototypes = None
    if args.use_improved_wrapper or args.borrow_from_train_data:
        num_prots = math.floor(args.fraction_prots *
                               dataloader['x_train'].shape[0])
        args.num_prots = num_prots
        print(f'Number of prototypes = {num_prots}')

        if args.use_improved_wrapper:
            # Our bank: stores (P, C, N, T) already transposed
            instance_prototypes = build_prototype_bank(
                dataloader['x_train'], num_prots, device
            )
        else:
            # Original bank: (P, T, N, C) — obtain_relevant_data_from_prototypes
            # expects the raw format with its own transpose inside
            instance_prototypes = obtain_instance_prototypes(
                args, dataloader['x_train']
            )

    # ── Inference phase ───────────────────────────────────────────────────────
    all_mae, all_rmse = [], []

    for split_run in range(args.random_node_idx_split_runs):
        print(f'\n--- Subset split run {split_run} ---')

        if args.use_improved_wrapper:
            mae, rmse, idx_nodes = run_improved_inference(
                args, engine, dataloader, instance_prototypes, device, scaler
            )
        else:
            mae, rmse, idx_nodes = run_original_inference(
                args, engine, dataloader, instance_prototypes, device, scaler
            )

        print(f'  Subset size: {len(idx_nodes)} / {args.num_nodes} '
              f'({100*len(idx_nodes)/args.num_nodes:.1f}%)')
        for h in range(args.seq_out_len):
            print(f'  Horizon {h+1:2d}: MAE={mae[h]:.4f}  RMSE={rmse[h]:.4f}')

        all_mae.append(mae)
        all_rmse.append(rmse)

    return all_mae, all_rmse


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    all_mae, all_rmse = [], []
    for i in range(args.runs):
        m1, m2 = main(i)
        all_mae.extend(m1)
        all_rmse.extend(m2)

    all_mae  = np.array(all_mae)
    all_rmse = np.array(all_rmse)
    amae     = np.mean(all_mae,  0)
    armse    = np.mean(all_rmse, 0)
    smae     = np.std(all_mae,   0)
    srmse    = np.std(all_rmse,  0)

    method = ('Improved (cosine+adaptive-T+joint)' if args.use_improved_wrapper
              else ('FDW' if (args.borrow_from_train_data and args.use_ewp)
              else ('UW'  if  args.borrow_from_train_data
              else  'Partial')))

    sep = '=' * 60
    print(sep)
    print('Method : ' + method)
    print('Dataset: ' + args.data + '  Subset: ' + str(args.lower_limit_random_node_selections) + '%')
    print(sep)
    for i in range(args.seq_out_len):
        print('Horizon %2d | MAE = %.4f +/- %.4f | RMSE = %.4f +/- %.4f' %
              (i + 1, amae[i], smae[i], armse[i], srmse[i]))
