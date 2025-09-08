from __future__ import absolute_import, division, print_function

import argparse
import random
import torch
import numpy as np
import wandb
import torch.nn as nn
from torch.optim import SGD
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from torch.nn import MSELoss
from pytorch_transformers import WarmupLinearSchedule
from pytorch_transformers.modeling_roberta import RobertaConfig
from networks.SentiLARE import RobertaForSequenceClassification
from utils.databuilder import set_up_data_loader, random_sampling
from utils.set_seed import set_random_seed, seed
from utils.metric import score_model
from config.global_configs import DEVICE
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, mean_squared_error, r2_score
from networks.subnet.HCL_Module import HCL_Total

def parser_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, choices=["mosi", "mosei"], default="mosi")
    parser.add_argument("--data_path", type=str, default='./dataset/MOSI_16_sentilare_unaligned_data.pkl')
    parser.add_argument("--max_seq_length", type=int, default=50)
    parser.add_argument("--train_batch_size", type=int, default=64)
    parser.add_argument("--batch_size", type=int, default=64, help="训练批次大小")
    parser.add_argument("--dev_batch_size", type=int, default=128)
    parser.add_argument("--test_batch_size", type=int, default=128)
    parser.add_argument("--n_epochs", type=int, default=500)
    parser.add_argument("--beta_shift", type=float, default=1.0)
    parser.add_argument("--dropout_prob", type=float, default=0.5)
    parser.add_argument("--model", type=str, choices=["bert-base-uncased", "xlnet-base-cased", "roberta-base"], default="roberta-base")
    parser.add_argument("--model_name_or_path", type=str, default='./pretrained_model/sentilare_model/sentilare_model', help="Path to pre-trained model or shortcut name")
    parser.add_argument("--learning_rate", type=float, default=6e-5)
    parser.add_argument("--weight_decay", type=float, default=0)
    parser.add_argument("--gradient_accumulation_step", type=int, default=1)
    parser.add_argument("--test_step", type=int, default=20)
    parser.add_argument("--max_grad_norm", type=int, default=2)
    parser.add_argument("--warmup_proportion", type=float, default=0.4)
    parser.add_argument("--seed", type=seed, default=6758, help="integer or 'random'")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--active_learning_interval", type=int, default=30, help="Interval (in epochs) for applying active learning")
    parser.add_argument("--num_active_learning_samples", type=int, default=20, help="Number of samples to select during each active learning step")
    parser.add_argument("--al_alpha_init", type=float, default=1.0, help="Initial curriculum factor alpha_init")
    parser.add_argument("--al_alpha_decay", type=float, default=0.1, help="Decay per AL round: alpha_i = alpha_init - (i-1)*decay")
    parser.add_argument("--al_grad_param_hint", type=str, default="classifier")
    
    return parser.parse_args()

def _select_head_params(model: nn.Module, hint: str = "classifier", fallback_top_k: int = 2):
    head_params = []
    for n, p in model.named_parameters():
        if p.requires_grad and (hint in n or "regressor" in n or "out_proj" in n):
            head_params.append((n, p))
    if len(head_params) == 0:
        # fallback: pick the last K trainable parameter tensors
        trainables = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
        head_params = trainables[-fallback_top_k:] if len(trainables) >= fallback_top_k else trainables
    # return only tensors in stable order
    return [p for (_, p) in head_params]

def score_model(preds, labels):
    # Basic metrics
    mae = mean_absolute_error(labels, preds)
    corr = np.corrcoef(preds, labels)[0, 1]
    
    # Convert preds and labels to binary for classification metrics
    binary_preds = [1 if p > 0 else 0 for p in preds]
    binary_labels = [1 if l > 0 else 0 for l in labels]
    
    # Accuracy and F1 score for classes Has0 and Non0
    has0_acc_2 = accuracy_score(binary_labels, binary_preds)  # Adjust this if needed
    has0_f1_score = f1_score(binary_labels, binary_preds)
    
    non0_acc_2 = accuracy_score(binary_labels, binary_preds)  # Adjust this if needed
    non0_f1_score = f1_score(binary_labels, binary_preds)
    
    return has0_acc_2, has0_f1_score, non0_acc_2, non0_f1_score, mae, corr

def prep_for_training(args, num_train_optimization_steps: int):
    config = RobertaConfig.from_pretrained(args.model_name_or_path, num_labels=1, finetuning_task='sst')
    model = RobertaForSequenceClassification.from_pretrained(
        args.model_name_or_path, config=config, pos_tag_embedding=True, senti_embedding=True, polarity_embedding=True)
    model.to(DEVICE)
    
    param_optimizer = list(model.named_parameters())
    no_decay = ["bias", "LayerNorm.weight"]
    AL_ARCF_params = ['AL_ARCF']
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in param_optimizer if not any(nd in n for nd in no_decay) and not any(nd in n for nd in AL_ARCF_params)],
            "weight_decay": args.weight_decay,
        },
        {"params": model.roberta.encoder.AL_ARCF.parameters(), 'lr': args.learning_rate, "weight_decay": args.weight_decay},
        {
            "params": [p for n, p in param_optimizer if any(nd in n for nd in no_decay) and not any(nd in n for nd in AL_ARCF_params)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = SGD(optimizer_grouped_parameters, lr=args.learning_rate, momentum=0.9)
    scheduler = WarmupLinearSchedule(
        optimizer,
        warmup_steps=args.warmup_proportion * num_train_optimization_steps,
        t_total=num_train_optimization_steps,
    )
    return model, optimizer, scheduler

def train_epoch(args, model: nn.Module, train_dataloader: DataLoader, optimizer, scheduler):
    model.train()
    preds = []
    labels = []
    tr_loss = 0

    nb_tr_steps = 0
    for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
        batch = tuple(t.to(DEVICE) for t in batch)
        input_ids, visual_ids, acoustic_ids, pos_ids, senti_ids, polarity_ids, visual, acoustic, input_mask, segment_ids, label_ids = batch
        visual = torch.squeeze(visual, 1)

        outputs = model(
            input_ids,
            visual,
            acoustic,
            visual_ids,
            acoustic_ids,
            pos_ids, senti_ids, polarity_ids,
            attention_mask=input_mask,
            token_type_ids=segment_ids,
        )
        logits = outputs[0]
        hcl_loss = HCL_Total(visual,
            acoustic,
            visual_ids,
            acoustic_ids)
        try:
            # Ensure logits are a tensor
            if not isinstance(logits, torch.Tensor):
                raise TypeError(f"Expected logits to be a tensor, but got {type(logits)}")
            loss_fct = MSELoss()
            loss = loss_fct(logits.view(-1), label_ids.view(-1))
            if args.gradient_accumulation_step > 1:
                loss = loss / args.gradient_accumulation_step
                hcl_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            tr_loss += loss.item()
            nb_tr_steps += 1
            if (step + 1) % args.gradient_accumulation_step == 0:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
            
            # Convert logits to numpy and ensure they are tensors
            logits = logits.detach().cpu().numpy()
            logits = np.squeeze(logits).tolist()
            label_ids = label_ids.detach().cpu().numpy()
            label_ids = np.squeeze(label_ids).tolist()
            preds.extend(logits)
            labels.extend(label_ids)
        
        except TypeError as e:
            print(f"Skipping step {step + 1} due to error: {e}")
            continue  # Skip to the next step if an error occurs

    preds = np.array(preds)
    labels = np.array(labels)

    return tr_loss / nb_tr_steps, preds, labels



def evaluate_epoch(args, model: nn.Module, dataloader: DataLoader):
    model.eval()
    preds = []
    labels = []

    loss = 0
    nb_steps = 0
    
    with torch.no_grad():
        for step, batch in enumerate(dataloader):
            batch = tuple(t.to(DEVICE) for t in batch)
            input_ids, visual_ids, acoustic_ids, pos_ids, senti_ids, polarity_ids, visual, acoustic, input_mask, segment_ids, label_ids = batch
            visual = torch.squeeze(visual, 1)
            
            outputs = model(
                input_ids,
                visual,
                acoustic,
                visual_ids,
                acoustic_ids,
                pos_ids, senti_ids, polarity_ids,
                token_type_ids=segment_ids,
                attention_mask=input_mask,
                labels=None,
            )
            logits = outputs[0]
            loss_fct = MSELoss()
            loss = loss_fct(logits.view(-1), label_ids.view(-1))
            if args.gradient_accumulation_step > 1:
                loss = loss / args.gradient_accumulation_step
            loss += loss.item()
            nb_steps += 1
            
            logits = logits.detach().cpu().numpy()
            label_ids = label_ids.detach().cpu().numpy()
            
            logits = np.squeeze(logits).tolist()
            label_ids = np.squeeze(label_ids).tolist()
            
            preds.extend(logits)
            labels.extend(label_ids)
        
        preds = np.array(preds)
        labels = np.array(labels)

        # Compute additional metrics
        has0_acc_2, has0_f1_score, non0_acc_2, non0_f1_score, mae, corr = score_model(preds, labels)
        
    return loss / nb_steps, preds, labels, has0_acc_2, has0_f1_score, non0_acc_2, non0_f1_score, mae, corr


class ActiveLearningDataset(Dataset):
    def __init__(self, dataset, sampled_data):
        self.dataset = dataset
        self.sampled_data = sampled_data

    def __len__(self):
        return len(self.dataset) + len(self.sampled_data)

    def __getitem__(self, idx):
        if idx < len(self.dataset):
            return self.dataset[idx]
        else:
            return self.sampled_data[idx - len(self.dataset)]

def active_learning_step(args, model, unlabeled_data_loader, num_active_learning_samples, al_round_idx: int):
    if not isinstance(unlabeled_data_loader, DataLoader):
        raise TypeError("unlabeled_data_loader should be an instance of DataLoader")

    device = DEVICE
    model.eval()

    head_params = _select_head_params(model, hint=getattr(args, "al_grad_param_hint", "classifier"))
    if len(head_params) == 0:
        raise RuntimeError("No trainable parameters found for gradient embeddings.")

    def _flatten_grads(grads):
        vecs = []
        for g in grads:
            if g is None:
                continue
            vecs.append(g.reshape(-1))
        if len(vecs) == 0:
            return torch.zeros(1, device=device)
        return torch.cat(vecs, dim=0)

    alpha = max(0.0, args.al_alpha_init - (al_round_idx - 1) * args.al_alpha_decay)

    abundance_scores = []
    availability_scores = []
    sample_indices = []

    with torch.no_grad():
        pass

    model.zero_grad(set_to_none=True)
    model.train()

    global_idx_offset = 0
    for batch in tqdm(unlabeled_data_loader, desc="SJ_ALSP scoring"):
        batch = tuple(t.to(device) for t in batch)
        input_ids, visual_ids, acoustic_ids, pos_ids, senti_ids, polarity_ids, visual, acoustic, input_mask, segment_ids, _ = batch
        visual = torch.squeeze(visual, 1)

        B = input_ids.shape[0]
        for b in range(B):
            ii    = input_ids[b:b+1]
            vi_id = visual_ids[b:b+1]
            ai_id = acoustic_ids[b:b+1]
            pi    = pos_ids[b:b+1]
            si    = senti_ids[b:b+1]
            poli  = polarity_ids[b:b+1]
            v     = visual[b:b+1]
            a     = acoustic[b:b+1]
            msk   = input_mask[b:b+1]
            seg   = segment_ids[b:b+1]

            for p in head_params:
                if p.grad is not None: p.grad = None
            outputs_f = model(ii, v, a, vi_id, ai_id, pi, si, poli, attention_mask=msk, token_type_ids=seg)
            logit_f = outputs_f[0].view(-1)

            gf = torch.autograd.grad(logit_f.sum(), head_params, retain_graph=True, allow_unused=True)
            gf_vec = _flatten_grads(gf).detach()

            grad_vectors = {}
            v0 = torch.zeros_like(v)
            a0 = torch.zeros_like(a)
            for p in head_params:
                if p.grad is not None: p.grad = None
            out_t = model(ii, v0, a0, vi_id, ai_id, pi, si, poli, attention_mask=msk, token_type_ids=seg)
            gt = torch.autograd.grad(out_t[0].view(-1).sum(), head_params, retain_graph=True, allow_unused=True)
            grad_vectors["t"] = _flatten_grads(gt).detach()

            for p in head_params:
                if p.grad is not None: p.grad = None
            out_v = model(ii, v, a0, vi_id, ai_id, pi, si, poli, attention_mask=msk, token_type_ids=seg)
            gv = torch.autograd.grad(out_v[0].view(-1).sum(), head_params, retain_graph=True, allow_unused=True)
            grad_vectors["v"] = _flatten_grads(gv).detach()

            for p in head_params:
                if p.grad is not None: p.grad = None
            out_a = model(ii, v0, a, vi_id, ai_id, pi, si, poli, attention_mask=msk, token_type_ids=seg)
            ga = torch.autograd.grad(out_a[0].view(-1).sum(), head_params, retain_graph=True, allow_unused=True)
            grad_vectors["a"] = _flatten_grads(ga).detach()

            c1 = 0.0
            c2 = 0.0
            eps = 1e-12
            gf_norm = torch.linalg.vector_norm(gf_vec) + eps

            for key in ["t", "v", "a"]:
                gm = grad_vectors[key]
                c1 += torch.linalg.vector_norm(gm + gf_vec).item()
                gm_norm = torch.linalg.vector_norm(gm) + eps
                c2 += torch.dot(gm, gf_vec).item() / (gm_norm.item() * gf_norm.item())

            abundance_scores.append(c1)
            availability_scores.append(c2)
            sample_indices.append(global_idx_offset + b)

        global_idx_offset += B

    import numpy as np
    abundance_scores = np.array(abundance_scores, dtype=np.float64)
    availability_scores = np.array(availability_scores, dtype=np.float64)

    def _minmax(x):
        xmin, xmax = float(np.min(x)), float(np.max(x))
        if xmax - xmin < 1e-12:
            return np.zeros_like(x)
        return (x - xmin) / (xmax - xmin)

    c1n = _minmax(abundance_scores)
    c2n = _minmax(availability_scores)
    final_scores = c1n + alpha * c2n

    top_idx = np.argsort(final_scores)[-num_active_learning_samples:][::-1]
    chosen_global = [sample_indices[i] for i in top_idx]

    sampled_data = [unlabeled_data_loader.dataset[i] for i in chosen_global]
    print(f"SJ_ALSP step (round {al_round_idx}): selected {len(sampled_data)} samples. alpha={alpha:.4f}")

    model.eval()
    return sampled_data


def main():
    args = parser_args()
    set_random_seed(args.seed)
    wandb.init(project="sentilare", config=args)

    train_data_loader, dev_data_loader, test_data_loader, unlabeled_data_loader = set_up_data_loader(args)
    num_train_optimization_steps = (len(train_data_loader) // args.gradient_accumulation_step) * args.n_epochs
    model, optimizer, scheduler = prep_for_training(args, num_train_optimization_steps)

    best_dev_score = -1e10
    initial_train_size = len(train_data_loader.dataset)
    print(f"Initial training dataset size: {initial_train_size}")

    al_round_idx = 1
    for epoch in range(args.n_epochs):
        if epoch + 1 in (161, 162, 163):
            print("Skipping epoch 161 and the associated active learning step.")
            continue

        print(f"Epoch {epoch + 1}/{args.n_epochs}")
        train_loss, train_preds, train_labels = train_epoch(args, model, train_data_loader, optimizer, scheduler)

        dev_loss, dev_preds, dev_labels, dev_has0_acc_2, dev_has0_f1_score, dev_non0_acc_2, dev_non0_f1_score, dev_mae, dev_corr = evaluate_epoch(args, model, dev_data_loader)
        if dev_has0_f1_score > best_dev_score:
            best_dev_score = dev_has0_f1_score
            print(f"New best model saved with dev Has0_F1_score: {best_dev_score}")

        test_loss, test_preds, test_labels, test_has0_acc_2, test_has0_f1_score, test_non0_acc_2, test_non0_f1_score, test_mae, test_corr = evaluate_epoch(args, model, test_data_loader)

        if (epoch + 1) % args.active_learning_interval == 0:
            print(f"Performing SJ_ALSP active learning at epoch {epoch + 1}")
            sampled_data = active_learning_step(
                args, model, unlabeled_data_loader, args.num_active_learning_samples, al_round_idx
            )
            updated_train_dataset = ActiveLearningDataset(train_data_loader.dataset, sampled_data)
            train_data_loader = DataLoader(updated_train_dataset, batch_size=args.train_batch_size, shuffle=True)

            updated_train_size = len(updated_train_dataset)
            print(f"Updated training dataset with {len(sampled_data)} new samples.")
            print(f"Total training dataset size after update: {updated_train_size}")

            al_round_idx += 1
    
    test_loss, test_preds, test_labels, test_has0_acc_2, test_has0_f1_score, test_non0_acc_2, test_non0_f1_score, test_mae, test_corr = evaluate_epoch(args, model, test_data_loader)
    print(f"Final test loss: {test_loss}")
    print(f"Final Test Has0_acc_2: {test_has0_acc_2}")
    print(f"Final Test Has0_F1_score: {test_has0_f1_score}")
    print(f"Final Test Non0_acc_2: {test_non0_acc_2}")
    print(f"Final Test Non0_F1_score: {test_non0_f1_score}")
    print(f"Final Test MAE: {test_mae}")
    print(f"Final Test Corr: {test_corr}")

if __name__ == "__main__":
    main()

