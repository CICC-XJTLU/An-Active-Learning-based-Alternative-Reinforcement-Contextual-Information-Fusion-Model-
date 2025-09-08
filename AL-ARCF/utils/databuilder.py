import pickle
import torch
import numpy as np
import random
from torch.utils.data import DataLoader, TensorDataset
from pytorch_transformers import RobertaTokenizer

class InputFeatures(object):
    def __init__(self, input_ids, visual_ids, acoustic_ids, pos_ids, senti_ids, polarity_ids, visual, acoustic, input_mask, segment_ids, label_id):
        self.input_ids = input_ids
        self.visual_ids = visual_ids
        self.acoustic_ids = acoustic_ids
        self.pos_ids = pos_ids
        self.senti_ids = senti_ids
        self.polarity_ids = polarity_ids
        self.visual = visual
        self.acoustic = acoustic
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id

class MultimodalConfig(object):
    def __init__(self, beta_shift, dropout_prob):
        self.beta_shift = beta_shift
        self.dropout_prob = dropout_prob

def convert_to_features(args, examples, max_seq_length, tokenizer):
    features = []

    for (ex_index, example) in enumerate(examples):
        (words, visual, acoustic, pos_ids, senti_ids, visual_ids, acoustic_ids), label_id, segment = example
        
        tokens, inversions = [], []
        for idx, word in enumerate(words):
            tokenized = tokenizer.tokenize(word)
            tokens.extend(tokenized)
            inversions.extend([idx] * len(tokenized))

        assert len(tokens) == len(inversions)

        aligned_pos_ids = []
        aligned_senti_ids = []

        for inv_idx in inversions:
            aligned_pos_ids.append(pos_ids[inv_idx])
            aligned_senti_ids.append(senti_ids[inv_idx])

        visual = np.array(visual)
        visual_ids = np.array(visual_ids)
        acoustic = np.array(acoustic)
        acoustic_ids = np.array(acoustic_ids)
        pos_ids = aligned_pos_ids
        senti_ids = aligned_senti_ids

        if len(tokens) > max_seq_length - 3:
            tokens = tokens[: max_seq_length - 3]
            words = words[: max_seq_length - 3]
            pos_ids = pos_ids[: max_seq_length - 3]
            senti_ids = senti_ids[: max_seq_length - 3]

        input_ids, visual_ids, acoustic_ids, pos_ids, senti_ids, polarity_ids, visual, acoustic, input_mask, segment_ids = prepare_sentilare_input(
            args, tokens, visual_ids, acoustic_ids, pos_ids, senti_ids, visual, acoustic, tokenizer
        )
        assert len(input_ids) == args.max_seq_length
        assert len(input_mask) == args.max_seq_length
        assert len(segment_ids) == args.max_seq_length

        features.append(
            InputFeatures(
                input_ids=input_ids,
                visual_ids=visual_ids,
                acoustic_ids=acoustic_ids,
                pos_ids=pos_ids,
                senti_ids=senti_ids,
                polarity_ids=polarity_ids,
                input_mask=input_mask,
                segment_ids=segment_ids,
                visual=visual,
                acoustic=acoustic,
                label_id=label_id,
            )
        )
    return features

def prepare_sentilare_input(args, tokens, visual_ids, acoustic_ids, pos_ids, senti_ids, visual, acoustic, tokenizer):
    CLS = tokenizer.cls_token
    SEP = tokenizer.sep_token
    tokens = [CLS] + tokens + [SEP] + [SEP]
    pos_ids = [4] + pos_ids + [4] + [4]
    senti_ids = [2] + senti_ids + [2] + [2]
    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    segment_ids = [0] * len(input_ids)
    input_mask = [1] * len(input_ids)

    pad_length = args.max_seq_length - len(input_ids)
    padding = [0] * pad_length

    input_ids += padding
    pos_ids += [4] * pad_length
    senti_ids += [2] * pad_length
    polarity_ids = [5] * len(input_ids)
    input_mask += padding
    segment_ids += padding

    return input_ids, visual_ids, acoustic_ids, pos_ids, senti_ids, polarity_ids, visual, acoustic, input_mask, segment_ids

def get_tokenizer(args):
    return RobertaTokenizer.from_pretrained(args.model_name_or_path, do_lower_case=False)

def get_appropriate_dataset(args, data):
    tokenizer = get_tokenizer(args)
    features = convert_to_features(args, data, args.max_seq_length, tokenizer)

    # 将列表转换为 numpy 数组
    all_input_ids = np.array([f.input_ids for f in features], dtype=np.int64)
    all_visual_ids = np.array([f.visual_ids for f in features], dtype=np.int64)
    all_acoustic_ids = np.array([f.acoustic_ids for f in features], dtype=np.int64)
    all_pos_ids = np.array([f.pos_ids for f in features], dtype=np.int64)
    all_senti_ids = np.array([f.senti_ids for f in features], dtype=np.int64)
    all_polarity_ids = np.array([f.polarity_ids for f in features], dtype=np.int64)
    all_input_mask = np.array([f.input_mask for f in features], dtype=np.int64)
    all_segment_ids = np.array([f.segment_ids for f in features], dtype=np.int64)
    all_visual = np.array([f.visual for f in features], dtype=np.float32)
    all_acoustic = np.array([f.acoustic for f in features], dtype=np.float32)
    all_label_ids = np.array([f.label_id for f in features], dtype=np.float32)

    # 将 numpy 数组转换为 PyTorch 张量
    dataset = TensorDataset(
        torch.tensor(all_input_ids),
        torch.tensor(all_visual_ids),
        torch.tensor(all_acoustic_ids),
        torch.tensor(all_pos_ids),
        torch.tensor(all_senti_ids),
        torch.tensor(all_polarity_ids),
        torch.tensor(all_visual),
        torch.tensor(all_acoustic),
        torch.tensor(all_input_mask),
        torch.tensor(all_segment_ids),
        torch.tensor(all_label_ids),
    )
    return dataset

def set_up_data_loader(args):
    with open(args.data_path, "rb") as handle:
        data = pickle.load(handle)

    num_samples = int(len(data["train"]) * 0.1)  # 使用10%的数据进行主动学习
    sampled_train_data = random_sampling(data["train"], num_samples)
    dev_data = data["dev"]
    test_data = data["test"]

    train_dataset = get_appropriate_dataset(args, sampled_train_data)
    dev_dataset = get_appropriate_dataset(args, dev_data)
    test_dataset = get_appropriate_dataset(args, test_data)

    # 创建 unlabeled_data_loader，用于主动学习
    unlabeled_dataset = get_appropriate_dataset(args, data["train"])  # 使用完整的训练数据集作为未标注数据
    unlabeled_data_loader = DataLoader(
        unlabeled_dataset, batch_size=args.train_batch_size, shuffle=False, drop_last=False
    )

    num_train_optimization_steps = (
        int(
            len(train_dataset) / args.train_batch_size /
            args.gradient_accumulation_step
        )
        * args.n_epochs
    )

    train_dataloader = DataLoader(
        train_dataset, batch_size=args.train_batch_size, shuffle=True, drop_last=True
    )

    dev_dataloader = DataLoader(
        dev_dataset, batch_size=args.dev_batch_size, shuffle=False
    )

    test_dataloader = DataLoader(
        test_dataset, batch_size=args.test_batch_size, shuffle=False,
    )

    return (
        train_dataloader,
        dev_dataloader,
        test_dataloader,
        unlabeled_data_loader,  # 返回 unlabeled_data_loader
    )

def random_sampling(data, num_samples):
    """
    从数据集中随机选择样本。
    :param data: 原始数据
    :param num_samples: 需要选择的样本数量
    :return: 随机选择的样本
    """
    return random.sample(data, num_samples)
