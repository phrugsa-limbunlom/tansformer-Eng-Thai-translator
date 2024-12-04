import os

import pandas as pd
import torch
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoTokenizer

from Tokenizer import Tokenizer
from model.Transformer import Transformer

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['TORCH_USE_CUDA_DSA'] = '1'

if __name__ == "__main__":
    # Preprocessing
    file_path = "dataset/small_task_master_1.csv"

    df = pd.read_csv(file_path)

    # tokenizer from pretrained models
    en_tokenizer = AutoTokenizer.from_pretrained('bert-base-cased')
    th_tokenizer = AutoTokenizer.from_pretrained("airesearch/wangchanberta-base-att-spm-uncased")

    """
        use tokenizer from pretrained models >> Tokenizer(en_tokenizer, th_tokenizer)
        by default, use custom tokenizer
    """
    src, trg, src_vocab_size, trg_vocab_size = Tokenizer().preprocessing(df)

    print(f"Source Dataset before Tokenization: {df["en_text"][0]}")

    print(f"Source Dataset after Tokenization: {src[0]}")

    print(f"Target Dataset before Tokenization: {df["th_text"][0]}")

    print(f"Target Dataset after Tokenization: {trg[0]}")

    print(f"EN Vocabulary Size: {src_vocab_size}")

    print(f"TH Vocabulary Size: {trg_vocab_size}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    src = [torch.tensor(arr, dtype=torch.long, device=device) for arr in src]
    trg = [torch.tensor(arr, dtype=torch.long, device=device) for arr in trg]

    src_padded_tensor = pad_sequence(src, batch_first=True, padding_value=0)
    trg_padded_tensor = pad_sequence(trg, batch_first=True, padding_value=0)

    src = torch.tensor(src_padded_tensor, dtype=torch.long, device=device)
    trg = torch.tensor(trg_padded_tensor, dtype=torch.long, device=device)

    # Training

    src_pad_idx = 0
    trg_pad_idx = 0

    model = Transformer(src_vocab_size, trg_vocab_size, src_pad_idx, trg_pad_idx).to("cuda")

    out = model(src, trg[:, :-1])
    print(out.shape)

    # criterion = nn.CrossEntropyLoss(ignore_index=-100)
    # optimizer = optim.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)