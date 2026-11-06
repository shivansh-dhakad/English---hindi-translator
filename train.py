import math
import warnings
from pathlib import Path

import torch
import torch.nn as nn
from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import WordLevelTrainer
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from config import get_config, get_weights_file_path, latest_weights_file_path
from dataset import TranslationDataset, casual_mask
from model import BuildTransformer


def get_all_sentences(ds, lang):
    for item in ds:
        yield item['translation'][lang]


def get_or_build_tokenizer(config, ds, lang):
    tokenizer_path = Path(config['tokenizer_file'].format(lang))
    if not tokenizer_path.exists():
        tokenizer_path.parent.mkdir(parents=True, exist_ok=True)
        tokenizer = Tokenizer(WordLevel(unk_token='[UNK]'))
        tokenizer.pre_tokenizer = Whitespace()
        trainer = WordLevelTrainer(special_tokens=['[UNK]', '[PAD]', '[SOS]', '[EOS]'], min_frequency=2)
        tokenizer.train_from_iterator(get_all_sentences(ds, lang), trainer=trainer)
        tokenizer.save(str(tokenizer_path))
        return tokenizer

    return Tokenizer.from_file(str(tokenizer_path))


def normalize_dataset(ds, lang_src, lang_tgt):
    return ds.map(
        lambda row: {'translation': {lang_src: row[lang_src], lang_tgt: row[lang_tgt]}},
        remove_columns=ds.column_names,
    )


def get_ds(config):
    src, tgt = config['lang_src'], config['lang_tgt']
    ds_raw = load_dataset('csv', data_files=config['data_file'], split='train')
    original_size = len(ds_raw)
    ds_raw = ds_raw.filter(
        lambda row: isinstance(row[src], str) and bool(row[src].strip())
        and isinstance(row[tgt], str) and bool(row[tgt].strip())
    )
    removed_rows = original_size - len(ds_raw)
    if removed_rows:
        print(f'Removed {removed_rows} rows with missing or empty translations')
    ds_raw = normalize_dataset(ds_raw, src, tgt)

    tokenizer_src = get_or_build_tokenizer(config, ds_raw, src)
    tokenizer_tgt = get_or_build_tokenizer(config, ds_raw, tgt)

    # drop sentence pairs that would not fit in seq_len (instead of crashing mid-training)
    src_texts = [x[src] for x in ds_raw['translation']]
    tgt_texts = [x[tgt] for x in ds_raw['translation']]
    src_lens = [len(e.ids) for e in tokenizer_src.encode_batch(src_texts)]
    tgt_lens = [len(e.ids) for e in tokenizer_tgt.encode_batch(tgt_texts)]
    keep = [i for i, (a, b) in enumerate(zip(src_lens, tgt_lens))
            if a <= config['seq_len'] - 2 and b <= config['seq_len'] - 2]
    print(f'Kept {len(keep)}/{len(ds_raw)} pairs with <= {config["seq_len"] - 2} tokens '
          f'(max src {max(src_lens)}, max tgt {max(tgt_lens)})')
    ds_raw = ds_raw.select(keep)

    train_ds_size = int(0.9 * len(ds_raw))
    val_ds_size = len(ds_raw) - train_ds_size
    train_ds_raw, val_ds_raw = random_split(
        ds_raw, [train_ds_size, val_ds_size], generator=torch.Generator().manual_seed(config['seed'])
    )

    train_ds = TranslationDataset(train_ds_raw, tokenizer_src, tokenizer_tgt, src, tgt, config['seq_len'])
    val_ds = TranslationDataset(val_ds_raw, tokenizer_src, tokenizer_tgt, src, tgt, config['seq_len'])

    pin = torch.cuda.is_available()
    nw = config['num_workers']
    train_dataloader = DataLoader(train_ds, batch_size=config['batch_size'], shuffle=True,
                                  num_workers=nw, pin_memory=pin, persistent_workers=nw > 0)
    val_dataloader = DataLoader(val_ds, batch_size=config['batch_size'], shuffle=False,
                                num_workers=nw, pin_memory=pin, persistent_workers=nw > 0)

    return train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt


def get_model(config, vocab_src_len, vocab_tgt_len):
    return BuildTransformer(
        vocab_src_len, vocab_tgt_len, config['seq_len'], config['seq_len'],
        d_model=config['d_model'], N=config['N'], h=config['h'],
        dropout=config['dropout'], d_ff=config['d_ff'],
    )


@torch.no_grad()
def greedy_decode(model, src, src_mask, tokenizer_tgt, max_len, device):
    sos = tokenizer_tgt.token_to_id('[SOS]')
    eos = tokenizer_tgt.token_to_id('[EOS]')
    enc_out = model.encode(src, src_mask)
    dec_in = torch.full((1, 1), sos, dtype=torch.int64, device=device)
    while dec_in.size(1) < max_len:
        tgt_mask = casual_mask(dec_in.size(1)).to(device)
        out = model.decode(enc_out, src_mask, dec_in, tgt_mask)
        next_tok = torch.argmax(model.project(out[:, -1]), dim=-1)
        dec_in = torch.cat([dec_in, next_tok.view(1, 1)], dim=1)
        if next_tok.item() == eos:
            break
    return dec_in.squeeze(0)


@torch.no_grad()
def run_validation(model, val_dataloader, loss_fn, tokenizer_src, tokenizer_tgt, config, device, amp_ctx, num_examples=3):
    model.eval()
    total_loss, n = 0.0, 0
    vocab = tokenizer_tgt.get_vocab_size()
    for batch in val_dataloader:
        enc_in = batch['encoder_input'].to(device)
        dec_in = batch['decoder_input'].to(device)
        enc_mask = batch['encoder_mask'].to(device)
        dec_mask = batch['decoder_mask'].to(device)
        label = batch['label'].to(device)
        with amp_ctx():
            enc_out = model.encode(enc_in, enc_mask)
            dec_out = model.decode(enc_out, enc_mask, dec_in, dec_mask)
            proj = model.project(dec_out)
            loss = loss_fn(proj.float().view(-1, vocab), label.view(-1))
        total_loss += loss.item()
        n += 1

    # a few greedy-decoded examples so you can eyeball quality
    batch = next(iter(val_dataloader))
    for i in range(min(num_examples, batch['encoder_input'].size(0))):
        src = batch['encoder_input'][i:i + 1].to(device)
        src_mask = batch['encoder_mask'][i:i + 1].to(device)
        with amp_ctx():
            pred = greedy_decode(model, src, src_mask, tokenizer_tgt, config['seq_len'], device)
        pred_text = tokenizer_tgt.decode(pred.tolist())
        print(f"\n  SRC : {batch['src_text'][i]}\n  TGT : {batch['tgt_text'][i]}\n  PRED: {pred_text}")
    model.train()
    return total_loss / max(n, 1)


def train_model(config):
    torch.manual_seed(config['seed'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device {device}' + (f' ({torch.cuda.get_device_name(0)})' if device.type == 'cuda' else ''))
    if device.type == 'cpu':
        print('WARNING: no GPU found. In Colab: Runtime > Change runtime type > T4 GPU')

    # mixed precision: bf16 on Ampere+ (A100/L4), fp16 + GradScaler on T4
    use_cuda = device.type == 'cuda'
    use_bf16 = use_cuda and torch.cuda.get_device_capability()[0] >= 8
    amp_dtype = torch.bfloat16 if use_bf16 else torch.float16
    amp_ctx = lambda: torch.autocast(device_type='cuda', dtype=amp_dtype, enabled=use_cuda)
    scaler = torch.cuda.amp.GradScaler(enabled=use_cuda and not use_bf16)
    if use_cuda:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

    Path(config['model_folder']).mkdir(parents=True, exist_ok=True)

    train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt = get_ds(config)
    vocab_tgt = tokenizer_tgt.get_vocab_size()
    model = get_model(config, tokenizer_src.get_vocab_size(), vocab_tgt).to(device)
    print(f'Parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M | '
          f'src vocab {tokenizer_src.get_vocab_size()} | tgt vocab {vocab_tgt}')

    writer = SummaryWriter(config['experiment_name'])
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], betas=(0.9, 0.98), eps=1e-9)

    total_steps = config['num_epoch'] * len(train_dataloader)
    warmup = config['warmup_steps']

    def lr_lambda(step):
        if step < warmup:
            return (step + 1) / warmup
        progress = (step - warmup) / max(1, total_steps - warmup)
        return 0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * min(progress, 1.0)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    initial_epoch, global_step = 0, 0
    preload = config['preload']
    model_filename = latest_weights_file_path(config) if preload == 'latest' else (
        get_weights_file_path(config, preload) if preload else None)
    if model_filename and Path(model_filename).exists():
        print(f'Preloading model {model_filename}')
        state = torch.load(model_filename, map_location=device)
        model.load_state_dict(state['model_state_dict'])
        optimizer.load_state_dict(state['optimizer_state_dict'])
        if 'scheduler_state_dict' in state:
            scheduler.load_state_dict(state['scheduler_state_dict'])
        initial_epoch = state['epoch'] + 1
        global_step = state['global_step']
    elif preload and preload != 'latest':
        raise FileNotFoundError(f'Checkpoint not found: {model_filename}')

    loss_fn = nn.CrossEntropyLoss(
        ignore_index=tokenizer_tgt.token_to_id('[PAD]'), label_smoothing=config['label_smoothing']
    ).to(device)

    for epoch in range(initial_epoch, config['num_epoch']):
        model.train()
        batch_iterator = tqdm(train_dataloader, desc=f'Epoch {epoch:02d}')
        for batch in batch_iterator:
            encoder_input = batch['encoder_input'].to(device, non_blocking=True)
            decoder_input = batch['decoder_input'].to(device, non_blocking=True)
            encoder_mask = batch['encoder_mask'].to(device, non_blocking=True)
            decoder_mask = batch['decoder_mask'].to(device, non_blocking=True)
            label = batch['label'].to(device, non_blocking=True)

            with amp_ctx():
                encoder_output = model.encode(encoder_input, encoder_mask)
                decoder_output = model.decode(encoder_output, encoder_mask, decoder_input, decoder_mask)
                proj_output = model.project(decoder_output)
                loss = loss_fn(proj_output.float().view(-1, vocab_tgt), label.view(-1))

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config['grad_clip'])
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            batch_iterator.set_postfix(loss=f'{loss.item():6.3f}')
            if global_step % 50 == 0:
                writer.add_scalar('train loss', loss.item(), global_step)
                writer.add_scalar('lr', scheduler.get_last_lr()[0], global_step)
            global_step += 1

        val_loss = run_validation(model, val_dataloader, loss_fn, tokenizer_src, tokenizer_tgt, config, device, amp_ctx)
        print(f'\nEpoch {epoch:02d} | validation loss {val_loss:.4f}')
        writer.add_scalar('validation loss', val_loss, global_step)
        writer.flush()

        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'global_step': global_step,
        }, get_weights_file_path(config, f'{epoch:02d}'))

    writer.close()


if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    train_model(get_config())
