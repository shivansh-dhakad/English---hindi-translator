import sys

import torch
from tokenizers import Tokenizer

from config import get_config, latest_weights_file_path
from dataset import casual_mask
from model import BuildTransformer


def load(config, device):
    tok_src = Tokenizer.from_file(config['tokenizer_file'].format(config['lang_src']))
    tok_tgt = Tokenizer.from_file(config['tokenizer_file'].format(config['lang_tgt']))
    model = BuildTransformer(
        tok_src.get_vocab_size(), tok_tgt.get_vocab_size(), config['seq_len'], config['seq_len'],
        d_model=config['d_model'], N=config['N'], h=config['h'],
        dropout=config['dropout'], d_ff=config['d_ff'],
    ).to(device)
    state = torch.load(latest_weights_file_path(config), map_location=device)
    model.load_state_dict(state['model_state_dict'])
    return model.eval(), tok_src, tok_tgt


@torch.no_grad()
def translate(text, model, tok_src, tok_tgt, config, device):
    ids = tok_src.encode(text).ids[: config['seq_len'] - 2]
    sos, eos, pad = (tok_src.token_to_id(t) for t in ('[SOS]', '[EOS]', '[PAD]'))
    src = torch.tensor([[sos] + ids + [eos]], device=device)
    src_mask = (src != pad).unsqueeze(0).int()
    enc = model.encode(src, src_mask)
    dec = torch.full((1, 1), tok_tgt.token_to_id('[SOS]'), dtype=torch.int64, device=device)
    while dec.size(1) < config['seq_len']:
        out = model.decode(enc, src_mask, dec, casual_mask(dec.size(1)).to(device))
        nxt = model.project(out[:, -1]).argmax(-1)
        dec = torch.cat([dec, nxt.view(1, 1)], dim=1)
        if nxt.item() == tok_tgt.token_to_id('[EOS]'):
            break
    return tok_tgt.decode(dec.squeeze(0).tolist())


if __name__ == '__main__':
    cfg = get_config()
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    m, ts, tt = load(cfg, dev)
    print(translate(' '.join(sys.argv[1:]) or 'How are you?', m, ts, tt, cfg, dev))
