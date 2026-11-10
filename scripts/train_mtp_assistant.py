#!/usr/bin/env python3
"""Fine-tune Core's MTP draft assistant on Core's own predictions (hfq7).

gaia_cpp drafts with google/gemma-4-12B-it-assistant (4 layers), trained by
Google on the stock 12B-it; on Core v2 only ~33-44% of its draft tokens are
accepted. This distils it onto v2: the 12B (frozen, NF4) reads GAIA-style
sequences once and gives, at every position, its last hidden state, its
shared K/V and its own next-token choice; the assistant learns to predict
that choice. Output can't change (the target verifies every token); only
acceptance, i.e. speed, does.

What the assistant sees at draft time (transformers' assistant candidate
generator, and llama.cpp's draft-mtp): for the last token x_p at position p,
input = [embed(x_p), h_{p-1}] (embed = the 12B's own scaled embedding, h = its
last hidden state), position p, attending the 12B's last-layer K/V for
positions < p. Later draft steps feed [embed(own draft), own projected state]
at the same position. Training batches every position of a sequence at once
with that causal mask; step 2 is unrolled where the data's next token is the
12B's own choice. --check compares the batched path with the reference
single-position path before training.

Run in gaia-study with the GPU free (GAIA in MEDITATION):
    python scripts/train_mtp_assistant.py --check
    python scripts/train_mtp_assistant.py --out /models/gemma-4-12B-GAIA-Core-v2-assistant
"""
import argparse
import json
import math
import os
import random
import time

import torch
import torch.nn.functional as F

TURN = "<|turn>{role}<turn|>\n{content}"


def house(messages):
    """GAIA's house turn format (gaia_cpp backend._build_prompt)."""
    return "\n".join(TURN.format(role=m["role"], content=m["content"]) for m in messages)


def load_texts(curriculum, sessions_path, max_chars=6000, seed=0):
    texts = []
    with open(curriculum) as f:
        for line in f:
            r = json.loads(line)
            if r.get("instruction") and r.get("output"):
                texts.append(house([{"role": "user", "content": r["instruction"]},
                                    {"role": "assistant", "content": r["output"]}]))
    n_curr = len(texts)
    if sessions_path and os.path.exists(sessions_path):
        sessions = json.load(open(sessions_path))
        for s in sessions.values():
            hist = [m for m in s.get("history", []) if m.get("role") in ("user", "assistant")
                    and isinstance(m.get("content"), str) and m["content"].strip()]
            for i, m in enumerate(hist):
                if m["role"] != "assistant" or len(m["content"]) < 40:
                    continue
                window = hist[max(0, i - 3): i + 1]
                t = house(window)
                texts.append(t[-max_chars:])
    random.Random(seed).shuffle(texts)
    print(f"texts: {len(texts)} ({n_curr} curriculum, {len(texts) - n_curr} from conversations)", flush=True)
    return texts


class Pair:
    def __init__(self, target_dir, assistant_dir, device="cuda"):
        from transformers import AutoTokenizer, Gemma4UnifiedForConditionalGeneration, \
            Gemma4UnifiedAssistantForCausalLM
        self.device = device
        self.tok = AutoTokenizer.from_pretrained(target_dir)
        self.target = Gemma4UnifiedForConditionalGeneration.from_pretrained(
            target_dir, device_map={"": 0}, dtype=torch.bfloat16)
        self.target.eval().requires_grad_(False)
        self.embed = self.target.get_input_embeddings()
        self.assistant = Gemma4UnifiedAssistantForCausalLM.from_pretrained(
            assistant_dir, dtype=torch.bfloat16, attn_implementation="sdpa").to(device)
        self.window = self.assistant.config.get_text_config().sliding_window

    def encode(self, text, max_len):
        ids = self.tok(text, return_tensors="pt", add_special_tokens=True).input_ids[:, -max_len:]
        return ids.to(self.device)

    @torch.no_grad()
    def target_pass(self, ids):
        out = self.target(input_ids=ids, output_hidden_states=True, return_shared_kv_states=True,
                          use_cache=False)
        H = out.hidden_states[-1]
        kv = {k: (v[0], v[1]) for k, v in out.shared_kv_states.items()}
        lab = out.logits.argmax(-1)          # 12B's choice for the token after each position
        E = self.embed(ids)
        return H, kv, lab, E

    def draft(self, inp, pos, kv):
        """Batched assistant forward: inp (1,Q,2D), pos (Q,), each query
        attending the 12B's K/V at positions < its own (sliding: within the
        window). Returns (logits, projected state)."""
        a = self.assistant
        T = kv["full_attention"][0].shape[2]
        k = torch.arange(T, device=inp.device)[None, :]
        p = pos[:, None]
        full = k < p
        swa = full & (k >= p - self.window)
        masks = {"full_attention": full[None, None], "sliding_attention": swa[None, None]}
        x = a.pre_projection(inp.to(a.pre_projection.weight.dtype))
        out = a.model(inputs_embeds=x, attention_mask=masks, position_ids=pos[None],
                      shared_kv_states=kv, use_cache=False)
        h = out.last_hidden_state
        return a.lm_head(h), a.post_projection(h)

    def reference(self, H, kv, E, p):
        """The candidate generator's own path for one position p (q_len 1)."""
        kv_p = {k: (v[0][:, :, :p], v[1][:, :, :p]) for k, v in kv.items()}
        inp = torch.cat([E[:, p:p + 1], H[:, p - 1:p]], dim=-1)
        out = self.assistant(inputs_embeds=inp, attention_mask=torch.ones(1, p + 1, device=inp.device, dtype=torch.long),
                             position_ids=torch.tensor([[p]], device=inp.device), shared_kv_states=kv_p,
                             use_cache=False)
        return out.logits[0, -1]

    def losses(self, ids, step2_weight=0.5):
        """Step-1 and step-2 CE against the 12B's choices, plus agreement counts."""
        H, kv, lab, E = self.target_pass(ids)
        T = ids.shape[1]
        pos = torch.arange(1, T, device=ids.device)               # draft positions p = 1..T-1
        inp1 = torch.cat([E[:, 1:], H[:, :-1]], dim=-1)           # [embed(x_p), h_{p-1}]
        logits1, state1 = self.draft(inp1, pos, kv)
        y1 = lab[:, 1:]                                           # 12B's next token after p
        loss1 = F.cross_entropy(logits1.float().flatten(0, 1), y1.flatten())
        acc1 = (logits1.argmax(-1) == y1).float()
        # Step 2 at p: [embed(12B's token after p), state1_p], same position, predicts the
        # 12B's token after p+1 - valid only where the data's x_{p+1} is that token.
        valid = (ids[:, 2:] == lab[:, 1:-1])                      # x_{p+1} == 12B's choice at p
        inp2 = torch.cat([self.embed(lab[:, 1:-1]), state1[:, :-1]], dim=-1)
        logits2, _ = self.draft(inp2, pos[:-1], kv)
        y2 = lab[:, 2:]
        ce2 = F.cross_entropy(logits2.float().flatten(0, 1), y2.flatten(), reduction="none")
        vm = valid.flatten().float()
        loss2 = (ce2 * vm).sum() / vm.sum().clamp(min=1)
        acc2 = ((logits2.argmax(-1) == y2).float() * valid.float())
        return loss1 + step2_weight * loss2, {
            "loss1": loss1.item(), "loss2": loss2.item(),
            "acc1": (acc1.sum().item(), acc1.numel()),
            "acc2": (acc2.sum().item(), int(valid.sum().item())),
        }


def check(pair, texts, max_len):
    """The batched mask path must reproduce the reference single-position path."""
    ids = pair.encode(texts[0], max_len)
    H, kv, lab, E = pair.target_pass(ids)
    T = ids.shape[1]
    pos = torch.arange(1, T, device=ids.device)
    with torch.no_grad():
        logits, _ = pair.draft(torch.cat([E[:, 1:], H[:, :-1]], dim=-1), pos, kv)
        worst, agree = 0.0, 0
        probes = [p for p in (1, 2, 17, T // 3, T // 2, T - 1) if 1 <= p < T]
        for p in probes:
            ref = pair.reference(H, kv, E, p).float()
            mine = logits[0, p - 1].float()
            worst = max(worst, (ref - mine).abs().max().item() / (ref.abs().max().item() + 1e-6))
            agree += int(ref.argmax() == mine.argmax())
    print(f"check: {agree}/{len(probes)} argmax agree, worst relative logit diff {worst:.4f} (T={T})", flush=True)
    return agree == len(probes) and worst < 0.05


@torch.no_grad()
def evaluate(pair, texts, max_len):
    pair.assistant.eval()
    a1 = n1 = a2 = n2 = 0
    for t in texts:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            _, m = pair.losses(pair.encode(t, max_len))
        a1 += m["acc1"][0]; n1 += m["acc1"][1]; a2 += m["acc2"][0]; n2 += m["acc2"][1]
    pair.assistant.train()
    return a1 / max(n1, 1), a2 / max(n2, 1)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--target", default="/models/Gemma4-12B-GAIA-Core-v2-NF4")
    ap.add_argument("--assistant", default="/models/gemma-4-12B-it-assistant")
    ap.add_argument("--curriculum", default="/gaia/GAIA_Project/knowledge/curricula/core_v2x_identity_v3/text.jsonl")
    ap.add_argument("--sessions", default="/shared/sessions.json")
    ap.add_argument("--out", default="/models/gemma-4-12B-GAIA-Core-v2-assistant")
    ap.add_argument("--max-len", type=int, default=512)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--warmup", type=int, default=100)
    ap.add_argument("--eval-n", type=int, default=150)
    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument("--check", action="store_true", help="only verify the batched path against the reference")
    a = ap.parse_args()

    torch.manual_seed(0)
    texts = load_texts(a.curriculum, a.sessions)
    held, train = texts[:a.eval_n], texts[a.eval_n:]
    pair = Pair(a.target, a.assistant)
    print(f"GPU after load: {torch.cuda.memory_allocated() / 2**30:.1f} GiB", flush=True)
    if not check(pair, texts, a.max_len):
        raise SystemExit("batched draft path disagrees with the reference path; not training")
    if a.check:
        return

    asst = pair.assistant
    # The tied embedding / lm_head (262k x 1024) stays as Google trained it.
    asst.model.embed_tokens.requires_grad_(False)
    asst.lm_head.requires_grad_(False)
    params = [p for p in asst.parameters() if p.requires_grad]
    for p in params:
        p.data = p.data.float()
    print(f"trainable: {sum(p.numel() for p in params) / 1e6:.1f}M params", flush=True)
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.0)
    total = int(len(train) * a.epochs)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / a.warmup) * 0.5 * (1 + math.cos(math.pi * min(s, total) / total)))

    base1, base2 = evaluate(pair, held, a.max_len)
    print(f"baseline held-out agreement: step1 {base1:.3f}, step2 {base2:.3f}", flush=True)
    best = (base1, -1)
    asst.train()
    t0, run = time.time(), []
    for step in range(total):
        ids = pair.encode(train[step % len(train)], a.max_len)
        if ids.shape[1] < 8:
            continue
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss, m = pair.losses(ids)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
        run.append(m["loss1"])
        if (step + 1) % 50 == 0:
            print(f"step {step + 1}/{total} loss1 {sum(run) / len(run):.3f} lr {sched.get_last_lr()[0]:.2e} "
                  f"{(time.time() - t0) / (step + 1):.2f}s/step", flush=True)
            run = []
        if (step + 1) % a.eval_every == 0 or step + 1 == total:
            e1, e2 = evaluate(pair, held, a.max_len)
            print(f"eval @{step + 1}: step1 {e1:.3f} (base {base1:.3f}), step2 {e2:.3f} (base {base2:.3f})", flush=True)
            if e1 > best[0]:
                best = (e1, step + 1)
                for p in params:
                    p.data = p.data.to(torch.bfloat16)
                asst.save_pretrained(a.out, safe_serialization=True)
                pair.tok.save_pretrained(a.out)
                for p in params:
                    p.data = p.data.float()
                print(f"  saved {a.out} (step1 {e1:.3f})", flush=True)
    print(f"done: best step1 agreement {best[0]:.3f} at step {best[1]} (baseline {base1:.3f})", flush=True)


if __name__ == "__main__":
    main()
