"""20-step QLoRA smoke test on Gemma 4 12B NF4: does training fit, how fast (2qus)."""
import glob, json, sys, time, torch
from transformers import AutoModelForImageTextToText, AutoProcessor
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
SEQ = int(sys.argv[1]) if len(sys.argv) > 1 else 1024
src = "/models/gemma-4-12B-it-NF4"
proc = AutoProcessor.from_pretrained(src); tok = proc.tokenizer
rows = []
for f in sorted(glob.glob("/gaia/GAIA_Project/knowledge/curricula/core_v2x_patch/*.jsonl")):
    rows += [json.loads(l) for l in open(f) if l.strip()]
text = "".join(tok.apply_chat_template([{"role": "user", "content": r["instruction"]},
                                         {"role": "assistant", "content": r["output"]}], tokenize=False) for r in rows)
ids = tok(text, return_tensors="pt")["input_ids"][0]
batches = [ids[i:i + SEQ] for i in range(0, len(ids) - SEQ, SEQ)]
print(json.dumps({"samples": len(rows), "tokens": len(ids), "seq": SEQ, "batches": len(batches)}), flush=True)
torch.cuda.reset_peak_memory_stats(); t = time.time()
m = AutoModelForImageTextToText.from_pretrained(src, device_map={"": 0}, dtype=torch.bfloat16)
m.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False}); m.enable_input_require_grads(); m.config.use_cache = False
cfg = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.0, task_type="CAUSAL_LM",
                 target_modules=r".*language_model.*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)")
m = get_peft_model(m, cfg)
trainable = sum(p.numel() for p in m.parameters() if p.requires_grad)
mb = lambda x: round(x / 2**20)
print(json.dumps({"load_s": round(time.time() - t, 1), "after_load_mb": mb(torch.cuda.memory_allocated()),
                  "load_peak_mb": mb(torch.cuda.max_memory_allocated()), "trainable_M": round(trainable / 1e6, 1)}), flush=True)
import bitsandbytes as bnb; import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
_inner = m.base_model.model  # Gemma4UnifiedForConditionalGeneration under PEFT
_softcap = _inner.config.get_text_config().final_logit_softcapping


def chunked_loss(input_ids, chunk=256):
    """Next-token loss without the full [seq, 262144] logits: each chunk's
    logits are rebuilt in backward (checkpoint), so only one chunk lives at a
    time. At 1024 tokens the full logits, their float copy and their gradient
    were ~2 GB of the 12.6 GB peak (2026-11-06)."""
    hidden = _inner.model(input_ids=input_ids, return_dict=True).last_hidden_state
    h, y = hidden[:, :-1], input_ids[:, 1:]
    w = _inner.lm_head.weight

    def piece(hc, yc):
        logits = hc @ w.t()
        if _softcap:
            logits = torch.tanh(logits / _softcap) * _softcap
        return F.cross_entropy(logits.float().reshape(-1, logits.shape[-1]), yc.reshape(-1), reduction="sum")
    total = sum(checkpoint(piece, h[:, i:i + chunk], y[:, i:i + chunk], use_reentrant=False)
                for i in range(0, h.shape[1], chunk))
    return total / y.numel()


CHUNKED = __import__("os").environ.get("CHUNKED", "1") == "1"
if CHUNKED:
    with torch.no_grad():
        b0 = batches[0][:256].unsqueeze(0).to(0)
        print(json.dumps({"loss_check_model": round(m(input_ids=b0, labels=b0).loss.item(), 4),
                          "loss_check_chunked": round(chunked_loss(b0).item(), 4)}), flush=True)
opt = bnb.optim.AdamW8bit([p for p in m.parameters() if p.requires_grad], lr=float(__import__("os").environ.get("LR", "1e-4")))
m.train(); torch.cuda.reset_peak_memory_stats(); losses, t0 = [], time.time()
print(json.dumps({"before_step_mb": mb(torch.cuda.memory_allocated())}), flush=True)
for step in range(20):
    b = batches[step % len(batches)].unsqueeze(0).to(0)
    loss = chunked_loss(b) if CHUNKED else m(input_ids=b, labels=b).loss
    out = type("Out", (), {"loss": loss})
    out.loss.backward()
    if step == 0:
        print(json.dumps({"after_backward_peak_mb": mb(torch.cuda.max_memory_allocated()), "reserved_mb": mb(torch.cuda.memory_reserved())}), flush=True)
    torch.cuda.empty_cache()
    MODE = __import__("os").environ.get("MODE", "adam")
    if MODE == "adam":
        opt.step()
    elif MODE == "sgd":
        with torch.no_grad():
            for p_ in m.parameters():
                if p_.grad is not None: p_.add_(p_.grad, alpha=-1e-4)
    opt.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    if step >= 2 and MODE != "adam": print(json.dumps({"mode": MODE, "ok_through_step": step + 1, "loss": losses[-1] if losses else None}), flush=True); break
    losses.append(round(out.loss.item(), 3))
    if step in (0, 4, 19):
        print(json.dumps({"step": step + 1, "loss": losses[-1], "s_per_step": round((time.time() - t0) / (step + 1), 2),
                          "peak_mb": mb(torch.cuda.max_memory_allocated()), "reserved_mb": mb(torch.cuda.memory_reserved())}), flush=True)
print(json.dumps({"losses": losses}), flush=True)
