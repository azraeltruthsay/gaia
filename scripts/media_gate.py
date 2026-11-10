#!/usr/bin/env python3
"""Media gate: the cognitive battery's vision + audio tests, against any Core engine, N times.

Run it for every new Core build (training, quantization, backend change) and compare
with the live Core. Single runs swing about +-2 on audio (temperature 0.3), so use
--runs 3 and compare means.

Reference, 2026-11-10, 12B v2 on the HF engine (GPU NF4), 3 runs:
    vision 15.0/16, audio 12.3/20

Run where the battery's fixtures can be read, e.g. inside gaia-core:
    docker cp scripts/media_gate.py gaia-core:/tmp/mg/
    docker cp gaia-doctor/cognitive_test_battery.py gaia-core:/tmp/mg/
    docker cp gaia-doctor/test_assets gaia-core:/tmp/mg/
    docker exec gaia-core python /tmp/mg/media_gate.py http://localhost:8092 --runs 3
    # a llama-server engine (audio goes as input_audio, not audio_url):
    docker exec gaia-core python /tmp/mg/media_gate.py http://llmm:8198 --llama --runs 3

Results go to /tmp (never the doctor's live results file).
"""
import argparse
import base64
import functools
import json
import os
import sys
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("engine", help="engine base URL serving /v1/chat/completions")
    p.add_argument("--label", default="engine")
    p.add_argument("--runs", type=int, default=3)
    p.add_argument("--llama", action="store_true", help="send audio as llama-server input_audio")
    p.add_argument("--llama-house", action="store_true",
                   help="llama-server /completion with the GAIA engine's own house-format prompt "
                        "(what a house-trained Core expects; --llama uses the GGUF's stock template)")
    p.add_argument("--temperature", type=float, default=0.3,
                   help="sampling temperature (battery default 0.3; 0 = greedy, for a per-test "
                        "comparison of two engines)")
    p.add_argument("--dump", default=None,
                   help="write every test's reply to this JSON file ({fixture: [reply per run]})")
    p.add_argument("--fixtures", default=None,
                   help="dir holding vision/ and audio/ (default: test_assets next to this script, "
                        "else gaia-doctor/test_assets in the repo)")
    a = p.parse_args()

    fx = a.fixtures
    if fx is None:
        for cand in (os.path.join(HERE, "test_assets"), os.path.join(HERE, "..", "gaia-doctor", "test_assets")):
            if os.path.isdir(cand):
                fx = cand
                break
    if not fx:
        sys.exit("no fixtures dir found; pass --fixtures")
    os.environ["DOCTOR_VISION_FIXTURE_DST"] = os.path.join(fx, "vision")
    os.environ["DOCTOR_AUDIO_FIXTURE_DST"] = os.path.join(fx, "audio")
    os.environ["COGNITIVE_RESULTS_PATH"] = f"/tmp/media_gate_{a.label}.json"

    for d in (HERE, os.path.join(HERE, "..", "gaia-doctor")):
        sys.path.insert(0, d)
    import cognitive_test_battery as b

    ep = a.engine.rstrip("/")
    temp = a.temperature

    # The GAIA engine path, as the battery sends it (image_url / audio_url data
    # URIs to /v1/chat/completions), with the temperature as a parameter.
    def _gaia(prompt, path, kind, engine_endpoint=ep, timeout=120, max_tokens=80):
        ext = os.path.splitext(path)[1].lower()
        mime = (b._VISION_MIME_BY_EXT.get(ext, "image/jpeg") if kind == "image_url"
                else b._AUDIO_MIME_BY_EXT.get(ext, "audio/wav"))
        data = base64.b64encode(open(path, "rb").read()).decode()
        payload = {"model": "/models/core", "messages": [{"role": "user", "content": [
            {"type": "text", "text": prompt},
            {"type": kind, kind: {"url": f"data:{mime};base64,{data}"}}]}],
            "max_tokens": max_tokens, "temperature": temp}
        req = urllib.request.Request(f"{engine_endpoint}/v1/chat/completions", json.dumps(payload).encode(),
                                     {"Content-Type": "application/json"})
        r = json.load(urllib.request.urlopen(req, timeout=timeout))
        return b._strip_think_tags(r.get("choices", [{}])[0].get("message", {}).get("content", ""))
    b.query_engine_with_image = lambda prompt, path, **kw: _gaia(prompt, path, "image_url", **kw)
    if a.llama_house:
        # The prompt the GAIA engine builds for one user message of [text, media]
        # (core.py _prepare_*_inputs, house format, enable_thinking=True):
        #   <bos><|turn>user<turn|>\n{text}<media>\n<|turn>assistant<turn|>\n
        # llama-server adds <bos> and wraps its media marker in the boa/eoa
        # (boi/eoi) tokens, as the HF processor does.
        # Recent llama-server builds randomize the media marker; /props says which.
        props = json.load(urllib.request.urlopen(f"{ep}/props", timeout=30))
        marker = props.get("media_marker") or "<__media__>"

        def _house(prompt, path, engine_endpoint=ep, timeout=120, max_tokens=80):
            text = f"<|turn>user<turn|>\n{prompt}{marker}\n<|turn>assistant<turn|>\n"
            body = {"prompt": {"prompt_string": text,
                               "multimodal_data": [base64.b64encode(open(path, "rb").read()).decode()]},
                    "n_predict": max_tokens, "cache_prompt": False,
                    # the HF engine's sampling for these requests: temperature 0.3 (battery),
                    # top_p 0.9 and repetition_penalty 1.1 (engine defaults), top_k 64 (model's
                    # generation_config); llama-server would otherwise use top_k 40 + min_p 0.05
                    "temperature": temp, "top_p": 0.9, "top_k": 64, "min_p": 0.0,
                    "repeat_penalty": 1.1, "repeat_last_n": 8192}
            req = urllib.request.Request(f"{engine_endpoint}/completion", json.dumps(body).encode(),
                                         {"Content-Type": "application/json"})
            r = json.load(urllib.request.urlopen(req, timeout=timeout))
            return b._strip_think_tags(r.get("content", ""))
        b.query_engine_with_image = _house
        b.query_engine_with_audio = _house
    elif a.llama:
        def _audio(prompt, audio_path, engine_endpoint=ep, timeout=120, max_tokens=80):
            data = base64.b64encode(open(audio_path, "rb").read()).decode()
            fmt = os.path.splitext(audio_path)[1].lstrip(".").lower() or "wav"
            payload = {"messages": [{"role": "user", "content": [
                {"type": "text", "text": prompt},
                {"type": "input_audio", "input_audio": {"data": data, "format": fmt}}]}],
                "max_tokens": max_tokens, "temperature": temp}
            req = urllib.request.Request(f"{engine_endpoint}/v1/chat/completions", json.dumps(payload).encode(),
                                         {"Content-Type": "application/json"})
            r = json.load(urllib.request.urlopen(req, timeout=timeout))
            return b._strip_think_tags(r["choices"][0]["message"]["content"])
        b.query_engine_with_audio = _audio
    if not (a.llama or a.llama_house):
        b.query_engine_with_audio = lambda prompt, path, **kw: _gaia(prompt, path, "audio_url", **kw)

    # Record every reply, keyed by fixture, for --dump.
    replies: dict[str, list[str]] = {}

    def _recorded(fn):
        def wrap(prompt, path, **kw):
            out = fn(prompt, path, **kw)
            replies.setdefault(f"{os.path.basename(path)} | {prompt}", []).append(out)
            return out
        return wrap
    b.query_engine_with_image = _recorded(b.query_engine_with_image)
    b.query_engine_with_audio = _recorded(b.query_engine_with_audio)

    totals: dict[str, list[int]] = {"vision": [], "audio": []}
    size: dict[str, int] = {}
    for run in range(1, a.runs + 1):
        for sec in totals:
            r = b.run_battery(endpoint="http://unused", section=sec, timeout=180, target="core")
            s = r["by_section"].get(sec, {})
            totals[sec].append(s.get("passed", 0))
            size[sec] = s.get("total", 0)
            fails = [f["id"] for f in r.get("failures", [])]
            print(f"{a.label} run {run} {sec}: {s.get('passed')}/{s.get('total')} "
                  f"(errors {s.get('errors')}) fails={fails}", flush=True)
    for sec, xs in totals.items():
        print(f"{a.label} {sec} mean {sum(xs) / len(xs):.1f}/{size[sec]} over {len(xs)} runs: {xs}")
    if a.dump:
        with open(a.dump, "w") as f:
            json.dump(replies, f, indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main()
