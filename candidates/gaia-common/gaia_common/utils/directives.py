"""Lightweight directive parser for GCP (POPULATE / SKETCH / CHEATSHEET / CODE.*)."""
from __future__ import annotations
import re
from typing import List, Dict, Any

# Regex blocks (multiline, greedy until >>> on its own line)
RE_BLOCK = re.compile(r"<<<(?P<body>.*?)>>>", re.DOTALL)

def parse(text: str) -> List[Dict[str, Any]]:
    """Return a list of directives with normalized fields."""
    out: List[Dict[str,Any]] = []
    for m in RE_BLOCK.finditer(text):
        body = m.group("body").strip()
        # POPULATE scratch.dataX WITH Something()
        if body.startswith("POPULATE"):
            # POPULATE scratch.dataA WITH GAIADevMatrix.get_open_tasks()
            m2 = re.match(r"POPULATE\s+scratch\.(?P<slot>data[A-E])\s+WITH\s+(?P<expr>.+)", body)
            if m2: out.append({"op":"POPULATE","slot":m2.group("slot"),"expr":m2.group("expr").strip()})
            continue
        # SKETCH WRITE <key>:\n<block>
        if body.startswith("SKETCH WRITE"):
            m2 = re.match(r"SKETCH WRITE\s+(?P<key>[A-Za-z0-9_.\-]+)\s*:\s*\n(?P<content>.*)", body, re.DOTALL)
            if m2: out.append({"op":"SKETCH_WRITE","key":m2.group("key"),"content":m2.group("content")})
            continue
        # SKETCH READ <key> INTO scratch.dataX
        if body.startswith("SKETCH READ"):
            m2 = re.match(r"SKETCH READ\s+(?P<key>[A-Za-z0-9_.\-]+)\s+INTO\s+scratch\.(?P<slot>data[A-E])", body)
            if m2: out.append({"op":"SKETCH_READ","key":m2.group("key"),"slot":m2.group("slot")})
            continue
        # CHEATSHEET.LOAD id="..." INTO cheats.X
        if body.startswith("CHEATSHEET.LOAD"):
            m2 = re.match(r'CHEATSHEET\.LOAD\s+id="(?P<cid>[^"]+)"\s+INTO\s+cheats\.(?P<slot>[A-E])', body)
            if m2: out.append({"op":"CHEATSHEET_LOAD","id":m2.group("cid"),"slot":m2.group("slot")})
            continue
        # CODE.READ path="..." INTO scratch.dataX
        if body.startswith("CODE.READ"):
            m2 = re.match(r'CODE\.READ\s+path="(?P<path>[^"]+)"\s+INTO\s+scratch\.(?P<slot>data[A-E])', body)
            if m2: out.append({"op":"CODE_READ","path":m2.group("path"),"slot":m2.group("slot")})
            continue
        # CODE.SPAN path="..." start=n end=m INTO scratch.dataX
        if body.startswith("CODE.SPAN"):
            m2 = re.match(r'CODE\.SPAN\s+path="(?P<path>[^"]+)"\s+start=(?P<start>\d+)\s+end=(?P<end>\d+)\s+INTO\s+scratch\.(?P<slot>data[A-E])', body)
            if m2: out.append({"op":"CODE_SPAN","path":m2.group("path"),"start":int(m2.group("start")),"end":int(m2.group("end")),"slot":m2.group("slot")})
            continue
        # CODE.TRIM FROM scratch.dataX KEEP="..." NOTE="..."
        if body.startswith("CODE.TRIM"):
            m2 = re.match(r'CODE\.TRIM\s+FROM\s+scratch\.(?P<slot>data[A-E])\s+KEEP="(?P<hint>[^"]+)"\s+NOTE="(?P<note>[^"]+)"', body)
            if m2: out.append({"op":"CODE_TRIM","slot":m2.group("slot"),"hint":m2.group("hint"),"note":m2.group("note")})
            continue
        # CODE.SUMMARIZE FROM scratch.dataX INTO scratch.dataY max_tokens=120
        if body.startswith("CODE.SUMMARIZE"):
            m2 = re.match(r'CODE\.SUMMARIZE\s+FROM\s+scratch\.(?P<src>data[A-E])\s+INTO\s+scratch\.(?P<dst>data[A-E])\s+max_tokens=(?P<mt>\d+)', body)
            if m2: out.append({"op":"CODE_SUMMARY","src":m2.group("src"),"dst":m2.group("dst"),"max_tokens":int(m2.group("mt"))})
            continue
        # CODE.SYMBOL path="..." symbol="FunctionOrClass" INTO scratch.dataX
        if body.startswith("CODE.SYMBOL"):
            m2 = re.match(r'CODE\.SYMBOL\s+path="(?P<path>[^"]+)"\s+symbol="(?P<sym>[^"]+)"\s+INTO\s+scratch\.(?P<slot>data[A-E])', body)
            if m2: out.append({"op":"CODE_SYMBOL","path":m2.group("path"),"symbol":m2.group("sym"),"slot":m2.group("slot")})
            continue
        # CODE.LIST_FILES path="..." INTO scratch.dataX
        if body.startswith("CODE.LIST_FILES"):
            m2 = re.match(r'CODE\.LIST_FILES\s+path="(?P<path>[^"]+)"\s+INTO\s+scratch\.(?P<slot>data[A-E])', body)
            if m2: out.append({"op":"CODE_LIST_FILES","path":m2.group("path"),"slot":m2.group("slot")})
            continue
        # BLUEPRINT.READ id="gaia_core" INTO scratch.dataA
        if body.startswith("BLUEPRINT.READ"):
            m2 = re.match(r'BLUEPRINT\.READ\s+id="(?P<bid>[^"]+)"\s+INTO\s+scratch\.(?P<slot>data[A-E])', body)
            if m2: out.append({"op":"BLUEPRINT_READ","id":m2.group("bid"),"slot":m2.group("slot")})
            continue
    return out