"""
Helpers for rendering GAIA Cognition Packets (GCP) as structured templates.

The template is intentionally compact: only populated sections are emitted,
and lengthy text fields are trimmed so the rendered block can be injected
directly into prompts without blowing the token budget.
"""
from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

from gaia_common.protocols.cognition_packet import CognitionPacket, DataField

MAX_INLINE_LEN = int(os.getenv("GAIA_GCP_INLINE_LIMIT", "600"))


def _trim(value: Any, limit: int = MAX_INLINE_LEN) -> Any:
	if isinstance(value, str):
		return value.strip()[:limit]
	return value


def _clean_dict(payload: Dict[str, Any]) -> Dict[str, Any]:
	# Preserve explicit False/0 values so governance/status fields remain visible.
	return {k: v for k, v in payload.items() if v not in (None, "", [], {})}


def packet_to_template_dict(packet: CognitionPacket, processed_data_field_keys: Optional[set] = None) -> Dict[str, Any]:
	"""Return a nested dict representing the packet in a transport-friendly format."""
	header = getattr(packet, "header", None)
	persona = getattr(header, "persona", None) if header else None
	context = getattr(packet, "context", None)
	content = getattr(packet, "content", None)
	intent = getattr(packet, "intent", None)

	header_payload = _clean_dict({
		"identity_id": getattr(persona, "identity_id", None) if persona else None,
		"persona_id": getattr(persona, "persona_id", None) if persona else None,
		"persona_role": getattr(getattr(persona, "role", None), "value", None) if persona else None,
		"tone_hint": getattr(persona, "tone_hint", None) if persona else None,
		"traits": getattr(persona, "traits", None) if persona else None,
		"session_id": getattr(header, "session_id", None) if header else None,
		"packet_id": getattr(header, "packet_id", None) if header else None,
		"origin": getattr(getattr(header, "origin", None), "value", None) if header else None,
	})
	routing = getattr(header, "routing", None) if header else None
	routing_payload = _clean_dict({
		"target_engine": getattr(getattr(routing, "target_engine", None), "value", None),
		"allow_parallel": getattr(routing, "allow_parallel", None),
		"priority": getattr(routing, "priority", None),
		"deadline_iso": getattr(routing, "deadline_iso", None),
		"queue_id": getattr(routing, "queue_id", None),
	})
	model = getattr(header, "model", None) if header else None
	model_payload = _clean_dict({
		"name": getattr(model, "name", None),
		"provider": getattr(model, "provider", None),
		"context_window_tokens": getattr(model, "context_window_tokens", None),
		"max_output_tokens": getattr(model, "max_output_tokens", None),
		"response_buffer_tokens": getattr(model, "response_buffer_tokens", None),
		"temperature": getattr(model, "temperature", None),
		"top_p": getattr(model, "top_p", None),
		"seed": getattr(model, "seed", None),
		"allow_tools": getattr(model, "allow_tools", None),
	})

	def _serialize_data_fields(fields: Optional[List[DataField]], processed_keys: Optional[set] = None) -> List[Dict[str, Any]]:
		items: List[Dict[str, Any]] = []
		processed_keys = processed_keys if processed_keys is not None else set()
		for field in fields or []:
			try:
				if getattr(field, "key", "") in processed_keys:
					continue # Skip if already processed
				items.append(_clean_dict({
					"key": getattr(field, "key", None),
					"value": _trim(getattr(field, "value", None)),
					"type": getattr(field, "type", None),
					"source": getattr(field, "source", None),
				}))
			except Exception:
				continue
		return [item for item in items if item]

	context_payload = {}
	if context:
		try:
			cs = [{
				"id": getattr(c, "id", None),
				"title": getattr(c, "title", None),
				"ref": getattr(c, "pointer", None),
			} for c in (getattr(context, "cheatsheets", []) or [])]
		except Exception:
			cs = []
		try:
			snippets = [{
				"id": getattr(s, "id", None),
				"role": getattr(s, "role", None),
				"summary": _trim(getattr(s, "summary", "")),
			} for s in (getattr(context, "relevant_history_snippet", []) or [])]
		except Exception:
			snippets = []
		try:
			constraints = getattr(context, "constraints", None)
			if constraints:
				constraints_payload = _clean_dict({
					"max_tokens": getattr(constraints, "max_tokens", None),
					"time_budget_ms": getattr(constraints, "time_budget_ms", None),
					"safety_mode": getattr(constraints, "safety_mode", None),
				})
			else:
				constraints_payload = {}
		except Exception:
			constraints_payload = {}
		context_payload = _clean_dict({
			"session_history_ref": getattr(getattr(context, "session_history_ref", None), "value", None),
			"cheatsheets": [c for c in cs if c.get("id")],
			"constraints": constraints_payload,
			"relevant_history": [s for s in snippets if s.get("summary")],
			"available_mcp_tools": getattr(context, "available_mcp_tools", None),
		})

	content_payload = _clean_dict({
		"original_prompt": _trim(getattr(content, "original_prompt", "")) if content else "",
		"data_fields": _serialize_data_fields(getattr(content, "data_fields", []) if content else [], processed_data_field_keys),
	})

	intent_payload = _clean_dict({
		"user_intent": getattr(intent, "user_intent", None),
		"system_task": getattr(getattr(intent, "system_task", None), "value", None),
		"read_only": next((bool(getattr(df, "value", False)) for df in getattr(content, "data_fields", []) if getattr(df, "key", "") == "read_only_intent"), False) if content else False,
	})

	reasoning = getattr(packet, "reasoning", None)
	reasoning_payload = {}
	if reasoning:
		reflections = [{
			"step": getattr(r, "step", None),
			"summary": _trim(getattr(r, "summary", "")),
			"confidence": getattr(r, "confidence", None),
		} for r in (getattr(reasoning, "reflection_log", []) or [])]
		sketchpad = [{
			"slot": getattr(s, "slot", None),
			"content": _trim(getattr(s, "content", None)),
			"type": getattr(s, "content_type", None),
		} for s in (getattr(reasoning, "sketchpad", []) or [])]
		evals = [{
			"name": getattr(e, "name", None),
			"passed": getattr(e, "passed", None),
			"score": getattr(e, "score", None),
			"notes": _trim(getattr(e, "notes", "")),
		} for e in (getattr(reasoning, "evaluations", []) or [])]
		reasoning_payload = _clean_dict({
			"reflection_log": [r for r in reflections if r.get("summary")],
			"sketchpad": [s for s in sketchpad if s.get("content")],
			"evaluations": [e for e in evals if e.get("name")],
		})

	governance = getattr(packet, "governance", None)
	safety = getattr(governance, "safety", None) if governance else None
	governance_payload = _clean_dict({
		"execution_allowed": getattr(safety, "execution_allowed", None),
		"dry_run": getattr(safety, "dry_run", None),
		"allowed_commands_whitelist_id": getattr(safety, "allowed_commands_whitelist_id", None),
	})

	metrics = getattr(packet, "metrics", None)
	token_usage = getattr(metrics, "token_usage", None) if metrics else None
	metrics_payload = _clean_dict({
		"prompt_tokens": getattr(token_usage, "prompt_tokens", None),
		"completion_tokens": getattr(token_usage, "completion_tokens", None),
		"total_tokens": getattr(token_usage, "total_tokens", None),
		"latency_ms": getattr(metrics, "latency_ms", None),
	})

	status = getattr(packet, "status", None)
	status_payload = _clean_dict({
		"finalized": getattr(status, "finalized", None),
		"state": getattr(getattr(status, "state", None), "value", None),
		"next_steps": getattr(status, "next_steps", None),
	})

	template = _clean_dict({
		"header": header_payload,
		"routing": routing_payload,
		"model": model_payload,
		"context": context_payload,
		"content": content_payload,
		"intent": intent_payload,
		"reasoning": reasoning_payload,
		"governance": governance_payload,
		"metrics": metrics_payload,
		"status": status_payload,
	})
	return template


def render_gaia_packet_template(packet: CognitionPacket, indent: str = "  ", processed_data_field_keys: Optional[set] = None) -> str:
	"""Render the packet template dict into a human-readable block."""
	template = packet_to_template_dict(packet, processed_data_field_keys)
	lines: List[str] = []

	def render_section(name: str, payload: Dict[str, Any]):
		if not payload:
			return
		lines.append(f"[{name}]")
		for key, value in payload.items():
			if isinstance(value, list):
				if not value:
					continue
				lines.append(f"{indent}{key}:")
				for item in value:
					if isinstance(item, dict):
						inner = ", ".join(f"{ik}={_trim(iv)}" for ik, iv in item.items() if iv not in (None, "", []))
						lines.append(f"{indent*2}- {inner}")
					else:
						lines.append(f"{indent*2}- {_trim(item)}")
			elif isinstance(value, dict):
				if value:
					lines.append(f"{indent}{key}:")
					for ik, iv in value.items():
						if iv not in (None, "", []):
							lines.append(f"{indent*2}{ik}: {_trim(iv)}")
			else:
				lines.append(f"{indent}{key}: {_trim(value)}")
		lines.append("")  # blank line between sections

	for section in ("header", "routing", "model", "context", "intent", "content", "reasoning", "governance", "metrics", "status"):
		render_section(section.upper(), template.get(section, {}))

	return "\n".join(line for line in lines if line.strip() or line == "")
