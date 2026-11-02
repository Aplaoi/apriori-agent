"""A deliberately small DeepSeek tool-calling agent for Weka Apriori."""
from __future__ import annotations

import json
import logging
import re
import time
from typing import Any, Callable

import requests

from config import (
    DEEPSEEK_API_KEY,
    DEEPSEEK_BASE_URL,
    DEEPSEEK_MODEL,
    DEFAULT_MAX_RULES,
    DEFAULT_MIN_CONFIDENCE,
    DEFAULT_MIN_SUPPORT,
)
from weka_runner import WekaError, run_apriori

logger = logging.getLogger(__name__)

RUN_APRIORI_TOOL = {
    "type": "function",
    "function": {
        "name": "run_apriori",
        "description": (
            "在已上传并预处理好的数据集上调用 Weka 的 Apriori 算法，挖掘频繁项集和关联规则。"
            "必须调用此工具后才能给出挖掘结论。"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "min_support": {
                    "type": "number",
                    "description": "最小支持度，0 到 1 之间的小数，例如 0.1 表示 10%。",
                },
                "min_confidence": {
                    "type": "number",
                    "description": "最小置信度，0 到 1 之间的小数，例如 0.6 表示 60%。",
                },
                "max_rules": {
                    "type": "integer",
                    "description": "最多返回多少条关联规则，1 到 100，默认 30。",
                },
            },
            "required": ["min_support", "min_confidence"],
        },
    },
}


class DeepSeekError(RuntimeError):
    """A user-safe DeepSeek API error (never contains the API key)."""


def _clean_message(message: dict[str, Any]) -> dict[str, Any]:
    cleaned: dict[str, Any] = {
        "role": "assistant",
        "content": message.get("content") or "",
    }
    if message.get("tool_calls"):
        cleaned["tool_calls"] = message["tool_calls"]
    return cleaned


def _chat(
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None = None,
    temperature: float = 0.2,
    max_tokens: int = 1800,
) -> dict[str, Any]:
    if not DEEPSEEK_API_KEY:
        raise DeepSeekError("未配置 DEEPSEEK_API_KEY，请在 .env 中填写。")

    payload: dict[str, Any] = {
        "model": DEEPSEEK_MODEL,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": False,
    }
    if tools:
        payload["tools"] = tools
        payload["tool_choice"] = "auto"

    headers = {
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
        "Content-Type": "application/json",
    }
    last_error = ""
    for attempt in range(2):
        try:
            response = requests.post(
                f"{DEEPSEEK_BASE_URL}/chat/completions",
                headers=headers,
                json=payload,
                timeout=(10, 120),
            )
        except requests.RequestException as exc:
            last_error = f"网络请求失败：{exc.__class__.__name__}"
        else:
            if response.status_code == 200:
                try:
                    return response.json()["choices"][0]["message"]
                except (KeyError, IndexError, ValueError) as exc:
                    raise DeepSeekError("DeepSeek 返回了无法解析的响应。") from exc
            if response.status_code in {429, 500, 502, 503, 504} and attempt == 0:
                last_error = f"HTTP {response.status_code}"
            else:
                detail = response.text.strip().replace("\n", " ")[:300]
                raise DeepSeekError(f"DeepSeek API 返回 HTTP {response.status_code}：{detail}")
        time.sleep(1.2)
    raise DeepSeekError(f"DeepSeek API 暂时不可用（{last_error}）。")


def _clamp_unit(value: Any, default: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    if 1 < number <= 100:
        number /= 100
    return max(0.001, min(number, 1.0))


def _extract_parameters(prompt: str) -> dict[str, Any]:
    """Deterministic fallback that also supplies sensible defaults to the agent."""
    text = prompt or ""

    def metric(keywords: list[str], default: float) -> float:
        for keyword in keywords:
            match = re.search(
                keyword + r"[^\d]{0,14}(\d+(?:\.\d+)?)\s*(%)?",
                text,
                flags=re.IGNORECASE,
            )
            if match:
                value = float(match.group(1))
                if match.group(2) or value > 1:
                    value /= 100
                return _clamp_unit(value, default)
        return default

    min_support = metric(["最小\\s*支持度", "支持度", "min\\s*support", "support"], DEFAULT_MIN_SUPPORT)
    min_confidence = metric(
        ["最小\\s*置信度", "置信度", "min\\s*confidence", "confidence"], DEFAULT_MIN_CONFIDENCE
    )

    max_rules = DEFAULT_MAX_RULES
    match = re.search(r"(?:最多|前|top|返回|输出)\s*(\d+)\s*条", text, flags=re.IGNORECASE)
    if not match:
        match = re.search(r"(\d+)\s*条(?:关联)?规则", text, flags=re.IGNORECASE)
    if not match:
        match = re.search(
            r"(?:top|max(?:imum)?|first)\s*(\d+)\s*(?:rules?)?",
            text,
            flags=re.IGNORECASE,
        )
    if match:
        max_rules = max(1, min(int(match.group(1)), 100))

    return {
        "min_support": min_support,
        "min_confidence": min_confidence,
        "max_rules": max_rules,
    }


def _normalize_tool_arguments(arguments: Any, fallback: dict[str, Any]) -> dict[str, Any]:
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except json.JSONDecodeError:
            arguments = {}
    if not isinstance(arguments, dict):
        arguments = {}
    return {
        "min_support": _clamp_unit(arguments.get("min_support"), fallback["min_support"]),
        "min_confidence": _clamp_unit(
            arguments.get("min_confidence"), fallback["min_confidence"]
        ),
        "max_rules": max(
            1,
            min(
                int(arguments.get("max_rules", fallback["max_rules"]) or fallback["max_rules"]),
                100,
            ),
        ),
    }


def _compact_result(result: dict[str, Any]) -> dict[str, Any]:
    itemsets = sorted(result["itemsets"], key=lambda entry: entry["support"], reverse=True)[:12]
    rules = sorted(
        result["rules"], key=lambda entry: (entry["confidence"], entry["lift"]), reverse=True
    )[:10]
    return {
        "status": "ok",
        "engine": result["engine"],
        "parameters": result["parameters"],
        "total_instances": result["total_instances"],
        "filter": result.get("filter"),
        "itemset_total": result["itemset_total"],
        "rule_total": result["rule_total"],
        "single_item_count": result["single_item_count"],
        "top_itemsets": [
            {"itemset": entry["display"], "support": round(entry["support"], 4)}
            for entry in itemsets
        ],
        "top_rules": [
            {
                "rule": entry["display"],
                "support": round(entry["support"], 4),
                "confidence": round(entry["confidence"], 4),
                "lift": round(entry["lift"], 4),
            }
            for entry in rules
        ],
    }


def _system_prompt(profile: dict[str, Any]) -> str:
    compact_profile = {
        "rows": profile.get("rows"),
        "columns": profile.get("columns"),
        "used_columns": profile.get("used_columns", [])[:30],
        "numeric_columns": profile.get("numeric_columns", []),
        "filter": profile.get("filter"),
        "dropped_columns": profile.get("dropped_columns", []),
        "notes": profile.get("notes", []),
    }
    return f"""你是「Weka Apriori 挖掘助手」，负责把用户的中文分析需求转成一次真实的数据挖掘任务。

当前已上传并预处理的数据集信息：
{json.dumps(compact_profile, ensure_ascii=False, indent=2)}

规则：
1. 你只能通过 run_apriori 工具获得挖掘结果，绝不能编造项集、规则、支持度、置信度或提升度。
2. 用户没有指定参数时使用：最小支持度 0.10、最小置信度 0.50、最多 30 条规则。
3. 支持度和置信度使用 0~1 的小数；如果用户说 10%，传 0.10。
4. 必须先调用 run_apriori，等工具返回后再作答。不要询问用户是否继续。
5. 最终回答用中文，简洁实用：先用一两句话概括结果，再列 3~6 条最值得关注的模式（引用工具返回的真实数值），最后说明参数和结果解读的注意事项。不要重复输出完整表格（前端会展示表格）。
6. 如果结果为空，直接说明在当前支持度/置信度下没有找到，并建议调整参数，不要虚构内容。
7. 如果数据集的 filter 不为空，说明数据已经按该条件过滤；回答开头必须写明筛选条件、原始行数和剩余行数，不要声称分析了全量数据。
"""


def _deterministic_answer(result: dict[str, Any], prompt: str, reason: str | None = None) -> str:
    params = result["parameters"]
    lines = [
        f"已在 {result['total_instances']} 条交易上调用 Weka Apriori 完成挖掘："
        f"最小支持度 {params['min_support']:.0%}，最小置信度 {params['min_confidence']:.0%}。",
        f"共找到 {result['itemset_total']} 个频繁项集、{result['rule_total']} 条关联规则。",
    ]
    filter_info = result.get("filter")
    if filter_info:
        lines.insert(
            0,
            f"已按「{filter_info['column']} = {filter_info['value']}」筛选，"
            f"原始 {filter_info['source_rows']} 行，保留 {filter_info['matched_rows']} 行。",
        )
    if result["itemsets"]:
        multi_itemsets = [entry for entry in result["itemsets"] if entry["size"] >= 2]
        top_itemset = max(
            multi_itemsets or result["itemsets"],
            key=lambda entry: (entry["support"], entry["size"]),
        )
        lines.append(
            f"支持度最高的频繁项集是「{top_itemset['display']}」（{top_itemset['support']:.1%}）。"
        )
    if result["rules"]:
        top_rule = max(result["rules"], key=lambda entry: (entry["confidence"], entry["lift"]))
        lines.append(
            f"最值得关注的规则是「{top_rule['display']}」，"
            f"置信度 {top_rule['confidence']:.1%}，提升度 {top_rule['lift']:.2f}。"
        )
    else:
        lines.append("当前置信度阈值下没有找到关联规则，可适当降低最小置信度后重试。")
    if reason:
        lines.append(f"（提示：{reason}，以上为本地规则生成的摘要。）")
    return "\n".join(lines)


def _summarize_with_model(
    prompt: str,
    profile: dict[str, Any],
    result: dict[str, Any],
) -> str:
    messages = [
        {"role": "system", "content": _system_prompt(profile)},
        {
            "role": "user",
            "content": (
                f"用户原始要求：{prompt}\n\n"
                "run_apriori 已执行，工具结果如下（这是唯一可信的数据来源）：\n"
                f"{json.dumps(_compact_result(result), ensure_ascii=False)}\n\n"
                "请直接给出最终中文总结，不要再调用工具。"
            ),
        },
    ]
    message = _chat(messages, tools=None, max_tokens=1400)
    return (message.get("content") or "").strip()


def analyze_with_agent(
    prompt: str,
    prepared: dict[str, Any],
    on_tool: Callable[[str, dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Run a bounded tool-calling loop; always return real Weka results."""
    prompt = (prompt or "").strip() or "分析这份数据的频繁项集和关联规则。"
    fallback_parameters = _extract_parameters(prompt)
    profile = prepared["profile"]
    tool_trace: list[dict[str, Any]] = []
    warnings: list[str] = []
    mining_result: dict[str, Any] | None = None
    final_text = ""
    mode = "agent"

    messages: list[dict[str, Any]] = [
        {"role": "system", "content": _system_prompt(profile)},
        {
            "role": "user",
            "content": (
                f"{prompt}\n\n"
                "请先调用 run_apriori（参数根据我的要求推断），再基于真实结果总结。"
            ),
        },
    ]

    for _round in range(4):
        try:
            message = _chat(messages, tools=[RUN_APRIORI_TOOL], max_tokens=1800)
        except DeepSeekError as exc:
            warnings.append(str(exc))
            mode = "fallback"
            break

        tool_calls = message.get("tool_calls") or []
        if not tool_calls:
            final_text = (message.get("content") or "").strip()
            break

        messages.append(_clean_message(message))
        for call in tool_calls:
            function = call.get("function") or {}
            name = function.get("name", "")
            call_id = call.get("id", "call_0")
            if name != "run_apriori":
                payload = {"status": "error", "error": f"未知工具：{name}"}
                messages.append(
                    {"role": "tool", "tool_call_id": call_id, "content": json.dumps(payload, ensure_ascii=False)}
                )
                tool_trace.append({"name": name, "status": "error", "error": f"未知工具：{name}"})
                continue

            parameters = _normalize_tool_arguments(function.get("arguments"), fallback_parameters)
            if on_tool:
                on_tool("run_apriori", parameters)
            try:
                mining_result = run_apriori(
                    prepared["arff_path"],
                    prepared["total_instances"],
                    prepared["attr_map"],
                    min_support=parameters["min_support"],
                    min_confidence=parameters["min_confidence"],
                    max_rules=parameters["max_rules"],
                    binary_attrs=prepared.get("binary_attrs"),
                )
            except WekaError as exc:
                payload = {"status": "error", "error": str(exc)}
                messages.append(
                    {"role": "tool", "tool_call_id": call_id, "content": json.dumps(payload, ensure_ascii=False)}
                )
                tool_trace.append(
                    {"name": name, "status": "error", "parameters": parameters, "error": str(exc)}
                )
                warnings.append(f"Weka 执行失败：{exc}")
                continue

            mining_result["parameters"]["num_bins"] = prepared.get("bins")
            mining_result["filter"] = prepared["profile"].get("filter")
            compact = _compact_result(mining_result)
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": call_id,
                    "content": json.dumps(compact, ensure_ascii=False),
                }
            )
            tool_trace.append(
                {
                    "name": name,
                    "status": "ok",
                    "parameters": parameters,
                    "itemset_total": mining_result["itemset_total"],
                    "rule_total": mining_result["rule_total"],
                }
            )

    if mining_result is None:
        mode = "fallback" if mode == "fallback" else "direct"
        parameters = fallback_parameters
        mining_result = run_apriori(
            prepared["arff_path"],
            prepared["total_instances"],
            prepared["attr_map"],
            min_support=parameters["min_support"],
            min_confidence=parameters["min_confidence"],
            max_rules=parameters["max_rules"],
            binary_attrs=prepared.get("binary_attrs"),
        )
        mining_result["parameters"]["num_bins"] = prepared.get("bins")
        mining_result["filter"] = prepared["profile"].get("filter")
        tool_trace.append(
            {
                "name": "run_apriori",
                "status": "ok",
                "parameters": parameters,
                "itemset_total": mining_result["itemset_total"],
                "rule_total": mining_result["rule_total"],
                "source": "fallback",
            }
        )

    if not final_text:
        try:
            final_text = _summarize_with_model(prompt, profile, mining_result)
        except DeepSeekError as exc:
            warnings.append(str(exc))
            mode = "fallback"
            final_text = ""

    if not final_text:
        final_text = _deterministic_answer(
            mining_result,
            prompt,
            reason="DeepSeek 未返回可用摘要" if warnings else None,
        )

    return {
        "answer": final_text,
        "mode": mode,
        "warning": "；".join(dict.fromkeys(warnings)) if warnings else None,
        "mining": mining_result,
        "tool_trace": tool_trace,
        "requested_parameters": fallback_parameters,
    }