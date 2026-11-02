"""Thin, dependency-light wrapper around the Weka 3.8 Apriori CLI.

The agent never re-implements Apriori: every mining result comes from Weka.
This module only prepares a clean ARFF file, invokes Weka, and turns its text
output into structured JSON for the UI / LLM.
"""
from __future__ import annotations

import csv
import math
import re
import subprocess
import uuid
from pathlib import Path
from typing import Any

import pandas as pd

from config import (
    DEFAULT_BINS,
    WEKA_JAR,
    WEKA_TIMEOUT_SECONDS,
    WORK_DIR,
    java_executable,
)

ITEM_COLUMN_NAMES = {
    "item",
    "items",
    "itemset",
    "basket",
    "baskets",
    "products",
    "product",
    "transaction_items",
    "商品",
    "商品列表",
    "购物篮",
}
ID_COLUMN_NAMES = {
    "id",
    "tid",
    "transaction_id",
    "transactionid",
    "transaction",
    "order_id",
    "orderid",
    "编号",
    "序号",
    "交易编号",
}


class WekaError(RuntimeError):
    """Raised when Weka cannot be found or fails to mine the dataset."""


def _java_command() -> list[str]:
    if WEKA_JAR is None:
        raise WekaError(
            "找不到 weka.jar。请在 .env 中设置 WEKA_JAR，例如 "
            r"WEKA_JAR=D:\Weka-3-8-7\weka.jar"
        )
    return [
        java_executable(),
        "--add-opens",
        "java.base/java.lang=ALL-UNNAMED",
        "-Dfile.encoding=UTF-8",
        "-Dstdout.encoding=UTF-8",
        "-Dstderr.encoding=UTF-8",
        "-Xmx1g",
        "-cp",
        str(WEKA_JAR),
    ]


def _run_java(args: list[str], timeout: int = WEKA_TIMEOUT_SECONDS) -> subprocess.CompletedProcess[str]:
    command = _java_command() + args
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
            check=False,
        )
    except FileNotFoundError as exc:
        raise WekaError("找不到 Java，请安装 Java 或在 .env 中设置 WEKA_JAVA。") from exc
    except subprocess.TimeoutExpired as exc:
        raise WekaError(f"Weka 运行超过 {timeout} 秒，已停止。请提高最小支持度后重试。") from exc
    return result


def _detect_encoding(path: Path) -> str:
    raw = path.read_bytes()[: 256 * 1024]
    for encoding in ("utf-8-sig", "utf-8", "gb18030"):
        try:
            raw.decode(encoding)
            return encoding
        except UnicodeDecodeError:
            continue
    return "utf-8"


def _detect_delimiter(sample: str) -> str:
    try:
        return csv.Sniffer().sniff(sample, delimiters=",;\t|").delimiter
    except csv.Error:
        return ","


def _read_csv(path: Path) -> pd.DataFrame:
    encoding = _detect_encoding(path)
    sample = path.read_text(encoding=encoding, errors="replace")[: 64 * 1024]
    delimiter = _detect_delimiter(sample)
    try:
        frame = pd.read_csv(
            path,
            dtype=str,
            keep_default_na=False,
            na_values=["", "?", "NA", "N/A", "null", "NULL"],
            encoding=encoding,
            sep=delimiter,
            engine="python",
        )
    except Exception as exc:  # pandas raises several parser-specific errors
        raise WekaError(f"CSV 读取失败：{exc}") from exc
    return frame


def _arff_to_csv(arff_path: Path, csv_path: Path) -> None:
    result = _run_java(
        ["weka.core.converters.CSVSaver", "-i", str(arff_path), "-o", str(csv_path)]
    )
    if result.returncode != 0 or not csv_path.exists():
        detail = (result.stderr or result.stdout).strip()
        raise WekaError(f"ARFF 解析失败：{detail[:800]}")


def _clean_cell(value: Any) -> str | None:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    text = str(value).strip()
    if text == "" or text.lower() in {"nan", "none", "null"}:
        return None
    return text


GENDER_COLUMN_NAMES = {"性别", "gender", "sex"}
GENDER_VALUE_GROUPS = {
    "女": {"女", "女性", "女生", "女人", "f", "female", "woman", "women"},
    "男": {"男", "男性", "男生", "男人", "m", "male", "man", "men"},
}
FILTER_VALUE_SUFFIXES = (
    "的数据",
    "的样本",
    "的记录",
    "的用户",
    "的受访者",
    "的人",
    "数据",
    "样本",
    "记录",
    "用户",
    "受访者",
    "的",
)
DIRECT_FILTER_RE = re.compile(
    r"(?:只|仅|单独)\s*(?:处理|分析|看|保留|筛选|使用|考虑|要|取|提取|选择)?\s*"
    r"(?P<column>[^\s,，。;；:：=]{1,20})\s*"
    r"(?:为|是|等于|=|:|：)\s*"
    r"(?P<value>[^\s,，。;；]{1,30})",
    re.IGNORECASE,
)
EXPLICIT_FILTER_RE = re.compile(
    r"(?:筛选|过滤|限制|where|only)\s*(?:出)?\s*"
    r"(?P<column>[^\s,，。;；:：=]{1,20})\s*"
    r"(?:为|是|等于|=|:|：)\s*"
    r"(?P<value>[^\s,，。;；]{1,30})",
    re.IGNORECASE,
)
GENDER_ONLY_RE = re.compile(
    r"(?:只|仅|单独|only)\s*(?:看|要|保留|处理|分析|筛选|使用)?\s*"
    r"(?P<value>女性|男性|女生|男生|female|male)\s*"
    r"(?:用户|样本|数据|记录|受访者|的)?",
    re.IGNORECASE,
)


def _norm_filter_token(value: Any) -> str:
    return re.sub(r"\s+", "", str(value)).strip("'\"“”‘’").lower()


def _is_gender_column(column: str) -> bool:
    normalized = _norm_filter_token(column)
    return normalized in GENDER_COLUMN_NAMES or any(
        token in normalized for token in ("性别", "gender", "sex")
    )


def _find_filter_column(candidate: str, columns: list[str]) -> str | None:
    normalized = _norm_filter_token(candidate).strip("的")
    if not normalized:
        return None
    for column in columns:
        if _norm_filter_token(column) == normalized:
            return column
    if normalized in GENDER_COLUMN_NAMES:
        for column in columns:
            if _norm_filter_token(column) in GENDER_COLUMN_NAMES:
                return column
    for column in columns:
        column_norm = _norm_filter_token(column)
        if len(column_norm) >= 2 and (
            column_norm in normalized or normalized in column_norm
        ):
            return column
    return None


def _strip_filter_value(raw: str) -> str:
    text = str(raw).strip().strip("'\"“”‘’")
    text = re.split(r"[，,。;；\n]", text)[0].strip()
    for suffix in FILTER_VALUE_SUFFIXES:
        if text.endswith(suffix):
            text = text[: -len(suffix)].strip()
            break
    return text.strip("：:= ")


def _normalize_filter_value(raw: str, column: str, series: pd.Series) -> str | None:
    text = _strip_filter_value(raw)
    if not text:
        return None
    values = [str(value) for value in series.map(_clean_cell).dropna().unique()]
    normalized_text = _norm_filter_token(text)

    for value in values:
        if _norm_filter_token(value) == normalized_text:
            return value

    if _is_gender_column(column):
        for group in GENDER_VALUE_GROUPS.values():
            if normalized_text not in {_norm_filter_token(item) for item in group}:
                continue
            for value in values:
                if _norm_filter_token(value) in group:
                    return value
    return None


def _extract_row_filter(
    prompt: str, frame: pd.DataFrame
) -> tuple[str, str, str] | None:
    """Extract one equality row filter from a Chinese/English prompt."""
    text = (prompt or "").strip()
    if not text or frame.empty:
        return None
    columns = [str(column) for column in frame.columns]

    for pattern in (DIRECT_FILTER_RE, EXPLICIT_FILTER_RE):
        for match in pattern.finditer(text):
            column = _find_filter_column(match.group("column"), columns)
            if column is None:
                continue
            raw_value = match.group("value")
            value = _normalize_filter_value(raw_value, column, frame[column])
            if value is None:
                available = [
                    str(item)
                    for item in frame[column].map(_clean_cell).dropna().unique()[:12]
                ]
                raise WekaError(
                    f"提示词要求筛选「{column} = {_strip_filter_value(raw_value)}」，"
                    f"但该列没有这个值。可选值：{'、'.join(available)}"
                )
            return column, raw_value, value

    for match in GENDER_ONLY_RE.finditer(text):
        column = _find_filter_column("性别", columns)
        if column is None:
            continue
        raw_value = match.group("value")
        value = _normalize_filter_value(raw_value, column, frame[column])
        if value is None:
            continue
        return column, raw_value, value
    return None


def _apply_row_filter(
    frame: pd.DataFrame, prompt: str
) -> tuple[pd.DataFrame, dict[str, Any] | None, str | None]:
    spec = _extract_row_filter(prompt, frame)
    if spec is None:
        return frame, None, None
    column, raw_value, value = spec
    source_rows = int(len(frame))
    series = frame[column].map(_clean_cell)
    mask = series.map(
        lambda item: item is not None and _norm_filter_token(item) == _norm_filter_token(value)
    )
    filtered = frame.loc[mask].reset_index(drop=True)
    if filtered.empty:
        raise WekaError(
            f"按「{column} = {value}」筛选后没有数据行，请检查提示词或数据内容。"
        )
    filter_info = {
        "column": column,
        "value": value,
        "source_rows": source_rows,
        "matched_rows": int(len(filtered)),
    }
    note = (
        f"根据提示词筛选「{column} = {value}」："
        f"{source_rows} → {len(filtered)} 行。"
    )
    return filtered, filter_info, note


def _expand_basket(frame: pd.DataFrame) -> tuple[pd.DataFrame, str | None]:
    """Turn a transaction/items table into a 0/1 basket when the shape is obvious."""
    if frame.empty:
        return frame, None
    lower_names = {str(column).strip().lower(): column for column in frame.columns}
    item_column = next((lower_names[name] for name in ITEM_COLUMN_NAMES if name in lower_names), None)
    if item_column is None:
        return frame, None

    other_columns = [column for column in frame.columns if column != item_column]
    if len(other_columns) > 1:
        return frame, None

    transactions: list[list[str]] = []
    for raw in frame[item_column].tolist():
        text = _clean_cell(raw)
        if not text:
            transactions.append([])
            continue
        parts = re.split(r"\s*[;,|]\s*", text) if re.search(r"[;,|]", text) else [text]
        items = [part.strip() for part in parts if part.strip()]
        transactions.append(items)

    vocabulary: list[str] = []
    seen: set[str] = set()
    for items in transactions:
        for item in items:
            if item not in seen:
                seen.add(item)
                vocabulary.append(item)

    if not vocabulary:
        return frame, None

    rows = []
    for items in transactions:
        present = set(items)
        rows.append({item: (1 if item in present else 0) for item in vocabulary})
    expanded = pd.DataFrame(rows, columns=vocabulary)
    note = f"检测到交易明细列「{item_column}」，已展开为 {len(vocabulary)} 个 0/1 商品列。"
    return expanded, note


def _safe_attribute_name(name: str, index: int) -> str:
    safe = re.sub(r"[^0-9A-Za-z_]+", "_", name).strip("_")
    if not safe:
        safe = f"attr_{index}"
    if safe[0].isdigit():
        safe = f"a_{safe}"
    return safe


def _quote_arff(value: str) -> str:
    text = value.replace("\r", " ").replace("\n", " ").replace("\\", "\\\\").replace("'", "\\'")
    return f"'{text}'"


def _is_numeric(series: pd.Series) -> bool:
    non_null = series.dropna()
    if non_null.empty:
        return False
    converted = pd.to_numeric(non_null, errors="coerce")
    return bool(converted.notna().all())


def _number_label(value: str) -> str:
    try:
        number = float(value)
    except ValueError:
        return value
    if number.is_integer():
        return str(int(number))
    return f"{number:g}"


def _looks_like_identifier(name: str, series: pd.Series) -> bool:
    normalized = name.strip().lower()
    if normalized in ID_COLUMN_NAMES:
        return True
    if any(token in normalized for token in ("编号", "序号")):
        return series.nunique(dropna=True) > 1
    if normalized.endswith(("_id", "transactionid", "orderid")):
        return series.nunique(dropna=True) > 1
    if re.search(r"(^|[_\s-])(id|tid|uuid)($|[_\s-])", normalized):
        return series.nunique(dropna=True) > 1
    return False


def _write_arff(
    frame: pd.DataFrame,
    arff_path: Path,
    relation: str,
) -> tuple[
    list[dict[str, str]],
    dict[str, str],
    list[str],
    list[str],
    list[dict[str, str]],
    list[str],
]:
    """Write a clean ARFF file and return column metadata."""
    used_attributes: list[dict[str, str]] = []
    attr_map: dict[str, str] = {}
    numeric_columns: list[str] = []
    nominal_columns: list[str] = []
    dropped_columns: list[dict[str, str]] = []
    binary_attrs: list[str] = []
    used_names: set[str] = set()

    binary_candidates: set[str] = set()
    for original_name in frame.columns:
        values = frame[original_name].dropna().astype(str).str.strip()
        if set(values.unique().tolist()) == {"0", "1"}:
            binary_candidates.add(str(original_name))
    basket_mode = len(binary_candidates) >= 3 and len(binary_candidates) / max(len(frame.columns), 1) >= 0.5

    for index, original_name in enumerate(frame.columns):
        name = str(original_name).strip()
        series = frame[original_name]
        non_null = series.dropna()

        if non_null.empty:
            dropped_columns.append({"name": name, "reason": "整列缺失"})
            continue
        if non_null.nunique() <= 1:
            missing_count = len(frame) - len(non_null)
            single_value = str(non_null.iloc[0]).strip()
            negative_values = {"0", "false", "no", "n", "否", "无", "none"}
            if missing_count == 0:
                dropped_columns.append({"name": name, "reason": "常量列，不提供区分信息"})
                continue
            if single_value.lower() in negative_values:
                dropped_columns.append(
                    {"name": name, "reason": "仅有 0/否 且存在缺失，无法形成有效正项"}
                )
                continue
            # A single non-missing value plus missing cells is still informative:
            # e.g. a survey column {是} where "?" means "did not select".
            # Keep it as a one-level nominal attribute so Weka can mine "col=是".
        if _looks_like_identifier(name, series):
            dropped_columns.append({"name": name, "reason": "疑似 ID 列，已排除"})
            continue

        safe = _safe_attribute_name(name, index + 1)
        base = safe
        suffix = 2
        while safe.lower() in used_names:
            safe = f"{base}_{suffix}"
            suffix += 1
        used_names.add(safe.lower())

        kind = "numeric" if _is_numeric(series) else "nominal"
        values = non_null.astype(str).str.strip()
        unique_count = values.nunique()

        if kind == "numeric":
            if unique_count <= 6:
                kind = "nominal"

        if kind == "nominal":
            levels = sorted({_number_label(v) for v in values.tolist()})
            if len(levels) > 40:
                if len(levels) / max(len(frame), 1) > 0.8:
                    dropped_columns.append({"name": name, "reason": "高基数文本列，疑似 ID"})
                    used_names.discard(safe.lower())
                    continue
                keep = set(levels[:39])
                dropped_columns.append(
                    {"name": name, "reason": f"类别过多，仅保留前 39 类（原 {len(levels)} 类）"}
                )
                levels = sorted(keep) + ["__OTHER__"]
            if not levels:
                dropped_columns.append({"name": name, "reason": "没有可用取值"})
                used_names.discard(safe.lower())
                continue
            used_attributes.append({"safe": safe, "original": name, "kind": "nominal", "levels": levels})
            nominal_columns.append(name)
            attr_map[safe] = name
            if basket_mode and set(levels) == {"0", "1"}:
                binary_attrs.append(safe)
        else:
            used_attributes.append({"safe": safe, "original": name, "kind": "numeric"})
            numeric_columns.append(name)
            attr_map[safe] = name

    if not used_attributes:
        raise WekaError("清洗后没有可用于 Apriori 的列，请检查文件内容。")
    if len(used_attributes) < 2:
        raise WekaError(
            "清洗后只剩 1 个有效属性，Weka Apriori 至少需要 2 个属性。"
            "请检查是否误把存在型列当成了常量列，或补充数据维度。"
        )

    mapping = {entry["safe"]: entry for entry in used_attributes}
    lines = [f"@relation {_quote_arff(relation)}", ""]
    for entry in used_attributes:
        if entry["kind"] == "nominal":
            levels = ",".join(_quote_arff(level) for level in entry["levels"])
            lines.append(f"@attribute {entry['safe']} {{{levels}}}")
        else:
            lines.append(f"@attribute {entry['safe']} numeric")
    lines.extend(["", "@data"])

    for _, row in frame.iterrows():
        cells: list[str] = []
        for entry in used_attributes:
            raw = row.get(entry["original"])
            text = _clean_cell(raw)
            if text is None:
                cells.append("?")
                continue
            if entry["kind"] == "numeric":
                try:
                    cells.append(f"{float(text):.15g}")
                except ValueError:
                    cells.append("?")
            else:
                label = _number_label(text)
                if entry["safe"] in binary_attrs:
                    # For 0/1 basket columns, "absent" is not an item: treat it as
                    # missing so Apriori only mines item presence, not item=0.
                    if label == "0":
                        cells.append("?")
                        continue
                    if label == "1":
                        cells.append(_quote_arff("1"))
                        continue
                if label not in entry["levels"]:
                    label = "__OTHER__"
                cells.append(_quote_arff(label))
        lines.append(",".join(cells))

    arff_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return (
        used_attributes,
        attr_map,
        numeric_columns,
        nominal_columns,
        dropped_columns,
        binary_attrs,
    )


def _count_arff_rows(arff_path: Path) -> int:
    in_data = False
    count = 0
    for line in arff_path.read_text(encoding="utf-8", errors="replace").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("%"):
            continue
        if stripped.lower().startswith("@data"):
            in_data = True
            continue
        if in_data:
            count += 1
    return count


def prepare_dataset(
    input_path: Path, bins: int = DEFAULT_BINS, prompt: str = ""
) -> dict[str, Any]:
    """Convert CSV/ARFF into a Weka-friendly ARFF and return a dataset profile."""
    suffix = input_path.suffix.lower()
    if suffix not in {".csv", ".arff"}:
        raise WekaError("只支持 .csv 或 .arff 文件。")

    work_dir = WORK_DIR / uuid.uuid4().hex
    work_dir.mkdir(parents=True, exist_ok=True)
    notes: list[str] = []

    if suffix == ".arff":
        csv_path = work_dir / "source.csv"
        _arff_to_csv(input_path, csv_path)
        frame = _read_csv(csv_path)
    else:
        frame = _read_csv(input_path)

    if frame.empty:
        raise WekaError("文件没有可读取的数据行。")
    frame.columns = [str(column).strip() for column in frame.columns]
    source_rows = int(len(frame))

    frame, filter_info, _ = _apply_row_filter(frame, prompt)

    frame, basket_note = _expand_basket(frame)
    if basket_note:
        notes.append(basket_note)

    for column in frame.columns:
        frame[column] = frame[column].map(lambda value: _clean_cell(value))

    relation = re.sub(r"[^0-9A-Za-z_]+", "_", input_path.stem) or "dataset"
    prepared = work_dir / "prepared.arff"
    attributes, attr_map, numeric_columns, nominal_columns, dropped, binary_attrs = _write_arff(
        frame, prepared, relation
    )
    if filter_info:
        for item in dropped:
            if item["name"] == filter_info["column"] and "常量" in item["reason"]:
                item["reason"] = "筛选后成为常量列，不再作为 Apriori 属性"
    if binary_attrs:
        notes.append(
            f"{len(binary_attrs)} 个 0/1 列按「商品存在」处理，0 视为缺失，只挖掘 1 的组合。"
        )

    final_arff = prepared
    if numeric_columns:
        discretized = work_dir / "discretized.arff"
        result = _run_java(
            [
                "weka.filters.unsupervised.attribute.Discretize",
                "-B",
                str(max(2, min(int(bins), 20))),
                "-i",
                str(prepared),
                "-o",
                str(discretized),
            ]
        )
        if result.returncode != 0 or not discretized.exists():
            detail = (result.stderr or result.stdout).strip()
            raise WekaError(f"数值属性离散化失败：{detail[:800]}")
        final_arff = discretized
        notes.append(
            f"数值列 {', '.join(numeric_columns)} 已按等宽分箱离散化为 {bins} 箱后挖掘。"
        )

    total_rows = _count_arff_rows(final_arff)
    if total_rows <= 0:
        raise WekaError("清洗后没有有效交易行。")

    preview_frame = frame.head(5)
    preview = []
    for _, row in preview_frame.iterrows():
        preview.append({str(k): _clean_cell(v) for k, v in row.items()})

    profile = {
        "source_format": suffix.lstrip("."),
        "rows": total_rows,
        "source_rows": source_rows,
        "columns": len(attributes),
        "used_columns": [entry["original"] for entry in attributes],
        "numeric_columns": numeric_columns,
        "nominal_columns": nominal_columns,
        "binary_columns": [attr_map[key] for key in binary_attrs],
        "filter": filter_info,
        "dropped_columns": dropped,
        "notes": notes,
        "preview": preview,
    }
    return {
        "work_dir": work_dir,
        "arff_path": final_arff,
        "profile": profile,
        "attr_map": attr_map,
        "binary_attrs": set(binary_attrs),
        "total_instances": total_rows,
    }


def _unquote_arff_value(value: str) -> str:
    text = value.strip()
    if len(text) >= 2 and text[0] == "'" and text[-1] == "'":
        text = text[1:-1]
    return text.replace("\\'", "'").replace("\\\\", "\\").strip()


def _display_item(raw_item: str, attr_map: dict[str, str], binary_attrs: set[str] | None = None) -> str:
    attribute, separator, value = raw_item.partition("=")
    if not separator:
        return _unquote_arff_value(raw_item)
    original_attribute = attr_map.get(attribute, attribute)
    unquoted_value = _unquote_arff_value(value)
    if binary_attrs and attribute in binary_attrs and unquoted_value == "1":
        return original_attribute
    return f"{original_attribute} = {unquoted_value}"


def _split_items(text: str, known_items: list[str]) -> list[str]:
    """Split Weka's space-joined item text using the known L(1) vocabulary."""
    text = text.strip()
    if not text:
        return []
    ordered = sorted(known_items, key=len, reverse=True)
    result: list[str] = []
    position = 0
    while position < len(text):
        while position < len(text) and text[position].isspace():
            position += 1
        if position >= len(text):
            break
        for item in ordered:
            end = position + len(item)
            if text.startswith(item, position) and (end == len(text) or text[end].isspace()):
                result.append(item)
                position = end
                break
        else:
            # Unknown token: keep the remainder as one item rather than losing data.
            remainder = text[position:].strip()
            if remainder:
                result.append(remainder)
            break
    return result


def _to_float(value: str, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


ITEMS_HEADER_RE = re.compile(r"Large Itemsets L\((\d+)\):")
ITEMS_COUNT_RE = re.compile(r"^(?P<items>.*\S)\s+(?P<count>\d+)\s*$")
RULE_RE = re.compile(
    r"^\s*\d+\.\s+(?P<lhs>.+?)\s+(?P<lhs_count>\d+)\s+==>\s+"
    r"(?P<rhs>.+?)\s+(?P<rhs_count>\d+)\s+"
    r"<conf:\((?P<confidence>[^)]+)\)>\s+"
    r"lift:\((?P<lift>[^)]+)\)\s+"
    r"lev:\((?P<leverage>[^)]+)\)\s+\[(?P<leverage_count>-?\d+)\]\s+"
    r"conv:\((?P<conviction>[^)]+)\)"
)


def parse_apriori_output(
    output: str,
    total_instances: int,
    attr_map: dict[str, str] | None = None,
    binary_attrs: set[str] | None = None,
) -> dict[str, Any]:
    """Parse Weka's Apriori console output into itemsets and association rules."""
    attr_map = attr_map or {}
    binary_attrs = binary_attrs or set()
    total_instances = max(int(total_instances), 1)

    reported_support = None
    reported_confidence = None
    cycles = None
    itemsets_raw: dict[tuple[str, ...], int] = {}
    rules: list[dict[str, Any]] = []
    known_items: list[str] = []
    current_size: int | None = None
    in_rules = False

    for line in output.splitlines():
        support_match = re.match(r"Minimum support:\s*([0-9.]+)\s*\((\d+) instances\)", line)
        if support_match:
            reported_support = float(support_match.group(1))
            continue
        confidence_match = re.match(r"Minimum metric <confidence>:\s*([0-9.]+)", line)
        if confidence_match:
            reported_confidence = float(confidence_match.group(1))
            continue
        cycle_match = re.match(r"Number of cycles performed:\s*(\d+)", line)
        if cycle_match:
            cycles = int(cycle_match.group(1))
            continue
        if "Best rules found:" in line:
            in_rules = True
            current_size = None
            continue

        header_match = ITEMS_HEADER_RE.search(line)
        if header_match:
            current_size = int(header_match.group(1))
            continue
        if line.strip().startswith("Size of set of large itemsets"):
            continue

        if current_size is not None and not in_rules:
            count_match = ITEMS_COUNT_RE.match(line)
            if count_match:
                item_text = count_match.group("items").strip()
                count = int(count_match.group("count"))
                if current_size == 1:
                    raw_items = [item_text]
                    known_items.append(item_text)
                else:
                    raw_items = _split_items(item_text, known_items)
                key = tuple(raw_items)
                itemsets_raw[key] = max(itemsets_raw.get(key, 0), count)
            continue

        if in_rules:
            rule_match = RULE_RE.match(line)
            if not rule_match:
                continue
            lhs_items = _split_items(rule_match.group("lhs"), known_items)
            rhs_items = _split_items(rule_match.group("rhs"), known_items)
            if not lhs_items or not rhs_items:
                continue
            lhs_count = int(rule_match.group("lhs_count"))
            rhs_count = int(rule_match.group("rhs_count"))
            confidence = _to_float(rule_match.group("confidence"))
            union_key = tuple(dict.fromkeys(lhs_items + rhs_items))
            union_count = itemsets_raw.get(union_key)
            if union_count is None:
                union_count = round(lhs_count * confidence)
            rule = {
                "antecedent": [_display_item(item, attr_map, binary_attrs) for item in lhs_items],
                "consequent": [_display_item(item, attr_map, binary_attrs) for item in rhs_items],
                "support": union_count / total_instances,
                "support_count": union_count,
                "antecedent_support": lhs_count / total_instances,
                "consequent_support": rhs_count / total_instances,
                "confidence": confidence,
                "lift": _to_float(rule_match.group("lift")),
                "leverage": _to_float(rule_match.group("leverage")),
                "conviction": (
                    None
                    if rule_match.group("conviction").strip().lower() in {"inf", "infinity"}
                    else _to_float(rule_match.group("conviction"))
                ),
                "display": (
                    " + ".join(_display_item(item, attr_map, binary_attrs) for item in lhs_items)
                    + "  ⇒  "
                    + " + ".join(_display_item(item, attr_map, binary_attrs) for item in rhs_items)
                ),
            }
            rules.append(rule)

    itemsets = []
    for key, count in itemsets_raw.items():
        itemsets.append(
            {
                "items": [_display_item(item, attr_map, binary_attrs) for item in key],
                "size": len(key),
                "count": count,
                "support": count / total_instances,
                "display": " + ".join(_display_item(item, attr_map, binary_attrs) for item in key),
            }
        )
    itemsets.sort(key=lambda entry: (entry["size"], entry["support"]), reverse=True)

    return {
        "itemsets": itemsets,
        "rules": rules,
        "total_instances": total_instances,
        "reported_min_support": reported_support,
        "reported_min_confidence": reported_confidence,
        "cycles": cycles,
        "single_item_count": len(known_items),
    }


def run_apriori(
    arff_path: Path,
    total_instances: int,
    attr_map: dict[str, str],
    min_support: float = 0.1,
    min_confidence: float = 0.5,
    max_rules: int = 30,
    binary_attrs: set[str] | None = None,
) -> dict[str, Any]:
    """Run Weka Apriori and return parsed, display-ready results."""
    min_support = max(0.001, min(float(min_support), 1.0))
    min_confidence = max(0.0, min(float(min_confidence), 1.0))
    max_rules = max(1, min(int(max_rules), 100))

    command = [
        "weka.associations.Apriori",
        "-I",
        "-N",
        "20000",
        "-C",
        f"{min_confidence:g}",
        "-M",
        f"{min_support:g}",
        "-D",
        "0.05",
        "-t",
        str(arff_path),
    ]
    result = _run_java(command, timeout=WEKA_TIMEOUT_SECONDS)
    output = (result.stdout or "") + "\n" + (result.stderr or "")
    log_dir = arff_path.parent
    (log_dir / "weka-apriori.log").write_text(output, encoding="utf-8", errors="replace")

    if "Cannot handle numeric" in output:
        raise WekaError("Weka 仍然遇到数值属性，离散化没有生效，请更换数据文件。")
    if "java.lang." in output and "Exception" in output and "Best rules found" not in output:
        summary = output.strip().splitlines()
        tail = "\n".join(summary[-12:])
        raise WekaError(f"Weka Apriori 运行失败：{tail[:1200]}")

    parsed = parse_apriori_output(output, total_instances, attr_map, binary_attrs)
    if not parsed["itemsets"] and "Best rules found" not in output:
        raise WekaError(
            "Weka 没有返回可解析的频繁项集。请提高最小支持度，或检查数据是否全是唯一值。"
        )

    all_itemsets = parsed["itemsets"]
    all_rules = parsed["rules"]
    all_rules.sort(key=lambda entry: (entry["confidence"], entry["lift"]), reverse=True)

    return {
        "engine": "Weka 3.8.7 · weka.associations.Apriori",
        "weka_jar": str(WEKA_JAR),
        "parameters": {
            "min_support": min_support,
            "min_confidence": min_confidence,
            "max_rules": max_rules,
            "num_bins": None,
        },
        "total_instances": total_instances,
        "reported_min_support": parsed["reported_min_support"],
        "reported_min_confidence": parsed["reported_min_confidence"],
        "cycles": parsed["cycles"],
        "single_item_count": parsed["single_item_count"],
        "itemset_total": len(all_itemsets),
        "rule_total": len(all_rules),
        "itemsets": all_itemsets[:300],
        "rules": all_rules[:max_rules],
        "log_path": str(log_dir / "weka-apriori.log"),
    }


def read_source_preview(path: Path, limit: int = 20) -> list[dict[str, Any]]:
    """Small helper used by tests / diagnostics."""
    suffix = path.suffix.lower()
    if suffix == ".arff":
        temp_csv = WORK_DIR / f"preview-{uuid.uuid4().hex}.csv"
        _arff_to_csv(path, temp_csv)
        frame = _read_csv(temp_csv)
        temp_csv.unlink(missing_ok=True)
    else:
        frame = _read_csv(path)
    return frame.head(limit).to_dict(orient="records")