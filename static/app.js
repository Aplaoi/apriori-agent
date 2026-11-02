(() => {
  "use strict";

  const $ = (selector) => document.querySelector(selector);
  const $$ = (selector) => Array.from(document.querySelectorAll(selector));

  const state = {
    file: null,
    result: null,
    activeTab: "itemsets",
    loading: false,
    progressTimers: [],
  };

  const elements = {
    engineStatus: $("#engineStatus"),
    engineStatusText: $("#engineStatusText"),
    dropzone: $("#dropzone"),
    fileInput: $("#fileInput"),
    dropCopy: $("#dropCopy"),
    fileChip: $("#fileChip"),
    fileName: $("#fileName"),
    fileSize: $("#fileSize"),
    fileClear: $("#fileClear"),
    demoButton: $("#demoButton"),
    promptInput: $("#promptInput"),
    runButton: $("#runButton"),
    runButtonLabel: $(".run-button-label"),
    progress: $("#progress"),
    progressText: $("#progressText"),
    emptyState: $("#emptyState"),
    errorBox: $("#errorBox"),
    errorText: $("#errorText"),
    resultsWrap: $("#resultsWrap"),
    answerBody: $("#answerBody"),
    modeBadge: $("#modeBadge"),
    toolTrace: $("#toolTrace"),
    warningBox: $("#warningBox"),
    statsGrid: $("#statsGrid"),
    itemsetCount: $("#itemsetCount"),
    ruleCount: $("#ruleCount"),
    itemsetMeta: $("#itemsetMeta"),
    ruleMeta: $("#ruleMeta"),
    itemsetBody: $("#itemsetBody"),
    ruleBody: $("#ruleBody"),
    itemsetChart: $("#itemsetChart"),
    ruleChart: $("#ruleChart"),
    downloadButton: $("#downloadButton"),
    datasetDetails: $("#datasetDetails"),
    datasetBody: $("#datasetBody"),
    resultFootnote: $("#resultFootnote"),
  };

  function escapeHtml(value) {
    return String(value ?? "")
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;")
      .replace(/'/g, "&#039;");
  }

  function formatNumber(value) {
    const number = Number(value);
    return Number.isFinite(number) ? new Intl.NumberFormat("zh-CN").format(number) : "—";
  }

  function formatPct(value, digits = 1) {
    const number = Number(value);
    return Number.isFinite(number) ? `${(number * 100).toFixed(digits)}%` : "—";
  }

  function formatBytes(bytes) {
    if (!Number.isFinite(bytes) || bytes <= 0) return "0 B";
    if (bytes < 1024) return `${bytes} B`;
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
    return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
  }

  function inlineMarkdown(text) {
    return escapeHtml(text)
      .replace(/`([^`]+)`/g, "<code>$1</code>")
      .replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>")
      .replace(/(^|[^*])\*([^*]+)\*/g, "$1<em>$2</em>");
  }

  function renderMarkdown(text) {
    const lines = String(text || "").replace(/\r\n/g, "\n").split("\n");
    const output = [];
    let listType = null;

    const closeList = () => {
      if (listType) {
        output.push(`</${listType}>`);
        listType = null;
      }
    };

    for (const rawLine of lines) {
      const line = rawLine.trim();
      if (!line) {
        closeList();
        continue;
      }
      const heading = line.match(/^#{1,4}\s+(.+)$/);
      if (heading) {
        closeList();
        output.push(`<h3>${inlineMarkdown(heading[1])}</h3>`);
        continue;
      }
      const unordered = line.match(/^[-*]\s+(.+)$/);
      if (unordered) {
        if (listType !== "ul") {
          closeList();
          output.push("<ul>");
          listType = "ul";
        }
        output.push(`<li>${inlineMarkdown(unordered[1])}</li>`);
        continue;
      }
      const ordered = line.match(/^\d+\.\s+(.+)$/);
      if (ordered) {
        if (listType !== "ol") {
          closeList();
          output.push("<ol>");
          listType = "ol";
        }
        output.push(`<li>${inlineMarkdown(ordered[1])}</li>`);
        continue;
      }
      closeList();
      output.push(`<p>${inlineMarkdown(line)}</p>`);
    }
    closeList();
    return output.join("");
  }

  function setFile(file) {
    if (!file) return;
    const extension = file.name.split(".").pop().toLowerCase();
    if (!["csv", "arff"].includes(extension)) {
      showError("只支持 .csv 或 .arff 文件，请重新选择。");
      return;
    }
    if (file.size > 20 * 1024 * 1024) {
      showError("文件超过 20 MB 限制，请换一个更小的文件。");
      return;
    }
    state.file = file;
    elements.fileName.textContent = file.name;
    elements.fileSize.textContent = formatBytes(file.size);
    elements.dropCopy.hidden = true;
    elements.fileChip.hidden = false;
    elements.fileChip.querySelector(".file-chip-icon").textContent = extension.toUpperCase();
    clearError();
  }

  function clearFile() {
    state.file = null;
    elements.fileInput.value = "";
    elements.fileChip.hidden = true;
    elements.dropCopy.hidden = false;
  }

  function showError(message) {
    elements.errorText.textContent = message;
    elements.errorBox.hidden = false;
    if (!state.result) {
      elements.emptyState.hidden = true;
      elements.resultsWrap.hidden = true;
    }
  }

  function clearError() {
    elements.errorBox.hidden = true;
    if (!state.result) {
      elements.resultsWrap.hidden = true;
      elements.emptyState.hidden = false;
    }
  }

  function setLoading(loading) {
    state.loading = loading;
    elements.runButton.disabled = loading;
    elements.demoButton.disabled = loading;
    elements.runButtonLabel.textContent = loading ? "正在挖掘…" : "开始挖掘";
    elements.progress.hidden = !loading;
    state.progressTimers.forEach((timer) => window.clearTimeout(timer));
    state.progressTimers = [];
    if (!loading) {
      elements.progressText.textContent = "正在解析数据…";
      return;
    }
    const messages = [
      "正在解析数据、清洗列并离散化数值…",
      "正在调用 Weka Apriori 搜索频繁项集…",
      "DeepSeek 正在解读结果，Matplotlib 正在绘图…",
    ];
    elements.progressText.textContent = messages[0];
    messages.slice(1).forEach((message, index) => {
      state.progressTimers.push(
        window.setTimeout(() => {
          elements.progressText.textContent = message;
        }, (index + 1) * 2800)
      );
    });
  }

  async function checkHealth() {
    try {
      const response = await fetch("/api/health");
      const data = await response.json();
      const ready = Boolean(data.weka_ready && data.deepseek_configured);
      elements.engineStatus.classList.toggle("is-ready", ready);
      elements.engineStatus.classList.toggle("is-error", !ready);
      if (!data.weka_ready) {
        elements.engineStatusText.textContent = "Weka 未就绪 — 检查 .env";
      } else if (!data.deepseek_configured) {
        elements.engineStatusText.textContent = "DeepSeek Key 未配置";
      } else {
        elements.engineStatusText.textContent = `Weka 已就绪 · ${data.deepseek_model}`;
      }
      elements.engineStatus.title = data.weka_jar || "";
    } catch {
      elements.engineStatus.classList.add("is-error");
      elements.engineStatusText.textContent = "无法连接本地服务";
    }
  }

  function renderToolTrace(trace) {
    elements.toolTrace.replaceChildren();
    if (!Array.isArray(trace) || trace.length === 0) {
      elements.toolTrace.hidden = true;
      return;
    }
    elements.toolTrace.hidden = false;
    const label = document.createElement("span");
    label.className = "tool-trace-label";
    label.textContent = "Tool calls";
    elements.toolTrace.appendChild(label);

    trace.forEach((call) => {
      const chip = document.createElement("span");
      chip.className = "tool-chip";
      if (call.status === "error") {
        chip.classList.add("is-error");
        chip.textContent = `${call.name} · error`;
      } else {
        const parameters = call.parameters || {};
        const support = Number(parameters.min_support);
        const confidence = Number(parameters.min_confidence);
        chip.textContent =
          `${call.name} · support ${Number.isFinite(support) ? formatPct(support, 0) : "—"}` +
          ` · conf ${Number.isFinite(confidence) ? formatPct(confidence, 0) : "—"}` +
          ` · ${formatNumber(call.itemset_total)} sets / ${formatNumber(call.rule_total)} rules`;
      }
      chip.title = JSON.stringify(call, null, 2);
      elements.toolTrace.appendChild(chip);
    });
  }

  function createStat(value, label, suffix = "") {
    const card = document.createElement("div");
    card.className = "stat-card";
    const valueNode = document.createElement("div");
    valueNode.className = "stat-value";
    valueNode.textContent = formatNumber(value);
    if (suffix) {
      const suffixNode = document.createElement("small");
      suffixNode.textContent = suffix;
      valueNode.appendChild(suffixNode);
    }
    const labelNode = document.createElement("div");
    labelNode.className = "stat-label";
    labelNode.textContent = label;
    card.append(valueNode, labelNode);
    return card;
  }

  function renderStats(data) {
    elements.statsGrid.replaceChildren(
      createStat(data.total_instances, "transactions"),
      createStat(data.itemset_total, "frequent itemsets"),
      createStat(data.rule_total, "association rules"),
      createStat(data.elapsed_seconds, "elapsed", "s")
    );
  }

  function supportCell(value, className = "") {
    const cell = document.createElement("td");
    const wrapper = document.createElement("div");
    wrapper.className = "support-cell";
    const valueNode = document.createElement("span");
    valueNode.className = "support-value";
    valueNode.textContent = formatPct(value);
    const track = document.createElement("div");
    track.className = "support-track";
    const fill = document.createElement("div");
    fill.className = `support-fill ${className}`.trim();
    fill.style.width = `${Math.max(2, Math.min(100, Number(value) * 100))}%`;
    track.appendChild(fill);
    wrapper.append(valueNode, track);
    cell.appendChild(wrapper);
    return cell;
  }

  function renderItemsets(itemsets, total, parameters) {
    elements.itemsetCount.textContent = total ? `(${formatNumber(total)})` : "";
    elements.itemsetMeta.textContent =
      `显示前 ${formatNumber(itemsets.length)} 个，共 ${formatNumber(total)} 个 · ` +
      `最小支持度 ${formatPct(parameters.min_support)} · 按项集大小与支持度排序`;
    elements.itemsetBody.replaceChildren();

    if (!itemsets.length) {
      const row = document.createElement("tr");
      const cell = document.createElement("td");
      cell.colSpan = 5;
      cell.className = "empty-table";
      cell.textContent = "当前参数下没有找到频繁项集，可尝试降低最小支持度。";
      row.appendChild(cell);
      elements.itemsetBody.appendChild(row);
      return;
    }

    itemsets.forEach((itemset, index) => {
      const row = document.createElement("tr");

      const rank = document.createElement("td");
      rank.className = "rank-col";
      rank.textContent = String(index + 1).padStart(2, "0");

      const items = document.createElement("td");
      const itemsText = document.createElement("div");
      itemsText.className = "itemset-text";
      itemsText.textContent = itemset.display;
      items.appendChild(itemsText);

      const size = document.createElement("td");
      size.className = "num-col";
      size.textContent = String(itemset.size);

      const support = supportCell(itemset.support);

      const count = document.createElement("td");
      count.className = "num-col";
      count.textContent = formatNumber(itemset.count);

      row.append(rank, items, size, support, count);
      elements.itemsetBody.appendChild(row);
    });
  }

  function renderRules(rules, total, parameters) {
    elements.ruleCount.textContent = total ? `(${formatNumber(total)})` : "";
    elements.ruleMeta.textContent =
      `显示前 ${formatNumber(rules.length)} 条，共 ${formatNumber(total)} 条 · ` +
      `最小置信度 ${formatPct(parameters.min_confidence)} · 按置信度与提升度排序`;
    elements.ruleBody.replaceChildren();

    if (!rules.length) {
      const row = document.createElement("tr");
      const cell = document.createElement("td");
      cell.colSpan = 5;
      cell.className = "empty-table";
      cell.textContent = "当前参数下没有找到关联规则，可尝试降低最小置信度或最小支持度。";
      row.appendChild(cell);
      elements.ruleBody.appendChild(row);
      return;
    }

    rules.forEach((rule, index) => {
      const row = document.createElement("tr");

      const rank = document.createElement("td");
      rank.className = "rank-col";
      rank.textContent = String(index + 1).padStart(2, "0");

      const ruleCell = document.createElement("td");
      const ruleText = document.createElement("div");
      ruleText.className = "rule-text";
      const antecedent = document.createElement("span");
      antecedent.textContent = rule.antecedent.join(" + ");
      const arrow = document.createElement("span");
      arrow.className = "rule-arrow";
      arrow.textContent = "⇒";
      const consequent = document.createElement("span");
      consequent.textContent = rule.consequent.join(" + ");
      ruleText.append(antecedent, arrow, consequent);
      ruleCell.appendChild(ruleText);

      const support = supportCell(rule.support);
      const confidence = supportCell(rule.confidence, "is-confidence");

      const lift = document.createElement("td");
      lift.className = "num-col";
      const liftText = document.createElement("span");
      liftText.className = `lift-value ${Number(rule.lift) > 1.2 ? "is-strong" : ""}`.trim();
      liftText.textContent = Number(rule.lift).toFixed(2);
      lift.appendChild(liftText);

      row.append(rank, ruleCell, support, confidence, lift);
      elements.ruleBody.appendChild(row);
    });
  }

  function addDatasetLine(parent, text, className = "") {
    const line = document.createElement("div");
    if (className) line.className = className;
    line.textContent = text;
    parent.appendChild(line);
  }

  function renderDataset(dataset) {
    elements.datasetBody.replaceChildren();
    addDatasetLine(
      elements.datasetBody,
      `${dataset.filename || "数据集"} · ${formatNumber(dataset.rows)} 行 × ${formatNumber(dataset.columns)} 列`
    );
    if (dataset.used_columns?.length) {
      addDatasetLine(elements.datasetBody, `参与挖掘：${dataset.used_columns.join("、")}`);
    }
    if (dataset.filter) {
      addDatasetLine(
        elements.datasetBody,
        `筛选条件：${dataset.filter.column} = ${dataset.filter.value}（${formatNumber(dataset.filter.source_rows)} → ${formatNumber(dataset.filter.matched_rows)} 行）`,
        "muted"
      );
    }
    if (dataset.numeric_columns?.length) {
      addDatasetLine(elements.datasetBody, `数值列（已离散化）：${dataset.numeric_columns.join("、")}`);
    }
    if (dataset.binary_columns?.length) {
      addDatasetLine(elements.datasetBody, `0/1 商品列：${dataset.binary_columns.join("、")}`);
    }
    if (dataset.dropped_columns?.length) {
      addDatasetLine(elements.datasetBody, "排除的列：");
      const list = document.createElement("ul");
      dataset.dropped_columns.forEach((item) => {
        const entry = document.createElement("li");
        entry.textContent = `${item.name} — ${item.reason}`;
        list.appendChild(entry);
      });
      elements.datasetBody.appendChild(list);
    }
    if (dataset.notes?.length) {
      dataset.notes.forEach((note) => addDatasetLine(elements.datasetBody, note, "muted"));
    }
    elements.datasetDetails.hidden = false;
    if (dataset.filter) {
      elements.datasetDetails.open = true;
    }
  }

  function renderResult(data) {
    state.result = data;
    elements.emptyState.hidden = true;
    elements.resultsWrap.hidden = false;
    clearError();

    elements.answerBody.innerHTML = renderMarkdown(data.answer);
    elements.modeBadge.textContent =
      data.mode === "agent" ? "DeepSeek · 工具调用" : data.mode === "direct" ? "DeepSeek · 直接挖掘" : "本地回退摘要";
    elements.modeBadge.classList.toggle("is-fallback", data.mode !== "agent");

    renderToolTrace(data.tool_trace);
    renderStats(data);
    renderItemsets(data.itemsets || [], data.itemset_total || 0, data.parameters || {});
    renderRules(data.rules || [], data.rule_total || 0, data.parameters || {});
    renderDataset(data.dataset || {});

    if (data.warning) {
      elements.warningBox.hidden = false;
      elements.warningBox.textContent = `提示：${data.warning}`;
    } else {
      elements.warningBox.hidden = true;
    }

    const version = encodeURIComponent(data.job_id || Date.now());
    elements.itemsetChart.src = `${data.charts.itemsets}?v=${version}`;
    elements.ruleChart.src = `${data.charts.rules}?v=${version}`;

    const reportedSupport = Number(data.reported_min_support);
    const requestedSupport = Number(data.parameters?.min_support);
    let footnote =
      `所有指标由 ${data.engine} 计算；可视化由 Python Matplotlib 生成。` +
      ` 规则提升度 lift > 1 表示前件与后件正相关。`;
    if (Number.isFinite(reportedSupport) && Number.isFinite(requestedSupport) && Math.abs(reportedSupport - requestedSupport) > 0.005) {
      footnote += ` Weka 最终采用的最小支持度为 ${formatPct(reportedSupport)}（规则数量上限可能使其提前停止）。`;
    }
    elements.resultFootnote.textContent = footnote;

    switchTab("itemsets");
    if (window.innerWidth < 1080) {
      elements.resultsWrap.scrollIntoView({ behavior: "smooth", block: "start" });
    }
  }

  function switchTab(tabName) {
    state.activeTab = tabName;
    $$(".tab").forEach((tab) => {
      const active = tab.dataset.tab === tabName;
      tab.classList.toggle("is-active", active);
      tab.setAttribute("aria-selected", String(active));
    });
    $$(".tab-panel").forEach((panel) => {
      panel.classList.toggle("is-active", panel.id === `tab-${tabName}`);
    });
  }

  async function runAnalysis() {
    if (state.loading) return;
    if (!state.file) {
      showError("请先选择一个 CSV 或 ARFF 数据文件。");
      elements.dropzone.focus();
      return;
    }
    const prompt = elements.promptInput.value.trim() || "分析这份数据的频繁项集和关联规则，使用默认参数。";
    clearError();
    setLoading(true);

    const formData = new FormData();
    formData.append("prompt", prompt);
    formData.append("file", state.file);

    try {
      const response = await fetch("/api/analyze", { method: "POST", body: formData });
      let data = null;
      try {
        data = await response.json();
      } catch {
        throw new Error(`服务返回了无法解析的响应（HTTP ${response.status}）。`);
      }
      if (!response.ok) {
        throw new Error(data?.detail || `分析失败（HTTP ${response.status}）。`);
      }
      renderResult(data);
    } catch (error) {
      showError(error instanceof Error ? error.message : String(error));
    } finally {
      setLoading(false);
    }
  }

  function downloadJson() {
    if (!state.result) return;
    const blob = new Blob([JSON.stringify(state.result, null, 2)], {
      type: "application/json;charset=utf-8",
    });
    const url = URL.createObjectURL(blob);
    const anchor = document.createElement("a");
    anchor.href = url;
    anchor.download = `apriori-result-${new Date().toISOString().slice(0, 19).replace(/[:T]/g, "-")}.json`;
    document.body.appendChild(anchor);
    anchor.click();
    anchor.remove();
    URL.revokeObjectURL(url);
  }

  function bindEvents() {
    elements.dropzone.addEventListener("click", (event) => {
      if (event.target.closest(".file-clear")) return;
      elements.fileInput.click();
    });

    elements.dropzone.addEventListener("keydown", (event) => {
      if (event.key === "Enter" || event.key === " ") {
        event.preventDefault();
        elements.fileInput.click();
      }
    });

    elements.fileInput.addEventListener("change", () => {
      setFile(elements.fileInput.files?.[0]);
    });

    ["dragenter", "dragover"].forEach((eventName) => {
      elements.dropzone.addEventListener(eventName, (event) => {
        event.preventDefault();
        elements.dropzone.classList.add("is-dragging");
      });
    });

    ["dragleave", "drop"].forEach((eventName) => {
      elements.dropzone.addEventListener(eventName, (event) => {
        event.preventDefault();
        elements.dropzone.classList.remove("is-dragging");
      });
    });

    elements.dropzone.addEventListener("drop", (event) => {
      const file = event.dataTransfer?.files?.[0];
      if (file) setFile(file);
    });

    elements.fileClear.addEventListener("click", (event) => {
      event.stopPropagation();
      clearFile();
    });

    elements.demoButton.addEventListener("click", async () => {
      try {
        const response = await fetch("/api/demo");
        if (!response.ok) throw new Error("示例数据加载失败");
        const blob = await response.blob();
        const file = new File([blob], "llm.arff", { type: "text/plain" });
        setFile(file);
        elements.promptInput.value =
          "找最小支持度为0.4，最小置信度为0.7的频繁项集和强规则，只处理性别为女的数据";
      } catch (error) {
        showError(error instanceof Error ? error.message : String(error));
      }
    });

    $$(".prompt-chips button").forEach((button) => {
      button.addEventListener("click", () => {
        elements.promptInput.value = button.dataset.prompt || "";
        elements.promptInput.focus();
      });
    });

    elements.runButton.addEventListener("click", runAnalysis);
    elements.downloadButton.addEventListener("click", downloadJson);

    $$(".tab").forEach((tab) => {
      tab.addEventListener("click", () => switchTab(tab.dataset.tab));
    });

    elements.promptInput.addEventListener("keydown", (event) => {
      if ((event.ctrlKey || event.metaKey) && event.key === "Enter") {
        event.preventDefault();
        runAnalysis();
      }
    });
  }

  bindEvents();
  checkHealth();
})();