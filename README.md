# Apriori Agent · Weka × DeepSeek

用 DeepSeek 的 function calling 做一个 Agent，让它在收到自然语言要求后调用 **Weka Apriori**，对上传的 CSV / ARFF 做频繁项集与关联规则挖掘，再用 **Matplotlib** 生成图表，最后用中文解释结果。

## 运行环境

- Python 3.10+（本项目在 3.12 上验证）
- Java 21+
- Weka 3.8.7
- DeepSeek API Key

## 快速启动

```shell
python -m pip install -r requirements.txt

# 3. 配置密钥（仓库里只有 .env.example，.env 已被 .gitignore 排除）
cp .env.example .env
nano .env

# 4. 启动
uvicorn app:app --host 127.0.0.1 --port 8180
```

打开 <http://127.0.0.1:8180>，点击“载入示例 llm.arff”即可试跑；它会自动填充 `data/llm.arff` 和提示词“支持度 0.4、置信度 0.7，只处理性别为女的数据”。

> 如果 8180 也被占用，换成任意未被系统排除的端口，例如 `--port 8765`。Windows 的 `netsh interface ipv4 show excludedportrange protocol=tcp` 可以查看被保留的端口段。


## Agent 工作流

```text
用户提示词 + 文件
        │
        ▼
FastAPI /api/analyze
        │  CSV/ARFF → 清洗 → 提示词筛选 → ARFF →（数值列）Discretize
        ▼
DeepSeek chat/completions（tools 中提供 run_apriori）
        │  tool_calls: run_apriori(min_support, min_confidence, max_rules)
        ▼
java -cp weka.jar weka.associations.Apriori -I -N 20000 -C ... -M ... -t data.arff
        │  解析 Large Itemsets L(k) 与 Best rules found
        ▼
Matplotlib 绘制 itemsets.svg / rules.svg
        │
        ▼
工具结果回填给 DeepSeek → 中文总结 → 前端表格 + 图表
```

行筛选只支持“单列等值”条件；类似“年龄大于 30 且性别为女”的复合条件不在当前简化版范围内。筛选信息会写入返回的 `dataset.filter`，界面也会展示原始行数和筛选后行数。

关键代码：

- `agent.py`：DeepSeek 工具定义、最多 4 轮的调用循环、参数归一化、降级逻辑。
- `weka_runner.py`：数据清洗、ARFF 生成、Weka 调用、控制台结果解析。
- `charts.py`：频繁项集/关联规则条形图。
- `app.py`：FastAPI 接口与静态文件服务。
- `static/`：无需构建工具的原生 HTML/CSS/JS 前端。
