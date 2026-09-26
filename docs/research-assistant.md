# 研究草案与探索结果导入

```powershell
python -m quant_agent.research_assistant paper.txt --template template.yaml --output ../drafts/momentum --study-id momentum-v1 --database ../studies/experiments.db --studies-root ../studies
```

默认离线：保存来源和SHA-256、复制选定模板、列出因子数据要求、检索相似登记研究并写研究笔记。离线模板模式不会声称理解论文内容。使用`--database experiments.db`读取登记定义和终态事件，使用`--studies-root`读取`study.json`、结果、失败和artifact；检索是透明的词项相似度，不是语义向量搜索。数据库以SQLite只读URI打开。

显式`--llm --model MODEL`并设置`QUANT_AGENT_LLM_OK=1`后，才通过配置的模型生成草案；遵守DATA_POLICY.md。只发送给定来源、因子目录及有限模板，不发送价格表或策略账本。模型只允许返回hypothesis/factors/variants/evidence_quotes，引用必须逐字出现在来源中，因子必须已注册或来自模板中已校验的`factor_expressions`。输入、费用、风险、研究模式和留出定义不能由模型改写。不自动运行草案。`evidence.json.model_invocation`区分真实配置provider、注入测试替身和未调用，测试不会声称调用过在线模型。

探索JSON导入结构：`{"source_text":"原文", "proposal":{"hypothesis":"待验证假设", "factors":{"momentum_20d":1}, "variants":[], "evidence_quotes":["原文"]}}`。支持Unicode及带引号原文，不依赖JSON序列化后的转义形式。

输出recipe.yaml、source.txt、evidence.json、research-note.md。受限公式先通过`quant-factors`校验；模型生成的Python不进入执行链。在线模型质量需要用户配置服务后另行评估，测试使用受控响应，不宣称在线调用已验证。

## 历史证据API与CLI

UI可调用：

```python
from pathlib import Path

from quant_agent import research_advice

advice = research_advice(
    "A股动量与低波动在成本后是否仍有增量",
    study_paths=[Path("studies/ashare/study.json")],
    database=Path("studies/ashare/experiments.db"),
    recipe=recipe,
)
```

`research_advice`和底层`research_history`返回匹配study、完成结果、失败、citation、引用复核、artifact摘要核验、最小对照和缺失数据。JSON文件引用固定到文件SHA-256和JSON Pointer；SQLite引用固定到表、主键、行摘要和JSON Pointer。`verify_citations`会重新读取源并复核来源与选中值。完成结果只有在声明的artifact全部存在、路径未逃逸attempt目录且SHA-256一致时，才可成为最小对照。

```powershell
quant-research-history "动量 低波动" `
  --database ../studies/ashare/experiments.db `
  --studies-root ../studies `
  --recipe recipe.yaml `
  --output advice.json
```

最小对照只描述候选与已核验历史control，并为`inputs`、`interval`、`costs`、`risk`和`holdout`记录锁定摘要。它不会改配方、选择胜者或启动实验；缺少完成结果、失败记录或完整artifact时，会在`missing_data`中明确列出。
