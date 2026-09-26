# 研究草案与探索结果导入

```powershell
python -m quant_agent.research_assistant paper.txt --template template.yaml --output ../drafts/momentum --study-id momentum-v1
```

默认离线：保存来源和SHA-256、复制选定模板、列出因子数据要求、检索相似登记研究并写研究笔记。离线模板模式不会声称理解论文内容。使用`--database experiments.db`进行透明的词项相似度检索；这不是语义向量搜索。

显式`--llm --model MODEL`并设置`QUANT_AGENT_LLM_OK=1`后，才通过配置的模型生成草案；遵守DATA_POLICY.md。只发送给定来源、因子目录及有限模板，不发送价格表或策略账本。模型只允许返回hypothesis/factors/variants/evidence_quotes，引用必须逐字出现在来源中，因子必须已注册。输入、费用、风险和留出定义不能由模型改写。不自动运行草案。

探索JSON导入结构：`{"source_text":"原文", "proposal":{"hypothesis":"待验证假设", "factors":{"momentum_20d":1}, "variants":[], "evidence_quotes":["原文"]}}`。支持Unicode及带引号原文，不依赖JSON序列化后的转义形式。

输出recipe.yaml、source.txt、evidence.json、research-note.md。新公式先显式开发、测试并注册；模型生成的Python不进入执行链。在线模型质量需要用户配置服务后另行评估，测试使用受控响应，不宣称在线调用已验证。
