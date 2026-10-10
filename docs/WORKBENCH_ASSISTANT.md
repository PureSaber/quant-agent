# Studio研究助手接口

`python -m quant_agent.workbench_assistant`接收Studio用户审阅过的证据摘要，返回带引用的文字解释。
它不会运行工具、生成可执行代码、修改研究或代替原有可信度核验。

## 安装与调用

继续使用本仓库`requirements.lock`及README里的独立环境安装步骤，锁文件已包含可选LLM运行依赖，无需重新解析依赖。

```powershell
python -m quant_agent.workbench_assistant --context reviewed-context.json --output advice.json
```

默认离线，输出`mode=evidence_only`、`model_invoked=false`和证据清单，不能把离线整理表述为模型推理。
在线模式需要`QUANT_AGENT_LLM_OK=1`、供应商凭据以及显式`--model`。Studio另外要求用户点击证据发送按钮。

## 输入与输出

输入JSON仅接受`question`与`evidence`。每条证据仅含`id`（E1–E99）、`text`和文本UTF-8字节的`sha256`。
问题最多4000字，证据最多30条、每条6000字，总JSON最多60KB；重复ID、未知字段、哈希不匹配均拒绝。

模型仅返回`findings`与`next_steps`。每条finding包含`text`、`citations`、`quotes`。
引用必须来自输入，逐字片段必须存在于引用来源；无效JSON、超长内容或伪造引用使调用失败，不降级伪装成有效模型结果。
CLI只创建新输出，不覆盖旧建议。失败返回1，日志保留错误类型，不输出可能包含授权头和证据内容的供应商异常全文。
建议仍是待验证判断，引用存在不等于推理正确，也不证明交易策略有效。

## 验证范围

测试覆盖默认离线、显式授权、引用与逐字片段、损坏证据、重复证据、供应商接口参数和错误日志脱敏。
供应商接口测试使用替身，不发起真实付费请求。Studio另有真实子进程集成测试。
没有提供联网模型验收结论；实际部署后还需在配置好模型的环境中人工核对建议质量。
