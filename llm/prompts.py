"""
提示语模板文件
集中管理所有代理使用的提示语
"""

# 元数据提取提示语 - 基础模板（动态配置）
METADATA_PROMPT_BASE = """你是一位经验丰富的学术文献管理专家，擅长从学术论文中准确提取元数据信息。

【任务】
请从以下论文文本中提取元数据信息。

【论文文本（前5000字）】
{{text}}

【提取要求】
1. **标题（title）**：提取完整的论文标题，保留原文格式和大小写
2. **作者（authors）**：提取所有作者的姓名，格式为 ["作者1", "作者2", ...]。若无法获取，则返回空数组 []
3. **发表年份（publication_year）**：提取4位数年份（如 "2024"）。若无法确定，返回 null
4. **期刊或会议（journal_or_conference）**：提取完整的期刊名称或会议名称。若无法获取，返回 null
5. **DOI（doi）**：提取完整的DOI链接或DOI标识符。若无DOI，返回 null
6. **关键词（keywords）**：提取论文中明确列出的关键词。若无关键词部分，返回空数组 []
7. **摘要（abstract）**：提取完整的摘要原文，保留换行和段落结构。若无摘要，返回 null

【重要提示】
- 对于无法获取的信息，字符串类型字段使用 null，数组类型字段使用 []
- 不要编造或推测缺失的信息
- 确保JSON格式正确，可被程序解析

{OUTPUT_FORMAT_SECTION}"""

# 多媒体内容提取提示语 - 基础模板（动态配置）
MULTIMEDIA_CONTENT_PROMPT_BASE = """你是学术抽取助手。请从论文中提取多媒体与参考文献信息（图表、公式、引用）。

论文文本：
{text}

要求：
1) 输出必须严格符合“输出格式”中的schema。
2) 不要臆造；若无信息，按schema允许的空值输出。
3) 仅输出JSON。

{OUTPUT_FORMAT_SECTION}"""

# JSON schema 修复提示语
JSON_SCHEMA_REPAIR_PROMPT = """你是JSON修复助手。根据校验错误修复JSON。

【校验错误】
{validation_errors}

【待修复JSON】
{json_data}

要求：仅输出修复后的JSON，不要解释。
"""

QUALITY_RATER_PROMPT = """你是质量评估助手。请对抽取结果进行打分（0-100）并列出问题。

论文文本：
{paper_text}

抽取结果：
{extracted_json}

输出JSON：{"score": 0, "issues": [], "needs_refine": false}
"""

CONTENT_REWRITE_PROMPT = """你是内容改写助手。请在不改变事实的前提下优化与修复抽取结果。

论文文本：
{paper_text}

抽取结果：
{extracted_json}

仅输出JSON，不要解释。
"""

KEYWORDS_PROMPT = """你是关键词提取助手。

论文文本：
{text}

请提取最多 {max_keywords} 个关键词，输出JSON：{"keywords": []}
"""

CITATION_PURPOSE_PROMPT = """你是引用作用分析助手。

引用：{citation}

上下文：{context}

输出JSON：{"purpose": ""}
"""

RESEARCH_NARRATIVE_PROMPT_BASE = """你是资深学术分析专家。只抽取 research_narrative 部分。

【证据片段库（已编号，必须从中选用）】
{evidence_pool}

【要求】
1) 输出必须符合“输出格式”中的schema，不得新增字段。
2) 每条 value 必须是完整、可独立阅读的句子。
3) 每条陈述必须给出 evidence_segment_ids（至少1个），且只能引用证据片段库中已有的ID。
4) 严禁臆造；若没有证据，不要输出该条陈述；methods/results/conclusions 的 main 可以为 null。
5) background / research_gaps / research_questions / hypotheses 为“多条陈述列表”。
6) methods / results / conclusions 为“主结论 + 关键支撑子结论”结构：
   - main：1条总括性主结论（可为 null）
   - supports：若干条支撑子结论（可为空数组）
7) citations 字段用于记录引用及其作用说明：
   - citations 列表中的 citation_id 应来自证据片段中的数字引用（如 [3]）。
   - citation_text 可置为 null（后续系统会补全）。
   - purpose 若无法判断，可暂时输出空字符串 ""。
8) logic_chains 必须是“多条链”，按研究问题/假设拆分：
   - question_ids/hypothesis_ids 对应 research_questions/hypotheses 中的 node_id
   - steps 为该链条的 node_id 顺序列表（背景→空白→问题/假设→方法main→结果main→结论main）
9) 仅输出JSON。

{OUTPUT_FORMAT_SECTION}"""


# Research narrative evidence map prompt (stage 1)
RESEARCH_NARRATIVE_EVIDENCE_PROMPT = """你是论文证据选择助手。请仅从给定的证据片段库中选择最相关的片段ID。

【证据片段库】
{evidence_pool}

【要求】
1) 只能输出证据库中已有的ID，不得编造。
2) 尽量覆盖全文与主要章节（引言/方法/结果/讨论/结论），每个章节至少选1条可用证据。
3) 背景/空白/问题/假设应尽量覆盖论文主线，必要时可多选。
4) 方法/结果/结论分为 main 与 supports：main 代表总括性陈述的证据。
5) 如果确实没有证据，请输出空数组。

【输出（仅JSON）】
{
  "background_ids": [],
  "research_gap_ids": [],
  "research_question_ids": [],
  "hypothesis_ids": [],
  "method_main_ids": [],
  "method_support_ids": [],
  "result_main_ids": [],
  "result_support_ids": [],
  "conclusion_main_ids": [],
  "conclusion_support_ids": []
}
"""


# Research narrative synthesis prompt (stage 2)
RESEARCH_NARRATIVE_SYNTH_PROMPT = """你是资深学术分析专家。只抽取 research_narrative 部分。

【证据片段库（已编号，必须从中选用）】
{evidence_pool}

【已选证据ID（来自上一步）】
{selected_ids}

【要求】
1) 输出必须符合“输出格式”中的schema，不得新增字段。
2) 每条 value 必须是完整、可独立阅读的句子。
3) 每条陈述必须给出 evidence_segment_ids（至少1个），且只能引用证据库中已有的ID。
4) 严禁臆造；若没有证据，不要输出该条陈述；methods/results/conclusions 的 main 可以为 null。
5) background / research_gaps / research_questions / hypotheses 为多条陈述列表。
6) methods / results / conclusions 为“主结论 + 关键支撑子结论”结构。
7) citations 字段用于记录引用及其作用说明；若无法判断可先置空字符串。
8) logic_chains 必须是“多条链”，按研究问题/假设拆分，steps 只包含主链 node_id（背景→空白→问题/假设→方法main→结果main→结论main），不要包含 supports。
9) 仅输出JSON。

{OUTPUT_FORMAT_SECTION}"""
