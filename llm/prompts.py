"""
提示语模板文件
集中管理所有代理使用的提示语
"""

# 元数据提取提示语
METADATA_PROMPT_BASE = """你是经验丰富的学术文献管理专家，擅长从论文中准确提取元数据。

【论文文本】
{text}

【提取要求】
1) title：完整标题，保留原文大小写与格式。
2) authors：作者姓名列表，如无法获取返回 []。
3) publication_year：四位年份，如无法确定返回 null。
4) journal_or_conference：期刊/会议名称，如无法获取返回 null。
5) doi：完整 DOI 或 DOI 链接，如无返回 null。
6) keywords：文中明确列出的关键词，如无返回 []。
7) abstract：完整摘要原文，如无返回 null。

【注意】
- 不要臆测或编造缺失信息。
- 严格输出 JSON。

{OUTPUT_FORMAT_SECTION}"""


# 多媒体内容提取提示语
MULTIMEDIA_CONTENT_PROMPT_BASE = """你是学术抽取助手，请从论文中提取图表、公式、引用等多媒体与参考文献信息。

【论文文本】
{text}

【要求】
1) 输出必须符合“输出格式”中的 schema。
2) 不要臆造；若无信息按 schema 允许的空值输出。
3) 仅输出 JSON。

{OUTPUT_FORMAT_SECTION}"""


# JSON schema 修复提示语
JSON_SCHEMA_REPAIR_PROMPT = """你是 JSON 修复助手，根据校验错误修复 JSON。

【校验错误】
{validation_errors}

【待修复 JSON】
{json_data}

仅输出修复后的 JSON，不要解释。
"""


# 质量评分提示语
QUALITY_RATER_PROMPT = """你是质量评估助手，请对抽取结果打分（0-100）并列出问题。

【论文文本】
{paper_text}

【抽取结果】
{extracted_json}

输出 JSON：{"score": 0, "issues": [], "needs_refine": false}
"""


# 内容改写提示语
CONTENT_REWRITE_PROMPT = """你是内容改写助手，请在不改变事实的前提下优化并修复抽取结果。

【论文文本】
{paper_text}

【抽取结果】
{extracted_json}

仅输出 JSON，不要解释。
"""


# 关键词提示语
KEYWORDS_PROMPT = """你是关键词提取助手。

【论文文本】
{text}

请提取最多 {max_keywords} 个关键词，输出 JSON：{"keywords": []}
"""


# 引用用途提示语
CITATION_PURPOSE_PROMPT = """你是引用作用分析助手。

引用：{citation}

上下文：{context}

输出 JSON：{"purpose": ""}
"""


# 证据扫描提示语（分批）
EVIDENCE_SCAN_PROMPT = """你是论文证据识别助手。请从当前批次片段中判断哪些片段属于论文主要内容，并将其ID划入对应类别。

【输入JSON】
{text}

【类别】
- background: 引言/背景/相关工作
- research_gap: 研究空白/不足/问题
- research_question: 研究问题
- objective: 研究目标
- hypothesis: 假设
- method: 方法/模型/实验设计
- result: 结果/发现/分析
- conclusion: 讨论/结论/展望

【要求】
1) 只能使用本批次片段的ID，不得编造。
2) 保持覆盖范围宽、尽量保留主要信息。
3) 只剔除明显无意义片段（如参考文献/致谢/版权/纯图表标题/编号）。
4) 每个片段最多归入一个类别。
5) methods/results/conclusions 必须区分 main/supports：main 选出最能概括该类别的1个核心片段，其余放 supports。
6) 如果没有对应片段，输出空数组。

【输出JSON】
{
  "background_ids": [],
  "research_gap_ids": [],
  "research_question_ids": [],
  "objective_ids": [],
  "hypothesis_ids": [],
  "method_main_ids": [],
  "method_support_ids": [],
  "result_main_ids": [],
  "result_support_ids": [],
  "conclusion_main_ids": [],
  "conclusion_support_ids": []
}
"""


# 主结论/主方法/主结果选择提示语
MAIN_ID_PICK_PROMPT = """你是主方法/主结果/主结论选择助手。请从候选片段中选出最能代表 main 的ID。

【输入JSON】
{text}

【要求】
1) 只能使用候选ID，不得编造。
2) 若没有合适候选，返回空字符串""。

【输出JSON】
{"method_main_id":"","result_main_id":"","conclusion_main_id":""}
"""


# 研究叙事合成提示语
RESEARCH_NARRATIVE_SYNTH_PROMPT = """你是资深学术分析专家。只抽取 research_narrative 部分。

【已选证据片段（带ID）】
{segments}

【已选证据ID分组（来自上一步）】
{selected_ids}

【要求】
1) 输出必须符合“输出格式”中的 schema，不得新增字段。
2) 每条 value 必须是完整、可独立阅读的句子。
3) 每条陈述必须给出 evidence_segment_ids（至少一个），且只能使用已选证据ID。
4) 严禁臆造；若无证据，不要输出该条陈述。
5) background / research_gaps / research_questions / research_objectives / hypotheses 为多条陈述列表。
6) methods / results / conclusions 为“主结论 + 关键支撑子结论”结构。
7) citations 字段用于记录引用及其作用说明；若无法判断可先置空字符串。
8) logic_chains 必须是“多条链”，steps 只包含主链 node_id（背景→空白→研究目的→方法main→结果main→结论main），不包含 supports。
9) 仅输出 JSON。

{OUTPUT_FORMAT_SECTION}"""
