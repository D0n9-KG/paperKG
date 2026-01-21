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
MULTIMEDIA_CONTENT_PROMPT_BASE = """你是一位专业的学术文档分析专家，擅长从学术论文中准确提取多媒体内容信息，包括图片、参考文献和数学公式。

【任务】
请从以下论文文本中提取所有多媒体内容信息，包括图片、参考文献和重要公式。

【论文文本】
{text}

【提取要求】

**一、图片信息提取（images）**
1. **图片信息组织**：
   - 找出所有图片引用（格式如：![alt](path)）
   - 按主图编号组织为字典结构，例如：
     * "1": 对应Figure 1的所有图片
     * "2": 对应Figure 2的所有图片（包括子图2a、2b等）

2. **图片信息提取**：
   - 为每个图片提取：
     * path：图片相对路径（从![alt](path)中提取的path部分）
     * caption：图片标题（完整标题）

3. **图片组织规则**：
   - 所有子图（如2a、2b）都放在主图编号"2"的数组中
   - 每个主图编号对应一个数组，包含该编号下的所有图片

**二、参考文献信息提取（references）**
1. **参考文献列表（reference_list）**：
   - 提取论文末尾的参考文献列表，尽可能完整
   - 为每篇参考文献提取：
     * id：参考文献编号（如"1"、"2"等）
     * citation：完整引用字符串
     * doi：无需提取，统一返回空字符串""

2. **参考文献总数（total_count）**：
   - **必须提供**：统计提取到的参考文献总数，值为reference_list的长度

**三、公式信息提取（formulas）**
1. **公式列表（formula_list）**：
   - 提取论文中的重要数学公式
   - 为每个重要公式提取：
     * formula_number：公式编号（如"1"、"2"等）
     * latex_content：公式内容（LaTeX格式）
     * description：公式描述或解释（可选）

2. **公式总数（total_count）**：
   - **必须提供**：统计提取到的公式总数，值为formula_list的长度

【重要提示】
1. **必填字段**：
   - references.total_count、formulas.total_count都是必填字段
   - 必须根据对应的列表长度计算并填写正确的数值

2. **DOI字段要求**：
   - 不要尝试提取DOI，统一返回空字符串""

3. **图片格式要求**：
   - 图片信息必须按主图编号组织为字典
   - 每个主图编号对应一个数组，包含该编号下的所有图片
   - 图片信息包含path和caption字段

4. **准确性原则**：
   - 宁缺毋滥：如果无法确定准确信息，使用合理的默认值
   - 保持一致性：相同类型的元素使用相同的提取策略
   - 基于上下文：根据附近的文本判断最合适的信息

5. **输出格式**：
   - 确保JSON格式正确，符合指定的schema
   - 所有必填字段必须提供值
   - 保持JSON结构简单清晰

【自我检查】
输出前请确认：
✓ 图片信息是否按主图编号正确组织？
✓ 参考文献列表是否尽可能完整？
✓ 每篇参考文献是否都包含doi字段（统一空字符串）？
✓ 重要公式是否都已提取？
✓ **所有total_count字段是否都已正确填写？**
✓ JSON格式是否正确？

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
EVIDENCE_SCAN_PROMPT = """你是论文证据识别助手。请从当前批次片段中识别“论文主干 + 关键技术细节”的证据片段，并将其ID划入对应类别。

【输入JSON】
{text}

【类别】
- background: 引言/背景/基础概念
- state_of_art: 研究现状/已有工作/已有方法
- research_gap: 研究空白/不足/问题
- research_question: 研究问题
- objective: 研究目标
- hypothesis: 假设
- method: 方法/模型/实验设计
- result: 结果/发现/分析
- conclusion: 讨论/结论/展望

【要求】
1) 只能使用本批次片段的ID，不得编造。
2) 保持覆盖范围宽：宁可多保留关键段落，也不要遗漏核心主线与技术细节。
3) 仅剔除明显无意义或非内容片段：作者/机构/致谢/参考文献列表/版权/目录/纯图注或纯编号。
4) 每个片段最多归入一个类别；若内容跨类，归入“更能代表其核心功能”的类别。
5) methods/results/conclusions 必须区分 main/supports：
   - main：选出“最能概括该类别整体贡献”的1个片段；
   - supports：补充细节与证据（方法步骤、参数、设置、对比、消融、局限、外推等）。
6) 对“包含公式/推导/参数/实验设置/数值求解步骤”的片段，优先纳入 method_support（若主要在阐述方法）。
7) 图表或表格：若包含关键结果或对比结论，可归入 result_support；若仅为图片标题或符号列表，剔除。
8) 优先保留含关键变量、定量结果、因果关系或“关键术语 + 结论”的片段。
9) 避免同义重复：相近片段只保留信息更完整者。
10) 如果没有对应片段，输出空数组。

【输出JSON】
{
  "background_ids": [],
  "state_of_art_ids": [],
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
2) 主片段要求：能概括该类别的核心贡献（方法=研究设计/建模/实验路线，结果=核心发现/关键数值趋势，结论=最终判断/贡献/意义）。
3) 优先选择覆盖范围广、信息密度高、包含关键变量/条件的片段。
4) 尽量避免选择纯图注/纯公式堆叠/纯步骤列表作为 main。
5) 若没有合适候选，返回空字符串""。

【输出JSON】
{"method_main_id":"","result_main_id":"","conclusion_main_id":""}
"""


# 研究叙事合成提示语
RESEARCH_NARRATIVE_SYNTH_PROMPT = """你是资深学术分析专家。只抽取 research_narrative 部分，并确保逻辑链能覆盖论文主要脉络与关键技术细节。

【已选证据片段（带ID）】
{segments}

【已选证据ID分组（来自上一步）】
{selected_ids}

【要求】
1) 输出必须符合“输出格式”中的 schema，不得新增字段。
2) 每条 value 必须是完整、可独立阅读的句子，优先“概念 + 结论/目的 + 关键限定条件”结构。
3) 每条陈述必须给出 evidence_segment_ids（至少一个），且只能使用已选证据ID。
4) 严禁臆造；若无证据，不要输出该条陈述。
5) background / state_of_art / research_gaps / research_questions / research_objectives / hypotheses 为多条陈述列表（并联），不要人为串联成单一链。
6) 如果证据充足，每一类至少给出 1 条；如原文未显式出现研究问题/假设，不要强行编造。
7) methods / results / conclusions 为“主结论 + 关键支撑子结论”结构：
   - methods.main：覆盖研究设计/方法路线/数据来源或实验设置；
   - results.main：覆盖核心发现或关键定量关系；
   - conclusions.main：覆盖最终判断/贡献/意义或局限性总结。
8) supports 用于补充细节：方法步骤、关键参数、模型假设、对照/消融、局限与未来工作等。
9) 避免重复或近义改写堆叠；优先选择信息更完整的表述。
10) 保留关键符号与变量名称（如 μ, J, φ, Θ 等），避免乱码或误替换。
11) 不要原样粘贴长公式或图注；必要时用文字概括其含义。
12) 需要生成 5 个汇聚节点：background_hub / state_hub / gap_hub / objective_hub / hypothesis_hub，用于接入主链。
13) logic_chains 中的 steps 仅包含汇聚节点 + method_main/result_main/conclusion_main，形成主链。
14) node_relations 可先输出空数组（关系由后处理补齐）。
15) citations 字段用于记录引用及其作用说明；若无法判断可先置空字符串。
16) 仅输出 JSON。

{OUTPUT_FORMAT_SECTION}"""


STATE_GAP_OBJECTIVE_LINK_PROMPT = """你是研究逻辑关系链接助手。请根据节点语义建立以下对应关系：
1) 现有工作（state_of_art） -> 缺口（research_gaps）
2) 缺口（research_gaps） -> 研究目的（research_objectives）

【输入JSON】
{text}

【要求】
1) 只能使用提供的节点ID，不得编造。
2) 可一对多或多对一；仅在“语义上明确相关”时建立链接。
3) 若无明显对应关系，可输出空列表。
4) 关系类型只能为 LEADS_TO_GAP 或 MOTIVATES_OBJECTIVE。

【输出JSON】
{
  "state_gap_links": [{"state_id": "", "gap_id": ""}],
  "gap_objective_links": [{"gap_id": "", "objective_id": ""}]
}
"""
