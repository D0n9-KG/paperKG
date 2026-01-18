"""
Crossref API客户端 - 异步实现
用于从Crossref获取论文元数据
"""
import asyncio
import aiohttp
import logging
from typing import Dict, Any, Optional, List
import time
import random
from datetime import datetime, timedelta
import json


logger = logging.getLogger(__name__)


class CrossrefClient:
    """异步Crossref API客户端"""
    
    def __init__(self, config: Dict | None = None):
        self.config = config or {}
        self.base_url = "https://api.crossref.org"
        self.user_agent = self.config.get("user_agent", "paper_KG/1.0")
        self.timeout = self.config.get("timeout", 10)
        self.max_retries = self.config.get("max_retries", 3)
        self.retry_delay = self.config.get("retry_delay", 1)
        self.concurrent_requests_limit = self.config.get("concurrent_requests_limit", 3)
        self.enable_cache = self.config.get("enable_cache", True)
        self.cache_ttl = self.config.get("cache_ttl", 3600)

        # 并发控制信号量
        self.semaphore = asyncio.Semaphore(self.concurrent_requests_limit)

        # 内存缓存
        self.cache = {}

        # 请求头
        self.headers = {
            "User-Agent": self.user_agent,
            "Accept": "application/json"
        }

        logger.info(f"Crossref客户端初始化完成 - 超时: {self.timeout}s, 重试: {self.max_retries}次, 并发限制: {self.concurrent_requests_limit}")
    
    def _get_cache_key(self, query_type: str, query_value: str) -> str:
        """生成缓存键"""
        return f"{query_type}:{query_value}"
    
    def _get_from_cache(self, cache_key: str) -> Optional[Dict]:
        """从缓存获取数据"""
        if not self.enable_cache:
            return None
        
        if cache_key in self.cache:
            cached_data, timestamp = self.cache[cache_key]
            # 检查是否过期
            if datetime.now().timestamp() - timestamp < self.cache_ttl:
                logger.debug(f"缓存命中: {cache_key}")
                return cached_data
            else:
                # 缓存过期，删除
                del self.cache[cache_key]
        
        return None
    
    def _save_to_cache(self, cache_key: str, data: Dict):
        """保存数据到缓存"""
        if self.enable_cache:
            self.cache[cache_key] = (data, datetime.now().timestamp())
            logger.debug(f"数据已缓存: {cache_key}")
    
    async def _make_request(self, url: str, params: Dict = None) -> Optional[Dict]:
        """发送HTTP请求（带重试机制和并发控制）"""
        last_exception = None

        async with self.semaphore:  # 并发控制
            for attempt in range(self.max_retries):
                start_time = time.time()

                try:
                    timeout = aiohttp.ClientTimeout(total=self.timeout)
                    async with aiohttp.ClientSession(timeout=timeout) as session:
                        async with session.get(url, headers=self.headers, params=params) as response:
                            elapsed = time.time() - start_time

                            if response.status == 200:
                                data = await response.json()
                                logger.debug(f"Crossref请求成功 - URL: {url}, 耗时: {elapsed:.2f}s")
                                return data
                            elif response.status == 404:
                                logger.debug(f"Crossref未找到资源 - URL: {url}")
                                return None
                            elif response.status == 429:  # 限流
                                retry_after = int(response.headers.get('Retry-After', self.retry_delay))
                                logger.warning(f"Crossref限流，{retry_after}秒后重试")
                                await asyncio.sleep(retry_after)
                                continue
                            else:
                                logger.warning(f"Crossref请求失败 - 状态码: {response.status}, URL: {url}")

                except asyncio.TimeoutError:
                    elapsed = time.time() - start_time
                    logger.warning(f"Crossref请求超时（尝试 {attempt + 1}/{self.max_retries}）: {url}, 耗时: {elapsed:.2f}s")
                    last_exception = "Timeout"

                except Exception as e:
                    elapsed = time.time() - start_time
                    logger.warning(f"Crossref请求异常（尝试 {attempt + 1}/{self.max_retries}）: {e}, URL: {url}, 耗时: {elapsed:.2f}s")
                    last_exception = str(e)

                # 等待重试（指数退避策略）
                if attempt < self.max_retries - 1:
                    # 指数退避：基础延迟 * 2^尝试次数 + 随机抖动，最大不超过60秒
                    retry_delay = min(self.retry_delay * (2 ** attempt) + random.uniform(0, 1), 60)
                    logger.debug(f"Crossref重试延迟: {retry_delay:.2f}秒（尝试 {attempt + 1}/{self.max_retries}）")
                    await asyncio.sleep(retry_delay)

        logger.error(f"Crossref请求失败（{self.max_retries}次尝试后）: {last_exception}")
        return None
    
    async def get_by_doi(self, doi: str) -> Optional[Dict]:
        """通过DOI获取论文元数据"""
        if not doi:
            return None
        
        # 清理DOI
        doi = doi.strip()
        if doi.startswith("https://doi.org/"):
            doi = doi[16:]
        elif doi.startswith("http://doi.org/"):
            doi = doi[15:]
        elif doi.startswith("doi:"):
            doi = doi[4:]
        
        # 检查缓存
        cache_key = self._get_cache_key("doi", doi)
        cached_data = self._get_from_cache(cache_key)
        if cached_data:
            return cached_data
        
        # 构建URL
        url = f"{self.base_url}/works/{doi}"
        
        # 发送请求
        data = await self._make_request(url)
        
        if data and "message" in data:
            result = data["message"]
            # 保存到缓存
            self._save_to_cache(cache_key, result)
            return result
        
        return None
    
    async def search_works(self, query: Dict) -> Optional[List[Dict]]:
        """搜索论文（模糊查询）"""
        if not query:
            return None
        
        # 构建查询参数
        params = {}
        
        # 标题查询
        if query.get("title"):
            params["query.title"] = query["title"]
        
        # 作者查询
        if query.get("author"):
            params["query.author"] = query["author"]
        
        # 年份查询
        if query.get("year"):
            params["filter"] = f"from-pub-date:{query['year']},until-pub-date:{query['year']}"
        
        # 限制结果数量
        params["rows"] = 5
        
        # 检查缓存
        cache_key = self._get_cache_key("search", json.dumps(query, sort_keys=True))
        cached_data = self._get_from_cache(cache_key)
        if cached_data:
            return cached_data
        
        # 构建URL
        url = f"{self.base_url}/works"
        
        # 发送请求
        data = await self._make_request(url, params)
        
        if data and "message" in data and "items" in data["message"]:
            results = data["message"]["items"]
            # 保存到缓存
            self._save_to_cache(cache_key, results)
            return results
        
        return None
    
    async def search_by_full_citation(self, citation: str) -> Optional[List[Dict]]:
        """通过完整引用字符串搜索论文"""
        if not citation:
            return None
        
        # 清理引用字符串：移除多余空格和换行符
        citation = ' '.join(citation.split())
        
        # 检查缓存
        cache_key = self._get_cache_key("full_citation", citation)
        cached_data = self._get_from_cache(cache_key)
        if cached_data:
            return cached_data
        
        # 构建查询参数：使用通用的query参数进行全文搜索
        params = {
            "query": citation,
            "rows": 5  # 获取前5个结果用于匹配
        }
        
        # 构建URL
        url = f"{self.base_url}/works"
        
        # 发送请求
        data = await self._make_request(url, params)
        
        if data and "message" in data and "items" in data["message"]:
            results = data["message"]["items"]
            # 保存到缓存
            self._save_to_cache(cache_key, results)
            return results
        
        return None
    
    async def get_metadata(self, doi: str = None, title: str = None, 
                          authors: List[str] = None, year: str = None,
                          full_citation: str = None) -> Optional[Dict]:
        """
        获取论文元数据（智能查询 - 混合策略）
        优先级：DOI查询 > 完整引用查询 > 标题+作者+年份查询 > 仅标题查询
        """
        
        # 1. DOI查询（最高优先级）
        if doi:
            result = await self.get_by_doi(doi)
            if result:
                logger.info(f"通过DOI查询成功: {doi}")
                return result
        
        # 2. 完整引用查询（新添加的优先级）
        if full_citation:
            results = await self.search_by_full_citation(full_citation)
            if results and len(results) > 0:
                logger.info(f"通过完整引用查询成功，找到{len(results)}个结果")
                # 使用智能匹配选择最佳结果
                best_match = self._select_best_match(results, full_citation, title, authors, year)
                if best_match:
                    logger.info(f"智能匹配选择最佳结果")
                    return best_match
                else:
                    # 如果没有找到好的匹配，返回第一个结果作为备选
                    logger.info(f"使用第一个结果作为备选")
                    return results[0]
        
        # 3. 标题+作者+年份查询（中等优先级）
        if title and authors:
            query = {
                "title": title,
                "author": " ".join(authors[:3]),  # 前3位作者
                "year": year
            }
            results = await self.search_works(query)
            if results and len(results) > 0:
                logger.info(f"通过标题+作者查询成功，找到{len(results)}个结果")
                # 返回第一个（最相关）的结果
                return results[0]
        
        # 4. 仅标题查询（最低优先级）
        if title:
            query = {"title": title}
            results = await self.search_works(query)
            if results and len(results) > 0:
                logger.info(f"通过标题查询成功，找到{len(results)}个结果")
                # 返回第一个结果
                return results[0]
        
        logger.warning("所有Crossref查询策略均失败")
        return None
    
    def _select_best_match(self, results: List[Dict], full_citation: str = None,
                          title: str = None, authors: List[str] = None, year: str = None) -> Optional[Dict]:
        """
        从多个结果中选择最佳匹配
        使用简单的启发式规则进行匹配评分
        """
        if not results:
            return None
        
        best_score = -1
        best_result = None
        
        for result in results:
            score = self._calculate_match_score(result, full_citation, title, authors, year)
            if score > best_score:
                best_score = score
                best_result = result
        
        # 设置阈值：如果最佳匹配分数太低，返回None
        if best_score < 0.3:  # 30%匹配度阈值
            logger.debug(f"最佳匹配分数太低: {best_score:.2f}")
            return None
        
        logger.debug(f"选择最佳匹配，分数: {best_score:.2f}")
        return best_result
    
    def _calculate_match_score(self, result: Dict, full_citation: str = None,
                              title: str = None, authors: List[str] = None, year: str = None) -> float:
        """
        计算匹配分数（0-1之间）
        基于标题相似度、作者匹配、年份匹配等
        """
        score = 0.0
        weights = {
            "title": 0.5,
            "year": 0.3,
            "authors": 0.2
        }
        
        # 1. 标题相似度
        if title:
            result_title = self._extract_result_title(result)
            if result_title and title:
                title_similarity = self._calculate_string_similarity(result_title.lower(), title.lower())
                score += title_similarity * weights["title"]
        
        # 2. 年份匹配
        if year:
            result_year = self._extract_result_year(result)
            if result_year and year:
                if result_year == year:
                    score += 1.0 * weights["year"]
                else:
                    # 年份接近（±1年）给部分分数
                    try:
                        year_int = int(year)
                        result_year_int = int(result_year)
                        if abs(year_int - result_year_int) <= 1:
                            score += 0.5 * weights["year"]
                    except ValueError:
                        pass
        
        # 3. 作者匹配（简化）
        if authors and len(authors) > 0:
            result_authors = self._extract_result_authors(result)
            if result_authors:
                # 检查第一个作者是否匹配
                first_author = authors[0].lower()
                result_first_author = result_authors[0].get("family_name", "").lower() if result_authors else ""
                if result_first_author and first_author and result_first_author in first_author or first_author in result_first_author:
                    score += 1.0 * weights["authors"]
        
        return min(score, 1.0)  # 确保分数不超过1.0
    
    def _extract_result_title(self, result: Dict) -> str:
        """从Crossref结果中提取标题"""
        titles = result.get("title", [])
        if titles and len(titles) > 0:
            title = titles[0]
            if isinstance(title, str):
                return title.strip()
        return ""
    
    def _extract_result_year(self, result: Dict) -> str:
        """从Crossref结果中提取年份"""
        # 尝试从多个日期字段中提取年份
        date_fields = ["published", "published-online", "published-print", "issued", "created"]
        
        for field in date_fields:
            date_data = result.get(field)
            if date_data and "date-parts" in date_data:
                date_parts = date_data["date-parts"]
                if date_parts and len(date_parts) > 0 and len(date_parts[0]) > 0:
                    year = str(date_parts[0][0])
                    if year.isdigit() and len(year) == 4:
                        return year
        
        return ""
    
    def _extract_result_authors(self, result: Dict) -> List[Dict]:
        """从Crossref结果中提取作者信息"""
        authors = []
        author_list = result.get("author", [])
        
        for author in author_list:
            given = author.get("given", "")
            family = author.get("family", "")
            
            author_info = {
                "given_name": given,
                "family_name": family,
                "full_name": f"{given} {family}".strip()
            }
            authors.append(author_info)
        
        return authors
    
    def _calculate_string_similarity(self, str1: str, str2: str) -> float:
        """
        计算两个字符串的相似度（简化版）
        使用基于共同单词的简单相似度计算
        """
        if not str1 or not str2:
            return 0.0
        
        # 将字符串拆分为单词
        words1 = set(str1.lower().split())
        words2 = set(str2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        # 计算Jaccard相似度
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        if union == 0:
            return 0.0
        
        return intersection / union


class CrossrefDataExtractor:
    """Crossref数据提取和转换器"""
    
    @staticmethod
    def extract_title(crossref_data: Dict) -> str:
        """提取标题"""
        titles = crossref_data.get("title", [])
        if titles and len(titles) > 0:
            # 使用第一个标题
            title = titles[0]
            if isinstance(title, str):
                return title.strip()
        return ""
    
    @staticmethod
    def extract_authors(crossref_data: Dict) -> List[Dict]:
        """提取作者信息"""
        authors = []
        author_list = crossref_data.get("author", [])
        
        for i, author in enumerate(author_list):
            given = author.get("given", "")
            family = author.get("family", "")
            
            author_info = {
                "given_name": given,
                "family_name": family,
                "full_name": f"{given} {family}".strip(),
                "sequence": "first" if i == 0 else "additional",
                "affiliations": author.get("affiliation", []),
                "orcid": author.get("ORCID", "").replace("https://orcid.org/", "")
            }
            authors.append(author_info)
        
        return authors
    
    @staticmethod
    def extract_year(crossref_data: Dict) -> str:
        """提取出版年份"""
        # 尝试从多个日期字段中提取年份
        date_fields = ["published", "published-online", "published-print", "issued", "created"]
        
        for field in date_fields:
            date_data = crossref_data.get(field)
            if date_data and "date-parts" in date_data:
                date_parts = date_data["date-parts"]
                if date_parts and len(date_parts) > 0 and len(date_parts[0]) > 0:
                    year = str(date_parts[0][0])
                    if year.isdigit() and len(year) == 4:
                        return year
        
        return ""
    
    @staticmethod
    def extract_journal_info(crossref_data: Dict) -> Dict:
        """提取期刊信息"""
        container_titles = crossref_data.get("container-title", [])
        journal_title = container_titles[0] if container_titles else ""
        
        return {
            "title": journal_title,
            "short_title": crossref_data.get("short-container-title", [""])[0],
            "volume": crossref_data.get("volume", ""),
            "issue": crossref_data.get("issue", ""),
            "pages": crossref_data.get("page", ""),
            "article_number": crossref_data.get("article-number", "")
        }
    
    @staticmethod
    def extract_identifiers(crossref_data: Dict) -> Dict:
        """提取标识符"""
        issn_list = crossref_data.get("ISSN", [])
        issn = issn_list[0] if issn_list else ""
        eissn = crossref_data.get("eISSN", "")
        
        return {
            "issn": issn,
            "eissn": eissn,
            "isbn": crossref_data.get("ISBN", [])
        }
    
    @staticmethod
    def extract_publisher_info(crossref_data: Dict) -> Dict:
        """提取出版商信息"""
        return {
            "name": crossref_data.get("publisher", ""),
            "location": crossref_data.get("publisher-location", ""),
            "member_id": crossref_data.get("member", "")
        }
    
    @staticmethod
    def extract_abstract(crossref_data: Dict) -> str:
        """提取摘要"""
        abstract = crossref_data.get("abstract", "")
        if isinstance(abstract, str):
            # 清理JATS XML标签（如果存在）
            import re
            abstract = re.sub(r'<[^>]+>', '', abstract)
            return abstract.strip()
        return ""
    
    @staticmethod
    def extract_keywords(crossref_data: Dict) -> List[str]:
        """提取关键词"""
        # Crossref通常不直接提供关键词，返回空数组
        return []
    
    @staticmethod
    def extract_categories(crossref_data: Dict) -> Dict:
        """提取分类信息"""
        return {
            "type": crossref_data.get("type", ""),
            "subtype": crossref_data.get("subtype", ""),
            "subjects": crossref_data.get("subject", []),
            "categories": crossref_data.get("categories", [])
        }
    
    @staticmethod
    def extract_citation_metrics(crossref_data: Dict) -> Dict:
        """提取引用指标"""
        return {
            "crossref_citations": crossref_data.get("is-referenced-by-count", 0),
            "references_count": crossref_data.get("references-count", 0),
            "score": crossref_data.get("score", 0.0)
        }
    
    @staticmethod
    def extract_open_access_info(crossref_data: Dict) -> Dict:
        """提取开放获取信息"""
        return {
            "is_open_access": crossref_data.get("is-open-access", False),
            "status": crossref_data.get("oa-status", ""),
            "url": crossref_data.get("oa-url", ""),
            "license": crossref_data.get("license", [])
        }
    
    @staticmethod
    def extract_funding_info(crossref_data: Dict) -> List[Dict]:
        """提取基金信息"""
        funders = crossref_data.get("funder", [])
        funding_info = []
        
        for funder in funders:
            info = {
                "funder": funder.get("name", ""),
                "funder_doi": funder.get("DOI", ""),
                "award": funder.get("award", [])
            }
            funding_info.append(info)
        
        return funding_info
    
    @staticmethod
    def extract_complete_metadata(crossref_data: Dict) -> Dict:
        """提取完整的Crossref元数据"""
        if not crossref_data:
            return {}

        return {
            # 基本标识
            "doi": crossref_data.get("DOI", ""),
            "url": crossref_data.get("URL", ""),
            "title": CrossrefDataExtractor.extract_title(crossref_data),
            "subtitle": crossref_data.get("subtitle", [""])[0] if crossref_data.get("subtitle") else "",
            "short_title": crossref_data.get("short-title", [""])[0] if crossref_data.get("short-title") else "",

            # 作者信息
            "authors": CrossrefDataExtractor.extract_authors(crossref_data),

            # 出版时间
            "publication_year": CrossrefDataExtractor.extract_year(crossref_data),
            "publication_date": {
                "published": crossref_data.get("published", {}).get("date-parts", [[]])[0] if crossref_data.get("published") else [],
                "published_online": crossref_data.get("published-online", {}).get("date-parts", [[]])[0] if crossref_data.get("published-online") else [],
                "published_print": crossref_data.get("published-print", {}).get("date-parts", [[]])[0] if crossref_data.get("published-print") else [],
                "issued": crossref_data.get("issued", {}).get("date-parts", [[]])[0] if crossref_data.get("issued") else []
            },

            # 期刊信息
            "journal_or_conference": CrossrefDataExtractor.extract_journal_info(crossref_data),

            # 标识符
            "identifiers": CrossrefDataExtractor.extract_identifiers(crossref_data),

            # 出版商
            "publisher": CrossrefDataExtractor.extract_publisher_info(crossref_data),

            # 摘要和关键词
            "abstract": CrossrefDataExtractor.extract_abstract(crossref_data),
            "keywords": CrossrefDataExtractor.extract_keywords(crossref_data),

            # 分类信息
            "categories": CrossrefDataExtractor.extract_categories(crossref_data),

            # 引用指标
            "citation_metrics": CrossrefDataExtractor.extract_citation_metrics(crossref_data),

            # 开放获取
            "open_access": CrossrefDataExtractor.extract_open_access_info(crossref_data),

            # 基金信息
            "funding": CrossrefDataExtractor.extract_funding_info(crossref_data),

            # 来源标记
            "metadata_source": {
                "primary_source": "crossref",
                "crossref_retrieved": datetime.now().isoformat(),
                "crossref_has_data": True,
                "llm_supplemented": False
            }
        }
