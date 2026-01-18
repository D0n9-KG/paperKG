"""
Neo4j 工具模块
负责 Neo4j 数据库连接和基本操作
"""

import os
from typing import Dict, Any, List, Optional
from neo4j import GraphDatabase
import logging
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)  # 强制设置neo4j_utils模块的日志级别，避免输出DEBUG信息


class Neo4jConnection:
    """Neo4j 数据库连接管理器"""

    def __init__(self, uri: str = None, user: str = None, password: str = None):
        """
        初始化 Neo4j 连接

        Args:
            uri: Neo4j 数据库 URI
            user: 用户名
            password: 密码
        """
        self.uri = uri or os.getenv('NEO4J_URI', 'bolt://localhost:7687')
        self.user = user or os.getenv('NEO4J_USERNAME', 'neo4j')
        self.password = password or os.getenv('NEO4J_PASSWORD', 'password')

        try:
            self.driver = GraphDatabase.driver(
                self.uri,
                auth=(self.user, self.password),
                # 连接池配置
                max_connection_lifetime=30 * 60,  # 30 minutes
                max_connection_pool_size=50,
                connection_acquisition_timeout=10.0,
                connection_timeout=10.0
            )
            logger.info(f"成功连接到 Neo4j 数据库: {self.uri}")
        except Exception as e:
            logger.error(f"Neo4j 连接失败: {e}")
            raise

    def close(self):
        """关闭数据库连接"""
        if hasattr(self, 'driver'):
            self.driver.close()
            logger.info("Neo4j 连接已关闭")

    def execute_query(self, query: str, parameters: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """
        执行 Cypher 查询

        Args:
            query: Cypher 查询语句
            parameters: 查询参数

        Returns:
            查询结果列表
        """
        if parameters is None:
            parameters = {}

        try:
            with self.driver.session() as session:
                result = session.run(query, parameters)
                return [record.data() for record in result]
        except Exception as e:
            logger.error(f"执行查询失败: {query}, 错误: {e}")
            raise

    def execute_write_query(self, query: str, parameters: Dict[str, Any] = None) -> Optional[Any]:
        """
        执行写操作查询

        Args:
            query: Cypher 查询语句
            parameters: 查询参数

        Returns:
            查询结果
        """
        if parameters is None:
            parameters = {}

        try:
            with self.driver.session() as session:
                result = session.run(query, parameters)
                # 对于写操作，通常返回 summary 或 None
                return result.consume().counters if result.consume() else None
        except Exception as e:
            logger.error(f"执行写查询失败: {query}, 错误: {e}")
            raise

    def create_node(self, label: str, properties: Dict[str, Any],
                   unique_property: str = None) -> Dict[str, Any]:
        """
        创建节点（如果不存在）

        Args:
            label: 节点标签
            properties: 节点属性
            unique_property: 唯一属性键

        Returns:
            创建的节点数据
        """
        if unique_property and unique_property in properties:
            # 使用 MERGE 创建唯一节点
            query = f"""
            MERGE (n:{label} {{{unique_property}: $unique_value}})
            ON CREATE SET n += $properties
            ON MATCH SET n += $properties
            RETURN n
            """
            unique_value = properties[unique_property]
            params = {
                'unique_value': unique_value,
                'properties': properties
            }
        else:
            # 直接创建节点
            query = f"""
            CREATE (n:{label})
            SET n = $properties
            RETURN n
            """
            params = {'properties': properties}

        result = self.execute_query(query, params)
        return result[0]['n'] if result else None

    def create_relationship(self, from_node_id: str, to_node_id: str,
                          relationship_type: str, properties: Dict[str, Any] = None) -> None:
        """
        创建关系（使用 MERGE 确保唯一性）

        Args:
            from_node_id: 起始节点 ID
            to_node_id: 结束节点 ID
            relationship_type: 关系类型
            properties: 关系属性
        """
        if properties is None:
            properties = {}

        # 使用 MERGE 确保关系唯一性
        query = f"""
        MATCH (a), (b)
        WHERE elementId(a) = $from_id AND elementId(b) = $to_id
        MERGE (a)-[r:{relationship_type}]->(b)
        SET r = $properties
        """

        params = {
            'from_id': from_node_id,
            'to_id': to_node_id,
            'properties': properties
        }

        self.execute_write_query(query, params)

    def find_node_by_property(self, label: str, property_key: str, property_value: Any) -> Optional[Dict[str, Any]]:
        """
        根据属性查找节点

        Args:
            label: 节点标签
            property_key: 属性键
            property_value: 属性值

        Returns:
            节点数据或 None
        """
        query = f"""
        MATCH (n:{label})
        WHERE n.{property_key} = $value
        RETURN n
        LIMIT 1
        """

        result = self.execute_query(query, {'value': property_value})
        return result[0]['n'] if result else None

    def merge_node(self, label: str, match_properties: Dict[str, Any],
                  set_properties: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        合并节点（存在则更新，否则创建）

        Args:
            label: 节点标签
            match_properties: 用于匹配的属性
            set_properties: 要设置的属性

        Returns:
            合并后的节点数据
        """
        if set_properties is None:
            set_properties = {}

        # 使用更标准的 MERGE 语法
        if len(match_properties) == 1:
            # 单属性匹配
            prop_name = list(match_properties.keys())[0]
            prop_value = match_properties[prop_name]

            query = f"""
            MERGE (n:{label} {{{prop_name}: ${prop_name}}})
            ON CREATE SET n += $set_properties
            ON MATCH SET n += $set_properties
            RETURN n, elementId(n) as node_id
            """

            params = {prop_name: prop_value, 'set_properties': set_properties}
        else:
            # 多属性匹配 - 使用 WHERE 子句
            match_conditions = [f"n.{k} = ${k}" for k in match_properties.keys()]

            query = f"""
            MERGE (n:{label})
            WHERE {' AND '.join(match_conditions)}
            ON CREATE SET n += $set_properties
            ON MATCH SET n += $set_properties
            RETURN n, elementId(n) as node_id
            """

            params = dict(match_properties)
            params['set_properties'] = set_properties

        logger.debug(f"MERGE 查询: {query}")
        logger.debug(f"参数: {params}")

        result = self.execute_query(query, params)

        if result:
            node_data = result[0]['n']
            node_id = result[0]['node_id']
            logger.debug(f"成功合并节点: {label}, ID: {node_id}, 数据: {node_data}")
            return node_data
        else:
            logger.debug(f"MERGE 失败: 没有返回结果")
            return None

    def get_node_id(self, label: str, property_key: str, property_value: Any) -> Optional[str]:
        """
        获取节点的内部 ID

        Args:
            label: 节点标签
            property_key: 属性键
            property_value: 属性值

        Returns:
            节点 ID 或 None
        """
        query = f"""
        MATCH (n:{label})
        WHERE n.{property_key} = $value
        RETURN elementId(n) as node_id
        """

        result = self.execute_query(query, {'value': property_value})
        return result[0]['node_id'] if result else None


# 全局连接实例
_neo4j_connection = None


def get_neo4j_connection() -> Neo4jConnection:
    """获取全局 Neo4j 连接实例"""
    global _neo4j_connection
    if _neo4j_connection is None:
        _neo4j_connection = Neo4jConnection()
    return _neo4j_connection


def close_neo4j_connection():
    """关闭全局 Neo4j 连接"""
    global _neo4j_connection
    if _neo4j_connection:
        _neo4j_connection.close()
        _neo4j_connection = None
