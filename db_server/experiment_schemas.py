"""
KnowledgeRAG 实验表结构配置工具
作者: XYZ-Algorithm-Team
用途: 为不同的实验创建灵活的表结构模板
"""

import os
import yaml
import json
from typing import Dict, List, Any, Optional
from pathlib import Path
from datetime import datetime

class SchemaTemplate:
    """表结构模板类"""
    
    def __init__(self, name: str, description: str = ""):
        self.name = name
        self.description = description
        self.tables: Dict[str, Dict] = {}
        self.created_at = datetime.now().isoformat()
        self.version = "1.0"
    
    def add_table(self, table_name: str, columns: List[Dict], 
                  indexes: List[Dict] = None, foreign_keys: List[Dict] = None):
        """添加表定义"""
        self.tables[table_name] = {
            'columns': columns,
            'indexes': indexes or [],
            'foreign_keys': foreign_keys or [],
            'created_at': datetime.now().isoformat()
        }
    
    def generate_sql(self) -> str:
        """生成SQL创建语句"""
        sql_statements = []
        
        # 添加头部注释
        sql_statements.append(f"-- {self.name} 表结构")
        sql_statements.append(f"-- 描述: {self.description}")
        sql_statements.append(f"-- 创建时间: {self.created_at}")
        sql_statements.append(f"-- 版本: {self.version}")
        sql_statements.append("")
        
        # 创建表
        for table_name, table_def in self.tables.items():
            sql_statements.append(f"-- 表: {table_name}")
            sql_statements.append(f"CREATE TABLE IF NOT EXISTS {table_name} (")
            
            # 列定义
            columns_sql = []
            for col in table_def['columns']:
                col_sql = f"    {col['name']} {col['type']}"
                
                # 添加约束
                if col.get('not_null', False):
                    col_sql += " NOT NULL"
                if col.get('auto_increment', False):
                    col_sql += " AUTO_INCREMENT"
                if col.get('primary_key', False):
                    col_sql += " PRIMARY KEY"
                if col.get('default') is not None:
                    if isinstance(col['default'], str):
                        col_sql += f" DEFAULT '{col['default']}'"
                    else:
                        col_sql += f" DEFAULT {col['default']}"
                if col.get('comment'):
                    col_sql += f" COMMENT '{col['comment']}'"
                
                columns_sql.append(col_sql)
            
            # 外键约束
            for fk in table_def.get('foreign_keys', []):
                fk_sql = f"    FOREIGN KEY ({fk['column']}) REFERENCES {fk['ref_table']}({fk['ref_column']})"
                if fk.get('on_delete'):
                    fk_sql += f" ON DELETE {fk['on_delete']}"
                if fk.get('on_update'):
                    fk_sql += f" ON UPDATE {fk['on_update']}"
                columns_sql.append(fk_sql)
            
            sql_statements.append(",\n".join(columns_sql))
            sql_statements.append(") ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;")
            sql_statements.append("")
            
            # 索引
            for idx in table_def.get('indexes', []):
                idx_type = idx.get('type', 'INDEX')
                idx_sql = f"CREATE {idx_type} idx_{table_name}_{idx['name']} ON {table_name}"
                if idx.get('columns'):
                    idx_sql += f" ({', '.join(idx['columns'])})"
                sql_statements.append(idx_sql + ";")
            
            sql_statements.append("")
        
        return "\n".join(sql_statements)
    
    def to_dict(self) -> Dict:
        """转换为字典"""
        return {
            'name': self.name,
            'description': self.description,
            'version': self.version,
            'created_at': self.created_at,
            'tables': self.tables
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'SchemaTemplate':
        """从字典创建"""
        template = cls(data['name'], data['description'])
        template.version = data.get('version', '1.0')
        template.created_at = data.get('created_at', datetime.now().isoformat())
        template.tables = data.get('tables', {})
        return template

class ExperimentSchemaManager:
    """实验表结构管理器"""
    
    def __init__(self, templates_dir: str = "db_server/schema_templates"):
        self.templates_dir = Path(templates_dir)
        self.templates_dir.mkdir(parents=True, exist_ok=True)
        
        # 创建默认模板
        self._create_default_templates()
    
    def _create_default_templates(self):
        """创建默认表结构模板"""
        
        # 1. 基础RAG模板
        basic_rag = SchemaTemplate("basic_rag", "基础RAG表结构")
        
        # 用户表
        basic_rag.add_table("users", [
            {"name": "id", "type": "BIGINT", "auto_increment": True, "primary_key": True},
            {"name": "name", "type": "VARCHAR(255)", "not_null": True, "comment": "用户名"},
            {"name": "email", "type": "VARCHAR(255)", "comment": "邮箱"},
            {"name": "created_at", "type": "DATETIME", "default": "CURRENT_TIMESTAMP"}
        ])
        
        # 文档表
        basic_rag.add_table("documents", [
            {"name": "id", "type": "BIGINT", "auto_increment": True, "primary_key": True},
            {"name": "user_id", "type": "BIGINT", "not_null": True},
            {"name": "title", "type": "VARCHAR(255)", "not_null": True},
            {"name": "content", "type": "LONGTEXT"},
            {"name": "created_at", "type": "DATETIME", "default": "CURRENT_TIMESTAMP"}
        ], foreign_keys=[
            {"column": "user_id", "ref_table": "users", "ref_column": "id", "on_delete": "CASCADE"}
        ])
        
        # 块表
        basic_rag.add_table("chunks", [
            {"name": "id", "type": "BIGINT", "auto_increment": True, "primary_key": True},
            {"name": "document_id", "type": "BIGINT", "not_null": True},
            {"name": "text", "type": "MEDIUMTEXT", "not_null": True},
            {"name": "sequence", "type": "INT", "default": 0},
            {"name": "created_at", "type": "DATETIME", "default": "CURRENT_TIMESTAMP"}
        ], foreign_keys=[
            {"column": "document_id", "ref_table": "documents", "ref_column": "id", "on_delete": "CASCADE"}
        ])
        
        self.save_template(basic_rag)
        
        # 2. 向量实验模板
        vector_exp = SchemaTemplate("vector_experiment", "向量实验表结构")
        
        # 向量表
        vector_exp.add_table("vectors", [
            {"name": "id", "type": "BIGINT", "auto_increment": True, "primary_key": True},
            {"name": "chunk_id", "type": "BIGINT", "not_null": True},
            {"name": "model_name", "type": "VARCHAR(100)", "not_null": True},
            {"name": "vector_dim", "type": "INT", "not_null": True},
            {"name": "milvus_id", "type": "VARCHAR(50)", "comment": "Milvus中的ID"},
            {"name": "created_at", "type": "DATETIME", "default": "CURRENT_TIMESTAMP"}
        ])
        
        # 检索日志表
        vector_exp.add_table("retrieval_logs", [
            {"name": "id", "type": "BIGINT", "auto_increment": True, "primary_key": True},
            {"name": "query_text", "type": "TEXT", "not_null": True},
            {"name": "model_name", "type": "VARCHAR(100)"},
            {"name": "top_k", "type": "INT", "default": 10},
            {"name": "results", "type": "JSON", "comment": "检索结果"},
            {"name": "duration_ms", "type": "FLOAT", "comment": "耗时（毫秒）"},
            {"name": "created_at", "type": "DATETIME", "default": "CURRENT_TIMESTAMP"}
        ])
        
        self.save_template(vector_exp)
        
        # 3. 灵活JSON模板
        flexible_json = SchemaTemplate("flexible_json", "灵活JSON字段表结构")
        
        # 灵活文档表
        flexible_json.add_table("flexible_documents", [
            {"name": "id", "type": "BIGINT", "auto_increment": True, "primary_key": True},
            {"name": "user_id", "type": "BIGINT", "not_null": True},
            {"name": "doc_type", "type": "VARCHAR(50)", "default": "unknown"},
            {"name": "metadata", "type": "JSON", "comment": "文档元数据"},
            {"name": "content", "type": "LONGTEXT"},
            {"name": "properties", "type": "JSON", "comment": "自定义属性"},
            {"name": "created_at", "type": "DATETIME", "default": "CURRENT_TIMESTAMP"},
            {"name": "updated_at", "type": "DATETIME", "default": "CURRENT_TIMESTAMP"}
        ])
        
        # 灵活实验表
        flexible_json.add_table("experiments", [
            {"name": "id", "type": "BIGINT", "auto_increment": True, "primary_key": True},
            {"name": "name", "type": "VARCHAR(100)", "not_null": True},
            {"name": "config", "type": "JSON", "comment": "实验配置"},
            {"name": "results", "type": "JSON", "comment": "实验结果"},
            {"name": "metrics", "type": "JSON", "comment": "评估指标"},
            {"name": "status", "type": "ENUM('running', 'completed', 'failed')", "default": "running"},
            {"name": "created_at", "type": "DATETIME", "default": "CURRENT_TIMESTAMP"}
        ])
        
        self.save_template(flexible_json)
        
        # 4. 图数据库模板
        graph_db = SchemaTemplate("graph_database", "图数据库表结构")
        
        # 节点表
        graph_db.add_table("nodes", [
            {"name": "id", "type": "BIGINT", "auto_increment": True, "primary_key": True},
            {"name": "node_id", "type": "VARCHAR(100)", "not_null": True},
            {"name": "node_type", "type": "VARCHAR(50)", "not_null": True},
            {"name": "properties", "type": "JSON"},
            {"name": "created_at", "type": "DATETIME", "default": "CURRENT_TIMESTAMP"}
        ])
        
        # 边表
        graph_db.add_table("edges", [
            {"name": "id", "type": "BIGINT", "auto_increment": True, "primary_key": True},
            {"name": "from_node", "type": "VARCHAR(100)", "not_null": True},
            {"name": "to_node", "type": "VARCHAR(100)", "not_null": True},
            {"name": "relation_type", "type": "VARCHAR(50)", "not_null": True},
            {"name": "weight", "type": "FLOAT", "default": 1.0},
            {"name": "properties", "type": "JSON"},
            {"name": "created_at", "type": "DATETIME", "default": "CURRENT_TIMESTAMP"}
        ])
        
        self.save_template(graph_db)
    
    def save_template(self, template: SchemaTemplate):
        """保存模板"""
        template_file = self.templates_dir / f"{template.name}.yaml"
        with open(template_file, 'w', encoding='utf-8') as f:
            yaml.dump(template.to_dict(), f, default_flow_style=False, allow_unicode=True)
    
    def load_template(self, name: str) -> Optional[SchemaTemplate]:
        """加载模板"""
        template_file = self.templates_dir / f"{name}.yaml"
        if not template_file.exists():
            return None
        
        with open(template_file, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
        
        return SchemaTemplate.from_dict(data)
    
    def list_templates(self) -> List[str]:
        """列出所有模板"""
        templates = []
        for file in self.templates_dir.glob("*.yaml"):
            templates.append(file.stem)
        return sorted(templates)
    
    def generate_schema_sql(self, template_name: str) -> Optional[str]:
        """生成模板的SQL"""
        template = self.load_template(template_name)
        if not template:
            return None
        
        return template.generate_sql()
    
    def create_custom_template(self, name: str, description: str, 
                              tables_config: Dict) -> SchemaTemplate:
        """创建自定义模板"""
        template = SchemaTemplate(name, description)
        
        for table_name, table_config in tables_config.items():
            template.add_table(
                table_name,
                table_config.get('columns', []),
                table_config.get('indexes', []),
                table_config.get('foreign_keys', [])
            )
        
        self.save_template(template)
        return template

def main():
    """命令行工具"""
    import argparse
    
    parser = argparse.ArgumentParser(description='实验表结构配置工具')
    parser.add_argument('--action', choices=['list', 'show', 'generate', 'create'], 
                       required=True, help='操作类型')
    parser.add_argument('--template', '-t', help='模板名称')
    parser.add_argument('--output', '-o', help='输出文件')
    parser.add_argument('--config', '-c', help='配置文件（JSON/YAML）')
    parser.add_argument('--name', '-n', help='新模板名称')
    parser.add_argument('--description', '-d', help='模板描述')
    
    args = parser.parse_args()
    
    manager = ExperimentSchemaManager()
    
    if args.action == 'list':
        templates = manager.list_templates()
        print(f"可用模板 ({len(templates)} 个):")
        for i, template in enumerate(templates, 1):
            print(f"{i}. {template}")
    
    elif args.action == 'show':
        if not args.template:
            print("请指定模板名称 --template <name>")
            return 1
        
        template = manager.load_template(args.template)
        if not template:
            print(f"未找到模板: {args.template}")
            return 1
        
        print(f"模板: {template.name}")
        print(f"描述: {template.description}")
        print(f"版本: {template.version}")
        print(f"创建时间: {template.created_at}")
        print(f"表数量: {len(template.tables)}")
        
        for table_name, table_def in template.tables.items():
            print(f"\n表: {table_name}")
            print(f"  列数: {len(table_def['columns'])}")
            print(f"  索引数: {len(table_def.get('indexes', []))}")
            print(f"  外键数: {len(table_def.get('foreign_keys', []))}")
    
    elif args.action == 'generate':
        if not args.template:
            print("请指定模板名称 --template <name>")
            return 1
        
        sql = manager.generate_schema_sql(args.template)
        if not sql:
            print(f"未找到模板: {args.template}")
            return 1
        
        if args.output:
            with open(args.output, 'w', encoding='utf-8') as f:
                f.write(sql)
            print(f"SQL已生成到: {args.output}")
        else:
            print(sql)
    
    elif args.action == 'create':
        if not args.name or not args.config:
            print("请指定模板名称和配置文件")
            print("用法: --name <name> --config <config.json>")
            return 1
        
        # 读取配置文件
        config_file = Path(args.config)
        if not config_file.exists():
            print(f"配置文件不存在: {args.config}")
            return 1
        
        with open(config_file, 'r', encoding='utf-8') as f:
            if config_file.suffix == '.json':
                config = json.load(f)
            else:
                config = yaml.safe_load(f)
        
        template = manager.create_custom_template(
            args.name,
            args.description or "",
            config
        )
        
        print(f"自定义模板创建成功: {template.name}")
        print(f"保存位置: {manager.templates_dir / (template.name + '.yaml')}")
    
    return 0

if __name__ == '__main__':
    import sys
    sys.exit(main()) 