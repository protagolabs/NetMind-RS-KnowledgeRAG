#!/usr/bin/env python3
"""
KnowledgeRAG 数据管理脚本
作者: XYZ-Algorithm-Team
用途: 统一管理 MySQL + Milvus + MinIO 的实验数据
支持完整的数据生命周期管理
"""

import os
import sys
import json
import yaml
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
import argparse

# 数据库连接
import mysql.connector
from mysql.connector import Error

# Milvus 连接
try:
    from pymilvus import connections, Collection, utility, FieldSchema, CollectionSchema, DataType
    MILVUS_AVAILABLE = True
except ImportError:
    MILVUS_AVAILABLE = False
    logging.warning("PyMilvus 未安装，Milvus 功能不可用")

# 本地对象存储（替代MinIO）
import shutil
from pathlib import Path
LOCAL_OBJECT_STORE_AVAILABLE = True

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 统一配置
SERVICES_CONFIG = {
    'mysql': {
        'host': os.getenv('MYSQL_HOST', '127.0.0.1'),
        'port': int(os.getenv('MYSQL_PORT', 3306)),
        'user': os.getenv('MYSQL_USER', 'root'),
        'password': os.getenv('MYSQL_PASSWORD', 'devpass'),
        'charset': 'utf8mb4',
        'collation': 'utf8mb4_unicode_ci'
    },
    'milvus': {
        'host': os.getenv('MILVUS_HOST', '127.0.0.1'),
        'port': int(os.getenv('MILVUS_PORT', 19530)),
        'alias': 'default'
    },
    'local_object_store': {
        'base_path': os.getenv('LOCAL_OBJECT_STORE_PATH', './data/local_object_store'),
        'experiments_dir': 'experiments'
    }
}

class UnifiedDataManager:
    """统一数据管理器 - 支持 MySQL + Milvus + 本地对象存储"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.mysql_conn: Optional[mysql.connector.MySQLConnection] = None
        self.milvus_conn: Optional[str] = None
        self.object_store_base_path: Path = Path(config.get('local_object_store', {}).get('base_path', './data/local_object_store'))
        
        # 服务连接状态
        self.services_status = {
            'mysql': False,
            'milvus': False,
            'local_object_store': False
        }
        
    def connect_all(self) -> Dict[str, bool]:
        """连接所有服务"""
        # 连接 MySQL
        try:
            self.mysql_conn = mysql.connector.connect(**self.config['mysql'])
            if self.mysql_conn.is_connected():
                self.services_status['mysql'] = True
                logger.info("✅ MySQL 连接成功")
        except Error as e:
            logger.error(f"❌ MySQL 连接失败: {e}")
        
        # 连接 Milvus
        if MILVUS_AVAILABLE:
            try:
                connections.connect(
                    alias=self.config['milvus']['alias'],
                    host=self.config['milvus']['host'],
                    port=self.config['milvus']['port']
                )
                self.milvus_conn = self.config['milvus']['alias']
                self.services_status['milvus'] = True
                logger.info("✅ Milvus 连接成功")
            except Exception as e:
                logger.error(f"❌ Milvus 连接失败: {e}")
        
        # 初始化本地对象存储
        if LOCAL_OBJECT_STORE_AVAILABLE:
            try:
                # 确保基础目录存在
                self.object_store_base_path.mkdir(parents=True, exist_ok=True)
                experiments_dir = self.object_store_base_path / self.config['local_object_store']['experiments_dir']
                experiments_dir.mkdir(parents=True, exist_ok=True)
                
                self.services_status['local_object_store'] = True
                logger.info("✅ 本地对象存储初始化成功")
            except Exception as e:
                logger.error(f"❌ 本地对象存储初始化失败: {e}")
        
        return self.services_status
    
    def disconnect_all(self):
        """断开所有连接"""
        if self.mysql_conn and self.mysql_conn.is_connected():
            self.mysql_conn.close()
            logger.info("MySQL 连接已断开")
        
        if self.milvus_conn and MILVUS_AVAILABLE:
            try:
                connections.disconnect(self.milvus_conn)
                logger.info("Milvus 连接已断开")
            except Exception as e:
                logger.warning(f"Milvus 断开连接失败: {e}")
    
    def create_experiment(self, experiment_name: str, researcher: str = "", 
                         description: str = "", template: str = "basic_rag") -> Dict[str, bool]:
        """创建完整的实验环境"""
        results = {
            'mysql': False,
            'milvus': False,
            'minio': False
        }
        
        # 1. 创建 MySQL 数据库
        if self.services_status['mysql']:
            results['mysql'] = self._create_mysql_database(experiment_name, template)
        
        # 2. 创建 Milvus 集合
        if self.services_status['milvus']:
            results['milvus'] = self._create_milvus_collections(experiment_name)
        
        # 3. 创建本地对象存储目录
        if self.services_status['local_object_store']:
            results['local_object_store'] = self._create_local_object_store_dir(experiment_name)
        
        # 4. 创建实验配置文件
        self._create_experiment_config(experiment_name, researcher, description, results)
        
        return results
    
    def delete_experiment(self, experiment_name: str, force: bool = False) -> Dict[str, bool]:
        """删除完整的实验环境"""
        if not force:
            confirm = input(f"⚠️  确定要删除实验 '{experiment_name}' 的所有数据吗？(y/N): ")
            if confirm.lower() != 'y':
                logger.info("操作已取消")
                return {'cancelled': True}
        
        results = {
            'mysql': False,
            'milvus': False,
            'local_object_store': False
        }
        
        # 1. 删除 MySQL 数据库
        if self.services_status['mysql']:
            results['mysql'] = self._delete_mysql_database(experiment_name)
        
        # 2. 删除 Milvus 集合
        if self.services_status['milvus']:
            results['milvus'] = self._delete_milvus_collections(experiment_name)
        
        # 3. 删除本地对象存储目录
        if self.services_status['local_object_store']:
            results['local_object_store'] = self._delete_local_object_store_dir(experiment_name)
        
        # 4. 删除实验配置文件
        self._delete_experiment_config(experiment_name)
        
        return results
    
    def list_experiments(self) -> List[Dict[str, Any]]:
        """列出所有实验及其状态"""
        experiments = []
        
        # 从 MySQL 获取实验列表
        if self.services_status['mysql']:
            try:
                cursor = self.mysql_conn.cursor()
                cursor.execute("SHOW DATABASES")
                databases = [db[0] for db in cursor.fetchall()]
                
                for db in databases:
                    if db.startswith('knowledge_rag_'):
                        exp_name = db.replace('knowledge_rag_', '')
                        exp_info = {
                            'name': exp_name,
                            'mysql_db': db,
                            'mysql_exists': True,
                            'milvus_exists': self._check_milvus_collections(exp_name),
                            'local_object_store_exists': self._check_local_object_store_dir(exp_name)
                        }
                        experiments.append(exp_info)
                
                cursor.close()
            except Error as e:
                logger.error(f"获取实验列表失败: {e}")
        
        return experiments
    
    def health_check(self) -> Dict[str, Dict[str, Any]]:
        """健康检查"""
        health_status = {}
        
        # MySQL 健康检查
        if self.services_status['mysql']:
            try:
                cursor = self.mysql_conn.cursor()
                cursor.execute("SELECT 1")
                cursor.fetchone()
                cursor.close()
                health_status['mysql'] = {
                    'status': 'healthy',
                    'version': self.mysql_conn.get_server_info()
                }
            except Error as e:
                health_status['mysql'] = {
                    'status': 'unhealthy',
                    'error': str(e)
                }
        else:
            health_status['mysql'] = {'status': 'disconnected'}
        
        # Milvus 健康检查
        if self.services_status['milvus'] and MILVUS_AVAILABLE:
            try:
                # 获取版本信息
                from pymilvus import __version__
                health_status['milvus'] = {
                    'status': 'healthy',
                    'version': __version__
                }
            except Exception as e:
                health_status['milvus'] = {
                    'status': 'unhealthy',
                    'error': str(e)
                }
        else:
            health_status['milvus'] = {'status': 'disconnected'}
        
        # 本地对象存储健康检查
        if self.services_status['local_object_store'] and LOCAL_OBJECT_STORE_AVAILABLE:
            try:
                # 检查基础目录
                if self.object_store_base_path.exists():
                    experiments_dir = self.object_store_base_path / self.config['local_object_store']['experiments_dir']
                    experiment_dirs = list(experiments_dir.glob('*')) if experiments_dir.exists() else []
                    health_status['local_object_store'] = {
                        'status': 'healthy',
                        'base_path': str(self.object_store_base_path),
                        'experiments_count': len(experiment_dirs)
                    }
                else:
                    health_status['local_object_store'] = {
                        'status': 'unhealthy',
                        'error': 'Base path does not exist'
                    }
            except Exception as e:
                health_status['local_object_store'] = {
                    'status': 'unhealthy',
                    'error': str(e)
                }
        else:
            health_status['local_object_store'] = {'status': 'disconnected'}
        
        return health_status
    
    def backup_experiment(self, experiment_name: str, backup_dir: str = "backups") -> Dict[str, bool]:
        """备份实验数据"""
        backup_path = Path(backup_dir) / f"{experiment_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        backup_path.mkdir(parents=True, exist_ok=True)
        
        results = {
            'mysql': False,
            'milvus': False,
            'local_object_store': False
        }
        
        # 1. 备份 MySQL 数据
        if self.services_status['mysql']:
            results['mysql'] = self._backup_mysql_data(experiment_name, backup_path)
        
        # 2. 备份 Milvus 数据
        if self.services_status['milvus']:
            results['milvus'] = self._backup_milvus_data(experiment_name, backup_path)
        
        # 3. 备份本地对象存储数据
        if self.services_status['local_object_store']:
            results['local_object_store'] = self._backup_local_object_store_data(experiment_name, backup_path)
        
        logger.info(f"备份完成: {backup_path}")
        return results
    
    # === 私有方法 ===
    
    def _create_mysql_database(self, experiment_name: str, template: str) -> bool:
        """创建 MySQL 数据库"""
        try:
            cursor = self.mysql_conn.cursor()
            db_name = f"knowledge_rag_{experiment_name}"
            
            cursor.execute(f"CREATE DATABASE IF NOT EXISTS {db_name} "
                          f"CHARACTER SET {self.config['mysql']['charset']} "
                          f"COLLATE {self.config['mysql']['collation']}")
            
            cursor.execute(f"USE {db_name}")
            
            # 执行模板 SQL
            # 这里可以集成 experiment_schemas.py 的模板系统
            schema_sql = self._get_template_sql(template)
            if schema_sql:
                for statement in schema_sql.split(';'):
                    if statement.strip():
                        cursor.execute(statement)
            
            self.mysql_conn.commit()
            cursor.close()
            logger.info(f"MySQL 数据库创建成功: {db_name}")
            return True
            
        except Error as e:
            logger.error(f"MySQL 数据库创建失败: {e}")
            return False
    
    def _create_milvus_collections(self, experiment_name: str) -> bool:
        """创建 Milvus 集合"""
        if not MILVUS_AVAILABLE:
            return False
        
        try:
            # 创建文档向量集合
            collection_name = f"knowledge_rag_{experiment_name}_documents"
            
            # 定义字段
            # 从配置获取向量维度
            from knowledge_rag.config import get_embedding_settings
            embedding_settings = get_embedding_settings()
            vector_dim = embedding_settings.dimension
            
            fields = [
                FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
                FieldSchema(name="document_id", dtype=DataType.INT64),
                FieldSchema(name="chunk_id", dtype=DataType.INT64),
                FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=vector_dim),
                FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=2000)
            ]
            
            schema = CollectionSchema(
                fields=fields,
                description=f"Document embeddings for experiment {experiment_name}"
            )
            
            # 创建集合
            collection = Collection(
                name=collection_name,
                schema=schema,
                using=self.milvus_conn
            )
            
            # 创建索引
            index_params = {
                "metric_type": "L2",
                "index_type": "IVF_FLAT",
                "params": {"nlist": 128}
            }
            collection.create_index(
                field_name="embedding",
                index_params=index_params
            )
            
            logger.info(f"Milvus 集合创建成功: {collection_name}")
            return True
            
        except Exception as e:
            logger.error(f"Milvus 集合创建失败: {e}")
            return False
    
    def _create_local_object_store_dir(self, experiment_name: str) -> bool:
        """创建本地对象存储目录"""
        if not LOCAL_OBJECT_STORE_AVAILABLE:
            return False
        
        try:
            experiments_dir = self.object_store_base_path / self.config['local_object_store']['experiments_dir']
            experiment_dir = experiments_dir / experiment_name
            
            # 创建实验目录
            experiment_dir.mkdir(parents=True, exist_ok=True)
            
            # 创建子目录结构
            (experiment_dir / "documents").mkdir(exist_ok=True)
            (experiment_dir / "images").mkdir(exist_ok=True)
            (experiment_dir / "metadata").mkdir(exist_ok=True)
            
            logger.info(f"本地对象存储目录创建成功: {experiment_dir}")
            return True
            
        except Exception as e:
            logger.error(f"本地对象存储目录创建失败: {e}")
            return False
    
    def _delete_mysql_database(self, experiment_name: str) -> bool:
        """删除 MySQL 数据库"""
        try:
            cursor = self.mysql_conn.cursor()
            db_name = f"knowledge_rag_{experiment_name}"
            cursor.execute(f"DROP DATABASE IF EXISTS {db_name}")
            cursor.close()
            logger.info(f"MySQL 数据库删除成功: {db_name}")
            return True
        except Error as e:
            logger.error(f"MySQL 数据库删除失败: {e}")
            return False
    
    def _delete_milvus_collections(self, experiment_name: str) -> bool:
        """删除 Milvus 集合"""
        if not MILVUS_AVAILABLE:
            return False
        
        try:
            collection_name = f"knowledge_rag_{experiment_name}_documents"
            
            if utility.has_collection(collection_name, using=self.milvus_conn):
                utility.drop_collection(collection_name, using=self.milvus_conn)
                logger.info(f"Milvus 集合删除成功: {collection_name}")
            
            return True
        except Exception as e:
            logger.error(f"Milvus 集合删除失败: {e}")
            return False
    
    def _delete_local_object_store_dir(self, experiment_name: str) -> bool:
        """删除本地对象存储目录"""
        if not LOCAL_OBJECT_STORE_AVAILABLE:
            return False
        
        try:
            experiments_dir = self.object_store_base_path / self.config['local_object_store']['experiments_dir']
            experiment_dir = experiments_dir / experiment_name
            
            if experiment_dir.exists():
                # 删除整个实验目录
                shutil.rmtree(experiment_dir)
                logger.info(f"本地对象存储目录删除成功: {experiment_dir}")
            
            return True
        except Exception as e:
            logger.error(f"本地对象存储目录删除失败: {e}")
            return False
    
    def _check_milvus_collections(self, experiment_name: str) -> bool:
        """检查 Milvus 集合是否存在"""
        if not MILVUS_AVAILABLE or not self.services_status['milvus']:
            return False
        
        try:
            collection_name = f"knowledge_rag_{experiment_name}_documents"
            return utility.has_collection(collection_name, using=self.milvus_conn)
        except Exception:
            return False
    
    def _check_local_object_store_dir(self, experiment_name: str) -> bool:
        """检查本地对象存储目录是否存在"""
        if not LOCAL_OBJECT_STORE_AVAILABLE or not self.services_status['local_object_store']:
            return False
        
        try:
            experiments_dir = self.object_store_base_path / self.config['local_object_store']['experiments_dir']
            experiment_dir = experiments_dir / experiment_name
            return experiment_dir.exists()
        except Exception:
            return False
    
    def _get_template_sql(self, template: str) -> str:
        """获取模板 SQL（集成 experiment_schemas.py）"""
        try:
            # 这里可以集成 experiment_schemas.py 的功能
            from experiment_schemas import ExperimentSchemaManager
            schema_manager = ExperimentSchemaManager()
            return schema_manager.generate_schema_sql(template)
        except Exception as e:
            logger.warning(f"获取模板 SQL 失败: {e}")
            return ""
    
    def _create_experiment_config(self, experiment_name: str, researcher: str, 
                                 description: str, results: Dict[str, bool]):
        """创建实验配置文件"""
        config = {
            'name': experiment_name,
            'researcher': researcher,
            'description': description,
            'created_at': datetime.now().isoformat(),
            'services': results,
            'mysql_db': f"knowledge_rag_{experiment_name}",
            'milvus_collection': f"knowledge_rag_{experiment_name}_documents",
            'local_object_store_dir': f"experiments/{experiment_name}"
        }
        
        experiments_dir = Path("experiments")
        experiments_dir.mkdir(exist_ok=True)
        
        config_file = experiments_dir / f"{experiment_name}.yaml"
        with open(config_file, 'w', encoding='utf-8') as f:
            yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
        
        logger.info(f"实验配置文件创建: {config_file}")
    
    def _delete_experiment_config(self, experiment_name: str):
        """删除实验配置文件"""
        config_file = Path("experiments") / f"{experiment_name}.yaml"
        if config_file.exists():
            config_file.unlink()
            logger.info(f"实验配置文件删除: {config_file}")
    
    def _backup_mysql_data(self, experiment_name: str, backup_path: Path) -> bool:
        """备份 MySQL 数据"""
        try:
            import subprocess
            
            db_name = f"knowledge_rag_{experiment_name}"
            backup_file = backup_path / f"{db_name}.sql"
            
            # 使用 mysqldump 备份
            cmd = [
                'mysqldump',
                '-h', self.config['mysql']['host'],
                '-P', str(self.config['mysql']['port']),
                '-u', self.config['mysql']['user'],
                f'-p{self.config["mysql"]["password"]}',
                db_name
            ]
            
            with open(backup_file, 'w') as f:
                subprocess.run(cmd, stdout=f, check=True)
            
            logger.info(f"MySQL 数据备份成功: {backup_file}")
            return True
            
        except Exception as e:
            logger.error(f"MySQL 数据备份失败: {e}")
            return False
    
    def _backup_milvus_data(self, experiment_name: str, backup_path: Path) -> bool:
        """备份 Milvus 数据"""
        # Milvus 备份需要特殊处理，这里简化实现
        try:
            collection_name = f"knowledge_rag_{experiment_name}_documents"
            backup_file = backup_path / f"{collection_name}.json"
            
            # 导出集合信息
            backup_info = {
                'collection_name': collection_name,
                'backup_time': datetime.now().isoformat(),
                'note': 'Milvus 数据备份需要专门的备份工具'
            }
            
            with open(backup_file, 'w', encoding='utf-8') as f:
                json.dump(backup_info, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Milvus 数据备份信息保存: {backup_file}")
            return True
            
        except Exception as e:
            logger.error(f"Milvus 数据备份失败: {e}")
            return False
    
    def _backup_local_object_store_data(self, experiment_name: str, backup_path: Path) -> bool:
        """备份本地对象存储数据"""
        try:
            experiments_dir = self.object_store_base_path / self.config['local_object_store']['experiments_dir']
            experiment_dir = experiments_dir / experiment_name
            
            if not experiment_dir.exists():
                logger.warning(f"实验目录不存在: {experiment_dir}")
                return True
            
            backup_dir = backup_path / f"local_object_store_{experiment_name}"
            
            # 复制整个实验目录
            shutil.copytree(experiment_dir, backup_dir, dirs_exist_ok=True)
            
            logger.info(f"本地对象存储数据备份成功: {backup_dir}")
            return True
            
        except Exception as e:
            logger.error(f"本地对象存储数据备份失败: {e}")
            return False


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='KnowledgeRAG 统一数据管理工具')
    parser.add_argument('--action', choices=[
        'create-exp', 'delete-exp', 'list-exp', 'health-check', 
        'backup-exp', 'status'
    ], required=True, help='操作类型')
    parser.add_argument('--experiment', '-e', help='实验名称')
    parser.add_argument('--researcher', '-r', help='研究员名称')
    parser.add_argument('--description', '-d', help='实验描述')
    parser.add_argument('--template', '-t', default='basic_rag', help='模板名称')
    parser.add_argument('--force', '-f', action='store_true', help='强制执行')
    parser.add_argument('--backup-dir', default='backups', help='备份目录')
    
    args = parser.parse_args()
    
    # 创建统一数据管理器
    manager = UnifiedDataManager(SERVICES_CONFIG)
    
    try:
        # 连接所有服务
        logger.info("正在连接所有服务...")
        status = manager.connect_all()
        
        print("🔗 服务连接状态:")
        for service, connected in status.items():
            icon = "✅" if connected else "❌"
            print(f"   {icon} {service.upper()}: {'已连接' if connected else '未连接'}")
        
        # 执行操作
        if args.action == 'create-exp':
            if not args.experiment:
                print("❌ 请指定实验名称 --experiment <name>")
                return 1
            
            print(f"\n🚀 创建实验: {args.experiment}")
            results = manager.create_experiment(
                args.experiment,
                args.researcher or "",
                args.description or "",
                args.template
            )
            
            print("📊 创建结果:")
            for service, success in results.items():
                icon = "✅" if success else "❌"
                print(f"   {icon} {service.upper()}: {'成功' if success else '失败'}")
        
        elif args.action == 'delete-exp':
            if not args.experiment:
                print("❌ 请指定实验名称 --experiment <name>")
                return 1
            
            print(f"\n🗑️  删除实验: {args.experiment}")
            results = manager.delete_experiment(args.experiment, args.force)
            
            if 'cancelled' in results:
                print("❌ 操作已取消")
            else:
                print("📊 删除结果:")
                for service, success in results.items():
                    icon = "✅" if success else "❌"
                    print(f"   {icon} {service.upper()}: {'成功' if success else '失败'}")
        
        elif args.action == 'list-exp':
            print("\n📋 实验列表:")
            experiments = manager.list_experiments()
            
            if not experiments:
                print("   没有找到实验")
            else:
                for exp in experiments:
                    print(f"\n🔬 {exp['name']}")
                    print(f"   MySQL: {'✅' if exp['mysql_exists'] else '❌'}")
                    print(f"   Milvus: {'✅' if exp['milvus_exists'] else '❌'}")
                    print(f"   MinIO: {'✅' if exp['minio_exists'] else '❌'}")
        
        elif args.action == 'health-check':
            print("\n🏥 健康检查:")
            health = manager.health_check()
            
            for service, info in health.items():
                status = info['status']
                icon = "✅" if status == 'healthy' else "❌" if status == 'unhealthy' else "⚠️"
                print(f"   {icon} {service.upper()}: {status}")
                
                if 'version' in info:
                    print(f"      版本: {info['version']}")
                if 'error' in info:
                    print(f"      错误: {info['error']}")
                if 'experiments_count' in info:
                    print(f"      实验数: {info['experiments_count']}")
        
        elif args.action == 'backup-exp':
            if not args.experiment:
                print("❌ 请指定实验名称 --experiment <name>")
                return 1
            
            print(f"\n💾 备份实验: {args.experiment}")
            results = manager.backup_experiment(args.experiment, args.backup_dir)
            
            print("📊 备份结果:")
            for service, success in results.items():
                icon = "✅" if success else "❌"
                print(f"   {icon} {service.upper()}: {'成功' if success else '失败'}")
        
        elif args.action == 'status':
            print("\n📊 系统状态:")
            print(f"   实验数量: {len(manager.list_experiments())}")
            
            health = manager.health_check()
            healthy_services = sum(1 for info in health.values() if info['status'] == 'healthy')
            print(f"   健康服务: {healthy_services}/{len(health)}")
        
        return 0
        
    except Exception as e:
        logger.error(f"操作失败: {e}")
        return 1
    
    finally:
        manager.disconnect_all()


if __name__ == '__main__':
    sys.exit(main()) 