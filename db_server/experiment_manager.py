#!/usr/bin/env python3
"""
KnowledgeRAG 实验管理工具
作者: XYZ-Algorithm-Team
用途: 为研究员提供便捷的实验环境管理界面
"""

import os
import sys
import json
import yaml
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional
import subprocess

# 添加src目录到Python路径
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from manage_table import ExperimentManager
from experiment_schemas import ExperimentSchemaManager

class ExperimentCLI:
    """实验管理命令行界面"""
    
    def __init__(self):
        self.experiment_manager = ExperimentManager({
            'host': os.getenv('MYSQL_HOST', '127.0.0.1'),
            'port': int(os.getenv('MYSQL_PORT', 3306)),
            'user': os.getenv('MYSQL_USER', 'root'),
            'password': os.getenv('MYSQL_PASSWORD', 'devpass'),
            'charset': 'utf8mb4',
            'collation': 'utf8mb4_unicode_ci'
        })
        
        self.schema_manager = ExperimentSchemaManager()
        self.current_experiment = None
        self.config_file = Path("current_experiment.yaml")
        
        # 尝试连接数据库
        if not self.experiment_manager.connect():
            print("⚠️  无法连接到数据库，请检查配置和服务状态")
            sys.exit(1)
        
        # 加载当前实验配置
        self._load_current_experiment()
    
    def _load_current_experiment(self):
        """加载当前实验配置"""
        if self.config_file.exists():
            try:
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    config = yaml.safe_load(f)
                    self.current_experiment = config.get('current_experiment')
            except Exception as e:
                print(f"⚠️  加载实验配置失败: {e}")
    
    def _save_current_experiment(self):
        """保存当前实验配置"""
        try:
            config = {
                'current_experiment': self.current_experiment,
                'updated_at': datetime.now().isoformat()
            }
            with open(self.config_file, 'w', encoding='utf-8') as f:
                yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
        except Exception as e:
            print(f"⚠️  保存实验配置失败: {e}")
    
    def create_experiment(
        self, name: str, 
        researcher: str = "", 
        description: str = "", 
        template: str = "basic_rag",
        force: bool = False
    ):
        """创建新实验"""
        try:
            # 检查实验是否已存在
            if name in self.experiment_manager.list_experiments():
                print(f"❌ 实验 '{name}' 已存在")
                return False
            
            # 生成schema文件
            schema_sql = self.schema_manager.generate_schema_sql(template)
            if not schema_sql:
                print(f"❌ 未找到模板: {template}")
                return False
            
            # 创建临时schema文件
            temp_schema = Path(f"temp_schema_{name}.sql")
            with open(temp_schema, 'w', encoding='utf-8') as f:
                f.write(schema_sql)
            
            # 创建实验
            success = self.experiment_manager.create_experiment(
                experiment_name=name,
                researcher=researcher,
                description=description,
                schema_file=str(temp_schema)
            )
            
            # 清理临时文件
            temp_schema.unlink()
            
            if success:
                print(f"✅ 实验 '{name}' 创建成功")
                print(f"📊 数据库: knowledge_rag_{name}")
                print(f"📄 配置文件: experiments/{name}.yaml")
                
                # 自动切换到新实验
                self.switch_experiment(name)
                return True
            else:
                print(f"❌ 实验 '{name}' 创建失败")
                return False
                
        except Exception as e:
            print(f"❌ 创建实验失败: {e}")
            return False
    
    def list_experiments(self):
        """列出所有实验"""
        experiments = self.experiment_manager.list_experiments()
        
        if not experiments:
            print("📋 没有找到实验环境")
            print("💡 使用 'create' 命令创建新实验")
            return
        
        print(f"📋 实验环境列表 ({len(experiments)} 个):")
        print("-" * 80)
        
        for i, exp_name in enumerate(experiments, 1):
            # 获取实验信息
            info = self.experiment_manager.get_experiment_info(exp_name)
            
            # 当前实验标记
            current_marker = "👉 " if exp_name == self.current_experiment else "   "
            
            print(f"{current_marker}{i}. {exp_name}")
            
            if info:
                print(f"      研究员: {info.get('researcher', 'N/A')}")
                print(f"      描述: {info.get('description', 'N/A')}")
                print(f"      创建时间: {info.get('created_at', 'N/A')}")
                
                if info.get('tables'):
                    table_count = len(info['tables'])
                    total_records = sum(info['tables'].values())
                    print(f"      表数量: {table_count}, 总记录: {total_records}")
                
                if info.get('notes'):
                    print(f"      笔记数: {len(info['notes'])}")
            
            print()
    
    def switch_experiment(self, name: str):
        """切换实验"""
        experiments = self.experiment_manager.list_experiments()
        
        if name not in experiments:
            print(f"❌ 实验 '{name}' 不存在")
            return False
        
        try:
            # 切换实验
            self.experiment_manager.switch_experiment(name)
            self.current_experiment = name
            self._save_current_experiment()
            
            print(f"✅ 已切换到实验: {name}")
            
            # 显示实验信息
            info = self.experiment_manager.get_experiment_info(name)
            if info:
                print(f"📊 数据库: knowledge_rag_{name}")
                print(f"👤 研究员: {info.get('researcher', 'N/A')}")
                print(f"📝 描述: {info.get('description', 'N/A')}")
            
            return True
            
        except Exception as e:
            print(f"❌ 切换实验失败: {e}")
            return False
    
    def delete_experiment(self, name: str, force: bool = False):
        """删除实验"""
        experiments = self.experiment_manager.list_experiments()
        
        if name not in experiments:
            print(f"❌ 实验 '{name}' 不存在")
            return False
        
        # 安全确认
        if not force:
            print(f"⚠️  即将删除实验: {name}")
            confirm = input("确定要删除吗？这将删除所有数据！(y/N): ")
            if confirm.lower() != 'y':
                print("❌ 操作已取消")
                return False
        
        try:
            # 删除实验
            success = self.experiment_manager.delete_experiment(name)
            
            if success:
                print(f"✅ 实验 '{name}' 删除成功")
                
                # 如果删除的是当前实验，清除当前实验
                if name == self.current_experiment:
                    self.current_experiment = None
                    self._save_current_experiment()
                    print("🔄 当前实验已清除")
                
                return True
            else:
                print(f"❌ 实验 '{name}' 删除失败")
                return False
                
        except Exception as e:
            print(f"❌ 删除实验失败: {e}")
            return False
    
    def show_experiment_info(self, name: str):
        """显示实验详细信息"""
        if name not in self.experiment_manager.list_experiments():
            print(f"❌ 实验 '{name}' 不存在")
            return False
        
        info = self.experiment_manager.get_experiment_info(name)
        if not info:
            print(f"❌ 无法获取实验信息: {name}")
            return False
        
        print(f"📊 实验详情: {name}")
        print("=" * 50)
        print(f"👤 研究员: {info.get('researcher', 'N/A')}")
        print(f"📝 描述: {info.get('description', 'N/A')}")
        print(f"🕐 创建时间: {info.get('created_at', 'N/A')}")
        print(f"🗄️  数据库: {info.get('database', 'N/A')}")
        print(f"📋 Schema文件: {info.get('schema_file', 'N/A')}")
        
        # 表信息
        if info.get('tables'):
            print(f"\n📊 数据表 ({len(info['tables'])} 个):")
            for table, count in info['tables'].items():
                print(f"   {table}: {count} 条记录")
        
        # 自定义表
        if info.get('custom_tables'):
            print(f"\n🔧 自定义表 ({len(info['custom_tables'])} 个):")
            for table, details in info['custom_tables'].items():
                print(f"   {table}: {details.get('created_at', 'N/A')}")
        
        # 实验笔记
        if info.get('notes'):
            print(f"\n📝 实验笔记 ({len(info['notes'])} 条):")
            for note in info['notes'][-5:]:  # 显示最近5条
                print(f"   [{note['timestamp']}] {note['note']}")
        
        return True
    
    def add_note(self, experiment_name: str, note: str):
        """添加实验笔记"""
        if experiment_name not in self.experiment_manager.list_experiments():
            print(f"❌ 实验 '{experiment_name}' 不存在")
            return False
        
        try:
            success = self.experiment_manager.add_experiment_note(experiment_name, note)
            if success:
                print(f"✅ 笔记已添加到实验: {experiment_name}")
                return True
            else:
                print(f"❌ 添加笔记失败")
                return False
        except Exception as e:
            print(f"❌ 添加笔记失败: {e}")
            return False
    
    def list_templates(self):
        """列出可用模板"""
        templates = self.schema_manager.list_templates()
        
        if not templates:
            print("📋 没有找到模板")
            return
        
        print(f"📋 可用模板 ({len(templates)} 个):")
        print("-" * 50)
        
        for i, template_name in enumerate(templates, 1):
            template = self.schema_manager.load_template(template_name)
            print(f"{i}. {template_name}")
            
            if template:
                print(f"   描述: {template.description}")
                print(f"   表数量: {len(template.tables)}")
                print(f"   版本: {template.version}")
            print()
    
    def show_template_info(self, template_name: str):
        """显示模板详细信息"""
        template = self.schema_manager.load_template(template_name)
        
        if not template:
            print(f"❌ 模板 '{template_name}' 不存在")
            return False
        
        print(f"📄 模板详情: {template_name}")
        print("=" * 50)
        print(f"📝 描述: {template.description}")
        print(f"📊 版本: {template.version}")
        print(f"🕐 创建时间: {template.created_at}")
        print(f"📋 表数量: {len(template.tables)}")
        
        print(f"\n📊 表结构:")
        for table_name, table_def in template.tables.items():
            print(f"   {table_name}:")
            print(f"     列数: {len(table_def['columns'])}")
            print(f"     索引数: {len(table_def.get('indexes', []))}")
            print(f"     外键数: {len(table_def.get('foreign_keys', []))}")
        
        return True
    
    def generate_template_sql(self, template_name: str, output_file: str = None):
        """生成模板SQL"""
        sql = self.schema_manager.generate_schema_sql(template_name)
        
        if not sql:
            print(f"❌ 模板 '{template_name}' 不存在")
            return False
        
        if output_file:
            try:
                with open(output_file, 'w', encoding='utf-8') as f:
                    f.write(sql)
                print(f"✅ SQL已生成到: {output_file}")
            except Exception as e:
                print(f"❌ 保存SQL失败: {e}")
                return False
        else:
            print(f"📄 模板 '{template_name}' 的SQL:")
            print("-" * 50)
            print(sql)
        
        return True
    
    def status(self):
        """显示当前状态"""
        print("📊 KnowledgeRAG 实验环境状态")
        print("=" * 50)
        
        # 数据库连接状态
        print(f"🔗 数据库连接: {'✅ 已连接' if self.experiment_manager.connection else '❌ 未连接'}")
        
        # 当前实验
        if self.current_experiment:
            print(f"🎯 当前实验: {self.current_experiment}")
            
            # 显示实验信息
            info = self.experiment_manager.get_experiment_info(self.current_experiment)
            if info:
                print(f"👤 研究员: {info.get('researcher', 'N/A')}")
                print(f"📊 数据库: knowledge_rag_{self.current_experiment}")
                if info.get('tables'):
                    table_count = len(info['tables'])
                    total_records = sum(info['tables'].values())
                    print(f"📋 表数量: {table_count}, 总记录: {total_records}")
        else:
            print("🎯 当前实验: 未选择")
        
        # 实验总数
        experiments = self.experiment_manager.list_experiments()
        print(f"📂 总实验数: {len(experiments)}")
        
        # 可用模板
        templates = self.schema_manager.list_templates()
        print(f"📄 可用模板: {len(templates)}")
        
        # 配置信息
        print(f"\n🔧 配置信息:")
        print(f"   MySQL Host: {os.getenv('MYSQL_HOST', '127.0.0.1')}")
        print(f"   MySQL Port: {os.getenv('MYSQL_PORT', '3306')}")
        print(f"   MySQL User: {os.getenv('MYSQL_USER', 'root')}")
    
    def interactive_mode(self):
        """交互模式"""
        print("🚀 KnowledgeRAG 实验管理工具 (交互模式)")
        print("输入 'help' 查看帮助，'exit' 退出")
        print("-" * 50)
        
        while True:
            try:
                # 显示当前实验
                prompt = f"[{self.current_experiment or '无'}] > "
                command = input(prompt).strip()
                
                if not command:
                    continue
                
                if command in ['exit', 'quit']:
                    print("👋 再见！")
                    break
                
                if command == 'help':
                    self._show_help()
                    continue
                
                # 解析命令
                args = command.split()
                cmd = args[0]
                
                if cmd == 'create':
                    if len(args) < 2:
                        print("❌ 用法: create <实验名> [研究员] [描述] [模板]")
                        continue
                    
                    name = args[1]
                    researcher = args[2] if len(args) > 2 else ""
                    description = args[3] if len(args) > 3 else ""
                    template = args[4] if len(args) > 4 else "basic_rag"
                    
                    self.create_experiment(name, researcher, description, template)
                
                elif cmd == 'list':
                    self.list_experiments()
                
                elif cmd == 'switch':
                    if len(args) < 2:
                        print("❌ 用法: switch <实验名>")
                        continue
                    self.switch_experiment(args[1])
                
                elif cmd == 'delete':
                    if len(args) < 2:
                        print("❌ 用法: delete <实验名>")
                        continue
                    self.delete_experiment(args[1])
                
                elif cmd == 'info':
                    if len(args) < 2:
                        if self.current_experiment:
                            self.show_experiment_info(self.current_experiment)
                        else:
                            print("❌ 请指定实验名或先切换到实验")
                        continue
                    self.show_experiment_info(args[1])
                
                elif cmd == 'status':
                    self.status()
                
                elif cmd == 'templates':
                    self.list_templates()
                
                elif cmd == 'note':
                    if len(args) < 2:
                        print("❌ 用法: note <笔记内容>")
                        continue
                    if not self.current_experiment:
                        print("❌ 请先切换到实验")
                        continue
                    note = " ".join(args[1:])
                    self.add_note(self.current_experiment, note)
                
                else:
                    print(f"❌ 未知命令: {cmd}")
                    print("💡 输入 'help' 查看帮助")
                
            except KeyboardInterrupt:
                print("\n👋 再见！")
                break
            except Exception as e:
                print(f"❌ 命令执行失败: {e}")
    
    def _show_help(self):
        """显示帮助信息"""
        print("🆘 命令帮助:")
        print("-" * 50)
        print("create <名称> [研究员] [描述] [模板]  - 创建新实验")
        print("list                                - 列出所有实验")
        print("switch <名称>                       - 切换实验")
        print("delete <名称>                       - 删除实验")
        print("info [名称]                         - 显示实验信息")
        print("note <内容>                         - 添加实验笔记")
        print("templates                           - 列出可用模板")
        print("status                              - 显示状态")
        print("help                                - 显示帮助")
        print("exit                                - 退出")
    
    def cleanup(self):
        """清理资源"""
        if self.experiment_manager.connection:
            self.experiment_manager.disconnect()

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='KnowledgeRAG 实验管理工具')
    parser.add_argument('--interactive', '-i', action='store_true', help='交互模式')
    parser.add_argument('--create', help='创建实验')
    parser.add_argument('--researcher', '-r', help='研究员名称')
    parser.add_argument('--description', '-d', help='实验描述')
    parser.add_argument('--template', '-t', default='basic_rag', help='模板名称')
    parser.add_argument('--list', '-l', action='store_true', help='列出实验')
    parser.add_argument('--switch', '-s', help='切换实验')
    parser.add_argument('--delete', help='删除实验')
    parser.add_argument('--info', help='显示实验信息')
    parser.add_argument('--status', action='store_true', help='显示状态')
    parser.add_argument('--templates', action='store_true', help='列出模板')
    parser.add_argument('--force', '-f', action='store_true', help='强制执行')
    
    args = parser.parse_args()
    
    # 创建CLI实例
    cli = ExperimentCLI()
    
    try:
        if args.interactive:
            cli.interactive_mode()
        elif args.create:
            cli.create_experiment(
                args.create,
                args.researcher or "",
                args.description or "",
                args.template
            )
        elif args.list:
            cli.list_experiments()
        elif args.switch:
            cli.switch_experiment(args.switch)
        elif args.delete:
            cli.delete_experiment(args.delete, args.force)
        elif args.info:
            cli.show_experiment_info(args.info)
        elif args.status:
            cli.status()
        elif args.templates:
            cli.list_templates()
        else:
            # 默认显示状态
            cli.status()
            print("\n💡 使用 --help 查看帮助，--interactive 进入交互模式")
        
        return 0
        
    except Exception as e:
        print(f"❌ 程序执行失败: {e}")
        return 1
    
    finally:
        cli.cleanup()

if __name__ == '__main__':
    sys.exit(main()) 