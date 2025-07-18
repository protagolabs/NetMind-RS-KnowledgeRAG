# KnowledgeRAG 实验管理系统

## 🚀 快速开始

### 1. 启动所有服务

以下的命令可以用 `docker` 也可以用 `docker-compose` 

```bash
cd db_server
docker-compose up -d
```

### 2. 验证服务状态
```bash
docker-compose ps
```

### 3. 查看服务日志
```bash
# 查看所有服务日志
docker-compose logs -f

# 查看特定服务日志
docker-compose logs -f mysql
docker-compose logs -f milvus
docker-compose logs -f python-app
```

### 4. 停止服务
```bash
# 停止服务
docker-compose down

# 停止服务并删除数据卷
docker-compose down -v
```

## 📋 服务信息

| 服务 | 端口 | 用途 | 连接信息 |
|------|------|------|----------|
| **MySQL** | 3306 | 关系型数据库 | `mysql://rag_user:rag_password@localhost:3306/knowledge_rag` |
| **Milvus** | 19530, 9091 | 向量数据库 | `localhost:19530` |
| **本地对象存储** | - | 对象存储 | `./data/local_object_store` |
| **Python** | - | 应用环境 | `docker exec -it knowledge_rag_python bash` |

## 🛠️ 核心工具

### 1. 实验管理器 (`experiment_manager.py`)
**用途**：提供友好的实验管理界面，支持实验的创建、切换、删除等操作。

#### 交互模式（推荐）
```bash
python experiment_manager.py --interactive
```

#### 命令行模式
```bash
# 创建实验
python experiment_manager.py --create my_experiment --researcher "张三" --description "向量搜索实验"

# 列出所有实验
python experiment_manager.py --list

# 切换实验
python experiment_manager.py --switch my_experiment

# 删除实验
python experiment_manager.py --delete my_experiment --force

# 查看实验信息
python experiment_manager.py --info my_experiment

# 显示状态
python experiment_manager.py --status
```

### 2. 统一数据管理器 (`experiment_data.py`)
**用途**：底层数据管理，支持 MySQL + Milvus + MinIO 三种数据源的统一管理。

#### 健康检查
```bash
python experiment_data.py --action health-check
```

#### 创建完整实验环境
```bash
python experiment_data.py --action create-exp --experiment test_exp --researcher "李四" --template basic_rag
```

#### 列出所有实验
```bash
python experiment_data.py --action list-exp
```

#### 删除实验（包含所有数据）
```bash
python experiment_data.py --action delete-exp --experiment test_exp --force
```

#### 备份实验数据
```bash
python experiment_data.py --action backup-exp --experiment test_exp
```

### 3. 表结构配置工具 (`experiment_schemas.py`)
**用途**：管理数据库表结构模板，支持不同实验场景的表结构定义。

#### 查看可用模板
```bash
python experiment_schemas.py --action list
```

#### 查看模板详情
```bash
python experiment_schemas.py --action show --template basic_rag
```

#### 生成模板SQL
```bash
python experiment_schemas.py --action generate --template basic_rag --output schema.sql
```

#### 创建自定义模板
```bash
python experiment_schemas.py --action create --name custom_template --config config.json
```

## 📄 预设模板

系统提供了多种预设的表结构模板，适用于不同的实验场景：

| 模板名称 | 用途 | 包含表 |
|----------|------|--------|
| `basic_rag` | 基础RAG实验 | users, documents, chunks |
| `vector_experiment` | 向量搜索实验 | vectors, retrieval_logs |
| `flexible_json` | 灵活JSON实验 | flexible_documents, experiments |
| `graph_database` | 图数据库实验 | nodes, edges |

## 🔄 典型工作流程

### 1. 创建新实验
```bash
# 方式1：交互式创建（推荐）
python experiment_manager.py --interactive

# 方式2：命令行创建
python experiment_manager.py --create my_experiment --researcher "张三" --template basic_rag
```

### 2. 切换到实验
```bash
python experiment_manager.py --switch my_experiment
```

### 3. 查看实验状态
```bash
# 查看当前实验状态
python experiment_manager.py --status

# 查看数据存储状态
python experiment_data.py --action health-check
```

### 4. 管理实验数据
```bash
# 备份实验数据
python experiment_data.py --action backup-exp --experiment my_experiment

# 删除实验（谨慎操作）
python experiment_manager.py --delete my_experiment
```

## 🗂️ 数据存储结构

每个实验会在三个数据源中创建对应的存储空间：

```
实验名称: my_experiment
├── MySQL 数据库: knowledge_rag_my_experiment
├── Milvus 集合: knowledge_rag_my_experiment_documents
└── 本地对象存储: ./data/local_object_store/experiments/my_experiment/
    ├── documents/      # 文档文件
    ├── images/         # 图片文件
    └── metadata/       # 元数据文件
```

## 🔧 常用命令

```bash
# 服务管理
docker-compose restart           # 重启服务
docker-compose ps               # 查看服务状态
docker-compose logs -f mysql    # 查看MySQL日志

# 环境管理
docker exec -it knowledge_rag_python bash   # 进入Python环境
docker network ls | grep knowledge_rag      # 查看网络

# 系统清理
docker system prune -f          # 清理未使用的资源
```

## 📚 环境变量配置

可以通过环境变量自定义连接配置：

```bash
# MySQL 配置
export MYSQL_HOST=127.0.0.1
export MYSQL_PORT=3306
export MYSQL_USER=root
export MYSQL_PASSWORD=devpass

# Milvus 配置
export MILVUS_HOST=127.0.0.1
export MILVUS_PORT=19530

# 本地对象存储配置
export LOCAL_OBJECT_STORE_PATH=./data/local_object_store
```

## 📁 数据持久化

所有数据保存在 `./data/` 目录：
- `./data/mysql/` - MySQL 数据文件
- `./data/milvus/` - Milvus 向量数据
- `./data/local_object_store/` - 本地对象存储

## 🚨 故障排查

### 依赖问题
```bash
# 安装Python依赖
pip install pymilvus mysql-connector-python pyyaml

# 检查依赖是否安装成功
python -c "import pymilvus; print('✅ PyMilvus 可用')"
python -c "import mysql.connector; print('✅ MySQL 可用')"
python -c "import pathlib; print('✅ 本地对象存储 可用')"
```

### 服务连接问题
```bash
# 检查服务健康状态
python experiment_data.py --action health-check

# 检查端口占用
netstat -tulpn | grep :3306   # MySQL
netstat -tulpn | grep :19530  # Milvus

# 停止占用端口的服务
sudo lsof -ti:3306 | xargs sudo kill -9
```

### 数据不一致问题
```bash
# 列出所有实验的数据存储状态
python experiment_data.py --action list-exp

# 检查特定实验的数据完整性
python experiment_manager.py --info my_experiment
```

### 权限问题
```bash
# 修复数据目录权限
sudo chown -R $(id -u):$(id -g) ./data/

# 修复实验配置文件权限
sudo chown -R $(id -u):$(id -g) ./experiments/
```

### 服务启动失败
```bash
# 查看详细日志
docker-compose logs mysql
docker-compose logs milvus
docker-compose logs python-app

# 重新构建和启动
docker-compose down
docker-compose up -d --force-recreate

# 检查服务状态
docker-compose ps
```

### 实验管理问题
```bash
# 如果实验管理器出现问题，可以重置配置
rm -f current_experiment.yaml

# 重新进入交互模式
python experiment_manager.py --interactive
```

## 📊 监控和维护

### 定期健康检查
```bash
# 创建健康检查脚本
cat > health_check.sh << 'EOF'
#!/bin/bash
echo "=== 服务健康检查 ==="
python experiment_data.py --action health-check

echo -e "\n=== 实验状态 ==="
python experiment_manager.py --status

echo -e "\n=== 磁盘使用情况 ==="
du -sh ./data/*
EOF

chmod +x health_check.sh
./health_check.sh
```

### 数据备份建议
```bash
# 定期备份重要实验
python experiment_data.py --action backup-exp --experiment important_exp

# 备份配置文件
cp -r experiments/ backups/experiments_$(date +%Y%m%d)
cp current_experiment.yaml backups/
```

## 🎯 版本信息

- MySQL: 8.0.42
- Milvus: v2.5.14
- MinIO: latest
- Python: 3.12-slim
- PyMilvus: 自动检测
- MinIO Python Client: 自动检测

## 📚 更多信息

### 相关文档
- [README_IMPLEMENTATION.md](../README_IMPLEMENTATION.md) - 实现说明
- [EXPERIMENT_GUIDE.md](../EXPERIMENT_GUIDE.md) - 实验指南
- [experiments.md](../experiments.md) - 详细项目规划

### 核心文件说明
- `experiment_manager.py` - 实验管理器，提供交互式界面
- `experiment_data.py` - 统一数据管理器，支持多数据源
- `experiment_schemas.py` - 表结构配置工具，管理模板
- `docker-compose.yml` - 服务编排文件

### 快速参考

#### 常用命令速查表
| 操作 | 命令 |
|------|------|
| 启动服务 | `docker-compose up -d` |
| 健康检查 | `python experiment_data.py --action health-check` |
| 交互模式 | `python experiment_manager.py --interactive` |
| 创建实验 | `python experiment_manager.py --create <name>` |
| 切换实验 | `python experiment_manager.py --switch <name>` |
| 查看状态 | `python experiment_manager.py --status` |
| 列出实验 | `python experiment_data.py --action list-exp` |
| 备份实验 | `python experiment_data.py --action backup-exp --experiment <name>` |

#### 目录结构
```
db_server/
├── docker-compose.yml          # 服务编排
├── experiment_manager.py       # 实验管理器
├── experiment_data.py          # 统一数据管理
├── experiment_schemas.py       # 表结构配置
├── schema_templates/           # 表结构模板
├── experiments/                # 实验配置文件
├── data/                       # 数据持久化
│   ├── mysql/                  # MySQL 数据
│   ├── milvus/                 # Milvus 数据
│   └── minio/                  # MinIO 数据
└── backups/                    # 备份文件
```

## 🎯 开始使用

如果您是第一次使用本系统，建议按以下步骤开始：

1. **启动服务**：`docker-compose up -d`
2. **检查健康**：`python experiment_data.py -o'kction health-check`
3. **进入交互模式**：`python experiment_manager.py --interactive`
4. **创建第一个实验**：使用交互模式创建实验
5. **开始研究**：使用实验环境进行数据处理和分析

**享受您的实验之旅！** 🚀 