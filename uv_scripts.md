# UV 项目管理脚本

## 常用命令

### 环境管理
```bash
# 同步依赖项（安装/更新）
uv sync

# 同步开发依赖项
uv sync --extra dev

# 添加新依赖
uv add package_name

# 添加开发依赖
uv add --dev package_name

# 移除依赖
uv remove package_name
```

### 运行项目
```bash
# 运行服务器
uv run python start_server.py

# 运行测试
uv run python test_api.py

# 运行示例脚本
uv run python example_doc_retrieval.py
```

### 开发工具
```bash
# 代码格式化
uv run black .

# 导入排序
uv run isort .

# 代码检查
uv run flake8 .

# 运行测试
uv run pytest
```

### 环境信息
```bash
# 查看已安装的包
uv pip list

# 查看项目信息
uv info

# 查看虚拟环境位置
uv venv --show-path
```

## 迁移说明

项目已成功从 `requirements.txt` 迁移到 uv 管理：

1. ✅ 创建了 `pyproject.toml` 配置文件
2. ✅ 迁移了所有依赖项到 `pyproject.toml`
3. ✅ 生成了 `uv.lock` 锁定文件
4. ✅ 测试了环境兼容性

## 项目结构
- `pyproject.toml` - 项目配置和依赖管理
- `uv.lock` - 依赖锁定文件（确保可重现的安装）
- `.venv/` - 虚拟环境目录
- `requirements.txt` - 旧的依赖文件（可以保留作为参考或删除）