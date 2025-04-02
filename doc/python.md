```markdown
### Python 3 默认版本与虚拟环境配置

1. **设置Python3为默认版本**
```bash
# 检查Python3版本
python3 --version

# 临时设置别名（仅在当前终端生效）
alias python='python3'
alias pip='pip3'

# 永久生效（写入bash配置文件）
echo "alias python='python3'" >> ~/.bashrc
echo "alias pip='pip3'" >> ~/.bashrc
source ~/.bashrc
```

2. **虚拟环境全流程操作**
```bash
# 查看当前虚拟环境路径
echo $VIRTUAL_ENV

# 删除旧虚拟环境
rm -rf myenv

# 创建新虚拟环境
python -m venv myenv

# 激活环境
source myenv/bin/activate

# 安装包示例
pip install numpy pandas
```

### Ollama 服务管理命令

```bash
# 检查服务状态
ps aux | grep ollama

# 启动服务（默认端口11434）
ollama serve

# 指定端口启动（解决端口冲突）
ollama serve --host 0.0.0.0:11435

# API连通性测试  不要走vpn 会502
curl http://localhost:11434/api/tags
```

### 故障排查命令

```bash
# 虚拟环境权限修复
chmod +x myenv/bin/activate

# 检查端口占用
lsof -i :11434

# 查看服务日志
tail -f ~/.ollama/logs/server.log
```
```