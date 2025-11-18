<<<<<<< HEAD
# Python与JavaScript包管理机制对比分析

## 核心差异分析

### JavaScript（Node.js）的模块系统
- **本地依赖优先**  
  Node.js默认将依赖安装在项目目录的 `node_modules` 下，每个项目的依赖完全独立，天然支持多版本共存
- **路径解析规则**  
  模块加载时优先从当前目录的 `node_modules` 逐级向上查找，无需手动配置路径

### Python的全局优先策略
- **全局 site-packages**  
  Python默认将第三方库安装在全局的 `site-packages` 目录中，所有项目共享同一环境，易引发版本冲突
- **路径管理依赖 sys.path**  
  通过环境变量和 `sys.path` 列表查找包，需要手动管理路径或使用虚拟环境隔离依赖

## JavaScript实现本地化的技术优势
### 模块系统设计
- Node.js的模块加载规则（优先本地 `node_modules`）与 `require()` 函数的路径解析机制，使得依赖隔离无需额外配置

### 包管理工具深度集成
- `npm`/`yarn` 从设计之初即支持项目级依赖管理，与语言运行时深度绑定
- Python的 `pip` 作为后期工具，需兼容历史设计

## 总结
Python依赖管理的"非本地化"源于其语言设计的历史路径和模块系统限制，而JavaScript的Node.js通过 `node_modules` 机制实现了开箱即用的本地依赖隔离。推荐Python开发者使用 `Poetry` 或 `Pipenv` 等现代工具实现类似体验。

---

# 浏览器沙盒环境解析

## 定义与核心机制
**沙盒环境（Sandbox）**是计算机安全领域的隔离机制，通过限制程序对系统资源（文件系统/网络/硬件等）的访问权限，防止恶意代码破坏宿主系统。

## 浏览器沙盒核心特性
### 隔离性
- **进程隔离**  
  现代浏览器（如Chrome）采用多进程架构，各网页/扩展程序运行在独立渲染进程中
- **作用域隔离**  
  通过同源策略限制JavaScript仅能访问同协议、同域名、同端口的资源
- **API访问限制**  
  禁止直接操作系统本地文件/硬件，只能通过标准化API进行受控访问

### 权限控制
- **资源访问限制**  
  通过内容安全策略（CSP）限制脚本加载来源
- **权限分级**  
  敏感操作（如摄像头访问）需用户显式授权

### 安全边界
- **虚拟化技术**  
  利用操作系统级隔离（如Linux seccomp、Windows Hyper-V）
- **异常监控**  
  实时检测高频网络请求/内存泄漏等危险行为

## 具体实现机制
### 架构设计
- **多进程架构**  
  主进程/渲染进程/GPU进程相互独立，渲染进程通过IPC与主进程通信

### 安全策略
- **同源策略**  
  阻止跨域数据访问（如跨站点Cookie窃取）
- **CSP策略**  
  通过HTTP头或`<meta>`标签限制脚本来源

### API封装
- **沙盒化Web API**  
  `localStorage`/`fetch`等API受域名隔离限制
- **动态代码限制**  
  禁用危险函数（如`eval()`），使用V8引擎内存隔离机制

## 意义与局限性
### 核心价值
- 防御恶意代码攻击
- 确保单页面崩溃不波及整体
- 提供安全调试环境

### 现存挑战
- 进程隔离带来的性能开销
- 零日漏洞可能导致沙盒逃逸

## 总结
浏览器沙盒通过**进程隔离+权限控制+虚拟化技术**构建安全边界，是现代Web安全的核心保障，尽管存在性能损耗和潜在风险，仍是不可替代的安全基石。
=======
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
>>>>>>> f24a3d5328409c16f73925739a17bc04729b8fda
