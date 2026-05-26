# SmartAnalysis

智能问数与预测模拟沙盘 — 通用领域数据分析 Web 应用。

支持上传任意 CSV/Excel 表格，自动完成数据探查、交互式数据清洗、AI 辅助探索分析、多元线性回归建模，并通过交互式滑块沙盘进行 What-If 模拟预测。

---

## 目录

- [架构概览](#架构概览)
- [项目结构](#项目结构)
- [快速启动](#快速启动)
- [五板块功能说明](#五板块功能说明)
  - [Tab 1：数据接入](#tab-1数据接入)
  - [Tab 2：数据理解](#tab-2数据理解)
  - [Tab 3：智能分析](#tab-3智能分析)
  - [Tab 4：模型训练](#tab-4模型训练)
  - [Tab 5：预测沙盘](#tab-5预测沙盘)
- [API 接口文档](#api-接口文档)
- [技术栈](#技术栈)
- [设计原则](#设计原则)

---

## 架构概览

```
                  ┌─────────────────────────────┐
                  │     后端 (FastAPI :8000)       │
                  │                             │
                  │  main.py                    │
                  │  services/                  │
                  │    ├─ data_engine.py        │
                  │    ├─ data_profiler.py      │
                  │    ├─ data_cleaner.py       │
                  │    ├─ analysis_registry.py  │
                  │    ├─ analysis_functions.py │
                  │    ├─ llm_agent.py          │
                  │    └─ ml_trainer.py         │
                  │                             │
                  │  uploads/                   │
                  └──────────┬──────────────────┘
                             │ HTTP JSON / SSE
┌────────────────────────────┼────────────────────────────┐
│     前端 (Vanilla JS)       │                            │
│                             │                            │
│  index.html  +  app.js  +  style.css                    │
│  ECharts 5.5 渲染                                         │
│  sessionStorage (LLM 配置 / 模型参数)                     │
└──────────────────────────────────────────────────────────┘
```

- **前端**：纯静态 HTML/CSS/JS，通过任意 HTTP 服务器托管（`python -m http.server` 或 nginx）
- **后端**：Python FastAPI，端口 8000，提供 RESTful JSON API + SSE 流式端点
- **状态管理**：前端 sessionStorage 保存 LLM 配置、登录令牌和模型参数
- **LLM 安全**：API Key 仅存浏览器端，每次请求随 Body 透传，后端使用后就清理
- **用户认证**：SQLite + 自实现 JWT，支持注册（注册后一键登录）、登录、管理员权限管理、用户自助注销
- **异步流式**：`POST /analyze` 使用 SSE（Server-Sent Events），四阶段流式输出

---

## 项目结构

```
project/
├── README.md
├── requirements.txt
│
├── backend/
│   ├── main.py                     # FastAPI 入口，11 个路由
│   ├── auth.db                     # SQLite 用户数据库（自动创建）
│   ├── services/
│   │   ├── __init__.py
│   │   ├── auth.py                 # 用户认证（注册/登录/JWT）
│   │   ├── data_engine.py          # 通用表格读取 + Schema 提取
│   │   ├── data_profiler.py        # 数据画像（列统计、异常检测、质量评分）
│   │   ├── data_cleaner.py         # 数据清洗操作引擎
│   │   ├── analysis_registry.py    # 分析函数注册表（24 个函数）
│   │   ├── analysis_functions.py   # 分析函数实现
│   │   ├── llm_agent.py            # LLM 代理（流式 + 规划 + 解读）
│   │   └── ml_trainer.py           # 线性回归训练管道（含诊断）
│   └── uploads/                    # 用户上传文件存放处
│
├── frontend/
│   ├── index.html                  # 5 个 Tab 面板 + 全局设置模态框
│   ├── app.js                      # 全前端逻辑
│   └── style.css                   # 设计系统
│
└── data/
    └── Online Retail.xlsx          # 示例数据集
```

---

## 快速启动

### 前置条件

- Python 3.9+
- 依赖安装：

```bash
pip install -r requirements.txt
```

### 启动

**终端 1 — 后端：**

```bash
cd backend
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

访问 `http://localhost:8000/docs` 查看 Swagger API 文档。

**终端 2 — 前端：**

```bash
cd frontend
python -m http.server 3000
```

浏览器打开 `http://localhost:3000`。首次使用需在左侧注册账号（注册成功后弹窗选择"立即登录"），然后配置 LLM。

---

## 五板块功能说明

### Tab 1：数据接入

- **拖拽上传** CSV / Excel 文件（.csv / .xls / .xlsx）
- **自动 Schema 提取**：列名、类型推断（数值/分类/日期/文本/布尔/标识符）、缺失率、样本数据。日期检测采用抽样+阈值（80%可解析），ID 列自动识别（命名模式 + 高基数整数），文本列加入平均长度判断
- **编码容错**：UTF-8 → GBK → GB18030 → Latin-1 逐级回退
- 左侧实时展示数据概览面板（行列数 + 字段列表 + 缺失率进度条）
- **条件解锁**：数据解析成功后，若 >= 2 列数值型则自动激活 Tab 4/5
- 上传成功后自动跳转至 Tab 2「数据理解」
- **注册即登录**：注册成功后弹窗询问"是否立即登录"，确认后自动登录进入应用
- **用户自助注销**：登录后可在账号区域点击注销，删除个人账号及数据

### Tab 2：数据理解

**步骤条串联两个阶段：**

**阶段 2a — 数据总览：**

| 模块 | 内容 |
|------|------|
| 摘要卡片 | 行数、列数、完整度%、重复行数 |
| 字段分类图 | 数值/分类/日期/文本/布尔/标识符 各 N 列（环形图） |
| AI 数据摘要 | LLM 自动生成：数据包含什么、适合/不适合回答什么问题 |
| 样例数据 | 前 5 行可滚动表格 |
| 字段详情卡片 | 每列的类型标签、缺失率、sparkline 迷你分布图、top 值 |

**阶段 2b — 数据清洗（AI 推荐 + 人工确认）：**

- **缺失值**：逐列统计 NaN 率，AI 推荐处理策略（数值用中位数，分类用众数），用户可逐列调整
- **异常值**：IQR + Z-score 双重检测，选项：保留 / 截断 / 剔除
- **重复行**：自动检测，AI 推荐删除
- **常量列**：自动识别，AI 推荐删除
- **低方差列**：唯一值占比 < 0.1% 近乎常量，建议删除
- **重复列检测**：数值列对皮尔逊相关系数 > 0.999，标记为冗余
- **缺失值关联模式**：检测两列缺失是否强关联（如 A 缺失时 B 也大概率缺失）
- **数据类型建议**：识别存为文本的日期、低基数数值→分类、文本布尔值等类型问题
- 一键「应用清洗」→ 清洗后的数据另存为新文件，自动更新全站状态

### Tab 3：智能分析

**分析笔记本模式**：左侧分析历史时间线 + 右侧分析卡片区。

**四阶段 SSE 流式分析流程**：

```
用户提问
  → 阶段 0: 概念翻译 + 列映射（AI 将"高价值客户"翻译为 P80 分位数，将"销售额"映射到 UnitPrice*Quantity）
  → 阶段 1: 分析规划（AI 从 20 个分析函数中选择 1-4 个，规划执行方案）
  → 阶段 2: 执行（Python 逐项执行分析，每完成一项实时推送结果 + 渲染图表）
  → 阶段 3: 解读（AI 流式输出自然语言结论 + 追问建议）
```

**内置 20 个分析函数，覆盖四层分析需求**：

| 层级 | 函数 | 说明 |
|------|------|------|
| 分布 | `describe_distribution` `describe_frequency` `detect_outliers` `test_normality` `analyze_composition` | 数值分布画像、分类频次、异常检测、正态检验、构成占比 |
| 关系 | `correlate` `cross_tabulate` `compare_groups` `compare_pairs` `correlation_matrix` `dimension_reduce` `cluster_analysis` | 相关性、交叉分析、分组比较、配对比较、PCA、聚类 |
| 趋势 | `timeseries_line` `timeseries_decompose` `timeseries_growth` `moving_average` `rank_top_n` `pareto_analysis` | 时序折线、三要素分解、增长率、移动平均、排名、帕累托 |
| 质量 | `profile_missing` `profile_duplicates` | 缺失值画像、重复值分析 |

AI 只做决策推荐（规划 + 解读），不生成代码；分析由预定义 Python 函数安全执行。

**AI 分析鲁棒性设计（2026-05-26 强化）**：

针对 LLM 输出不稳定的固有问题，实施了三层防御：

| 层次 | 位置 | 机制 |
|------|------|------|
| **Prompt 层** | `llm_agent.py` `_COMBINED_PROMPT` | 丰富列描述（类型标签 + 范围 + 高频值 + 样本值），引导 LLM 推断列的业务含义再匹配用户意图；硬性约束 actions 与 column_mappings 的列名一致性；computed 列必须先声明再引用；无法分析时引导 LLM 诚实说明而非硬编 |
| **校验层** | `main.py` `/analyze` | 执行前遍历每个 action 的 params，通过 registry 的 input_schema 识别哪些参数是列引用，与 `analysis_df` 实际列名比对。不存在的列立即返回明确错误（含"当前可用列: [...]"），阻止 KeyError 传播到 LLM 解读阶段 |
| **解读层** | `llm_agent.py` `_INTERPRETATION_PROMPT` | 映射日志和列缺失信息注入 LLM 解读上下文，LLM 可解释具体哪些列缺失及原因；全部失败时用"根据上传数据，无法完成本次分析，因为……"诚实说明，不编造结论 |

### Tab 4：模型训练

- **目标变量 Y**：单选下拉框（仅数值列）
- **影响因素 X**：多选 chip 组件 + "全选所有"/"清空"批量操作
- **互斥防御**：切换 Y 时自动从 X 中移除同名列
- **工业级 5 阶段清洗管道**：数值化 → Y 缺失删行 → X 中位数填补 → 拟合 + 统计量提取
- **结果展示**：R² 环形仪表盘 + 回归方程 + 系数列表（正负分色）+ 标准化特征重要度
- **Ridge/Lasso 标准化**：模型对比前对训练集应用 StandardScaler，确保正则化项在不同量纲特征间公平施加
- **极值绑定**：返回每个特征的 P1/P99 分位数作为滑块范围，避免极端异常值撑大滑块
- **训练前预览**：`/preview-train` 端点先检查数据质量，展示有效样本数、NaN 丢弃率、非数值列警告后再训练
- **高基数检测**：自动识别 CustomerID 等唯一值占比 >50% 的特征，标记为"可能是 ID 列"并警告
- **非数值拦截**：选了 Country、Description 等文本列作为特征时，后端明确返回"不支持用于回归分析"而非系统级报错

### Tab 5：预测沙盘 (What-If Simulation)

- **左列**：每个特征一个滑块卡片（range + number 双向同步，默认值 = 均值）
- **右列**：超大预测数字 + CountUp 滚动动画 + ECharts 仪表盘

### 全局错误处理

- **统一错误弹窗**：所有用户可见的错误（上传失败、训练失败、SSE 异常、401 过期）均通过居中模态弹窗展示，含标题 + 详情 + 建议 + 关闭按钮，不再出现原生 `alert()` 或系统级报错原文
- **SSE 流异常兜底**：LLM 调用失败、列映射失败、解读失败三个阶段均有 `event: error` 事件推送给前端，用户能明确知道哪个阶段出了问题
- **上传 50 MB 限制**：前端 + 后端双重校验，超限弹窗提示

### 权限控制

- **管理员页面隔离**：仅 admin 角色可见「用户管理」Tab 标签页，非 admin 登录后标签页自动隐藏
- **路由级守卫**：切换 Tab 时校验 admin 权限，非授权访问被拦截并提示
- **退出自动跳转**：管理员退出登录后若当前在管理页面，自动跳转至数据接入
- **账号自助注销**：普通用户可自主注销账号，DELETE `/users/{username}` 支持 admin 删任意用户 + 普通用户自删

### UI 设计优化

- **侧边栏统一**：数据接入 (17rem) 与智能分析 (16rem) 侧边栏宽度协调
- **预测沙盘比例**：滑块面板占比优化，右侧图表区域更宽敞
- **页面宽度放宽**：模型训练 (max-w-4xl) 和用户管理 (max-w-5xl) 在大屏幕上不再局促

---

## API 接口文档

Base URL: `http://localhost:8000`

### `GET /ping`

健康检查。

### `POST /register`

注册新账号。注册成功后在弹窗中选择"立即登录"即可自动登录。

**请求：** `{"username": "...", "password": "..."}`
**响应：** `{"ok": true, "user": {"username": "...", "role": "user"}}`

### `POST /login`

登录并获取 JWT 令牌（7 天有效）。

**请求：** `{"username": "...", "password": "..."}`
**响应：** `{"ok": true, "token": "eyJ...", "username": "...", "role": "..."}`

### `GET /users`（需登录 + admin 角色）

管理员查看所有注册用户列表。

**请求头：** `Authorization: Bearer <token>`
**响应：** `{"users": [{"id": 1, "username": "admin", "role": "admin", "created_at": "..."}]}`

### `DELETE /users/{username}`（需登录）

删除用户账号。admin 可删除任意用户，普通用户仅可注销自己的账号。

**请求头：** `Authorization: Bearer <token>`
**响应：** `{"ok": true}`

### `POST /upload`

上传 CSV/Excel 文件，返回表结构 Schema。

### `POST /data-profile`

数据画像。返回逐列分布统计、sparkline 数据、异常值计数、数据质量指标。

### `POST /clean-data`

执行清洗操作（缺失填充、异常处理、去重、常量列删除），返回清洗后的新文件路径和 Schema。

### `POST /analyze` (SSE)

AI 辅助探索分析。请求体包含用户问题、文件信息和 LLM 配置。响应为 `text/event-stream`：

```
event: translation   → 概念翻译 + 列映射
event: plan          → 分析计划 + 函数选择
event: progress      → 执行进度
event: result        → 单步分析结果（图表数据 + 统计 + 表格）
event: interpretation → AI 自然语言解读（逐 token 流式）
event: done          → 完成
event: error         → 错误信息
```

### `POST /preview-train`

训练前数据质量预览（2026-05-26 新增）。请求体同 `/train`，返回有效样本数、NaN 率、非数值列警告。

### `POST /chat`

自然语言问答（同步，用于 AI 数据摘要等轻量场景）。

### `POST /advanced-analysis`

自动化 EDA（直方图、热力图、分类饼图、时序折线图）。保留兼容。

### `POST /train`

训练多元线性回归模型。

---

## 技术栈

| 层 | 技术 | 用途 |
|---|---|---|
| 后端框架 | FastAPI (Python) | REST API + SSE 流式 + 自动文档 |
| 数据处理 | Pandas + NumPy | 表格读写、清洗、统计 |
| 科学计算 | SciPy | 统计检验（t-test, ANOVA, 卡方, Shapiro-Wilk 等） |
| 机器学习 | scikit-learn | LinearRegression + SimpleImputer + PCA + KMeans |
| LLM 集成 | httpx (async streaming) | OpenAI-compatible API，流式 token 转发 |
| 前端渲染 | Vanilla HTML/CSS/JS | 零框架依赖 |
| 样式 | Tailwind CSS (CDN) + 自定义 CSS | 浅色极简设计系统 |
| 图表 | ECharts 5.5 | 仪表盘、热力图、散点图、柱状图、折线图、箱线图等 |
| 字体 | Inter (Google Fonts) + Noto Sans SC | 全局排版 |

---

## 设计原则

1. **零硬编码列名**：数据处理管道完全由 Schema 驱动，适配任意 CSV/Excel 表格
2. **前后端严格解耦**：后端只返回 JSON，前端负责全部渲染和状态管理
3. **AI 决策 + Python 执行**：AI 负责理解意图、规划方案、解读结果；Python 负责安全执行分析计算。AI 不生成代码
4. **LLM 可插拔**：API Key / Base URL / Model 通过全局设置模态框配置，支持任意 OpenAI-compatible 服务
5. **安全第一**：API Key 仅存浏览器 sessionStorage，后端不落盘；AI 不生成代码，分析函数参数由后端校验
6. **工业级容错**：编码回退链、静默数值化、中位数填补、JSON 序列化安全 — 脏数据不崩溃
7. **性能优先**：50k+ 行采样、Excel → CSV 缓存、rAF 批量图表更新、SSE 流式输出、CountUp 动画
8. **领域无关**：从"销售沙盘"完全通用化为"预测模拟沙盘"，支持金融/医疗/工业/气象等任意领域
9. **用户主导分析**：分析不是预设的静态图表，而是由用户提问驱动，AI 辅助规划，结果可解释

---

## 近期更新 (2026-05-26)

### 字段分类引擎增强

- **日期检测容错**：从整列 `pd.to_datetime(errors="raise")` 改为抽样 200 个值 + 80% 成功率阈值，少量脏数据不再导致整列误判
- **ID 列自动识别**：新增 `identifier` 类型，自动识别 `*_id`, `*_no`, `*编号`, `*code` 等命名模式以及高基数整数列（唯一值占比 > 90%），自动排除在训练特征之外
- **文本判断强化**：object 列增加平均字符串长度判断（>80 字符优先判为 text），避免短文本高唯一值小数据集被误判为 categorical

### 数据质量检测扩展

- **低方差列检测**：唯一值占比 < 0.1% 的近乎常量列
- **重复列检测**：皮尔逊相关系数 > 0.999 的数值列对
- **缺失值关联模式**：检测列间缺失的强关联关系（条件概率 > 80%）
- **反向类型建议**：低基数数值 (nunique ≤ 5) → 提示改为分类；文本布尔值 (是/否, Yes/No 等) → 提示改为布尔

### ML 训练改进

- **Ridge/Lasso 标准化**：`_compare_models()` 中使用 `StandardScaler` 对训练/测试集标准化后再拟合，确保正则化项在量纲不同的特征间公平施加
