# DEA-DD — 双目标柔性作业车间调度 (FJSP) 算法实现

**目标**：最小化 **makespan** (`f1`) 与 **总加权提前/拖期 (TWTE)** (`f2`)，并考虑
**随机加工时间**（对数正态，蒙特卡洛 simheuristic）。本工程是对 `算法设计.md`
中 DEA-DD（双解码 + 自适应）算法的完整 Python 实现。

## 环境

- **conda 虚拟环境名**：`DEA-DD`（Python 3.8，`D:\Environment\anaconda\envs\DEA-DD`）
- **依赖**：`numpy`、`matplotlib`（见 `requirements.txt`）

```powershell
conda create -n DEA-DD python=3.8
conda activate DEA-DD
pip install -r requirements.txt
```

## 运行

```powershell
# 使用内置合成算例（10 工件 × 6 机器，Mk01 规模）快速冒烟：
python -m dea_dd.main --N 50 --N_eo 20 --Iter_EO 40 --gen 100 --R_short 20 --R_long 10000

# 使用已下载的 Brandimarte 算例：
python -m dea_dd.main --instance data/Brandimarte/BrandimarteMk1.fjs --seed 0
```

完整参数（默认值即 §11 推荐）：`--N 100 --N_eo 40 --Iter_EO 80 --gen 400
--R_short 20 --R_long 10000 --ul 0.25 --seed 0`。

除命令行传参外，也可在 Python 中以编程方式调用：

```python
from dea_dd.params import Params
from dea_dd.main import run

params = Params(N=50, N_eo=20, Iter_EO=40, gen=100, R_short=20, R_long=10000)
run(params, inst_path="data/Brandimarte/新建文件夹/BrandimarteMk1.fjs")
```

### 关键命令行参数

| 参数 | 默认 | 含义 |
| --- | --- | --- |
| `--instance` | 合成算例 | `.fjs` 文件路径（缺省则用内置 10×6 合成算例） |
| `--N` | 100 | 主种群规模 |
| `--N_eo` / `--Iter_EO` | 40 / 80 | 每个 w 子问题的 EO 粒子数 / 迭代数 |
| `--gen` | 400 | 阶段二 NSGA-II 代数 |
| `--R_short` / `--R_long` | 20 / 10000 | 迭代短仿真 / 最终长仿真情景数 |
| `--ul` | 0.25 | 加工时间不确定度 UL ∈ {0.1, 0.25, 0.5} |
| `--seed` | 0 | 随机种子 |

## 算法流程（三阶段）

1. **阶段一 · 多 w 切比雪夫分解预热**（`algorithms/eo.py` / `algorithms/preheat.py`）：
   EO 实值双变量编码 `(u, v)`，在 `W = {0, 1, 0.1, 0.9, 0.25, 0.75, 0.5}`（两端优先）
   上逐 w 单目标优化，每个 w 按归一化拥挤距离取 K 个种子，形成初始种群（固定名义加工时间）。
   - 机器坐标越界**只用反射修复**，不用截断（`core/encoding.py` 的 `reflect_coord`）。
2. **阶段二 · 改进 NSGA-II 主搜索**（`algorithms/nsga2.py` / `algorithms/local_search.py`）：
   POX/UX 交叉、动态变异位数（式 16）、非支配排序 + 归一化拥挤距离、**延迟触发的自适应局部搜索**
   （左移 L1-L3 缩 makespan，右移 R1a/R1b 消早到；成功率轮盘赌，式 17/18）。
   随机对数正态加工时间 + 蒙特卡洛 `R_short`（CRN）。
3. **阶段三 · 长仿真**（`simulation/stochastic.py` 的 `mc_evaluate_long`）：对非支配前沿每解做
   `R_long = 10000` 次仿真，输出期望 ± 标准差与稳健性 gap `Δo = (f_s − f_d) / f_d`。

**双解码**：左移解码（active，无空闲）算 `f1`；右移解码（`RightShiftRepair`）算
`f2`，把早到工件贴到交货期、不新增拖期。

## 模块结构（按功能分层）

```
dea_dd/
  params.py              全局参数（N / N_eo / gen / UL … 默认值即 §11）
  main.py                总流程编排、CLI 与编程式入口
  compare.py             对比实验入口（DEA-DD vs 传统 NSGA-II）
  core/                  核心数据与解码
    instance.py          Instance、.fjs 解析、交货期生成 (Bagheri & Zandieh 2011)
    encoding.py          MS+OS 编码、EO 实值粒子、反射修复、粒子解码
    decoding.py          左移解码 / 右移解码 / 双解码评估、TWTE
    normalization.py     逐代动态 min-max 归一化 + 切比雪夫标量化 (式 9-11)
    pareto.py            支配 / 非支配排序 / 拥挤距离
  algorithms/            搜索算法
    eo.py                均衡优化器 EO (式 13-15)
    preheat.py           阶段一多 w 预热与种子选择
    nsga2.py             阶段二交叉 / 变异 / 环境选择
    local_search.py      双向邻域 + 自适应算子选择 (式 17/18)
  baselines/             基线对比算法
    nsga2.py             传统 NSGA-II 基线（随机初始化 + 固定变异，无 EO/局部搜索）
  simulation/
    stochastic.py        对数正态加工时间 + 蒙特卡洛评估 (式 19/19a/20)
  diagnostics/           诊断与输出
    metrics.py           HV / IGD / C-metric / NPS / 稳健性 gap
    plots.py             5 类诊断图（相同 w 的点连线）
    logging_setup.py     logging 控制台 + 文件（不用 print）
    generator.py         合成算例生成（离线回退）
```

## 日志

全部关键步骤用 `logging` 输出（**未使用 `print`**），同时写入 **控制台** 与
**按算法区分的日志文件**（滚动追加）：DEA-DD → `logs/dea_dd.log`、传统
NSGA-II → `logs/nsga2.log`、对比汇总 → `logs/compare.log`。除常规阶段信息外，
日志还额外输出：

- EO 阶段每个 w 生成的候选解中**不重复解个数**；
- EO 阶段合并去重后**共计不重复解个数**；
- NSGA-II 阶段最终生成的不重复解个数（非支配前沿规模）。

> 日志与图片的输出路径都按 **DEA-DD 项目根目录** 锚定（`dea_dd/paths.py` 的
> `PROJECT_ROOT`），因此从控制台（任意 cwd）或 PyCharm 运行都会写到同一处。

## 诊断图（DEA-DD 输出到 `outputs/dea_dd/`，基线输出到 `outputs/nsga2/`）

1. `eo_convergence.png` — EO 阶段 makespan / TWTE 随迭代变化
2. `eo_w_fronts.png` — EO 阶段每个 w 生成解的帕累托前沿（相同 w 的点连线）
3. `eo_initial_front.png` — EO 阶段初始种群（种子）的帕累托前沿（相同 w 连线）
4. `nsga2_convergence.png` — NSGA-II 阶段 makespan / TWTE 随代数变化
5. `final_front.png` — 最终（R=10000 长仿真）帕累托前沿

## 测试

```powershell
python -m unittest discover -s tests -v
```

57 个用例覆盖编码解码、EO、预热、交叉/变异、局部搜索、随机层、指标与传统
NSGA-II 基线；玩具例（§3.1.1）验证 `Cmax = 13`、左移 `TWTE = 4`、右移
`TWTE = 1`。

## 对比实验（传统 NSGA-II 基线）

为验证 DEA-DD 各项创新点的收益，提供**传统 NSGA-II 基线**
（`dea_dd/baselines/nsga2.py`）与一键对比入口 `dea_dd/compare.py`。基线去掉
DEA-DD 的三大类创新：

- **无 EO 预热**（直接随机初始化）；
- **无局部搜索**（只用交叉/变异）；
- **固定变异位数**（固定改 2 位，不做式 (16) 动态变异）。

与 DEA-DD **完全共享**编码、双解码、随机口径蒙特卡洛评估（CRN）与重叠参数
`N / p_c / p_m / gen_max / UL / R_short / R_long / seed`（即同一个 `Params`），
保证是「同一赛道」下的公平对比。

```powershell
# 单独运行传统 NSGA-II 基线（图写入 outputs/nsga2/）
python -m dea_dd.baselines.nsga2 --instance data/Brandimarte/BrandimarteMk1.fjs --gen 100

# 一键对比两者：--instance 既可传单个 .fjs，也可传目录（非递归，只扫顶层 .fjs）
# 每个算例默认重复 10 次（每次 seed 递增 1），第 i 次结果隔离到
#   outputs/<算例名>/run_XXX/（dea_dd/、nsga2/、front_comparison.png），
# 逐算例汇总 → outputs/<算例名>/summary.csv，跨算例汇总 → outputs/summary_all.csv
python -m dea_dd.compare --instance data/Brandimarte/BrandimarteMk1.fjs --gen 100 --runs 10
python -m dea_dd.compare --instance data/Brandimarte --gen 100 --runs 10   # 扫整个目录
```

`compare.py` 每次运行的指标：两个前沿各自的 **NPS** 与 **HV**（共用同一参考点，
否则不可比），以及双向覆盖率 **C(DEA-DD | NSGA-II)** 与 **C(NSGA-II | DEA-DD)**。
每个算例重复 `--runs` 次（缺省 10，第 i 次 seed = 基准 seed + i - 1）；单算例汇总在
`summary.csv`、多算例跨例汇总在 `summary_all.csv`，日志里也有 mean ± std。输出目录
按 `outputs/<算例名>/run_XXX/` 逐算例隔离。

```python
from dea_dd.compare import compare
from dea_dd.params import Params

# 程序化对比：复用同一个 Params，保证重叠参数一致；runs 控制每算例重复次数
compare(Params(N=100, gen=400, R_short=20, R_long=10000),
        inst_path="data/Brandimarte/新建文件夹/BrandimarteMk1.fjs", runs=10)
# inst_path 传目录 → 对顶层所有 .fjs 逐个对比，各自输出隔离
compare(Params(N=100, gen=400, R_short=20, R_long=10000),
        inst_path="data/Brandimarte", runs=10)
```

## 数据集

已从 `https://github.com/Lei-Kun/FJSP-benchmarks` 下载并放置 **15 个 Brandimarte
算例** 到 `data/Brandimarte/BrandimarteMk1.fjs` … `BrandimarteMk15.fjs`。`.fjs`
解析器支持该格式（首行 `工件数 机器数 平均候选数`，机器号 1-based 自动转 0-based）。