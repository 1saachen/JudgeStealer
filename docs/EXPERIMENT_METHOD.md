# 实验方法：有限预算主动选样与平滑训练

本文根据当前实验目录中的代码、启动脚本和实验规范，整理离散 1--10 分 judge 与三阶段生成式 SFT 方法。

## 1. 实验目标与总体流程

实验研究：在只能有限次查询 victim judge 的情况下，如何训练本地 surrogate 去拟合 victim 的评分、偏好和排序行为。

给定未标注候选池，整体流程如下：

1. 随机抽取初始化样本并查询 victim；
2. 用已查询样本训练临时 pointwise proxy；
3. 用 proxy 计算剩余候选的 acquisition score；
4. 选择 score 最高的一批并查询 victim；
5. 增量更新 proxy，重复上述过程直到耗尽预算；
6. 丢弃临时 proxy，从基础 checkpoint 重新训练最终 surrogate；
7. 在 question ID 不重叠的固定验证集上评估。



## 2. 数据、预算与三阶段训练

每个问题包含五个候选回答。候选 answer triple 由三个回答组成，并为每个回答查询 pointwise 1--10 分。预算以 answer units 计数：

$$
\mathrm{budget}=600
\quad\Longrightarrow\quad
200\ \mathrm{triples}
\quad\Longrightarrow\quad
600\ \mathrm{pointwise\ labels}.
$$

选中的 triple 用于三阶段生成式 SFT：

1. Stage 1：pointwise score generation；
2. Stage 2：pairwise preference generation，pointwise replay ratio 为 0；
3. Stage 3：listwise ranking 为主任务，同时以 1:1 比例回放 pointwise 和 pairwise。


## 3. 样本转换与交换增强

选中的 answer triple 会被转换成 pointwise、pairwise 和 listwise 三种训练数据。设选中 $N$ 个 triple，每个 triple 包含回答 A、B、C。

### 3.1 Pointwise 转换

每个 triple 拆成三个独立评分样本：

$$
(A,y_A),\quad(B,y_B),\quad(C,y_C).
$$

其中 $y_A,y_B,y_C\in\{1,\ldots,10\}$ 是 victim 分数。每个 triple 产生 3 个 pointwise 样本，预算 600 时共 600 个样本。prompt 只包含一个候选回答和评分指令，不把 gold score 放入输入。

### 3.2 Pairwise 转换

每个 triple 生成三个回答对：

$$
(A,B),\quad(A,C),\quad(B,C).
$$

标签由两个回答的 pointwise 分数决定：左侧分数高则选择左侧，右侧分数高则选择右侧，分数相等则为 tie。因此未增强时每个 triple 产生 3 个 pairwise 样本。

### 3.3 Pairwise 交换增强

开启 pairwise order augmentation 后，每个 pair 额外生成交换顺序的副本：

$$
(A,B,y)\longrightarrow(B,A,\operatorname{swap}(y)).
$$

标签变换为：

$$
\operatorname{swap}(A)=B,\qquad
\operatorname{swap}(B)=A,\qquad
\operatorname{swap}(tie)=tie.
$$

因此每个 triple 的 pairwise 样本数从 3 变为 6。交换只改变 prompt 中的回答顺序和对应 label，不重新查询 victim，也不增加 query budget；它用于减少左右位置偏置。

当前三阶段入口默认开启该增强；只有显式传入 `--no-pairwise-order-augmentation` 时才关闭。

### 3.4 Listwise 转换与交换增强

未增强时，每个 triple 生成一个 listwise 样本，标签由三个 score 计算出的排序关系决定，例如 $A>B=C$。

开启 listwise order augmentation 后，枚举三个回答的全部 $3!=6$ 个排列：

$$
(A,B,C),\ (A,C,B),\ (B,A,C),\ (B,C,A),\ (C,A,B),\ (C,B,A).
$$

每次排列都会重新构造 prompt，并根据排列后新的 Assistant A/B/C 位置重新计算 ranking label。因此不是简单复制原始 ranking 字符串。每个 triple 的 listwise 样本数从 1 变为 6，同样不消耗额外查询。

当前三阶段入口默认枚举六种排列；只有显式传入 `--no-listwise-order-augmentation` 时才关闭。

### 3.5 样本数量

当 $N=200$ 时：

| 任务 | 无增强 | 开启顺序增强 |
|---|---:|---:|
| Pointwise | $3N=600$ | 600 |
| Pairwise | $3N=600$ | $6N=1200$ |
| Listwise | $N=200$ | $6N=1200$ |

上述样本均由同一批 selected triples 派生，不计入 victim query budget。

## 4. 主动选样方法

### 4.1 当前主方法：直接使用 pointwise proxy

代码中的 selector 类型包括 pointwise_proxy（当前主方法）、random（随机基线）以及 bert、shared_llama、shared_llama_two_stage（历史 selector 或消融）。临时 proxy 使用 classifier-head pointwise 预测；最终模型仍是生成式 causal-LM SFT。

### 4.2 Pointwise 分类熵



对候选回答 $x$，pointwise proxy 输出 $K=10$ 个分数类别的概率 $p_k=p(y=k|x)$。回答级不确定性使用归一化熵：

$$
H(x)
=-\frac{\sum_{k=1}^{K}p_k\log p_k}{\log K}.
$$

熵越大，表示 proxy 对该回答的评分越不确定，候选越值得查询。

### 4.3 Triple-level acquisition score



对 answer triple $t$，代码先计算三个回答的 entropy，再按“均值为主、最大值保留困难回答”的方式聚合：

$$
A(t)
=0.75\cdot\frac{1}{3}\sum_{x\in t}H(x)
 +0.25\cdot\max_{x\in t}H(x).
$$





### 4.4 主动学习循环

以预算 600、初始化 80 个 triple、每轮 batch size 20 为例：

1. 随机初始化 80 个 triple，并查询其 240 个 pointwise 分数；
2. 用初始化数据 warmup pointwise proxy，默认 3 epochs；
3. 对所有剩余候选 triple 计算 $A(t)$；
4. 选择 acquisition score 最高的 20 个 triple；
5. 查询其 60 个 pointwise 分数，并增量更新 proxy 1 epoch；
6. 重复选样和更新，直到得到 200 个 triple；
7. 保存 selected_triples 文件，删除临时 proxy；
8. 从基础 checkpoint 重新训练最终三阶段 surrogate。



## 5. 生成式 score-token smoothing



### 5.1 smoothing 作用位置

生成式 1--10 分任务中，smoothing 只作用于 pointwise score token 的损失。prompt、解释文本以及 Stage 2 的 pairwise、Stage 3 的 listwise token 仍使用普通 causal-LM cross entropy。



### 5.2 经验全局先验

设 $K=10$ 个分数类别在当前已见 pointwise 标签中的计数为 $h_k$，smooth_prior 为正的加性伪计数 $\beta$，则经验先验为：

$$
\pi_k
=\frac{h_k+\beta}
       {\sum_{j=1}^{K}(h_j+\beta)}.
$$

若启用 uniform mix，代码进一步计算：

$$
\pi_k^{\mathrm{mix}}
=(1-\rho)\pi_k+\rho\frac{1}{K},
$$

其中 $\rho$ 是 uniform mix 比例。


### 5.3 Pointwise score loss

设真实分数类别为 $y$，模型对候选分数 $k$ 的完整 score sequence 负对数似然为 $\mathrm{NLL}_\theta(k\mid x)$。代码在 pointwise score 位置使用：

$$
\mathcal L_{\mathrm{score}}
=(1-\alpha_t)\,\mathrm{NLL}_\theta(y\mid x)
 +\alpha_t\sum_{k=1}^{K}
   \pi_k^{\mathrm{mix}}\,
   \mathrm{NLL}_\theta(k\mid x).
$$

等价的类别级软目标是：

$$
q_k
=(1-\alpha_t)\mathbf{1}[k=y]
 +\alpha_t\pi_k^{\mathrm{mix}}.
$$

对于多 token 的分数，代码沿每个候选序列的前缀重新前向并累加完整 sequence NLL，而不是只比较第一个 token。

### 5.4 先验来源与冻结

- Online prior：训练过程中持续把当前 pointwise 标签加入 $h_k$；
- Stage-1 prior：使用 Stage 1 的 pointwise 标签初始化 $h_k$；
- Frozen prior：初始化后不再更新 $h_k$。

稳定 smoothing 消融通常采用 Stage-1 初始化、Stage-3 冻结，并按已见 pointwise replay 样本数调度 alpha，而不是按混合 batch 的顺序调度。

### 5.5 Alpha warmup

令 $s$ 为 pointwise replay 已处理样本数，$s_0$ 为开始 smoothing 的样本数，$W$ 为 warmup 长度，配置上限为 $\alpha$，则：

$$
\alpha_s=
\begin{cases}
0, & s<s_0,\\
\alpha\min\left(1,\frac{s-s_0+1}{W}\right),
& s\ge s_0,\ W>0,\\
\alpha, & s\ge s_0,\ W=0.
\end{cases}
$$

固定 Stage-1 prior 的实验通常设置前 200 个 pointwise replay 样本不平滑，之后在 200 个样本内线性 warmup。

### 5.6 自适应和可训练 alpha

Entropy-adaptive 版本按模型当前 score 分布熵缩放单样本 alpha：

$$
\alpha_i
=\alpha_t\cdot
\frac{H(p_\theta(\cdot\mid x_i))}{\log K}.
$$

可训练 alpha 使用：

$$
\alpha=\alpha_{\max}\sigma(a),
$$

并加入保持其接近初始值的正则项：

$$
\mathcal L_{\alpha}
=\lambda(\alpha-\alpha_0)^2.
$$

这两种设置均属于消融，不是默认主配置。

## 6. Selector 与 smoothing 的关系

在 pointwise_proxy 主动选样模式中，临时 proxy 的 warmup 和每轮 update 都会接收 smoothing 设置。因此重新运行 selection 时，alpha 可能改变 proxy 的 entropy ranking，进而改变 selected triples。

若要测量 smoothing 的纯训练效应，必须复用完全相同的 selected_triples 文件，只改变 alpha。否则最终差异是 selected data 改变和训练目标改变的联合效果。

推荐区分两类实验：

- End-to-end smoothing：smoothing 同时用于 acquisition proxy 和最终 surrogate；
- Fixed-selection smoothing：固定 selected triples，只在最终三阶段训练中使用 smoothing。




