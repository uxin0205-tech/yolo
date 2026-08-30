# Full35 Activation 完整数学推导

## 1. 范围与符号

本文覆盖程式 registry 中实际评估的六种 activation：`silu`、`hardswish`、`relu`、
`qsilu_pq`、`poly_quality`、`poly_shift`。其中后三个为本工作中以约束方程推导的候选；
SiLU、Hardswish、ReLU 是控制组，但也完整列出其公式与性质，避免量化工作列只看到名称而不知道
不变量、连续性及尾端行为。

记

\[
u=|x|,\qquad [v]_+=\max(v,0).
\]

我们关注的核心分解是

\[
A(x)=\frac{x}{2}+H(|x|).
\]

因为第二项是偶函数，所以不论 \(H\) 的具体形式为何，都有

\[
A(x)-A(-x)
=\left(\frac{x}{2}+H(|x|)\right)
-\left(-\frac{x}{2}+H(|x|)\right)=x. \tag{1}
\]

这是所有 proposed profiles 的结构性不变量，不是训练后才近似成立的统计性质。若再要求
\(H(0)=0\)，便有 zero anchor \(A(0)=0\)；若某个 \(T>0\) 之后
\(H(u)=u/2\)，便得到 exact ReLU tails：\(x\ge T\) 时 \(A(x)=x\)，
\(x\le -T\) 时 \(A(x)=0\)。

## 2. SiLU 控制组

### 2.1 从 sigmoid 推导偶残差

SiLU 定义为

\[
\operatorname{SiLU}(x)=x\sigma(x),\qquad
\sigma(x)=\frac{1}{1+e^{-x}}.
\]

由 \(\sigma(-x)=1-\sigma(x)\)，可直接得到

\[
\operatorname{SiLU}(x)-\operatorname{SiLU}(-x)
=x\sigma(x)+x\sigma(-x)=x. \tag{2}
\]

又因为

\[
\sigma(x)=\frac{1+\tanh(x/2)}{2},
\]

所以

\[
\operatorname{SiLU}(x)
=\frac{x}{2}+\frac{x}{2}\tanh(x/2)
=\frac{x}{2}+\frac{|x|}{2}\tanh(|x|/2). \tag{3}
\]

最后一步成立是因为 \(x\tanh(x/2)\) 为偶函数。式 (3) 正是我们采用
\(x/2+H(|x|)\) 的来源，其中

\[
H_{\mathrm{SiLU}}(u)=\frac{u}{2}\tanh(u/2).
\]

### 2.2 导数与实现成本

\[
\operatorname{SiLU}'(x)
=\sigma(x)+x\sigma(x)(1-\sigma(x)). \tag{4}
\]

SiLU 是光滑、含负输出谷、非单调的函数，但 sigmoid／exp 对整数 datapath 并不便宜。因此后续
候选保留式 (1)、zero anchor、负谷与 ReLU tails，同时把非线性改写为低阶多项式或分段二次式。

## 3. Hardswish 控制组

定义

\[
\operatorname{Hardswish}(x)
=x\frac{\operatorname{ReLU6}(x+3)}{6}
=\begin{cases}
0,&x\le -3,\\
\dfrac{x(x+3)}{6},&-3<x<3,\\
x,&x\ge 3.
\end{cases} \tag{5}
\]

在中心区间，

\[
\frac{x(x+3)}{6}=\frac{x}{2}+\frac{x^2}{6},
\]

所以 Hardswish 也满足式 (1)，且在 \(|x|\ge3\) 是 exact ReLU tails。其导数为

\[
\operatorname{Hardswish}'(x)=
\begin{cases}
0,&x<-3,\\
\dfrac{2x+3}{6},&-3<x<3,\\
1,&x>3.
\end{cases} \tag{6}
\]

在 \(-3\) 的左右导数分别为 \(0\) 与 \(-1/2\)，在 \(3\) 的左右导数分别为
\(3/2\) 与 \(1\)，因此它只有 \(C^0\)，不是 \(C^1\)。它只有一个中心二次式且门槛固定，硬体简单；
但 Full35 的 10-epoch recovery 仍有 BBAT bat pose 超过 accuracy gate，不能只凭公式简单就宣称胜出。

## 4. ReLU 诊断控制组

\[
\operatorname{ReLU}(x)=[x]_+
=\frac{x+|x|}{2}. \tag{7}
\]

因此它是 \(H(u)=u/2\) 的退化情形，也满足式 (1)。其导数为

\[
\operatorname{ReLU}'(x)=
\begin{cases}0,&x<0,\\1,&x>0,\end{cases}
\]

并在原点不可微。ReLU 仅作为「完全拿掉负谷与平滑过渡」的诊断下界；本实验 zero-shot 八项
mAP50-95 全为 0，所以没有进入昂贵 recovery。

## 5. SIPA：受约束积分多项式族

### 5.1 先设计导数，再积分

令正规化位置 \(z=u/T\in[0,1]\)。我们不直接拟合函数值，而先设定偶残差的导数

\[
q_a(z)=az+(9-\tfrac92a)z^2+(6a-16)z^3
+(\tfrac{15}{2}-\tfrac52a)z^4. \tag{8}
\]

四个系数来自下列四个线性约束：

\[
q_a(0)=0,\quad q_a(1)=\frac12,\quad q_a'(1)=0,
\quad \int_0^1q_a(t)\,dt=\frac12. \tag{9}
\]

这些约束各有明确作用：

- \(q_a(0)=0\)：使原点左右一阶导数同为 \(1/2\)。
- \(q_a(1)=1/2\)：接上 \(H(u)=u/2\) 的 tail slope。
- \(q_a'(1)=0\)：接上尾端的零二阶导数。
- 积分为 \(1/2\)：使 \(H(T)=T/2\)，函数值接上 exact ReLU tail。

积分得到

\[
R_a(z)=\int_0^zq_a(t)\,dt
=\frac a2z^2+(3-\tfrac32a)z^3
+(\tfrac32a-4)z^4+(\tfrac32-\tfrac12a)z^5. \tag{10}
\]

于是定义

\[
H_{T,a}(u)=T R_a\!\left(\frac{\min(u,T)}{T}\right)
+\frac{[u-T]_+}{2}, \tag{11}
\]

\[
A_{T,a}(x)=\frac{x}{2}+H_{T,a}(|x|). \tag{12}
\]

### 5.2 不变量与连续性的逐项证明

由 \(R_a(0)=0\)，式 (12) 有 \(A(0)=0\)。由偶残差结构，式 (1) 精确成立。
由式 (9) 与 (10)，

\[
H(T^-)=TR_a(1)=T/2=H(T^+),
\]

\[
H'(T^-)=q_a(1)=1/2=H'(T^+),
\]

\[
H''(T^-)=q_a'(1)/T=0=H''(T^+).
\]

所以两个门槛及原点都达到 \(C^2\)。在 \(u\ge T\) 时，式 (11) 化为

\[
H(u)=TR_a(1)+(u-T)/2=u/2,
\]

因此 exact ReLU tails 成立。参数 \(a=q_a'(0)\) 控制原点附近曲率；\(T\) 控制负谷宽度与尾端门槛。

把式 (10) 的 \(z=u/T\) 展开，可得到不需 runtime 除法的直接式

\[
H(u)=c_2u^2+c_3u^3+c_4u^4+c_5u^5,\quad 0\le u\le T, \tag{13}
\]

其中 \(c_k\) 等于式 (10) 中 \(z^k\) 的系数除以 \(T^{k-1}\)。

### 5.3 `poly_quality` 的完整代入

选择

\[
a=4,\qquad T=\frac{109}{16}=6.8125.
\]

式 (8) 与 (10) 化为

\[
q(z)=4z-9z^2+8z^3-\frac52z^4, \tag{14}
\]

\[
R(z)=2z^2-3z^3+2z^4-\frac12z^5. \tag{15}
\]

其导数可因式分解为

\[
q'(z)=2(1-z)^2(2-5z). \tag{16}
\]

所以尾端二阶连续一目了然。代入 \(T=109/16\)，直接系数为

\[
H(u)=\frac{32}{109}u^2
-\frac{768}{11881}u^3
+\frac{8192}{1295029}u^4
-\frac{32768}{141158161}u^5,\quad u\le\frac{109}{16}. \tag{17}
\]

负半轴写成 \(A(-Tz)=T(R(z)-z/2)\)。极值条件是 \(q(z)=1/2\)，其中一根为
\(z=1/5\)，因此负谷值为

\[
T\left(R(1/5)-1/10\right)=-0.27904. \tag{18}
\]

解析导数范围为 \([-27/250,277/250]=[-0.108,1.108]\)。这组门槛优先贴近 SiLU 品质，
直接系数不是 dyadic，仍需要常数乘法器或近似。

### 5.4 `poly_shift` 的完整代入

选择

\[
a=\frac92,\qquad T=8.
\]

式 (8) 与 (10) 化为

\[
q(z)=\frac92z-\frac{45}{4}z^2+11z^3-\frac{15}{4}z^4, \tag{19}
\]

\[
R(z)=\frac94z^2-\frac{15}{4}z^3+\frac{11}{4}z^4-\frac34z^5. \tag{20}
\]

由于 \(T=8\)，直接式全部成为 dyadic：

\[
H(u)=\frac9{32}u^2-\frac{15}{256}u^3
+\frac{11}{2048}u^4-\frac3{16384}u^5,\quad 0\le u\le8, \tag{21}
\]

\[
H(u)=u/2,\quad u\ge8. \tag{22}
\]

例如 \(9/32=1/4+1/32\)、\(15/256=1/16-1/256\)，可用 shift-add 表示常数；
但式 (21) 仍需形成 \(u^2,u^3,u^4,u^5\)，所以「dyadic」不等于「零乘法器」。本次 Full35
结果显示它的 uniform zero-shot 与 10-epoch recovery 都未过 gate，因此 region／mixed search
依预注册依赖被封锁，没有拿缺失结果补造结论。

## 6. `qsilu_pq`：C1 分段二次 qSiLU

### 6.1 设计目标

SIPA 的五次式虽可做 dyadic 常数化，但仍有高次数据路径。`qsilu_pq` 改用门槛
\(0,1,2,4,8\) 的分段二次偶残差：只形成一次 \(u^2\)，保持式 (1)、zero anchor、负谷、
非单调导数与 \(|x|\ge8\) exact ReLU tails。它是 SiLU 近似，不是把 SiLU 换成 GELU。

### 6.2 从截断平方基底推导

采用截断平方基底

\[
h(u)=\frac1{1024}\sum_{i=0}^{4}k_i[u-t_i]_+^2, \tag{23}
\]

\[
(t_0,t_1,t_2,t_3,t_4)=(0,1,2,4,8),
\]

\[
(k_0,k_1,k_2,k_3,k_4)=(228,-136,-114,17,5). \tag{24}
\]

每个 \([u-t]_+^2\) 在门槛处的函数值与一阶导数都连续，所以式 (23) 自动是 \(C^1\)。
前三个 exact-tail 线性条件是

\[
\sum_i k_i=228-136-114+17+5=0, \tag{25}
\]

\[
\sum_i k_it_i=-136-228+68+40=-256, \tag{26}
\]

\[
\sum_i k_it_i^2=-136-456+272+320=0. \tag{27}
\]

当 \(u\ge8\) 时展开式 (23)：二次项由式 (25) 消失，线性项为

\(-2\sum k_it_i/1024=1/2\)，常数项由式 (27) 消失，因此

\[
h(u)=u/2,\qquad u\ge8. \tag{28}
\]

再定义

\[
Q(x)=\frac{x}{2}+h(|x|), \tag{29}
\]

便得到 exact ReLU tails 与 \(Q(x)-Q(-x)=x\)。

### 6.3 逐区间展开

把式 (23) 依门槛展开，得到程式实际使用的四段：

\[
h(u)=
\begin{cases}
\dfrac{57}{256}u^2,&0\le u<1,\\[2mm]
\dfrac{23}{256}u^2+\dfrac{17}{64}u-\dfrac{17}{128},&1\le u<2,\\[2mm]
-\dfrac{11}{512}u^2+\dfrac{91}{128}u-\dfrac{37}{64},&2\le u<4,\\[2mm]
-\dfrac{5}{1024}u^2+\dfrac{37}{64}u-\dfrac{5}{16},&4\le u<8,\\[2mm]
\dfrac12u,&u\ge8.
\end{cases} \tag{30}
\]

以第二区为例，\((228u^2-136(u-1)^2)/1024\) 展开为
\(23u^2/256+17u/64-17/128\)；其余区段按相同方式逐个加入新的截断平方项。

所有系数均是 dyadic。看似较重的 \(91/128\) 也可写成

\[
\frac{91}{128}=1-\frac14-\frac1{32}-\frac1{128}, \tag{31}
\]

即四个 signed shift 项。融合实现可共用一个 \(u^2\)，再依门槛选择系数；目前 PyTorch eager
reference 会同时形成多个候选张量，所以显存行为不等同未来 fused RTL／kernel。

### 6.4 函数与导数误差

以 dense grid 对原 SiLU 比较，当前冻结候选的数值为：

| 区间／项目 | 数值 |
| --- | ---: |
| \([-8,8]\) function MAE | 0.0051297 |
| \([-8,8]\) function RMSE | 0.0060978 |
| \([-8,8]\) function max error | 0.0108319 |
| \([-12,12]\) function MSE | \(2.51284\times10^{-5}\) |
| derivative MAE | 0.0072994 |
| derivative max error | 0.0342158 |
| derivative range | \([-0.125,1.125]\) |

它通过 float、Q16.10 bit-true 与 ONNX 数值 gate；但尚无特定 FPGA／ASIC synthesis、latency、
power 或 resource 实测。因此可称「shift-friendly 候选」，不可称已证明的硬体加速。

## 7. 保持不变量的 fixed-point integerization

令输入整数 code 为 \(n\)，正负输入共用同一个偶残差查表／多项式 \(\widehat H(|n|)\)，定义

\[
\widehat A(n)=\left\lfloor\frac n2\right\rfloor+\widehat H(|n|). \tag{32}
\]

由于

\[
\left\lfloor\frac n2\right\rfloor
-\left\lfloor\frac{-n}{2}\right\rfloor=n,
\]

偶残差又会精确抵消，所以对所有可表示整数 code 都有

\[
\widehat A(n)-\widehat A(-n)=n. \tag{33}
\]

尾端若使用互补 rounding

\[
\widehat H(u)=\left\lceil\frac u2\right\rceil,
\]

则正输入为 \(\lfloor n/2\rfloor+\lceil n/2\rceil=n\)，负输入为 0，奇数 code 也保持 0 LSB
的 ReLU tail 误差。后续量化工作必须另外枚举 Q8／Q12／Q16 全输入码，并验证 overflow、
saturation、rounding、compiler IR、HLS 与 RTL bit-exact；目前不能把公式证明扩大成 PTQ/QAT AP
或板上成本已经获证。

## 8. 数学贡献边界与最终定位

本阶段最值得继续验证的是：zero-anchored、SiLU-like non-monotone、式 (1) symmetry-constrained、
\(C^2\) exact-tail 的低自由度积分多项式族，以及保持式 (33) 的整数化。`qsilu_pq` 是本专案
冻结的 \(C^1\) 分段二次候选；已存在其他分段二次 SiLU／Swish 硬体近似，因此不能把「分段二次
近似 SiLU」本身宣称为新颖。SIPA 的各个单独元素也有 prior art，学术主张必须以组合约束、
bit-exact 证据与 Full35／量化／硬体结果共同支撑。完整 prior-art 来源与名称冲突见
[qSiLU 硬体近似研究](qsilu-hardware-approximation.md)与
[SIPA–BCSP 新颖性稽核](sipa-bcsp-novelty-audit.md)。

最后，BCSP 在本轮只是节省实验成本的 bounded placement 计划。由于 `poly_shift` 的 uniform
prerequisite 未过，14 个 region／mixed jobs 正确封锁；本阶段没有产生足以把 BCSP 升格为新算法
贡献的跨资料集 normalized-regret 搜寻结果。
