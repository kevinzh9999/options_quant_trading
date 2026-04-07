# **A股主要宽基指数均值回归策略在不同波动率区间下的可行性与投资价值研究**

## **引言**

在量化投资与金融工程的广阔领域中，资产价格的运行规律通常被划分为趋势追踪（动量）与均值回归两种最基本的微观形态。传统的有效市场假说（EMH）认为，金融资产的价格变动遵循随机游走规律，历史价格信息无法用于预测未来走势，任何试图通过历史轨迹获取超额收益的尝试最终都将被交易成本所吞噬。然而，大量的实证研究、统计套利实践以及行为金融学的深度观察表明，真实的金融市场在特定的时间尺度、特定的资产类别以及特定的波动率区间内，呈现出极其显著的均值回归（Mean Reversion）特征 1。均值回归理论的核心假设在于：资产价格、收益率乃至波动率本身，在受到外部宏观经济冲击或微观流动性挤兑而偏离其长期均衡水平后，最终会受到一种内在引力的作用，逐步向其长期历史均值收敛 4。

中国A股市场作为一个典型的以散户投资者为重要参与力量的新兴资本市场，其微观交易结构与欧美成熟发达市场存在着本质的差异。由于个人投资者在市场自由流通市值中占比较高，市场在宏观政策发布、海外市场波动或突发性事件面前，常表现出更强的情绪化波动、羊群效应（Herd Behavior）以及对局部信息的过度反应 2。这种群体性的非理性过度反应，往往会导致资产价格在短期内严重脱离其内在基本面价值，从而在客观上为系统性的均值回归策略提供了异常丰厚的统计套利空间。特别是针对A股市场的几个核心宽基指数——上证50、沪深300、中证500与中证1000，它们在成份股市值分布、微观流动性、行业集中度、对外部政策不确定性的敏感度以及内在年化波动率特征上，呈现出极其清晰的阶梯式差异 7。这种差异不仅决定了各指数在面对冲击时的偏离幅度，更深刻地影响着它们向均值收敛的速度与路径。

本研究旨在全面且深入地剖析均值回归策略在A股主流宽基指数上的应用可行性，打破传统量化研究中将策略置于全天候环境下的粗放评估模式，重点考察并界定不同波动率区间（Volatility Regimes）对回归策略有效性及超额回报的底层影响机制。通过结合经典的连续时间随机过程理论、前沿的波动率聚类深度学习模型，以及国内外主流金融工程研究机构（如中金公司、华泰证券、摩根大通等）的最新量化研究成果，本文将构建一套严谨的策略推荐与参数动态选择体系，旨在为面对复杂多变市场的系统性交易者提供坚实的数理理论支撑与高价值的实战指导。

## **均值回归的经典金融理论与数学微观机制**

要论证均值回归策略的有效性，必须首先确立其在统计学与随机微积分领域的理论根基。金融资产价格序列并非简单的白噪声，其内部蕴含着复杂的记忆性与分布特征。

## **随机游走理论的局限与Hurst指数的统计检验**

在时间序列分析框架下，判断一个金融资产或价差序列是否具备潜在的均值回归特性，通常需要依赖严格的统计学假设检验。传统的计量经济学工具如增强型迪基-富勒检验（ADF）和菲利普斯-佩隆检验（PP）常用于检验序列的平稳性，但Hurst指数（Hurst Exponent）在衡量时间序列的长期记忆性与分形特征方面展现出了更为卓越的刻画能力。数学上，当Hurst指数 ![][image1] 时，价格序列呈现纯粹的几何布朗运动，即满足有效市场假说下的随机游走特性；当 ![][image2] 时，序列具有正向的长期记忆性，表现出动量效应或趋势延续性；而当 ![][image3] 时，序列则展现出强烈的反持续性，即均值回归特征 1。

针对新兴市场宽基指数的实证数据严密论证了均值回归现象的普遍存在。在长达数十年的回测周期内，包含A股及其他新兴市场核心指数在内的资产，其Hurst指数在大量观测窗口中显著低于0.5，这从根本上证实了逆势交易与均值回归策略的数理可行性 1。这种统计学上的反持续性，要求市场参与者在制定决策与交易策略时，必须具备长期视角，学会在市场极度狂热或恐慌导致的偏离极值处寻找回归的利润空间。

## **Ornstein-Uhlenbeck (OU) 过程与回归半衰期的参数标定**

在连续时间随机过程理论中，Ornstein-Uhlenbeck（OU）过程是刻画均值回归行为的最具代表性且被广泛应用的数学模型。无论是在利率期限结构建模（如Vasicek模型）、配对交易的价差拟合，还是在波动率自身的演化描述中，OU过程都占据着核心地位 10。其标准形式的随机微分方程（SDE）定义如下：

![][image4]  
在此方程中，![][image5] 代表 ![][image6] 时刻的资产价格（或两个高度协整资产之间的价差），![][image7] 代表该时间序列的长期均衡均值水平，![][image8] 为均值回归的速度（即反向拉扯的强度参数），![][image9] 为过程的瞬时波动率，而 ![][image10] 则是标准的维纳过程（布朗运动）带来的随机游走增量 10。

OU过程深刻且直观地揭示了均值回归策略的内在驱动逻辑：当资产价格 ![][image5] 向上严重偏离并远高于长期均值 ![][image7] 时，漂移项 ![][image11] 会迅速变为一个绝对值不断增大的负值，从而对价格产生强烈的向下回归拉力；反之亦然，当价格超跌时，正向的漂移项将驱动价格反弹 10。参数 ![][image8] 的大小直接决定了偏离状态能够维持的时间长短以及回归的剧烈程度。在量化交易的工程实践中，研究人员通常通过极大似然估计（MLE）或最小二乘法校准出这些参数，并据此计算出“半衰期”（Half-life）——即资产价格的期望偏离程度自然衰减减半所需的时间。半衰期的计算公式为 ![][image12] 1。对于不同的A股宽基指数或跨品种价差，由于其流动性深度和投资者结构的差异，其半衰期存在显著的不同，这要求策略在参数选择时必须基于当前的市场微观结构进行动态的滚动标定。

## **波动率聚类效应、GARCH族模型与RealRECH的演进**

金融市场的均值回归特性不仅仅体现在资产的绝对价格路径上，更深层次地体现在其波动率特征的演化规律上。金融时间序列普遍存在显著的“波动率聚类”（Volatility Clustering）现象，即市场的动荡往往自我繁衍，大波动日倾向于紧接着大波动日，而平静的低波动期也倾向于自我延续 12。这种异方差性对传统的恒定波动率假设提出了严峻挑战。

广义自回归条件异方差（GARCH）模型及其衍生家族是描述这一聚类现象及波动率均值回归过程的核心计量经济学工具。在经典的 GARCH(1,1) 模型中，条件方差的演化受滞后残差平方（ARCH项，代表外部冲击）与滞后条件方差（GARCH项，代表内部记忆）的共同影响。大量针对新兴市场指数的学术研究表明，这些市场的收益率序列不仅呈现出比发达市场更高的风险收益特征，而且其GARCH模型的参数 ![][image13] 和 ![][image14] 之和通常严格小于1，这是波动率序列具备长期均值回归特性的必要数学条件。随着 ![][image15] 的和不断趋近于1，系统对新冲击的记忆时间变长，波动率冲击的持续性显著增强，导致均值回归的过程相应变慢 14。

尽管长周期来看波动率始终被视为一种不可逆转的均值回归资产，但其在短期内可能在某一极高波动或极低波动的状态下维持超预期的时间 16。为了更精准地捕捉这种复杂的动态，学术界近期引入了已实现递归条件异方差（RealRECH）模型。该模型创造性地将长短期记忆人工神经网络（LSTM）单元集成到传统的已实现GARCH（RealGARCH）模型中，以捕捉金融时间序列中传统参数模型无法拟合的长期依赖性与非线性特征。针对A股市场的实证检验显示，RealRECH模型在面对具有更高波动特性和更复杂微观结构的中证500和中证1000指数时，展现出了极其优异的样本外预测能力，这为在中小盘宽基指数上实施基于波动率预测的均值回归策略提供了最前沿的模型支持 17。

## **A股主要宽基指数的微观结构与异质性波动特征剖析**

要成功部署均值回归策略，量化分析师必须首先深刻理解标的资产由于编制规则、市值分布不同而导致的微观结构差异。上证50、沪深300、中证500与中证1000四大核心宽基指数，在A股市场中各自扮演着截然不同的角色，其流动性、行业集中度以及衍生品生态构成了极其分化的风险面貌 7。

下表直观地展示了四大宽基指数在多个维度的关键定量与定性特征对比：

| 指数名称 | 样本覆盖与市值定位 | 市场风格偏好与行业特征 | 近年历史年化波动率均值 | 核心衍生品生态配套 | 流动性深度与微观风险特征描述 |
| :---- | :---- | :---- | :---- | :---- | :---- |
| **上证50 (SSE 50\)** | 沪市排名前50大市值股票 | 超大盘、价值、金融地产为主 | 约 20.9% | 股指期货(IH)、ETF期权 | 具备极高的流动性与抗跌性。头部股票集中度极高，单只个股对指数影响大，整体波动率最低 7。 |
| **沪深300 (CSI 300\)** | 沪深两市前300大市值股票 | 大中盘、核心资产、相对均衡 | 约 21.1% | 股指期货(IF)、ETF/股指期权 | 流动性极其充裕，波动率适中，作为中国宏观经济的晴雨表，受国内外宏观经济数据扰动明显 7。 |
| **中证500 (CSI 500\)** | 剔除沪深300后的前500只 | 中盘、成长、周期制造、新经济 | 约 22.8% | 股指期货(IC)、ETF/股指期权 | 具备较高的成长性与较高波动率，行业分布较离散，能更好地反映细分行业龙头的整体走势 7。 |
| **中证1000 (CSI 1000\)** | 剔除中证800后的前1000只 | 小微盘、专精特新、高弹性 | 约 25.1% | 股指期货(IM)、股指期权 | 代表A股最高成长潜力的同时也伴随最高的系统性风险与波动率，部分成份股存在流动性折价现象 7。 |

## **大盘价值阵营：上证50与沪深300的平稳与宏观敏感性**

上证50指数集中了上海证券交易所规模最大、流动性最好、最具代表性的50只龙头企业。其最大的特征在于权重的极度集中，前十大成份股的权重占比通常超过53%，这意味着指数的整体表现严重依赖于少数几家巨头（尤其是金融与能源板块）的走势 19。由于这些成熟期的大盘股内生波动率较低，上证50指数的历史年化波动率仅为20.9%左右 9。此外，上证50 ETF期权及IH股指期货等衍生品市场的繁荣，进一步重塑了现货市场的微观结构。实证研究表明，期权等衍生品的引入虽然在上市初期可能引发短暂的震荡，但长期来看，大规模的期现套利和做市商的动态对冲行为，显著平抑了上证50现货市场的波动，增强了其在熊市等下行波动区间的抗跌性 23。在期权定价层面，上证50 ETF期权的隐含波动率曲线常呈现出独特的右偏“假笑”形态（Smirk skewed to the right），这不同于美国市场常见的左偏结构，表明中国期权交易者在危机期间往往对市场的强力干预与快速复苏抱有强烈的均值回归预期 25。

沪深300指数则覆盖了沪深两市前300大市值公司，是A股市场应用最广泛的业绩基准。其年化波动率约为21.1% 9。由于其涵盖了消费、金融、工业等国民经济支柱产业，沪深300的均值回归特征往往与宏观经济周期的均值回归深度绑定。学术界利用GARCH-MIDAS模型进行的高频数据研究发现，沪深300指数对外部经济政策不确定性（EPU）极为敏感。尤其是美国经济政策的波动，通过贸易链条、跨国资本流动以及投资者风险偏好等传导机制，显著增加了沪深300中可选消费、金融和工业板块的长期波动率 8。

## **中小盘成长阵营：中证500与中证1000的高弹性与超额套利空间**

相较于大盘指数，中证500指数聚焦于中小市值成长股，年化波动率跃升至22.8%左右。中证500更能反映中国经济转型期中细分行业及制造业龙头的表现。有趣的是，研究指出中证500及中证1000等中低市值指数对德国等以制造业和供应链为核心的经济体的政策不确定性表现出更强的敏感度，这反映了中小盘制造业在国际贸易网络中的特定位置 8。

中证1000指数则进一步向下潜入小微盘领域，年化波动率高达25.1%，是四大指数中最具活力与风险的标的 9。高波动率赋予了中证1000极高的弹性与潜在风险溢价。由于小盘股市场的机构配置比例相对较低，散户资金主导的特征更为明显，导致其在受到资金面扰动或情绪面冲击时，极易发生严重的“超买”或“超卖”。这种因流动性分化和情绪过激导致的非理性定价偏差，使得中证1000在特定的波动率区间下，成为了执行被动攻击型均值回归、或反转因子选股策略的最理想试验田 7。回测数据显示，基于机器学习与均值回归逻辑构建的中证1000量化选股策略，能够在2022年至2024年的宽幅震荡周期中，实现高达13.68%的年化超额收益率，且超额最大回撤控制在极为优异的4.38%水平 27。在算法参数方面，使用随机森林（Random Forest）进行中证1000的区间预测时，将决策树数量增加至150棵，同时严格限制最大深度为5以防过拟合，被证明是处理该指数庞大样本特征的最优结构 28。

## **波动率区间（Volatility Regimes）对策略有效性的决定性影响**

要构建高胜率的均值回归系统，仅仅依赖资产价格的偏离程度是远远不够的。量化交易的圣杯在于对“波动率区间”（Volatility Regime）的精准识别与状态切换。金融市场是一个高度复杂的非线性动力学系统，其通常会在不同的波动率状态之间发生剧烈的马尔可夫链式切换（Markov Regime Switching），这种状态转移直接决定了价格动态的根本性质，进而裁决了动量与均值回归策略的生死 30。

## **动量与均值回归的非对称互补性**

学术研究、跨资产类别的长周期实盘回测以及业界的主流认知一致表明，动量（Momentum）策略与均值回归（Mean Reversion）策略在不同波动率区间下的表现呈现出完美的互补与非对称性 30。

1. **低波动率区间（Trending & Calm Markets）**：在低波动率环境下，市场通常由充裕的系统性流动性、稳定的宏观预期和机构的被动资产配置主导。此时，资产价格呈现出强烈的动量效应与顺滑的趋势延续性。基于晶格气体模型（Lattice Gas Model）的理论解释指出，在此区间内，由于投资者普遍存在的羊群行为（Herding Behavior），微弱的趋势倾向于不断自我强化并持续存在 2。如果在这种“钝刀子割肉”的单边趋势市中盲目应用传统的均值回归策略（即不断地尝试“摸顶抄底”或逆势加仓），量化系统将面临致命的灾难。因为在低波动趋势中，价格极少产生向历史均值的深度回归，而是不断以极小的回撤创出新高或新低。实证数据显示，在低波动且呈现单边倾向的市场中，均值回归策略的期望收益往往为负，有效性完全丧失 2。  
2. **高波动率区间（Volatile & Choppy Markets）**：当市场遭遇重大的基本面冲击、宏观政策转向（如央行急剧收紧流动性）或突发性地缘政治黑天鹅事件时，波动率急剧飙升，市场瞬间跃迁入高波动区间。此时，市场深度变浅，算法做市商和流动性提供者的库存风险急剧增加，甚至出现流动性枯竭，导致资产价格在单方向上容易因为短期的恐慌盘或贪婪盘而砸出极端的非理性价位 16。正是这种由极度恐慌或贪婪引发的过度偏离，触发了极强的金融物理学“橡皮筋效应”。高波动率环境下，价格被动触发大幅反向均值回归的概率呈指数级增加，且回归速度极快。此时，均值回归策略（如基于RSI极限超卖的买入、布林带极端突破后的反向操作）的胜率和盈亏比将达到最佳的甜点区 36。研究明确指出，在面对剧烈波动的股票或指数标的时，传统的短期时间序列动量策略极易频繁出现假突破而失效，而基于短期反转（Reversal）和均值回归逻辑的系统则能够斩获丰厚的超额阿尔法 30。

## **状态转移的量化识别与降维分析**

最致命的交易风险往往发生在**波动率状态转移的瞬间（Regime Shift）**。当市场从长达数月的低波动死水稳态，突然发生结构性断裂（Structural Breaks）跃迁至高波动状态时，传统均值回归系统原有的参数（如参考移动均线周期、布林带标准差乘数）会瞬间失效。历史回测无一例外地显示，绝大多数均值回归模型在波动率区间突变的初期会遭受严重的最大回撤（Drawdown），随后在新的高波动稳态建立并确认后，才开始大幅收复失地并创造利润 39。

为了在A股宽基指数中精准、实时地识别这些区间转换，前沿的量化模型摒弃了单一的历史收盘价标准差，转而采用主成分分析（PCA）对多种高级波动率度量指标（如对日内高低价敏感的Parkinson波动率、包含跳空缺口特征的Garman-Klass波动率、以及更全面的Yang-Zhang波动率）进行降维提取。分析显示，第一主成分（PC1）通常能够解释高达68%的系统性市场方差，代表了市场的系统性风险水平；而第二主成分（PC2）则能敏锐地捕获带有负面动量的动荡趋势 40。通过实时监控这些主成分的时间序列变化，交易算法能够动态地调整均值回归策略的敞口与止损阈值。

## **主流金融工程研究机构的实证发现与行为学解释**

在A股市场这一高度本土化、散户化特征显著的特定生态下，国内及国际主流金融工程研究机构对于四大宽基指数的均值回归效应有着深刻的微观行为学洞察与宏观结构学解释。

## **散户行为偏差与处置效应（中金公司研究视角）**

中金公司（CICC）研究部的深度报告指出，尽管近年来机构化进程不断加速，但A股市场中个人投资者（散户）在自由流通市值中的占比依然维持在极高的水平（约占54%） 6。行为金融学的海量样本分析表明，A股个人投资者群体中普遍存在着一种根深蒂固的“伪均值回归信念”以及严重的“处置效应”（Disposition Effect）。具体表现为：投资者极度倾向于过早地兑现并卖出近期表现优异的获利股票（Fading winners），同时却长期死抱、甚至越跌越买那些近期表现糟糕、处于明显下降通道的亏损股票（Buying losers） 6。这种广泛存在的群体性交易认知偏误，在微观的订单簿层面直接构成了阻止价格沿单边趋势无限延伸的巨大阻力。这种非理性的流动性提供机制，为宽基指数（尤其是散户参与度极高的中证500和中证1000）在短期内频繁发生的高频均值回归，提供了源源不断的基础流动性与反作用力。

## **市场风格轮动作为宏观级均值回归（华泰证券视角）**

华泰证券金工团队在其年度及高频策略研报中多次强调，A股市场存在着极具中国特色的风格轮动特征（如大小盘的极致切换、价值与成长属性的周期性博弈），而这种轮动在本质上可以被视为一种宏观级别的均值回归现象。当市场资金在代表小盘高弹性的中证1000或微盘股中抱团达到极度拥挤、估值偏离度创出历史极值状态时，微观结构的脆弱性将导致抱团迅速松动，资金会无可避免地向代表大盘价值的上证50或沪深300进行高低切换，反之亦然 41。宽基指数之间的这种跷跷板效应，为构建基于相对强弱正常化（Relative Strength Normalization）的跨品种均值回归模型提供了坚实的宏观逻辑支撑。

## **衍生品期权隐含波动率的绝对回归（摩根大通与中信视角）**

随着A股金融衍生品生态的日益完善，特别是中证1000股指期货（IM）及期权（MO）、上证50ETF期权体系的逐步成型，基于指数波动率本身的均值回归策略正成为外资与内资顶尖量化机构的核心利润来源。摩根大通（JPMorgan）与国内头部券商的量化战略衍生品报告深刻揭示：当指数现货经历暴跌或遭遇系统性抛售引发剧烈震荡时，期权的隐含波动率（IV）会随市场恐慌情绪产生非理性的脉冲式飙升 43。鉴于波动率资产本身存在着金融学中最刚性的长期均值回归物理定律，无论现货价格未来走向何方，极端高企的隐含波动率最终都必将不可避免地向其历史均值回落 16。基于此，摩根大通等机构强烈建议在标的指数（如中证1000）隐含资金费率极度扩张或波动率位于历史极高分位时，采用低Delta的期权结构组合——例如构建1x1.5的看涨期权比例价差并卖出深度虚值看跌期权以实现零成本建仓，从而在享受标的分散性收益的同时，充分榨取隐含波动率均值回归所带来的丰厚利润 43。

## **高胜率可行的均值回归策略推荐与动态参数选择体系**

基于上述对均值回归古典理论基础、宽基指数微观异质性、波动率区间决定性作用以及顶尖投行研究成果的系统性综合剖析，本文针对A股四大主要宽基指数，提出以下四套具体的、具有极高实战可行性的量化交易策略框架，并给出针对性的参数选择与优化建议。

## **策略一：基于高波动率区间硬过滤的动态Z-Score极值反转策略**

**适用核心标的**：中证500、中证1000（中小盘的高波动性赋予了策略更充裕的回撤容忍空间与更具爆发力的反弹幅度）。

**策略底层逻辑**： 传统的均值回归指标体系（如单纯的RSI超卖、布林带下轨买入）之所以在实盘中容易导致毁灭性回撤，根源在于其在单边低波动趋势市中会持续给出致命的虚假逆势信号。因此，构建具备正期望值的系统，必须引入刚性的“波动率区间过滤器”（Volatility Regime Filter）。规则设定为：当且仅当市场明确处于“高波动且缺乏连贯单边方向”的震荡区间时，算法才被允许解锁并启动均值回归交易模块 30。

**参数设置与执行步骤**：

1. **波动率状态识别引擎（Regime Filter）**：实时计算标的宽基指数过去 20 个交易日的真实波动幅度均值（ATR）或历史年化波动率（HV）。设定自适应阈值：若当前短周期 ATR 大于过去 252 个交易日（约一年）长期 ATR 的均值加上 1 到 1.5 个标准差，则系统明确判定市场已发生状态转移，进入高波动区间，允许均值回归策略激活 45。  
2. **动态均值锚点与偏离度（Z-Score）量化**：放弃固定的均线周期，采用过去 ![][image16] 或 ![][image17] 天的指数加权移动平均（EMA）作为反映近期记忆的基准均值 ![][image18] 45。实时计算当前收盘价格的统计学偏离度 Z-Score：![][image19]，其中 ![][image20] 为同一观测窗口下的标准差。  
3. **交易信号生成与执行**：  
   * **做多介入（Long Entry）**：当 ![][image21] 且市场确认处于高波动区间过滤器开启状态时，果断买入标的指数ETF或构建多头期权敞口，此举旨在系统性地吸收市场非理性恐慌抛盘带来的流动性溢价。  
   * **均值平仓（Mean Exit）**：当 ![][image22] 值由负转正，回归至 ![][image23] 轴（即价格精确触及 EMA 均线基准）时，必须纪律性地获利了结，绝不将其转化为长期趋势持有仓位 46。

**策略优势与参数调优**：通过这一刚性的波动率过滤机制，系统能够近乎完美地规避在低波动慢牛或阴跌慢熊中被“趋势机器”无情碾压的尾部风险。对于中证1000而言，由于其散户参与度极高且流动性时常断层，其价格在受到冲击时更容易触及 ![][image24] 甚至 ![][image25] 的极端Z-Score水平，从而为量化系统提供远高于大盘指数的盈亏比。

## **策略二：基于Ornstein-Uhlenbeck过程的跨品种配对统计套利（Statistical Arbitrage Pairs Trading）**

**适用核心标的**：中证500期指（IC） 与 中证1000期指（IM）的跨品种价差；或高度同源的沪深300（IF）与上证50（IH）的价差。

**策略底层逻辑**： 中证500与中证1000指数在宏观基本面驱动因子、流动性松紧环境上高度同源，两者的绝对价格序列虽不平稳，但其价差序列（Price Spread）通常是一个具有强烈均值回归特性的平稳时间序列。根据严格的协整检验（Cointegration Test），二者的价差轨迹能够非常完美地被前文述及的 Ornstein-Uhlenbeck 连续随机过程所拟合 47。利用现代随机控制理论中的双重最优停时框架（Optimal Double Stopping Problem），量化系统可以通过OU过程的参数，精确推导出包含交易成本在内的理论最优建仓与平仓阈值边界，彻底消除传统移动均线滚动窗口计算带来的致命滞后性 49。

**参数设置与执行步骤**：

1. **参数动态校准（Calibration）**：利用极大似然估计法（MLE）或对过去 6 个月的高频历史价差数据 ![][image26] 进行一阶自回归拟合：![][image27]。据此解析推导并每日更新OU过程的核心动力学参数：长期均衡均值 ![][image28]，回归速度 ![][image29]，以及残差瞬时波动率 ![][image9] 49。  
2. **半衰期监控与断裂预警**：每日计算理论半衰期 ![][image12]。若测算半衰期在 5 到 15 个交易日的合理范围内，说明均值回归的内生动力充足，套利逻辑成立。若监控到半衰期突然拉长至数月，则发出极高等级预警，表明两指数间的协整关系极大概率已被宏观结构性断裂所破坏，系统应立即停止基于历史数据的一切套利交易 1。  
3. **最优交易阈值的触发**：将当前的资金成本、期指保证金比例及滑点摩擦代入双重停时偏微分方程，解出非线性的最优开仓上/下界（通常不等于简单的 ![][image30]）与最优平仓边界（注意：理论最优平仓线绝非简单的回归至 ![][image7] 轴，而是会为补偿资金时间价值留出提前平仓的提前量）。当实时价差突破计算出的最优开仓上轨时，系统自动做空相对高估的期指（例如做空IC），同时等市值做多相对低估的期指（做多IM） 47。

**策略优势**：此策略实现了真正的市场中性（Market Neutral），完全免疫A股整体牛熊交替的系统性风险（Beta），在单边暴跌的极度恐慌市或宽幅震荡市中表现尤为强韧。在标的选择上，虽然沪深300与上证50的配对因成份股重叠度高而回归确定性极强，但受制于两者的低波动率，套利空间往往过窄；相比之下，中证500与中证1000由于存在显著的年化波动率差值（22.8% vs 25.1%），其价差序列的拉伸与压缩幅度更大，为资金容量与获利空间提供了更优的解 9。

## **策略三：在线机器学习框架下的被动攻击型均值回归（PAMR）组合优化配置**

**适用核心标的**：由上证50、沪深300、中证500、中证1000四大宽基指数ETF共同构成的投资组合矩阵。

**策略底层逻辑**： 在多资产横向配置领域，传统的趋势跟踪动量模型往往受制于A股频繁的假突破和快速的风格轮动。被动攻击型均值回归策略（Passive Aggressive Mean Reversion, PAMR）创造性地引入了在线机器学习算法，不再预测单一资产的涨跌，而是主动利用多重金融资产在时间序列上的均值回归相关性进行资产权重的逐日动态重分配 51。其核心哲学极其反直觉但行之有效：如果在 ![][image6] 时刻投资组合中某指数（如中证1000）发生异常暴涨，PAMR模型基于深度学习的统计预期其在 ![][image31] 时刻大概率将面临回调压力，于是算法会“极其激进地（Aggressive）”降低该暴涨指数的仓位权重，并将释放的资金惩罚性地转移分配给当期表现最差、处于下跌状态的指数（如上证50）；而当组合的日收益率安静地维持在模型预期的安全边界条件内时，算法则“顺从被动地（Passive）”保持现有权重不作任何调整。

**参数设置与执行步骤**：

1. **矩阵初始化**：构建涵盖上述四大宽基指数ETF的量化投资组合，赋予初始均等权重（各占25%）。  
2. **损失函数与敏感度阈值设定**：在机器学习模型中设定一个至关重要的敏感性阈值参数 ![][image32]（代表算法对短期震荡容忍度的超参数）。当某日投资组合的综合加权收益低于设定的 ![][image33] 边界时，立即触发权重更新惩罚机制 51。  
3. **拉格朗日优化求解**：系统利用拉格朗日乘数法快速求解出一个二次规划问题——即在满足次日均值回归期望收益率大幅反弹的严格约束条件下，寻找到距离当前持仓权重变化幅度最小的全新资产分配向量。这不仅最大化了对均值回归利润的捕获，同时通过最小化调仓幅度极大地控制了交易摩擦成本 51。  
4. **执行频次**：根据交易通道速度与费率，设定为严格的日频定点或周频动态再平衡（Rebalancing）。

**策略优势**：在缺乏类似于美股长达十年的长期单边长牛、多呈现结构性风格轮动与宽幅震荡市特征的A股资本市场生态中，PAMR在线算法能够极其敏锐地自动化捕获各指数间“高抛低吸”的复合超额阿尔法，其多重回测表现（无论是夏普比率还是绝对收益）往往在极高置信度下碾压传统的买入持有（Buy and Hold）或常规动量组合 51。

## **策略四：极端高波动率状态驱动的宽基期权卖方均值回归策略（Short Volatility）**

**适用核心标的**：流动性日益深厚的中证1000股指期权（MO）及上证50ETF期权体系。

**策略底层逻辑**： 现货指数的绝对价格在遭遇史诗级系统性危机时（如2008年次贷危机或2020年熔断），可能会长时间沉沦甚至发生永久性损毁而永远无法回归历史最高均值。但是，“隐含波动率（IV）”本身作为一个衡量人类预期恐慌程度的纯数学衍生变量，其均值回归的物理属性是绝对且极度刚性的 16。当宏观黑天鹅事件导致市场暴跌、恐慌情绪无限蔓延时，期权的隐含波动率会产生巨大的风险溢价，远超标的资产后续实际发生的真实波动率水平。随着恐慌情绪的边际衰竭和市场干预政策的落地，极其高昂的隐含波动率必然以极快的速度、呈断崖式向其长期历史均值无情回归 44。

**参数设置与执行步骤**：

1. **波动率极值监控引擎**：构建实时监控体系，追踪四大指数期权的合成 VIX（或同等期权隐含波动率综合代理指标）。当且仅当 IV 数据点急剧飙升突破其过去三年历史数据的 90% 甚至 95% 极端分位数上限时，方可触发高级别交易信号 16。  
2. **非对称宽跨式组合构建（Short Strangle）**：以高弹性的中证1000期权为例，在量化模型综合研判其短期暴跌动能枯竭、转入高位偏向震荡且波动率结构明确趋于下降的黄金节点，构建卖出宽跨式期权组合。为控制尾部极值风险，应选择卖出深度虚值的合约。例如，在距离行权到期日适中的月份，同时卖出执行价位于当前现货指数 ![][image34] 甚至 ![][image35] 标准差之外（Delta 通常在 0.10 到 0.15 之间）的虚值看涨期权，以及对称或略显偏度的下限虚值看跌期权 9。  
3. **多维时间价值收割与动态对冲**：通过在极端恐慌的顶点卖出合约，系统将瞬间收取极其丰厚的超额权利金。随后，交易者只需耐心等待波动率刚性均值回归带来的期权隐含价值迅速内爆衰减，从而在无需判断现货未来准确方向的前提下，尽情收割 Vega（波动率下降）与 Theta（时间流逝）带来的双重绝对收益。若期间标的指数发生出乎意料的暴力反转反弹，可通过配置少量对应方向的股指期货进行机械的 Delta 动态对冲，锁定利润底线。

## **风险管理与极端状态转移的回撤控制**

尽管严密的均值回归策略在匹配正确的波动率区间下能够展现出惊人的胜率与收益，但其内生的尾部风险（Tail Risk）绝对不容任何专业交易者忽视。防范陷入万劫不复的“均值回归陷阱”，必须构建极其冷酷的量化风控体系：

1. **左侧交易的无限资金管理陷阱**：均值回归在交易哲学上本质属于逆势交易（左侧接飞刀）。如果在单边暴跌的雪崩中，系统错误地过早运用该策略并不计成本地向下摊薄成本，将会遭遇彻底的毁灭。经典的现代风控应对机制是采用将核心持仓规模与实时波动率严格负相关的反向挂钩技术（Volatility Scaling）。当VIX等波动率前端指标打破历史极值上限并维持陡峭的上升斜率（确认进入毁灭性的状态转移阶段）时，风控模块必须强制接管，无条件削减多头仓位甚至斩仓，绝不允许在深套的无底洞中继续加仓以博取虚幻的回归 39。  
2. **止损悖论的化解与时间停机机制**：均值回归策略面临着一个量化界著名的核心逻辑悖论——从单纯的数学期望上看，如果资产价格向下偏离均值越远、亏损越严重，其蕴含的未来反向回归势能和潜在利润空间就越大，此时在最低点平仓止损似乎是极其反数学常识的；然而，如果拒不止损，一旦该资产的宏观基本面发生了不可逆的结构性断裂（Structural Break，如公司破产、行业逻辑彻底毁灭），将直接导致账户清零爆仓 3。解决这一生死悖论的唯一量化之道，在于结合“刚性时间止损”与“底层基本面状态实时监测”。具体而言，如果在根据OU过程推导出的回归半衰期（例如15个交易日）界限耗尽之后，标的资产价差仍未呈现出任何实质性的收敛回归迹象，模型必须在逻辑上假定其长期均衡均值（![][image7]）本身已经发生了永久性的漂移，此时不论账面亏损多大，系统必须无条件触发平仓离场指令 1。  
3. **深度前沿：基于长短期记忆网络的动态混合切换机制**：在追求极致夏普比率的机构量化体系中，绝不应孤注一掷地单一依赖均值回归。全球顶尖的机器学习应用实践表明，最佳的生存法则是在深度神经网络架构中，将擅长识别并紧咬持续顺向趋势的“慢速动量模块（Slow Momentum）”与专注于剥削局部价格过度震荡的“快速均值回归模块（Fast Mean-Reversion）”进行有机结合。通过引入在线变化点检测（Changepoint Detection, CPD）和长短期记忆网络（LSTM），系统能够在低波动市场平稳期机械地跟随动量吃尽趋势利润，而在捕捉到趋势反转的断裂点或局部波动率剧烈跳升时，瞬间将仓位逻辑翻转回快速均值回归状态。长达数十年的回测强有力地证明，这种由AI驱动的动态混合自适应机制，能够极其夸张地将投资组合的整体夏普比率（Sharpe Ratio）提升33%至66%之多 33。

## **综合结论与前瞻**

中国A股市场，由于其极具特色的散户群体行为主导地位、复杂的微观流动性特征差异以及强力的外部政策扰动敏感度，为量化投资者展现出了一个相较于欧美高度成熟市场而言，更为广阔且富集着惊人均值回归交易超额阿尔法的生态圈。通过综合运用古典连续时间随机过程理论与前沿的波动率聚类深度学习模型进行深度降维分析，我们得出以下极具战略指导意义的核心结论：

第一，均值回归策略的有效性与超额回报绝对不是全天候无条件存在的，它极其严苛地依赖于标的宽基指数自身的内生波动率基因以及整个市场当前所处的宏观波动率区间。大盘价值核心资产的上证50与沪深300指数，由于绝对波动率较低且长期受制于庞大衍生品体系的套利平抑，更适合构建短周期、极窄区间的精细化期现微盘回归套利；而中小盘成长属性的中证500与中证1000指数，因其高达22%至25%的年化波动率、以及由资金过度拥挤或散户主导的情绪化剧烈折溢价，为深度的动态Z-Score极值反转策略及在线机器学习PAMR调仓系统，提供了A股市场中无可替代的最佳弹性土壤与爆发力。

第二，对波动率区间（Volatility Regimes）的实时识别与刚性过滤，是裁决均值回归策略成败的绝对生命线。在波动率长期低迷的阴跌慢熊或单边逼空慢牛的趋势市中，必须果断切断一切逆势回归企图，将主导权交还给动量因子；而唯有在遭遇突发冲击、波动率飙升且方向混沌的宽幅震荡市中，均值回归才能展现出摧枯拉朽的高胜率与盈亏比。利用Ornstein-Uhlenbeck过程结合严格的双重停时边界计算偏离度与回归半衰期，能够从根本上有效替代传统技术指标（如RSI、布林带）严重的主观性与致命滞后性。

第三，在多维度品种与高阶衍生品的实战应用版图中，全面基于OU等连续随机过程模型构建中证500与中证1000的跨品种配对统计套利（Pairs Trading），以及在市场陷入极度恐慌、波动率创出历史极值处无情做空期权波动率本身（Short Volatility Strangle），已经成为当前外资投行与国内顶尖私募量化机构在极端恶劣市场环境下，依然能够获取稳健绝对中性收益的最前沿重器。

展望量化交易的未来纪元，随着A股市场机构化演进的不断深化、高频交易监管的重塑以及更多类似于中证1000甚至更小微盘宽基衍生品工具的深度下沉与繁荣，过去那种依赖简单移动均线与固定技术指标的古典价量均值回归策略的生存空间，必将遭受降维打击与残酷挤压。未来的量化交易者必须毫不犹豫地拥抱并转向借助深度神经网络机制（如LSTM结合CPD变化点检测机制与动态参数自适应网络），以及构建跨越现货、股指期货与期权波动率曲面的立体化多维回归套利矩阵，方能在瞬息万变、充满结构性断裂的波动率残酷周期轮回中，永远占据统计学的不败高地，持续攫取低相关性、跨越牛熊的稳健超额收益。

#### **Works cited**

1. Exploring Mean Reversion Dynamics in Financial Markets: Insights from Hurst Exponent Analysis \- UI Scholars Hub, accessed March 21, 2026, [https://scholarhub.ui.ac.id/cgi/viewcontent.cgi?article=1180\&context=icmr](https://scholarhub.ui.ac.id/cgi/viewcontent.cgi?article=1180&context=icmr)  
2. Trends and Reversion in Financial Markets on Time Scales from Minutes to Decades \- arXiv, accessed March 21, 2026, [https://arxiv.org/html/2501.16772v1](https://arxiv.org/html/2501.16772v1)  
3. Machine Learning and Algorithmic Trading of a Mean-Reversion Strategy from the Cloud for Liquid ETFs on Robinhood \- Jerome Fisher Program in Management & Technology, accessed March 21, 2026, [https://fisher.wharton.upenn.edu/wp-content/uploads/2019/06/Thesis\_Fan-Zhang.pdf](https://fisher.wharton.upenn.edu/wp-content/uploads/2019/06/Thesis_Fan-Zhang.pdf)  
4. Mean Reversion Strategies: Introduction, Trading, Strategies and More – Part I, accessed March 21, 2026, [https://www.interactivebrokers.com/campus/ibkr-quant-news/mean-reversion-strategies-introduction-trading-strategies-and-more-part-i/](https://www.interactivebrokers.com/campus/ibkr-quant-news/mean-reversion-strategies-introduction-trading-strategies-and-more-part-i/)  
5. Comparison of the Performances for the Mean-Reversion Strategy for Exchange Rate, accessed March 21, 2026, [https://bcpublication.org/index.php/BM/article/view/3971](https://bcpublication.org/index.php/BM/article/view/3971)  
6. three essays on the trading behaviour of individual investors \- Durham E-Theses, accessed March 21, 2026, [http://etheses.dur.ac.uk/14242/1/Ph.D.\_Thesis.pdf?DEF+](http://etheses.dur.ac.uk/14242/1/Ph.D._Thesis.pdf?DEF+)  
7. High-frequency lead-lag relationships in the Chinese stock index futures market: tick-by-tick dynamics of calendar spreads \- arXiv.org, accessed March 21, 2026, [https://arxiv.org/html/2501.03171v1](https://arxiv.org/html/2501.03171v1)  
8. Economic Policy Uncertainty and Stock Market Volatility: Evidence from China's CSI Indices, accessed March 21, 2026, [https://dspace.cuni.cz/bitstream/handle/20.500.11956/196855/120498449.pdf?sequence=1](https://dspace.cuni.cz/bitstream/handle/20.500.11956/196855/120498449.pdf?sequence=1)  
9. 中证1000股指期权介绍和上市首日交易策略 \- 新浪财经, accessed March 21, 2026, [https://finance.sina.com.cn/money/future/wemedia/2022-07-21/doc-imizirav4840628.shtml](https://finance.sina.com.cn/money/future/wemedia/2022-07-21/doc-imizirav4840628.shtml)  
10. Ornstein-Uhlenbeck Process for Mean Reversion \- QuestDB, accessed March 21, 2026, [https://questdb.com/glossary/ornstein-uhlenbeck-process-for-mean-reversion/](https://questdb.com/glossary/ornstein-uhlenbeck-process-for-mean-reversion/)  
11. Ornstein–Uhlenbeck process \- Wikipedia, accessed March 21, 2026, [https://en.wikipedia.org/wiki/Ornstein%E2%80%93Uhlenbeck\_process](https://en.wikipedia.org/wiki/Ornstein%E2%80%93Uhlenbeck_process)  
12. Volatility Clustering Explained: Why Market Chaos Breeds More Chaos \- Stock Titan, accessed March 21, 2026, [https://www.stocktitan.net/articles/volatility-clustering-explained](https://www.stocktitan.net/articles/volatility-clustering-explained)  
13. Stochastic Volatility Models Simulation and Statistical Model | by Simon Leung \- Medium, accessed March 21, 2026, [https://medium.com/@simonleung5jobs/stochastic-volatility-models-and-statistical-simulation-51ff813084b1](https://medium.com/@simonleung5jobs/stochastic-volatility-models-and-statistical-simulation-51ff813084b1)  
14. (PDF) STOCK RETURNS, VOLATILITY AND MEAN REVERSION IN EMERGING AND DEVELOPED FINANCIAL MARKETS \- ResearchGate, accessed March 21, 2026, [https://www.researchgate.net/publication/325482252\_STOCK\_RETURNS\_VOLATILITY\_AND\_MEAN\_REVERSION\_IN\_EMERGING\_AND\_DEVELOPED\_FINANCIAL\_MARKETS](https://www.researchgate.net/publication/325482252_STOCK_RETURNS_VOLATILITY_AND_MEAN_REVERSION_IN_EMERGING_AND_DEVELOPED_FINANCIAL_MARKETS)  
15. Stock returns, volatility and mean reversion in emerging and developed financial markets, accessed March 21, 2026, [https://journals.vilniustech.lt/index.php/TEDE/article/view/1690](https://journals.vilniustech.lt/index.php/TEDE/article/view/1690)  
16. Volatility's a mean-reverting asset but speed, level are unclear | Insights \- Bloomberg News, accessed March 21, 2026, [https://www.bloomberg.com/professional/insights/trading/volatilitys-a-mean-reverting-asset-but-speed-level-are-unclear/](https://www.bloomberg.com/professional/insights/trading/volatilitys-a-mean-reverting-asset-but-speed-level-are-unclear/)  
17. Predicting the volatility of Chinese stock indices based on realized recurrent conditional heteroskedasticity | PLOS One \- Research journals, accessed March 21, 2026, [https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0308967](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0308967)  
18. 5-minute high-frequent data for SSE 50, CSI300, CSI500 and CSI 1000 indices \- Dryad, accessed March 21, 2026, [https://datadryad.org/dataset/doi:10.5061/dryad.18931zd65](https://datadryad.org/dataset/doi:10.5061/dryad.18931zd65)  
19. empirical study on the effectiveness of the dogs of the dow strategy for chinese retail investors, accessed March 21, 2026, [https://waseda.repo.nii.ac.jp/record/75743/files/WasedaBusinessSchool\_mba\_2022\_0409\_57200545.pdf](https://waseda.repo.nii.ac.jp/record/75743/files/WasedaBusinessSchool_mba_2022_0409_57200545.pdf)  
20. Q4: What are the differences in compilation methodology between CSI 300 Index and SSE Composite Index?|CFFEX, accessed March 21, 2026, [http://www.cffex.com.cn/en\_new/rdhd/20101124/15095.html](http://www.cffex.com.cn/en_new/rdhd/20101124/15095.html)  
21. Economic Policy Uncertainty and Stock Market Volatility: Evidence from China's CSI Indices, accessed March 21, 2026, [https://dspace.cuni.cz/bitstream/handle/20.500.11956/196855/120498449.pdf?sequence=1\&isAllowed=y](https://dspace.cuni.cz/bitstream/handle/20.500.11956/196855/120498449.pdf?sequence=1&isAllowed=y)  
22. High-frequency lead-lag relationships in the Chinese stock index futures market \- arXiv, accessed March 21, 2026, [https://arxiv.org/pdf/2501.03171](https://arxiv.org/pdf/2501.03171)  
23. Index option trading and equity volatility: Evidence from the SSE 50 and CSI 500 stocks, accessed March 21, 2026, [https://ideas.repec.org/a/eee/reveco/v73y2021icp60-75.html](https://ideas.repec.org/a/eee/reveco/v73y2021icp60-75.html)  
24. Research on the Impact of SSE 50 Index Futures on the Volatility of the Target Stock Market, accessed March 21, 2026, [https://francis-press.com/papers/10750](https://francis-press.com/papers/10750)  
25. How Do Chinese Option-Traders “Smirk” on China: Evidence from SSE 50 ETF options \- ACFR, accessed March 21, 2026, [https://acfr.aut.ac.nz/\_\_data/assets/pdf\_file/0010/265366/How-do-Chinese-option-traders-smirk-on-China-Evidence-from-SSE-50-ETF-option.pdf](https://acfr.aut.ac.nz/__data/assets/pdf_file/0010/265366/How-do-Chinese-option-traders-smirk-on-China-Evidence-from-SSE-50-ETF-option.pdf)  
26. 5-minute series of CSI300, SSE50 and CSI500 index. \- ResearchGate, accessed March 21, 2026, [https://www.researchgate.net/figure/minute-series-of-CSI300-SSE50-and-CSI500-index\_fig1\_332666795](https://www.researchgate.net/figure/minute-series-of-CSI300-SSE50-and-CSI500-index_fig1_332666795)  
27. 【大浪淘沙】每周研报精选（2025年第2期）（202501013, accessed March 21, 2026, [https://finance.sina.com.cn/roll/2025-02-18/doc-inekwcvp6103496.shtml](https://finance.sina.com.cn/roll/2025-02-18/doc-inekwcvp6103496.shtml)  
28. Novel Portfolio Construction Based on Traditional Stock Index \- SciTePress, accessed March 21, 2026, [https://www.scitepress.org/publishedPapers/2024/132080/pdf/index.html](https://www.scitepress.org/publishedPapers/2024/132080/pdf/index.html)  
29. Novel Portfolio Construction Based on Traditional Stock Index \- SciTePress, accessed March 21, 2026, [https://www.scitepress.org/Papers/2024/132080/132080.pdf](https://www.scitepress.org/Papers/2024/132080/132080.pdf)  
30. Volatility and Market Regimes: How Changing Risk Shapes Market Behavior (with Python Examples) | by Trading Dude | Medium, accessed March 21, 2026, [https://medium.com/@trading.dude/volatility-and-market-regimes-how-changing-risk-shapes-market-behavior-with-python-examples-190de97917d8](https://medium.com/@trading.dude/volatility-and-market-regimes-how-changing-risk-shapes-market-behavior-with-python-examples-190de97917d8)  
31. Momentum vs Mean Reversion Strategies for Challenges | For Traders, accessed March 21, 2026, [https://www.fortraders.com/blog/momentum-vs-mean-reversion-strategies-for-challenges](https://www.fortraders.com/blog/momentum-vs-mean-reversion-strategies-for-challenges)  
32. Momentum vs Mean Reversion: Which Dominates in a Choppy Market? \- Bookmap, accessed March 21, 2026, [https://bookmap.com/blog/momentum-vs-mean-reversion-which-dominates-in-a-choppy-market](https://bookmap.com/blog/momentum-vs-mean-reversion-which-dominates-in-a-choppy-market)  
33. Slow Momentum with Fast Reversion: A Trading Strategy Using Deep Learning and Changepoint Detection | Portfolio Management Research, accessed March 21, 2026, [https://www.pm-research.com/content/iijjfds/4/1/111](https://www.pm-research.com/content/iijjfds/4/1/111)  
34. Comparison of the Performances for Mean Reversion Strategy for DDAIF, Ford and VWAGY, accessed March 21, 2026, [https://www.researchgate.net/publication/369433950\_Comparison\_of\_the\_Performances\_for\_Mean\_Reversion\_Strategy\_for\_DDAIF\_Ford\_and\_VWAGY](https://www.researchgate.net/publication/369433950_Comparison_of_the_Performances_for_Mean_Reversion_Strategy_for_DDAIF_Ford_and_VWAGY)  
35. How Volatility and Turnover Affect Return Reversals \- Alpha Architect, accessed March 21, 2026, [https://alphaarchitect.com/how-volatility-and-turnover-affect-return-reversals/](https://alphaarchitect.com/how-volatility-and-turnover-affect-return-reversals/)  
36. Mean Reversion vs Momentum Forex Strategies Guide \- LiteFinance, accessed March 21, 2026, [https://www.litefinance.org/blog/for-beginners/trading-strategies/mean-reversion-vs-momentum-strategies/](https://www.litefinance.org/blog/for-beginners/trading-strategies/mean-reversion-vs-momentum-strategies/)  
37. Mastering Mean Reversion: The Power of Statistical Gravity in Markets | Macro Ops: Unparalleled Investing Research, accessed March 21, 2026, [https://macro-ops.com/mastering-mean-reversion/](https://macro-ops.com/mastering-mean-reversion/)  
38. Machine Learning-Based Adaptive Time Series Momentum Strategies in Equity Index Futures: A Comparative Analysis Between S\&P 500 and CSI 300 Futures Markets \- Preprints.org, accessed March 21, 2026, [https://www.preprints.org/manuscript/202603.1400](https://www.preprints.org/manuscript/202603.1400)  
39. Volatility Mean Reversion Stategy : r/algotrading \- Reddit, accessed March 21, 2026, [https://www.reddit.com/r/algotrading/comments/1roxy1x/volatility\_mean\_reversion\_stategy/](https://www.reddit.com/r/algotrading/comments/1roxy1x/volatility_mean_reversion_stategy/)  
40. Volatility and Regimes. : r/quant \- Reddit, accessed March 21, 2026, [https://www.reddit.com/r/quant/comments/1k9pkkp/volatility\_and\_regimes/](https://www.reddit.com/r/quant/comments/1k9pkkp/volatility_and_regimes/)  
41. 华泰证券A股策略：适度回归性价比与景气度, accessed March 21, 2026, [https://www.stcn.com/article/detail/3338486.html](https://www.stcn.com/article/detail/3338486.html)  
42. Annual Report \- HKEXnews, accessed March 21, 2026, [https://www.hkexnews.hk/listedco/listconews/sehk/2025/0428/2025042801747.pdf](https://www.hkexnews.hk/listedco/listconews/sehk/2025/0428/2025042801747.pdf)  
43. JPM Bram Kaplan \- 2023 Equity Derivatives Outlook | PDF | Recession \- Scribd, accessed March 21, 2026, [https://www.scribd.com/document/663120496/JPM-Bram-Kaplan-2023-Equity-Derivatives-Outlook](https://www.scribd.com/document/663120496/JPM-Bram-Kaplan-2023-Equity-Derivatives-Outlook)  
44. Volatility and the Alchemy of Risk \- CAIA, accessed March 21, 2026, [https://caia.org/sites/default/files/03\_volatility\_4-2-18.pdf](https://caia.org/sites/default/files/03_volatility_4-2-18.pdf)  
45. Mean Reversion Trading: Understanding Strategies & Indicators | FTO \- Forex Tester Online, accessed March 21, 2026, [https://forextester.com/blog/mean-reversion-trading/](https://forextester.com/blog/mean-reversion-trading/)  
46. Mean Reversion Strategies: A Guide to Profitable Trading \- Trade with the Pros, accessed March 21, 2026, [https://tradewiththepros.com/mean-reversion-strategies/](https://tradewiththepros.com/mean-reversion-strategies/)  
47. An Application of the Ornstein-Uhlenbeck Process to Pairs Trading \- arXiv, accessed March 21, 2026, [https://arxiv.org/html/2412.12458v1](https://arxiv.org/html/2412.12458v1)  
48. Optimal Strategy of the Dynamic Mean-Variance Problem for Pairs Trading under a Fast Mean-Reverting Stochastic Volatility Model \- MDPI, accessed March 21, 2026, [https://www.mdpi.com/2227-7390/11/9/2191](https://www.mdpi.com/2227-7390/11/9/2191)  
49. Trading Under the Ornstein-Uhlenbeck Model \- Read the Docs, accessed March 21, 2026, [https://hudson-and-thames-arbitragelab.readthedocs-hosted.com/en/latest/optimal\_mean\_reversion/ou\_model.html](https://hudson-and-thames-arbitragelab.readthedocs-hosted.com/en/latest/optimal_mean_reversion/ou_model.html)  
50. Speculative Futures Trading under Mean Reversion \- RePEc, accessed March 21, 2026, [https://ideas.repec.org/p/arx/papers/1601.04210.html](https://ideas.repec.org/p/arx/papers/1601.04210.html)  
51. PAMR: Passive-Aggressive Mean Reversion Strategy for Portfolio Selection \- Institutional Knowledge (InK) @ SMU, accessed March 21, 2026, [https://ink.library.smu.edu.sg/context/sis\_research/article/3295/viewcontent/PAMR\_Passive\_Aggressive\_Mean\_Reversion\_Strategy\_for\_Portfolio\_Selection.pdf](https://ink.library.smu.edu.sg/context/sis_research/article/3295/viewcontent/PAMR_Passive_Aggressive_Mean_Reversion_Strategy_for_Portfolio_Selection.pdf)  
52. Mean Reversion Strategies \- Jonathan Ho \- Medium, accessed March 21, 2026, [https://johoblogs.medium.com/mean-reversion-strategies-a0e54ddd5af8](https://johoblogs.medium.com/mean-reversion-strategies-a0e54ddd5af8)

[image1]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAEQAAAAYCAYAAABDX1s+AAAB+ElEQVR4Xu2WOywFQRSGDxqviE6UlAoJKq9oNEjotAqNRCnRKdBR0BOCiEKjUZEQovCIkAhR6jziEUSQCOe/M3Mz95jZDZG9ivmSv9j/P7M7MzuZGaJAIBD4O2ZYj6xPS1c6y2Fdi+yO1a7zpBki1dcXVq/IfNSw1lm1rFxWJWuStWIXuTADdrFBKiuRQYKcstas5xPWjvXso5UyfyiEnxwJVgIKD2WgiZqsJMCPcH0fXqk0BS2sVdY0a4RVnBm76Sf18k4ZaJC9STNBjsg/IRhoFE2sYWnGgT3D9UHQTCrD7GYL3wr1+TaN9IsJMS/uYnWQ2jTbtHZ1VpCuTh7fwH2+TT1rkVTdEqkNeSujwgGKz1kDQoM6i/uoTTVrwaN51hxrltTphuU+pZpF4uuDz7fB6XIhPLTZFl4as3/4jlJk2dw/gG/gPj+OS4pod0P+sIFUNiqDhPEN3OfHsUmqXZnwU0S9FOc+skIZRFDBGvuh4ngidx/hnUlT4BrfgfbyhU95Oviv9w9DN7n7Aa/OesYAsQXYoGZZeO/a/8YEqQAflOBU+S8TAtCPPut5XHs2pr9Vloc7DI5eQzmpmh7LSx0/WIb3rFvWA+tDZ0WsZ+0hQ80rqSM5m5gftMc6JtUn3LBt0Md94QFc+9HWrAxc5wOBQCAQCCTPFzPipIEG4pCcAAAAAElFTkSuQmCC>

[image2]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAEQAAAAYCAYAAABDX1s+AAACSUlEQVR4Xu2WvUsdQRTFr4rfIiGN2GpnFbWJGkljY4TY2VokRUhKwc4iCQTEQv8AsQnBwsYmlYKiRNCIKIgmYAo7TeIHGoIR/Ljn3dnHvOvMvn3vyWsyPzjFnjOzO7Oz3LtEgUAgcH9Msc5YN5YOTVbC+qmyY9YzkxebEZK1/mW9UJmPVtY8q41VympiTbBm7UEuog27WCDJ6nVQRHZYc9b1NuuLde2jhzIPFMIhx4IvAQM3dGCIe1kuHmujQHAQrufDe6BNxVPWZ9Yk6x2rLjN284bk5s91YED2T5sx9JLMea2DPNkk/wvBRuN4wnqrzWygZrgeCLpJMrzdXOkgmTuqgxzxfaE+36aL8ngh0Y37WX0kRROnDK2arDo9OneaWResaR0kxLdxn2+DQ/lEMg7PR0FeyhjhAIO/s4aUhk2W7aFJecg6omTF0Ma3Bp9vg+6yrzzMWVZemqh++FopslzqRxIqWT9IOkeZylz4Nu7zs3FAMfN+kT/sJMne66BA0NVWSGpXjcpc+Dbu87OxSDKvQfkp4m6Kvo8syaKTUMHaY31jlassjnNyrxHerjYVrv2tG69K+anPFcF9/X/4QO34TfKDlw8D5F4HvHbrGhtECbDBmBnlXRr/DuMkAR6oQVcp9IWgu6D+fNRBHmAdr6zrMePZROttsTz8w6D1RjSSjBm0vFT7wWd4QlL1T1lXJqtl/TEeMoxBy0RLTsoj1jXrgw4KIDqgNdYWyZpQi2ywxq/KAyjemBt9GfidLyovtREIBAKBwP/JLbMUqCzoTt0HAAAAAElFTkSuQmCC>

[image3]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAEQAAAAYCAYAAABDX1s+AAACQ0lEQVR4Xu2WvUuWURjGb23JDyRyCMEhXIQ2c7KUGlxSqEnDqaElaCgR3BoqxMHB/oAQl3BoaWlSUJSGPpACKRzdSukDFVHB6r68z5Hz3pz7eZ7X901Czg+u4bmuc85zvjjnECUSiUT1mGJtsv4E+uayGta6yn6w+lx+0jwi6esO667KLDpYc6zLrFpWG+sZ61VYKIYfcIx5kqxJByfIZ9Zs8L3CehN8W/RS6YJCWORMsBNQcFkHjqzJqgT8t1mbEbAQsf/DO6dNxTXWa9Zz1hNWY2kc5z5J4zd14EC2p80KOE+ySth5RfhI9oRgoFl0sx5rMw+cGbEfgh6SDLNbKe2sfdYLHeRg7VDLD7lKx5gQ3/AtVj/JoXnD6a3L6o5Kl891kjYmlF8Ua+CWH9JFsgAoN0NyIC+WlIiAwqusEaVRl+X91GKIpO6wDsrE6oPlh+B2WVMe6iwp7wh/flhXKbJyz4+HJPVu6+CYWAO3/Dy+Uka9DbLDKyTZUx3kMEZSD+dPNbAGbvl5LJDUu6D8Q7Iaxb2PrF4HBfE7ZVAHZbJF8T7C+6JNRWx8H5x3Vvl0xgX/+v0xQNLOAx0UBBMa6we8zuAbA8QREIIyL5WHmy7WHk2SvYK4Vao1IR6c+L9Z4zooAPpxL/jGjaX75vt7KfDwhsHV62khKXMn8A6vH2zDn6zvrF+sA5c1sLadhwxldkmu5GpxkeT6my61M/EL9I71iaRPeOmGoI/vlQfw7EddvzPwnP8vwRZv1WYikUgkEqeHv18iqAC/6l2wAAAAAElFTkSuQmCC>

[image4]: <data:image/png;base64,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>

[image5]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABYAAAAYCAYAAAD+vg1LAAAA+klEQVR4Xu2TwQoBQRzG/1IOpBy9gfISLnKSwtEjODrIWV7CxcWBp3CWwplyIwdEolD4ppltZ//tZhmc9le/mvm+dnZ3Zpco4N9U4AU+NDdaf2XdXOt8kSN54ZLlMXiHUZa/hfVUPDOmT3KhppqLcdiuPydE9lOfYcJZm3EjuXCGF6YMSS684IUJbVgn90N0Q3wxL6nCrhpbhyhu4kUeTnnIycKRNtcP0YsJLPJQJwXXPCT7j0uyPK1yy56zJorADslSjDk1kt2KFwrXt5nBPdzCAzw5a9qpXPRifIQNrS+Qj/39hDEs8fAbWNsg9juuF6a04ACWeRHwW55WckGri9M16wAAAABJRU5ErkJggg==>

[image6]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAcAAAAYCAYAAAA20uedAAAAcklEQVR4XmNgGOTgGxCfQheEgf9AXIAuCAL6DBBJJmRBGyD2AuLdUElfKB8MioC4BCrxFsoHYRQAksxFFwQBXQaIJCO6BAisYYBIYgUgiXfogjAAkgQ5CgaOILHBkipQ9k9kCRDoYYAo+AHELGhywwEAAMS4F/hUVNxNAAAAAElFTkSuQmCC>

[image7]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAwAAAAXCAYAAAA/ZK6/AAAAnElEQVR4XmNgGAWDEdQDsRiaGCcUYwX/gdgdi9haNDEw8GaASKIDkJgxuiAInGPA1BCGRQwOQBK9WMR+oImBgTADRFIKTRwk1g5lT0eWmMQAkVRBEiuBivEDsQ4QhyLJMfyDSt6DKpgNxH1QMR8g/oVQCgEgiR4g1gXiaCBmRJILRGKDAcz9AugSuADMaqLBHwYcQYcLMKMLDCwAALFXIR+o4jgvAAAAAElFTkSuQmCC>

[image8]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAoAAAAXCAYAAAAyet74AAAAn0lEQVR4XmNgGHQgDYgL0AWRAQcQ/wdicSC2AeKfqNIIAFJkhcZnR+KDwRYg/o0mBlLoiiwgDBUMRBaEihUhC2yDCiIDFaiYO7IgSOAvEDNB+YxAPA0qDgcsUIF7QHwACYPEUBQmQQXkkQWhYteRBZZCBZHBIixiDNlYBEF8kE0oAGQlssI+IP6GxEcBX4E4BIiroGy8wBmIRdEFRzYAABkLJxqYpSHRAAAAAElFTkSuQmCC>

[image9]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAwAAAAXCAYAAAA/ZK6/AAAAcElEQVR4XmNgGAWDFfABsTsQe6FhDCAPxP/x4HSEUgYGXqhgAZLYKyD+h8RHASDFG9DEUqDiGGAqA3aJOwzYxRlOMWCXAIl1oAuCQAUDpoYVQPwLTQwF/AXifCAWBuJ9QHwfVRo7UAfiSHTBUUAOAADB0RznVYHwAAAAAABJRU5ErkJggg==>

[image10]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAACIAAAAYCAYAAACfpi8JAAABqklEQVR4Xu2UvytGYRTHj1JEUpIkg8LAYBMLC4Mi5ccig0mUjUkpEhtlsCmS39n8AzLZ/FoMskgZzH4L5+s5z3uPc+97uxm8qfdT33rO93uf857u87yXKMs/ppxVac2/ZJ31KRpnNbEelAddy7NDrBeT7UsGHk1WrbJE1JHbmKs83yyKuGyX1WrNpOxRuHHcj8Vl/u39CjTFcWjuxbesUvpBzq2RhAXWrKzRdEpl4Fj8HONfim8HKWKtGC+WSdarrGspaJqXesKxIX6V8vxrv6XwIM+mjmWYXIN85V2IZ5km57dLXcZakvWRZAVSt7D6ZZ0IbL6L8J6MBwbJZaNSv6tsTbIGqT9UZim0Rge5zSPGhzdjPNBMLltkdZLb78F9QtbDmmNVqEzTxTqz5jaFj6BGPH1UnlJy2QHrzWQDkk2wbkymOSU37A+WKTzIpvK2dCAggzCUplFl9l8F6inIIXzkUuCs9CB9UnvPDgngXVmTKSaX7djAENXzmzYKfnxMPFw01CX+IUXaRhSfgW6KuB+Z4ITVa81M4N8Y7gu+uhljnnVI7i5myZKYLxNzepKptrwFAAAAAElFTkSuQmCC>

[image11]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAFIAAAAYCAYAAABp76qRAAACqElEQVR4Xu2Yu2sUURTGj+ITjYKCQS2MIYKNhWKlgviKhaioCIqNKFqLoH9ADKRIHbFSrCxSCCJio6YIKvjEQgtRQdBG4wNf+PZ8zrnJyTd3sju7uNlZ9wcfu/d3du6de5m9s7MiTZo0ClM0c1g2OEtZlOKw5ghL4geL/4DbmmUsY0zT/Na0atZovo4uD/NTM5FlnfJakjn5rLLaIfKY11SrZYHPzWDJ+EFCmzs+rblArgiExZpE/rGmi9xYdGo+sfRc1Hwnh4E3RVwROSbJuV927pTmhGuXC/pZyBLMlaS4gzzcUdfeba6ohKsS4D5w3tXygL1ykCW4JOkF6jC32bnnktGB4wwLSfqqB55IMqduzQOq5WGPpNfrL5D+BjJBc9K8B21s0Flg0fkYAFfW3e4fM19GX5XVkOoDmy/kU82AS2xAtDeQ89yS9DG7Im68mC4j88LFUg2pOR0wuYg83KOIW0zOg/odcg/Nl2JljlQKzuO4vfZTLS/oAz8Xh+kz6cFNB24Bebg2ch7Ut0bcVXIxtuVIJfirEO95zkw59RYvek163mvekAP43DqWRlh8Bm4jyxqDB4uZro3fjjivsbap2Fw8qfoWknh+Tn3IgD/I0sBdkI/b59xeGZ8bzitNO7l5kpzXN/IA3ieLaM1LvN/u2p6XmmssjTBw+Pr4jR18ttdasUKSsV9wwQjn5q/UwF3JXgOQeQPFQSj80iyhmme/ZHQgiR+wV+SseSwg2mutXQswJramt5oPmh5Xa9N8tNqQ5p2kH/my5hi4rrnBMi+xQXZK3BeVUnNBfTLLvOCx6hy5+1J68KKwXkZ+9j3zBWO1xPfWiuBFQ/smuSKDhbrH0sATIP+DVDH4a+2La6NzPH41Olc0y1lWyyzNbJYNjv+vtkmTOuYPNxi/SrSq/VgAAAAASUVORK5CYII=>

[image12]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAHIAAAAYCAYAAAAmsqlBAAAEBUlEQVR4Xu2ZW8hUVRTHVypqkqLilTQt7cEHL9CLD6IEgSSKlKZPRXkJBQUVRfFN0QdvWQ+hlqIWVN4SNCK6YahhKiIi+iIKKlqWpUaZmZf1n72PrvnPOuebcT7ma8bzg8V39m+fy8xes2/nE8nJycmpVyawaIJhLMqkB4sckb4anVg+Ap9qjGBZBvdYlEHWNfguWzT6kW9YvpDQIIiRVFcpYzS+Zql00Dgp4RlnNJ4rri7wrMYfLDNorbGXZWS7xo/x+K5G/4dVxfytcZhlndMcifR6CBL0iyl/K+G81cYl/KUxjmUKn2g8yVJ5W0LyEj7Q+NWUi8AHmcOyzqk2kYs1jrOUcN+5jvOS/rz43iPtPPhepvx+dCUMlVDRiivqnGoTieuHsxQ/aTuim0kewGMozuIZjZUsld+k9FnH2GECxxzwTazAEIByo2AT2U1CI6+Nf8Fojc80Xo5lhhswYbyU9sh98rANGXgvSZbTLCK4dr3jij7bPI35UV6NZUSjYBOJxciH0WGOOR89EgyH4cqCH3RaIj1KGteAxdItlsQdFsprEu7Z2bg20dk5+gGomM2yArB6q4bJGh+nxEcSltybNTZpbNR4r3BV09hEWscNft1xSxyXRtLg+B4e6FFZ9xqr8TpL5XcJ1+0zcTS6FclJCYNjxRNcEcFDsoaF9hoDTHmXhPthSY7ldEuSlshz5PDr5obe6jgPtBvOm84VhoWSfa+bLCK4hq/b7bgCO8Wv6CphmP1JshNp9z3fmeMp4t+3luD5oxx3itzF6C2bHeeBc15kSSyQ7HtdYhHBNXsc9x+5AqhAF07jB8lOpH0Q7nWIyh1NOY2XJDyj3FgWLmuStESeIHchessixzHYe2N7kYCRaaopJ6yT9Hthi4NdgweumWXKmCvh7FbkAajAgifhgDkGWYkcJGFo9cBWJu3D1wo8n3sLHM/pXo98xXGWs1L6XhQLpu7kwFcSku6R9QzUvWrKmMvtqFcETh4Yj72VFRK5imUkWfl54ItuYFlD2kr4bhPJw6EHWq5Fz3gOIDGo88ID/l2WElag2PqlgVX2tnj8lsa/pq4EvFbCg/6RcGMGiVzDMrKcReQN8T94rcBnviwhYfiLHyj2d1jUwKEHYrsF8Bdl+CtS/HIc7fKCKSdw8spJZBeWEl7JPcWSSFau33NFpaBR3mEp6UttjPcz4jF6RTtTV298qXGQZYX0kewE1wwk0utd3piPle7nGm9qTJPQEPVOtY2N/37YeS6ht6SvPZqdGxKGHrzv+5PqJlEZ8DBTbSP8H8AKOZmnKqWnxs8sI0dYtARLWTQ4+zWGsCyDrB/ybRYtAVakjxvJvF8u/EaJeZpFrcErKftKLicnpzm4D2DpG4t1qxjPAAAAAElFTkSuQmCC>

[image13]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAwAAAAZCAYAAAAFbs/PAAAAiklEQVR4XmNgGAXDAswG4v9AfABK16HIooFfQHwBTQykyRHK/o0scYMBIokOQGIgg0BgN7rEc2QBKPjHAJEzB+JomKADVNAdJoAEHjFA5FBsX40ugASuMUDkJJEFG6CC2MBFBhxyIEFVNLF7QLwWKgcCfUhyDEJA/JcB4d4ZSHL3oWLxSGKjgLoAACgOJkiAxK1GAAAAAElFTkSuQmCC>

[image14]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAwAAAAYCAYAAADOMhxqAAAAxElEQVR4XtWQOwoCQRBE21gMDTT2GIIopt5FMDEQBDMxVQ/hKfwlnkED76AY+atiZ4fZ2lmN90FB82a6Z2izUlFDlshID2JskIWrW8gnOMsxRtri2DAR5zmrsKRhppKskapKSxoqKkn61zrScfUzqHPskIYljQwv3zM3AnoWn8TGqUqyV+HYIg+VpGjXb+SmkpxUODhopbKPXFSCoRW8fERelt01F8DLzcB50ilXVzNzfxqhaENR+P+uyl8cVPxjoKJkfAEDOycZaGYRWwAAAABJRU5ErkJggg==>

[image15]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAADAAAAAYCAYAAAC8/X7cAAABmklEQVR4Xu2Vu0rEQBSGjxfwgjaCgiBW2ggWgiAKwiqInaWNhW+gnYIgrhewsxQrX0AEH0AQ7CwtFCsFQQRfwAt4+Q9nsns8TsKYuMsu5oOPJP/JTDLJZEKUk/NvaIa7cB02mlrNswyP3H43/IQd5XJtMwFXTXYGT01WEfhJZeXVBuAZntuwEmQdwDyctCFJv/02jJiBJ3DJFlKQdQD8pJlOOA2b4ANcKZ2hGCW5YMEdr7njiH21H0rWAby4Lfej/cEwSaHL5JxFX/+7LgTivVggbXDDhuCG5CP+Bl/o0Ybgg6Q2BhdMTdND8gat3NZm7JA0S2QLttgQFMk8mIILZnXouKeE16YYgHMeuZ3N2Clplohv9WEuyNwPT5G4G7wmqfXaQiBx/YbAb98H93mlg6ILfVxSfC2EtG3byd92hCRvsAUOB012C49djdlTtVB8NxHCDnwiWT4j+kj6G1dZCV59eJXhE9gDVbtz2aLKQkk7gDe3jWYAe1guV4+0A4j7gKsO/zl/C8//TRvWE9uw1Yb1RMg/Iifnr/gC58RbP2xuOesAAAAASUVORK5CYII=>

[image16]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAD8AAAAYCAYAAABN9iVRAAAB9klEQVR4Xu2WOywGQRDHx6uQiJJCQyIagkKnICHRKUSj1YlCp1RRoJJQeZMoREIiIp6FiohCodBoVMQzXokQZm723Nx8e3xf4bsv7C/5527+s3e3l93ZXQCHw/GfmUI9oD6MZkJZ5h2CPKkpnE4bm8Dfv0V1qpxPOeoAuN22ykUif87GLvCL44L6lWfuO0x8GaQ9GozvU6tiK1moddQycOO2cNrjx5f8ItS3FeWtAfepVXgUd4mYeEXtKy9ED6rO3EeN/ps20gh9m/rULrxK412YuMjEdJX4pRLJtbi/A25cKLwK1KCI000Jak55jcD9PDJxn4k102D3v5BJqmuKT4W3gCoQcSawAdzPKhNTWdh+cgzsvgfV+6ry9NSPfNhCNWo+QjR6s8CjQbvMBGqcH0uJHOA+HQpvz3iaEWCfZk8Cst6lRw8Mm5gWjUziBYLp7kOz0/bzo8B+rk4QN9ow+KNPC0u/ysXJCWpRmxBd85Ng9z2iElvAuTNUvsp9RxlqKEUlyxJqQHnn5loP3N+kV3uaCjvaNGRDMPqZQC8k7uHFwAuaD/VVn1EewTK7adG4Aj4KRvEMXF9xQ8dpfyC0WkQ72gHkeYQWc2pTKjyvZu6B93fa1+nsbqMG1a3NGNA/LEUzVHKMegIuEco3h9MOh8PhcPxpPgEkDJQrH6MihAAAAABJRU5ErkJggg==>

[image17]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAD8AAAAYCAYAAABN9iVRAAACBUlEQVR4Xu2WzytmYRTHjx+FEtnYMGVKUgoLOwuLmYW1ZjVl9rLwF7BSwkoNG8aPmbKQGqWIwWIyYmkhERbKzs8YJDLnvOe5r/OceR6vWXjvxPOpb+8933Pufc+99zz3XoBAIPCaGUado+6NRq0scwcPedI7O50WTlHNqCJUIeoD6sSqYMpRa8B9LqicF3lyLpaADxwXsr9IpVYFQIPxI2pV7CQDNYv6DlzcZKcTpDzIM0P/34UaAP/kUU2L8m5Qq8qzaEPVmW3f3b/VRppx9SQpBq6hX8m88b0ciW1aW1RcILwKVLeI4+DRE0A6wF0zAm4/iUzSuqZ4S3jjqHwRxwH1tI3aQK0AT2K2yE+ZGk0/uP0EtN6nladH37uzg2rUN4++osaA7wa9ZYZQg7xbSqiHHBHPGC/ip4oj+oD9Ep0g5HqXHu3Qa2J6aPxvVAL32G5imk7XyX8G9uWUJDnWhiG6+1WoTpWLgywVZwL3t2li35r/Am4/gS/xAzi3h8pTucd4i+r5R6ViB7iXXOHRM4i8ZRPXm/jJT3sahUVtGqIr69wxzeyjfiuvEbi3j8KjWH+jXIBjummMDoE/BX1coq60GQNvULvKu4a/e5sD+3uEHuZ0QcqEBxOoM+D3O73X6dvdRQ2qVZsx8Qn4RA7M7y87nWQdeEomgeve2+lAIBAIBF40fwCbfJAgLGYREgAAAABJRU5ErkJggg==>

[image18]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABIAAAAXCAYAAAAGAx/kAAAA1klEQVR4XmNgGAWjgPqgHojF0MQ4oZgk8B+I3bGIrUUTwwu8GSCa0AFIzBhdEB84x4BpUBgWMYIApKEXi9gPNDEQ+AvEwuiCIAASBGmSQhMHibVD2dOBmBuI5aDi8kDMDJWDg0kMEEkVJLESqBg/EOsAcSgQc0HVgFwkiFCKAP8YIJruMUA0zgbiPqiYDxD/QigFi09A4qMAkIYeINYF4mggZkSSC0RigwDB8BFAl8ABYLEozoAWRjAvEAtAakEuPowu8YcBexTjAiBDeNEFQQAjCocHAADCMSnNlzWgjQAAAABJRU5ErkJggg==>

[image19]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAI4AAAAYCAYAAAAswsVWAAAEFUlEQVR4Xu2aR4gUQRSGnwEVc0QMuIonMSF6EBUWUVEEA4gBFV0MiAfBi3gSPYgHcwAvJlQMB0FREPEiiuEiiuGgF10Es5gDZuvfqtou/q2erllnem23P3jMzP+qusJUv3rVMyI5OTk5OYVpoawzi42EShYC6c9CY+QHC42IncoWspjAYGXLWSw1T5X9duy9slfKXjpa19rS6fNTWVMWDY+UfZGonx9F9x2vVltQWzq7vFE2gMUC4DstO5jcMywqror2TWNHiuxXdppFD+jnNRYVO0T7RrMjY3QSPY5QcPOXlQplD1hU7Bbd0Q3sSJmQyZonutxEdiiGivb5xpg1fimbwaKHRcrGsVhqziprRRpCOyb7POlpM1PCFs49iS+3XrTvFDsyyGZlX1n0gAVWdjjzHiZ6oqtJbwiQv1xm0QP6G7dwrK8ZO1JmFwuKbhKfu/noIfHjdLnFAtFedHSeTFZvcNxFx5Bs/gugL0tZ9IByV1hU7BPtG8GOlGkruh9tSIe2grQkkhYOTmA9WTRUSHQj+WxZVDQcrHx7gWIZouxwjB1SdlDZAdGJ7l5le3S1RNCXpL16luhyOK7j5PHOvId2winXkGySuvOKUyq0YiIOQJ1+LDpwO5Z2on0rHe2FlGBbs4umCenT6XOaJE0SKJTfMHzH+8CDRkSoEBtu6iSBxwkwF3v4KBbUmcSioYPom9UH6nGet8ToPhAlE/km+gJo2AWhaxRpaYI+9WWRsAs+hJByWFxTA22KqZME2t3i0XwpAaJloSfkqBfX7kXx53LIX3xjPy5+HesgMRI9Fl15EDvEf1EfiAobi7QQ0P5YFgmUeciiBzyLSkoay0EX0X3EcxgXaDjxMUlzDj9SAx9xdc+J3wftLYuKbaKff8WCFYrKc9gh+sv9wGLKoG+LWXSoEl1mPukMyriWJkhWuc1eRkPegfdIkJFvhPQRvuYsit46V7FoOCp1r2lzLJywXNw+PCNfDQidcGJgLiMlikKzyZc2T5RdYNHBTnYIoeVKDUI+2h7oaO7iuO7oW0Xf7XHY6OXjOQsOXM8ehOY6mktcGzVgf8V+agdmDZ+hf46KNhhV4h/Ea9GnJ7wi1GIsvnzBxXedNEC7N8wrzP4sUm0+u6ccJNCF8hs8xf/EogG/zRVijER9uC91t05LRwnIb7JAKb5wJLK3WUwBe6fzoSOOpLF+F/2rN4Mn/RNYrCeIerwLZZKTyo6xWCSXJNp2V7uOMrNdkheDBXmLLXvXdRhw2ou7Vin/coKo111Za9FpS6aJm7BQ+ojeGo6QXm7wJSBKhILt5g6LBvxNojeLhpss/AU46iPvWkd6JmkpyTnMvwjykfEs1oO1Ev9zwBrRESInBhwbQ3OF/41Cp9tKFnJycnKyzx9l6RomHbCQFwAAAABJRU5ErkJggg==>

[image20]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABEAAAAYCAYAAAAcYhYyAAAArUlEQVR4XmNgGAWjgDzAB8TuQOyFhokC8kD8Hw9ORyjFDngZIAoLkMReAfE/JD5BADJgA5pYClQcG+BBFwD5F5viFQzYxfkZsLhwBwN2xSCxD+iCQNAPxBPRBZcxYBoiAhUDxRQyQA7oF8gSwlBBGGCC8qOQxJABuoVwYMOAsOEmEAuiSsOBAAOW8CAV9AHxJHRBUsFfIBYHYi4gtkCTIxr4AvEZIG5AEx8FaAAANcwqmlANoRcAAAAASUVORK5CYII=>

[image21]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAFAAAAAYCAYAAABtGnqsAAACPklEQVR4Xu2Xz0vUQRjGn8KDICSYeNFUUIigQ4ZKKmGHzilKWZcuHaRT9g94K8HooJQ3sUudPASCUAShB38cI7qmBBFUePBHPyTS9/X1u84+O9+vu+5uazAfeFjmed6Zne/s7MwuEAgEAgGXL6IdR+ui76Jvjledqv4/qBetwub+XnQqLU2mWbQM6/uGMi9aOMOmsADLejg45vSLFp32CrJ/jm5YbcQFamfQIPrIpjAB6/iQgzw5ITrNZoHReV/3eIkLsY/W3CVvW7REXopZUTl5t2EDvSY/H6pEX0VvOSgCvsWKjqNz5LvUwGr01UXXgcdL0UTtFljxKvlH5SzsE3zOQRHRHXSTvN+w50o6C4fhX6gp+P0MdJdo4U8OjsAV2FiPyC8Vvl3JvIS/5in8fhonkd2bHMYt2Bj3OSgh0QLotyGJefiffwzm13LgEi2eHvIuvdSOYwjWf4CDPGjNQXHohaXzaufAwwv4F/AJzC/jIELPKC2oJH9Q1EleHA9gY1zmIA+u5SAfuhl0Tmc4iCHuDJyE39/jMyw8zwESOiUQ7cQbHJQAnUeF076DzEvTpQvWJ+tbeA4W8I2ljIo22MwB/R2mY9/j4B/xC5lfuQ/UbhNdJE/n3EfepmiNPDyGFY+TfwkHu7IQ51mH6K9ohIMi8gk2f59cfN4r0R+nHR0DjY63h/5U0UJ9OPcNtK3+j4PSgtAIG/NZul0UeNHiFlBvZxXzTrQlmob1uZoelxb911PHZiAQCAQCgcDh7AJgbpxYFWFuFQAAAABJRU5ErkJggg==>

[image22]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAA8AAAAXCAYAAADUUxW8AAAArklEQVR4XmNgGAUg8ByI/yPhT0D8BohfI4mJwFWjAZDkZnRBIDjGAJHzR5eAAXkgvocuCATTGCAa29AlkME2IOZAE4tjgGjchSaOAZTR+IYMEI0P0MQJAiEGiMbv6BKEABMDImRJBjCNjGjiAWh8FKDPANH0GF0CCI4zQFyEFQgyQDT+RJdggLgApxdgktgUcDFAxFehS8AATOMdIL4FxHeB+BmSOAjzwFWPgqEGAPfgLRK7egCXAAAAAElFTkSuQmCC>

[image23]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAoAAAAWCAYAAAD5Jg1dAAAAqklEQVR4XmNgGFqAG4h3AfF/ID4NxIyo0hAgzQBRwAnlC0P5THAVUPAViFeiiZ0B4h9oYmDdYWhiVVBxOLCDCtggCwJBPFRcCCZQABUwgglAQShU3Bwm0AQV0IMJQEEgVDwaJhAEFTCGCUAByM0gcX2YgBZUwBkmAAVpUHF2ZEGQQA6yABBMgIqjgO9AfA1N7B0Q30cTY2BhQLU+CsrHCUDWnQDiYHSJYQUA4z4lJYXNoC8AAAAASUVORK5CYII=>

[image24]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAACgAAAAXCAYAAAB50g0VAAABY0lEQVR4Xu2VvUoEMRSFr+AWgi6sNoKID2AnyCrY+Qx29r6AFlY2Fja2Wmxlqyz4AtaChaDY2YhgYaGojfjvPWRkw8nNJmMlkg8O7Hy5M7lJZndFCoW/y4BmiWWELc2b5l2zR2MptjWvmjvNAo31ZUizytLgUTNTfR7TfFXJAfduetfP4habxbBmjSXR1txqRjw3K67BU89ZLEq4kFHDRcGkqQaxWjzwgnzOLuJYrRq4ZZYWOQ2CA02TXE6DGMd7y8Cfs7TIbZCZFzcJGu8Hap5YivN4FwNZJ5PuNhOMf7I0QN09S+nNkeQ3O9jVfLCMgCYeWEr+Ams3uCL2hDHQyAtLcf6SpUWdBuc0V+RSxxQ7Srhdlha5DY5rzlhKOPkGXe9IWIN/L7gGeZOcBgeltxOcY6/usHL7ngNw0971idjf7ODhqUy426RjjP1kvaoBLc2NhL+XU+JqjzTX4moKhcK/4huU7XYDk0YVRwAAAABJRU5ErkJggg==>

[image25]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAACgAAAAXCAYAAAB50g0VAAABdklEQVR4Xu2VTStFURSGl4GBQqFMJD9AJgZKoWTgF5j5Dya4QxMDKWXEVCbKyB9QysyQkczEkBAJ+Vivde6173vWPWc7I2k/9dbdz17n7HX2+bgiicTfpU0zx7IFy5oPzadmp3mqlA3Nq+ZGM0FzhXRoFlk6nGsmg/GTWKMx3GtWg/GzZi0YF9KpWWLpgGbChhay8V7gPGYkfyG9jmtJl8Q3iF2rU8scbl0RuK1eM3DzLD1iG2Rw27AInuEiUPPGUsyfsvSo0uC62AIjPOGAugeWYh7PYk7+JoN2WIN+zabmWHOl6WuedsF5blnKzxqlVNlBgLcSC0zxBIGaO5ZiHp+sUqo2iGcvZhcw/8JSzF+w9IhpcFzshGPkYxv0auC2WXrENHgmdsIj8t7iKzTeknxNfffbybvENDgq+U/Frtgis4E7yNx+4ADccDA+Ef/NblxxbAbssG/wfw33qHnPfk8H86BHc63pJj8kVn+ouRSrSSQS/4ovld90QTK/zm4AAAAASUVORK5CYII=>

[image26]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAEIAAAAYCAYAAABOQSt5AAACy0lEQVR4Xu2W26tNURTGR+4ipBQpCilye/DgQR7kwSUibx4RT+SB5JIjyoNr8kLCKTxQ/AkevJByKUqJnKSUcilyv33fmXPsNdZYc++z1zknWVq/+tprfmOsuceea8y5l0hNTU1JpkNnoJXG226uq8JgaJY324E3/oLOQqOgRdBvaC/0weRVAdbL2qnS8KaF3pTg7/ZmBWDdpR9gpzRfPfrslqrBund5sydatVEz/19mpoS6h/hAT+hCHPMBx0Fokxnvgy5CA403Djon4Zwhy6DTks9RWOge6CQ0xsUU3s/5JvuAg9v6CjQfuiq9fIBHJFsMFYu3nIKmQN+gW9BnaAQ0TUL+0Jj3RsJi0HsBLZBQpC/sJfQEGh3HjPstSG9bvL4D/YBeZeEGNu9+HH/KwuXYKsXFYKGK/hD+s/gfxfFtqAOaAM2I3moT533KVykW+lDyXcF7npmxejsTXiqP3dpnlki2GApblNDz3ULvO7Qljs9HTxlgrjdIiI2P44nQDehmI0PkhBQXe2r0bNdwq/o8vgfR0w5tm7XeiHDv+y/hVrA/QqF3yY25dVLoAnOr8R1lRT7cjX8I5ELCS+VdTniE22qsN5VVku0tzw4pTngo4W2M3hzjcdxhxpZU8R7G/VlA72PCS+V9cR5p+Z13oevejPyU4hZg+/sJmcc9r/CMYM4w41mY6+dQhsdPxvlvYqG3P153GS+VdyBed0Gvo9fyAWhwpPOvSfqtzE/GIvzkbHnvWeZJMT5JQusqb6GnZsxO4D1zoXUS5iA+j/9ozJst+bzjEs6dpvAvjAfZewkTvIufnSZH4WIxpocWxQXzPIYeeNOxXLI5eJasz4e7Ycszrq1/NI7vNTIC7eSxa5ueD2U5LMUnWRX6tW6uar9O+JcYJFndj2ygN/AlhpM9hza7WBXgGcMXtj7BF5Ol0GIJe3tNPlxTU/Mf8ge/ENbxF9PKIgAAAABJRU5ErkJggg==>

[image27]: <data:image/png;base64,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>

[image28]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAFEAAAAYCAYAAACC2BGSAAACG0lEQVR4Xu2XTUgVURiGP8t+XLSLlroxcxnlom24CPyrFGrjwop00T4KglwJBhKuXCiKuiuqhQq1rl0ogqAFIZKKm8BNoNHv99755t7jO+deHEFzvOeBl3u/55u5nDkzd86MSCAQONwcZxFIxyvNMXJVmovkAiX4S/WmOfaBIlzQPGEp0QS+Zxnws8LCwCQ2sgz4+cFCuScp/spPNefI4YaKZJW7mueaCm54uK25yVJZlWgSKzUvNV07ugQ2vOZxWK2yRp1EY6+x+o/VPfktkvxkYcSLSr/V7zTfCu0CzeK/ZOEusyTOaCaLZEIzrhnTjGpGNMPRbvvGaYnGfd9xreZKPf99YWFgv0dOXW0uwZwkG7c87iBpSBGXLUmOGyeUnUufppal0inJ/a6aqyefk7PkPpn/X7SliAvGvOFx38m5FDvOz5Lsvfa4HJA3PO4DOR+nNM9SZr+4ItG4+d4H95hcDBbONywN7DfvcYlJbDd5gjxck33PyuJyVpIHiL8pHE72ec2dnW2Ztp6Pbc2QU5+U6LcuOS7HgjUeOO63ObxDDki2HnMwbtzLQLzIxBO7aJ8uPOkug5p1p8Yq/9ap8+BHvtonsmYeG6N+YXVWwAn/JdHYZ8xhIlB3xxsZeBTqJcdMSWFuOqiXB80WlmXCMou9gMWk1OV81PG95qXmo5TvJOIF4zrLvbCkeciyTCj13BjYJeW6DgSOJP8APlmJyd3kku0AAAAASUVORK5CYII=>

[image29]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAH8AAAAYCAYAAADTTCLxAAADMklEQVR4Xu2Z28sNURjGX2eS5FAkkj5KuSAkSdyQIuQ/EBI3wh3KDVFciogIuXOFJBcO925IUcQupyRyyPn4Pq01317fM2tm9uy9v9l7af3qqT2/tWbPrHn3zKyZLRKJRCKRbGazCIDBmrEsi9is2c4yQMZrZrJsgrmaqyy7iCGaWSwt7zWjWPoYrvmrmaBZrPnetzkYlogZB3KD2sqCA/uLpQPOrh8sK+SN1MeaRV5bL+i0iJaHOcuh0Y7i/9bMY6k8lPpBb+jg9iPYft6JekBzh6XLFc1PcvjS5eRCotXij5Tiwn6U4j5FfGZREmx/H0sCfQawBOPENK4jD7eTXEi0Wvz7mmssiU4Xv0fM9kdwA4Er2H6WAJMZHsB061aQDwku/noxl8CLdhmTpJOaHUkHAusvY0l0uvjnpb59fN7ttLkcl4y5C1bGL2OgXcbl4Zj1IcPFxy8fDqmJmQVjrFj+VO/WSyPj73Txk/FcsssHxdSSwVwutZ+YrUI+0dxyknxpHjh4+LX5ck5zVnNGc1pzSsxZViVcfPDcejzZJODM57EO8jgf3VB8zNfYzSE30fo+bLByKnm4B+SqYn6J5OErfs16l60eN83jfJQpPgrC+4989ThktFktk8lito2JacJQ67Y4LiG1n0c9EhM/uEnkq2JNieSBMdwk99h6l00eh5OBnY8yxV8p6f1HvnkcUnT8T0h629usG0MecF857JEfNO/I+cA7gEMlUyW+4j+y3mWjxwGfY8oUP4tmL/svJb1t1I4dwIu7lF9FEu+CU50CBeO4Ta5mvYvvzAc+x3Sy+Lil8a0Z+4Kzn1koGfvpSnxe6yyHDMZyl1zyOtQFj0dweJXrAreUHPNF0t9XlmaLn9yeEzC2p86yyxHJ2E8UGw1/NDOoLUTwbP5a80zMpRGXQvBWzGwfHu3oh0e8F9a90ly3fUFNc9lZdsF66I/1EHwHbpU9bqcGabb4YK+Y2iG7qM0Fb2/xGBhpkAWScba0mVaK3yhVjOO/A1fDdvwtnAfeKfQnezT3WEaKmSLmUSxk4lnfAqs1F1gGAuYlRS+LIgUU/cHTjeCfPly5IpFIxPIPuZH4nbPdNfoAAAAASUVORK5CYII=>

[image30]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAADQAAAAYCAYAAAC1Ft6mAAABbElEQVR4Xu2WTSsFYRTHDwslL7GwEh9AUT6A3GQh2clXkOVNrG0spLC0sGRlZeMDWFgQa9l6i5RYKQv8z33OmPMcY+bObe6gnl/9ms7/OfNMp5l7Z4gCgUCzaIFzNszgzAYZXMEZ2AH74AJ88zoKpB0u2TCBE/ihzIM+L7LN6yiQTrhswxQWqbGBNuEWHDZrhdNF5QxUGn9xoH44BaeNdVHWQM/wGF7CF3/5iwn6/lvTDsWtDtuQ5YA7zaPRgTRPCdmoZJMqe4fnqq6LMu6Qhf+2eY8xlXFdVTWzJ3kufmOgCrk9tqVel9pyT8l5Ks0eiB8b288vWc5WpX4l12fhnkMbZlH0QCum5t5bk61JPij1HTyNl2tUKP06P5J3oOjx4E8YywG5tX2VzUqu4Z4LVY9LFtEj9YjKEuGmPPL7IIK/vfiZvoHXcnyEu6qnl9zd6FYZs0Fuvwc57vjLNeYpvu4RbPWXA4FAIPCP+QRpO36pLg7VgAAAAABJRU5ErkJggg==>

[image31]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAACgAAAAYCAYAAACIhL/AAAAA5klEQVR4Xu2TPQoCMRBGRxG8iK3YWHoCQS9hLYh4DI9iJVjZaiMW3kEFBbX0p9IZkoUwZLOZ1WkkDx4k82XhYzcLkEgkXM58wLmjGz5UZoe+HYPQgREfRrLgAyFzKCjYAnOgyoNI1Ap20C66BHOgZ/dS1AqO0QmY8Gr3pBS1ghkUDvlQgGrBJpiwwgMP7RzXnhkZe6eDBWcQCBn9HLeeGVk3jxUSLEjBjQ+FqH5iCuhHyVg561jUCzbs+uUGAr4tSC+FenivxBRM+ERrLIulbMEHekIP6B49ohd04B76BWULJhKJv+YDpPxBzc42UAMAAAAASUVORK5CYII=>

[image32]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAACsAAAAYCAYAAABjswTDAAABXElEQVR4Xu2VvS4FQRTHDxEUEqGh0Ah6DRIvgEhEqdHcWsdD0HkIEuIFJBri6wkUvloKHy0Rwf/cOXsze3b2mp25objzS37F+Z/M3rOze2eJEolELDX4CD/hpuoFMw6/qIUXBAfwwap34KtVRzMK3+C+bgTwDbsc2bzKohmAz/BMNzzZJjOYhrM7HWZ0k7mTOd3whNffkvmBHtVrxgeVD1vIJyS8hBtwNd8O4gI+wX7dcOAcihz5igRDdtgCeGfv4ZVuOCgMJRRyLq7JvNyZMQzCF3is8mYUhhJy+ZQU/PgPLUMYg+9wVzc84HP112HXpehstKszS+YaW7pRgSMqH5bP8zqTElT552YskVm7phsBDFP5sLyhDfgRnFs17/KIVbvg421Zh5HwDu5Z9SK5b6D+nmbvxwnsyLf/DD5vb+ApmVn68u02oJfMY/JxQdb8G/xZnfF0WtYkEm3FD0GFVr4jDrrRAAAAAElFTkSuQmCC>

[image33]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABcAAAAYCAYAAAARfGZ1AAAAhElEQVR4XmNgGAWjYBSQCiSA2BOIldElKAENQPwfiJcDcTEQq6HIUgAOMkAMxgmEgdiESKwO1QMCcgwQg2OBmAWKmZHkwUAeiP2IxLZQPSCwlQFi+A4kPAdJniLwmYFAkFACNjDQ0HAQABkeiMSXQWJTDNiB+CUDxBIQnooqPQpGwYADAJVLGTtX0+r4AAAAAElFTkSuQmCC>

[image34]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAACgAAAAXCAYAAAB50g0VAAABE0lEQVR4XmNgGAWjYHiBM+gCBMAjIPYBYm4gFgXiDCD+haKCCuAkEP9HwqQAZH0wzIaiAg9gBeJSdEE8oIiBPAf2AXE/EOuiyREEHAz0cSDZgIdhkDuQl4E+DvwAxEeB+CYQf0SVxg/o5UBk8A6LGBiYYMH2QDwJizgIYwPkOBAdgIoZkBm26BJ+WHA4EC/AIg7C2AA1HOjAADFjOpo4VkDrKP7HgKkeVGiDxFrQxLECajuwHo0PUvsUTawDKi6HJo4VkOrALgaI4aAqCx1sYIDIrUISC4aKIwOQmutoYjgBsQ4E1Z0vgPgJED+G0q+BeDGSGkEGSGjxIYmBQC8DxFEvofRsVGn8gFgHjoJRMAoGEgAAiqROHA49mHYAAAAASUVORK5CYII=>

[image35]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAACgAAAAXCAYAAAB50g0VAAABYUlEQVR4Xu2UvytGURzGH0UZUFhs/gCbCKUMdqvF7i8wmCwGi8XAYGEl5R8wK4Mim0XKYCAs8tv36Vw5Ped73ENK6X7qqfd+7vd9z3Pvue8FGhr+B0uWJ8uzZVPO1bFsebRcWcbl3K9waxmsPvda3qqUwO8uRsf3CBdbRJtlTqUwYrm0dEZuGKHgYeQ8JpFeSI/jsrSjviCvlj94Ir7kLnJbvRm6GZUeHagvSLYtXeJKCvI8n1uF/lilB7etpKAyhrAIi38FZ+5UIng+i7X8tCAXeFXpwLlrlcjc/SEnE5YVxzM5diwvKjOwxI1KZC5wysm0ZcPxjMcs/AVzsMiDSgR/qtLjO1s8ajkTl2yT4G4lgltT6VFasM9ypBLp4gtyvIp0pqVyfAfXUlKwFZ93QrMfze1WbityhG4gOj6A/892KSm4jrTYR+ajuW7LBdL3ZT/C7J7lHGGmmJKCDQ0Nf807Cf1kYcb5Ha4AAAAASUVORK5CYII=>