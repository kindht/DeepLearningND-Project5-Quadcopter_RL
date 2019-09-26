# DeepRL 四轴飞行器控制器

# 回顾
## 问题 1：请描述你在 task.py 中指定的任务。你如何设计奖励函数？

回答：指定了起飞任务, 奖励设计:

1. 当智能体z坐标高于目标z坐标时给予奖励 
2. 智能体z方向的速度给予奖励
## 问题 2：请简要描述你的智能体，你可以参考以下问题：

* 你尝试了哪些学习算法？哪个效果最好？
* 你最终选择了哪些超参数（比如 α，γ，ϵ 等）？
* 你使用了什么样的神经网络结构（如果有的话）？请说明层数、大小和激活函数等信息。  

回答：

- 使用DDPG算法，因为该算法更适用于解决连续状态和动作空间的问题
- 超参数参考DDPG论文, 修改：模型学习率，折扣gamma，soft update的tao值, batchsize和replaybuffer size, OU noise的sigma参数
- Actor和Critic 参考项目示例，全连接神经网络，分别有2个隐藏层，参考论文使用400/300的隐藏单元（Critic 中包含两个子网络，结构相同）
- 激活函数全部用ReLU； Actor的输出层激活保留Sigmoid, 没有使用论文中的tanh

## 问题 3：根据你绘制的奖励图，描述智能体的学习状况。

* 学习该任务是简单还是困难？
* 该学习曲线中是否存在循序渐进或急速上升的部分？
* 该智能体的最终性能有多好？（比如最后十个阶段的平均奖励值）

回答:  
- 前期项目理解和建立模型感觉比较困难，设计合适的奖励函数后，起飞任务即可完成，任务本身并不复杂
- 该学习曲线不存在循序渐进，而是在后期急速上升（虽然偶有比较低的分数)
- 600个阶段之后，平均奖励值即可基本稳定在高位 (-1.6 左右）

## 问题 4：请简要总结你的本次项目经历。你可以参考以下问题：

* 本次项目中最困难的部分是什么？（例如开始项目、运行 ROS、绘制、特定的任务等。）
* 关于四轴飞行器和你的智能体的行为，你是否有一些有趣的发现？

回答  
本次项目，整体上算法的理解和调参都感觉有些困难，虽然实际修改代码的量并不多：

1. 刚开始对于项目的理解有些困难，通过反复查看示例代码，慢慢梳理出逻辑  
2. 理解整体逻辑后，建立智能体（in DDPG_Agent.py)和DDPG_model(行动者/评论者模型）以及 helper.py, 修改Agent代码，先跑通训练过程，绘制奖励图，然后逐句理解代码的作用，再尝试设计任务  
3. 原以为需要自定义Actor模型的损失函数，多花了些时间才理解  
4. 尝试修改reward函数, 默认任务，即，指定飞行目标
5. 确定飞行目标任务后，进行模型和算法调参，参考deepmind的论文的超参数设定
6. 鉴于之前项目使用dropout很难拟合的教训，考虑到本项目模型很简单（仅2层），没有应用正则，也没有batch normal（BN用于比较深层的网络）
7. 初步调参后，尝试训练500次, 相比之前，无明显进步，运行了1000次，效果更差，飞行轨迹非常曲折，奖励值也没有上升趋势...
8. 再重新修改reward函数, 在step()中设置了对z方向的奖励，起飞任务成功！  

有趣的发现：

1. 飞行器起初飞不起来，观察数据图形，x,y变化都比较小，只有z迅速变小，于是考虑到需要在reward和step中重点奖励z坐标高度
2. 强化学习的算法，数学建模和代码实现，有些难度，但其核心理念却很好理解，智能体的行为就像人的学习过程，初期懵懵懂懂，随机探索环境，采取行动，再根据环境的反馈，调整自己的行为，逐步地适应环境。行动者/评论者模型，把这个过程模拟得淋漓尽致，钦佩最初设计者们的巧思
3. 算法中最关键的是反馈，我们如何设计奖励和惩罚，决定了智能体接下来的行动 （损失函数的设计也很关键）