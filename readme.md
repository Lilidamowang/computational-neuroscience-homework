# 系统与计算神经科学大作业
2022年12月，系统与计算神经科学大作业代码实现。
## 基本信息
选择的baseline模型有两个：RIMs以及Mucko。其中RIMs为对大脑功能分区进行模拟的脑启发深度学习模型，RIMs模型就是一个模块化的结构，它由多组循环神经元组成。每组神经元能够独立地处理一些转换动态，并且在特定的时间与其他神经元发生交流。每个子系统都只针对一部分问题，在其他子系统状态发生变化时，自身在不相关的情况下也能保持鲁棒性，不受其他子系统的影响。这种设计方法能够提高模型的鲁棒性和泛化能力。

Mucko是一个多模态处理模型，主要用于视觉问答任务。在本仓库中，引入Mucko所创建的多模态记忆图来模仿人类获取的知识和记忆，联合RIMs一起处理，用于探究RIMs等类脑模型在处理现实任务时的表现。

代码的引用如下：
* mucko: https://github.com/jlian2/mucko
* RIMs: https://github.com/dido1998/Recurrent-Independent-Mechanisms

## 代码介绍
代码共包含3个主要部分：数据集构建、模型、训练
* 数据集构建：
    dataset包下为ok-vqa数据集的读取及输入的python实现
* 模型：
    所有的模型均在model包下，其中RIMs模型对应RIM.py文件。model_baseline.py为Mucko的模型实现。model下的其他文件为构建模型时所需要的基础模型。
* 训练：
    训练及实验部分在main.py中，通过运行main.py复现论文中的实验。