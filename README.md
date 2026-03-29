# agentwithclipmind
我的一个思路，我们去测试一下
# 第一次实验
这个数据集太烂了，会倾向于不去预测变化的rebuild之类的
问题 1：No_Change 样本占主导
大量样本的 object_labels, action_labels, location_labels 都是空数组 []
这意味着在 multi-hot 编码后，这些标签全是 全零向量
模型学会了"不预测任何东西"来最小化 location 损失
问题 2：Rebuild 动作样本极少
rebuild 只有 355 个正样本
模型很难学会这个动作，会倾向于预测其他更常见的动作
问题 3：Location 标签在 No_Change 样本中全为空
No_Change 样本没有 location 信息（空数组）
这导致模型在大量样本上学习"location = 全零"
当遇到有变化的样本时，模型很难预测 location
