# 机器学习实验：基于 Word2Vec 的情感预测

## 实验信息
- **实验名称**：Bag of Words Meets Bags of Popcorn
- **竞赛链接**：https://www.kaggle.com/competitions/word2vec-nlp-tutorial/overview
- **学生姓名**：孙意
- **学生学号**：112304260123
- **班级**：数据1231
- **实验日期**：2026-04-15

## 仓库结构
```
├── code/                 # 代码文件
│   ├── preprocessing.py  # 文本预处理模块
│   ├── bow_model.py      # Bag of Words 模型
│   ├── word2vec_model.py # Word2Vec 模型
│   ├── main.py           # 主程序
│   └── requirements.txt  # 依赖包
├── report/               # 实验报告
│   ├── 实验报告.md        # 完整实验报告
│   └── readme_机器学习实验2模板.md # 实验模板
├── results/              # 实验结果
│   ├── submission_bow.csv      # Bag of Words 模型预测结果
│   └── submission_word2vec.csv # Word2Vec 模型预测结果
├── labeledTrainData.tsv/ # 训练集数据
├── testData.tsv/         # 测试集数据
├── unlabeledTrainData.tsv/ # 无标签数据
└── README.md             # 本文件
```

## 实验简介
本实验基于 Kaggle 竞赛 "Bag of Words Meets Bags of Popcorn"，实现了电影评论情感分析任务。使用了两种模型：

1. **Bag of Words (TF-IDF) 模型**
   - 交叉验证 AUC：0.9523
   - 特征：TF-IDF 向量化，5000 维特征

2. **Word2Vec 模型**
   - 训练集 AUC：0.9505
   - 特征：300 维 Word2Vec 词向量

## 环境要求
- Python 3.7+
- 依赖包：见 code/requirements.txt

## 使用方法

### 1. 安装依赖
```bash
pip install -r code/requirements.txt
```

### 2. 运行实验
```bash
python code/main.py
```

### 3. 提交结果
将 results/ 目录下的 CSV 文件提交到 Kaggle 竞赛页面。

## 实验结果
- **Bag of Words 模型**：AUC = 0.9523
- **Word2Vec 模型**：AUC = 0.9505

## 注意事项
- 数据文件较大，已按原始结构存放
- 模型文件（.pkl 和 .model）未上传到仓库
- 每次实验后请及时提交代码和报告到 GitHub
