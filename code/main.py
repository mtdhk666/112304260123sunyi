import pandas as pd
import numpy as np
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

from preprocessing import TextPreprocessor
from bow_model import BOWModel
from word2vec_model import Word2VecModel

tqdm.pandas()

def load_data():
    print("正在加载数据...")
    train = pd.read_csv('../labeledTrainData.tsv/labeledTrainData.tsv', sep='\t', quoting=3)
    test = pd.read_csv('../testData.tsv/testData.tsv', sep='\t', quoting=3)
    unlabeled = pd.read_csv('../unlabeledTrainData.tsv/unlabeledTrainData.tsv', sep='\t', quoting=3)
    print(f"训练集大小: {train.shape}")
    print(f"测试集大小: {test.shape}")
    print(f"无标签集大小: {unlabeled.shape}")
    return train, test, unlabeled

def preprocess_data(train, test, unlabeled):
    print("\n正在预处理数据...")
    preprocessor = TextPreprocessor(method='stem')
    
    print("处理训练集...")
    train['clean_review'] = train['review'].progress_apply(preprocessor.clean_text)
    train['tokens'] = train['review'].progress_apply(preprocessor.tokenize)
    
    print("处理测试集...")
    test['clean_review'] = test['review'].progress_apply(preprocessor.clean_text)
    test['tokens'] = test['review'].progress_apply(preprocessor.tokenize)
    
    print("处理无标签集...")
    unlabeled['clean_review'] = unlabeled['review'].progress_apply(preprocessor.clean_text)
    unlabeled['tokens'] = unlabeled['review'].progress_apply(preprocessor.tokenize)
    
    return train, test, unlabeled

def train_bow_model(train):
    print("\n=== 训练 Bag of Words 模型 ===")
    bow_model = BOWModel(vectorizer_type='tfidf', max_features=5000, ngram_range=(1, 2))
    
    print("训练模型...")
    bow_model.fit(train['clean_review'], train['sentiment'])
    
    cv_scores = bow_model.cross_validate(train['clean_review'], train['sentiment'], cv=5)
    print(f"交叉验证 AUC 分数: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")
    
    bow_model.save()
    print("Bag of Words 模型已保存")
    return bow_model

def train_word2vec_model(train, unlabeled):
    print("\n=== 训练 Word2Vec 模型 ===")
    all_sentences = list(train['tokens']) + list(unlabeled['tokens'])
    
    w2v_model = Word2VecModel(
        vector_size=300, 
        window=5, 
        min_count=10, 
        workers=4,
        sg=1,
        epochs=10,
        classifier='lr'
    )
    
    print("训练 Word2Vec 词向量...")
    w2v_model.train_word2vec(all_sentences)
    
    print("训练分类器...")
    w2v_model.fit(train['tokens'], train['sentiment'], train_word2vec=False)
    
    train_auc = w2v_model.evaluate(train['tokens'], train['sentiment'])
    print(f"训练集 AUC: {train_auc:.4f}")
    
    w2v_model.save()
    print("Word2Vec 模型已保存")
    return w2v_model

def generate_submission(model, test, model_type='bow', filename='submission.csv'):
    print(f"\n生成 {model_type} 提交文件...")
    
    if model_type == 'bow':
        y_pred_proba = model.predict_proba(test['clean_review'])
    else:
        y_pred_proba = model.predict_proba(test['tokens'])
    
    submission = pd.DataFrame({'id': test['id'], 'sentiment': y_pred_proba})
    output_path = f'../results/{filename}'
    submission.to_csv(output_path, index=False, quoting=3)
    print(f"提交文件已保存为 {output_path}")
    return submission

def main():
    print("="*50)
    print("电影评论情感分析 - Bag of Words + Word2Vec")
    print("="*50)
    
    train, test, unlabeled = load_data()
    train, test, unlabeled = preprocess_data(train, test, unlabeled)
    
    bow_model = train_bow_model(train)
    submission_bow = generate_submission(bow_model, test, 'bow', 'submission_bow.csv')
    
    w2v_model = train_word2vec_model(train, unlabeled)
    submission_w2v = generate_submission(w2v_model, test, 'word2vec', 'submission_word2vec.csv')
    
    print("\n" + "="*50)
    print("实验完成！")
    print("="*50)
    print("\n生成的文件:")
    print("- submission_bow.csv (Bag of Words 模型预测结果)")
    print("- submission_word2vec.csv (Word2Vec 模型预测结果)")
    print("\n请将这两个文件提交到 Kaggle 平台进行评分！")

if __name__ == "__main__":
    main()
