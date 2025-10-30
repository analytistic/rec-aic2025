# itemCF.py
# 基于ItemCF的图书推荐脚本，用于全球校园人工智能算法大赛赛题
# 修改版：适配样本数据，融入user/item信息，处理时间/空值
# 运行：python itemCF.py
# 依赖：pip install pandas numpy tqdm scipy implicit scikit-learn

import warnings

warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np
from tqdm import tqdm
from collections import defaultdict
import scipy.sparse as sparse
import implicit
from datetime import datetime
import zipfile
from sklearn.preprocessing import OneHotEncoder


# 自定义时间解析函数（处理不标准格式和空值）
def parse_time(time_str):
    if pd.isnull(time_str) or time_str == '':
        return np.nan
    # 尝试多种格式
    formats = ['%Y/%m/%d %H:%M', '%Y-%m-%d%H:%M:%S', '%Y-%m-%d %H:%M:%S', '%Y-%m-%d%H:%M']
    for fmt in formats:
        try:
            return datetime.strptime(time_str.replace(' ', ''), fmt).timestamp()
        except ValueError:
            pass
    print(f"警告: 无法解析时间 '{time_str}'，返回NaN")
    return np.nan


# 加载数据
def load_data(inter_path='inter_preliminary.csv', user_path='user.csv', item_path='item.csv'):
    # 加载交互数据
    inter_df = pd.read_csv(inter_path)
    inter_df.rename(columns={'借阅时间': 'borrow_time'}, inplace=True)  # 重命名以匹配脚本
    inter_df['ts'] = inter_df['borrow_time'].apply(parse_time)  # 解析时间
    inter_df['ts'].fillna(inter_df['ts'].median(), inplace=True)  # 填充缺失时间（鲁棒性）
    inter_df = inter_df.sort_values(by=['user_id', 'ts'])  # 按用户和时间排序

    # 处理续借次数（作为权重）
    inter_df['续借次数'].fillna(0, inplace=True)

    # 加载用户数据（可选融合）
    user_df = pd.read_csv(user_path)
    user_df.rename(columns={'借阅人': 'user_id'}, inplace=True)

    # 加载图书数据（可选融合）
    item_df = pd.read_csv(item_path)

    return inter_df, user_df, item_df


# 计算离线指标（P/R/F1近似）
def get_metrics(recom_dict, val_df):
    val_books = val_df.groupby('user_id', as_index=False)['book_id'].agg(set)
    tp, fp, fn = 0, 0, 0
    for _, row in tqdm(val_books.iterrows(), total=len(val_books)):
        user = row['user_id']
        true_books = row['book_id']
        pred_books = recom_dict.get(user, set())
        tp += len(pred_books & true_books)
        fp += len(pred_books - true_books)
        fn += len(true_books - pred_books)
    p = tp / (tp + fp) if (tp + fp) > 0 else 0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0
    return p, r, f1


if __name__ == "__main__":
    print("开始加载和处理数据...")
    inter_df, user_df, item_df = load_data()

    # 时间拆分：假设最后一天为验证集（模拟）
    inter_df['datetime'] = inter_df['ts'].apply(lambda x: datetime.fromtimestamp(x))
    inter_df['day'] = inter_df['datetime'].dt.day
    df_val_offline = inter_df[inter_df['day'] == inter_df['day'].max()].reset_index(drop=True)
    df_train_offline = inter_df[inter_df['day'] < inter_df['day'].max()].reset_index(drop=True)

    print("构建 user2books 字典（添加续借权重）...")
    df_train_offline = df_train_offline[['user_id', 'book_id', '续借次数']]
    user_books = df_train_offline.groupby('user_id', as_index=False).agg({'book_id': list, '续借次数': list})
    user2books_dict = {}
    for _, row in user_books.iterrows():
        user2books_dict[row['user_id']] = list(zip(row['book_id'], row['续借次数']))  # (book, renew_count)

    print("计算 book_Sim（Item相似度，添加续借权重）...")
    book_Sim = defaultdict(lambda: defaultdict(float))
    for user, books in tqdm(user2books_dict.items()):
        for i, (item, renew_i) in enumerate(books):
            for j, (related_item, renew_j) in enumerate(books):
                if related_item == item:
                    continue
                # 添加续借权重和时间衰减（越近越重要）
                weight = (renew_i + renew_j + 1) * np.exp(abs(i - j) * -0.1)  # 简单衰减
                book_Sim[item][related_item] += weight

    print("计算热门图书（Top-100，融合item分类）...")
    sta = df_train_offline.groupby('book_id', as_index=False)['user_id'].count()
    sta.columns = ['book_id', 'count']
    sta = sta.sort_values('count', ascending=False).reset_index(drop=True)
    top100_book = list(sta['book_id'][:100])

    # 融合item分类（可选：添加分类相似作为bonus）
    item_df['一级分类'] = item_df['一级分类'].fillna('未知')
    class_sim = defaultdict(float)  # 简单示例：相同一级分类加分
    for book1 in top100_book:
        class1 = item_df[item_df['book_id'] == book1]['一级分类'].values[0]
        for book2 in top100_book:
            class2 = item_df[item_df['book_id'] == book2]['一级分类'].values[0]
            if class1 == class2 and book1 != book2:
                class_sim[(book1, book2)] = 1.0  # 可扩展为更复杂相似

    print("生成推荐（融合user DEPT）...")
    # One-hot DEPT（用户特征融合）
    ohe = OneHotEncoder(sparse_output=False)
    dept_onehot = ohe.fit_transform(user_df[['DEPT']].fillna('未知'))
    user_dept_dict = dict(zip(user_df['user_id'], dept_onehot))

    target_users = df_val_offline['user_id'].unique()  # 模拟测试用户

    top_k = 100
    recom_list_all = []
    for each_user in tqdm(target_users):
        user_dept = user_dept_dict.get(each_user, np.zeros(dept_onehot.shape[1]))  # 用户DEPT向量
        if each_user in user2books_dict:
            books_tmp = user2books_dict[each_user]
            recom_tmp = defaultdict(float)
            for each_book, _ in books_tmp:
                dict_tmp = book_Sim[each_book]
                candidate_books = sorted(dict_tmp.items(), key=lambda x: x[1], reverse=True)[:top_k]
                for element in candidate_books:
                    bonus = class_sim.get((each_book, element[0]), 0)  # item分类bonus
                    recom_tmp[element[0]] += element[1] + bonus
            recom_list = sorted(recom_tmp.items(), key=lambda x: x[1], reverse=True)[:1]
            recom_books_list = [[each_user, x[0]] for x in recom_list]
            if not recom_list:
                recom_books_list = [[each_user, top100_book[0]]]
            recom_list_all.extend(recom_books_list)
        else:
            recom_books_list = [[each_user, top100_book[0]]]
            recom_list_all.extend(recom_books_list)

    print("生成提交文件...")
    recom_df = pd.DataFrame(recom_list_all, columns=['user_id', 'book_id'])
    recom_df.to_csv('submission.csv', index=False, header=True, encoding='utf-8')

    with zipfile.ZipFile('submission.zip', 'w') as zipf:
        zipf.write('submission.csv')
    print('submission.zip 已生成！')

    print("计算离线评估...")
    recom_df_eval = recom_df.groupby('user_id', as_index=False)['book_id'].agg(set)
    recom_dict = dict(zip(recom_df_eval['user_id'], recom_df_eval['book_id']))
    p, r, f1 = get_metrics(recom_dict, df_val_offline)
    print(f'Precision: {p}, Recall: {r}, F1: {f1}')

    # 可选：运行BPR矩阵分解（融合到recom_tmp中）
    print("可选：运行BPR矩阵分解...")
    df_train_offline['user_label'], _ = pd.factorize(df_train_offline['user_id'])
    df_train_offline['book_label'], _ = pd.factorize(df_train_offline['book_id'])
    sparse_book_user = sparse.csr_matrix(
        (np.ones(len(df_train_offline)), (df_train_offline['book_label'], df_train_offline['user_label'])))

    epoch, emb_size = 10, 32  # 小数据集用小参数
    bpr_model = implicit.bpr.BayesianPersonalizedRanking(factors=emb_size, iterations=epoch, regularization=0.001,
                                                         random_state=42)
    bpr_model.fit(sparse_book_user)
    print("BPR训练完成！（可进一步融合嵌入到相似度）")