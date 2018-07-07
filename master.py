import os
import copy
import jieba
import logging
import os.path
import time
from pyltp import Segmentor
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDClassifier
from sklearn import metrics
from sklearn.linear_model.logistic import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn import tree
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors.nearest_centroid import NearestCentroid
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2, mutual_info_classif
from sklearn.feature_selection import SelectFromModel
from sklearn.feature_selection import f_regression
from sklearn.ensemble import VotingClassifier
from sklearn.externals import joblib
import numpy as np
stop_words = []  # 停用词表
train_tags = []  # 训练集标签
train_text = []  # 与训练集标签相对应的句子
test_tags = []  # 测试集标签
test_text = []  # 测试集...
vali_tags = []  # 验证集
vali_text = []
seg_train_text = []  # 分词后的训练集
seg_test_text = []  # 分词后的测试集
seg_vali_text = []  # 分词后的验证集


def devide():  # 将标签与句子分离
    with open('./data/trainingSet.txt', encoding='utf-8') as f:
        train_set = f.readlines()
    with open('./data/testSet.txt', encoding='utf-8') as f:
        test_set = f.readlines()
    with open('./data/validationSet.txt', encoding='utf-8') as f:
        vali_set = f.readlines()
    for line in train_set:
        t = line.split('\t', 1)  # 分割标签与句子
        train_tags.append(t[0])
        train_text.append(t[1].rstrip())
    for line in test_set:
        t = line.split('\t', 1)
        test_tags.append(t[0])
        test_text.append(t[1].rstrip())
    for line in vali_set:
        t = line.split('\t', 1)
        vali_tags.append(t[0])
        vali_text.append(t[1].rstrip())


def segment():  # 分词
    LTP_DATA_DIR = 'e:\\ltp_data'  # ltp模型目录的路径
    cws_model_path = os.path.join(LTP_DATA_DIR, 'cws.model')
    segmentor = Segmentor()  # 初始化实例
    segmentor.load(cws_model_path)  # 加载模型
    for t in train_text:  # 对训练集分词
        words = ' '.join(segmentor.segment(t))
        seg_train_text.append(words)
    for t in test_text:  # 对测试集分词
        words = ' '.join(segmentor.segment(t))
        seg_test_text.append(words)
    for t in vali_text:
        words = ' '.join(segmentor.segment(t))
        seg_vali_text.append(words)
    segmentor.release()


def get_stop_words():  # 获取停用词列表

    with open('./keywords/stopwords.txt', encoding='utf-8') as f:
        stop_word = f.read()
    string = stop_word.rstrip().split('\n')
    for s in string:
        stop_words.append(s)
    # print(stop_words[1:10])


def validate(i):
    clf = joblib.load('train_model' + str(i) + '.m')
    predicted = clf.predict(seg_vali_text)
    accuracy = metrics.accuracy_score(vali_tags, predicted)  # 准确率
    precision = metrics.precision_score(
        vali_tags, predicted, average='weighted')
    recall = metrics.recall_score(vali_tags, predicted, average='weighted')
    f1 = metrics.f1_score(vali_tags, predicted, average='weighted')
    ''''
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    rq = time.strftime('%Y%m%d%H%M', time.localtime(time.time()))
    log_path = (os.path.dirname(os.getcwd())) + '/Logs/'
    log_name = log_path + rq + '.log'
    logfile = log_name
    fh = logging.FileHandler(logfile, mode='w')
    fh.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(message)s")
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    logger.info("%.3f" % (accuracy))
    logger.info("%.3f" % (precision))
    logger.info("%.3f" % (recall))
    logger.info("%.3f" % (f1))
    logger.info(i)'''
    print("%.3f" % (accuracy), end='\t')
    print("%.3f" % (precision), end='\t')
    print("%.3f" % (recall), end='\t')
    print("%.3f" % (f1))
    # print('准确率:', accuracy, '\n')
    # print('classification report:')
    # print(metrics.classification_report(vali_tags, predicted))
    # print('混淆矩阵:')
    # print(metrics.confusion_matrix(test_tags, predicted))'''


def train():
    #dimension = list(range(1000, 15001, 1000))
    # dimension.append(15922)
    steps = [('vect', CountVectorizer(stop_words=stop_words)),
             ('tfidf', TfidfTransformer())]
    steps.append(('select', SelectKBest(chi2, k=8000)))
    #+++++++++++++++++++++++++++++++++++++++++++++++准确率    精确度    召回率
    mnb = ('clf', MultinomialNB())  # ++++++++++++++0.6026    0.46     0.60
    svc = ('clf', SVC(kernel='linear', C=2))  # +++++++0.7016    0.68     0.70
    sgd = ('clf', SGDClassifier())  # +++++++0.7103    0.69     0.71
    lsvc = ('clf', LinearSVC())  # +++++++++++++++++0.6972    0.67     0.70
    logis = ('clf', LogisticRegression())  # +++++++0.6317    0.58     0.63
    rfc = ('clf', RandomForestClassifier())  # ++++ +0.6870    0.65     0.69
    knn = ('clf', KNeighborsClassifier())  # +++++++0.5953    0.35     0.6
    dtc = ('clf', tree.DecisionTreeClassifier())  # 0.6667    0.65     0.67
    gdbt = ('clf', GradientBoostingClassifier())  # ++0.6972    0.66    0.70
    # nc = ('clf', NearestCentroid())
    '''
    # 检验k值
    print('classifier=SVC')
    print('\t\t准确率\t', '精确度\t', '召回率\t', 'f1分数')
    for i in range(len(dimension)):

        print('k=' + str(dimension[i]), end=' ')
        steps.append(('select', SelectKBest(chi2, k=dimension[i])))
        steps.append(lsvc)
        text_clf = Pipeline(steps)
        text_clf.fit(seg_train_text, train_tags)
        joblib.dump(text_clf, 'train_model' + str(i) + '.m')
        validate(i)
        steps.pop()
        steps.pop()
   '''

    steps.append(sgd)
    text_clf1 = Pipeline(steps)
    steps.pop()
    steps.append(svc)
    text_clf2 = Pipeline(steps)
    steps.pop()
    steps.append(lsvc)
    text_clf3 = Pipeline(steps)
    # 模型聚合
    eclf = VotingClassifier(estimators=[(
        'lr', text_clf1), ('rf', text_clf2), ('gnb', text_clf3)], voting='hard')
    eclf.fit(seg_train_text, train_tags)
    #text_clf3.fit(seg_train_text, train_tags)
    predicted = eclf.predict(seg_test_text)

    # predicted = text_clf3.predict(seg_test_text)
    # print('++++++++++模型聚合++++++++++')
    pre = np.mean(predicted == test_tags)  # 准确率
    print('准确率:', pre)
    print('classification report:')
    print(metrics.classification_report(test_tags, predicted))
    print('混淆矩阵:')
    print(metrics.confusion_matrix(test_tags, predicted))


def rate():
    num = 0
    for tag in train_tags:
        if tag == '0':
            num += 1
    return num / len(train_tags)


def main():
    devide()
    segment()
    get_stop_words()
    train()
    # print(rate())


main()
