from sklearn.feature_extraction.text import HashingVectorizer
from models.classic_ml.classic_ml import LogisticRegressionModel, SVMModel, RandomForestModel
from models.CNN.CNN import TextCNNModel
from preprocessing.textpreprocessor import TextPreprocessor


vectorizer = HashingVectorizer(n_features=1316)

logistic_regression_model = LogisticRegressionModel()
if not logistic_regression_model.load('LogisticRegression_model.pkl'):
    raise ValueError("Logistic Regression model не был загружен корректно.")

svm_model = SVMModel()
if not svm_model.load('SVM_model.pkl'):
    raise ValueError("SVM model не был загружен корректно.")

random_forest_model = RandomForestModel()
if not random_forest_model.load('RandomForest_model.pkl'):
    raise ValueError("Random Forest model не был загружен корректно.")

CNN_model = TextCNNModel(input_dim=1316, num_classes=2)
if not CNN_model.load_model('CNN.pkl'):
    raise ValueError("CNN model не был загружен корректно.")

text_preprocessor = TextPreprocessor()