from sklearn import datasets
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from xetrack import Tracker

from warnings import filterwarnings

filterwarnings('ignore', category=ConvergenceWarning)


def test_ml_example():
    tracker = Tracker(Tracker.IN_MEMORY, params={
                      'dataset': 'iris', 'model': 'logistic regression'})
    iris = datasets.load_iris()
    X_train, X_test, y_train, y_test = train_test_split(
        iris.data, iris.target, test_size=0.3, random_state=0)

    lr = LogisticRegression(solver='newton-cg').fit(X_train, y_train)
    result = tracker.log(
        {'accuracy': float(lr.score(X_test, y_test)), 'lr': lr, 'extra': {'foo': 'bar'}})
    assert len(tracker.get(result['lr']).predict(X_test))
