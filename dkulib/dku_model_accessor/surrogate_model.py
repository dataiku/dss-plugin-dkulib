# -*- coding: utf-8 -*-
import logging
from dku_model_accessor.constants import DkuModelAccessorConstants
from dku_model_accessor.preprocessing import Preprocessor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

logger = logging.getLogger(__name__)


class SurrogateModel(object):
    """
    In case the chosen saved model uses a non-tree based algorithm (and thus does not have feature importance), we fit this surrogate model
    on top of the prediction of the former one to be able to retrieve the feature importance information.
    """

    def __init__(self, prediction_type):
        self.check(prediction_type)
        self.feature_names = None
        self.target = None
        self.prediction_type = prediction_type
        # TODO should we define some params of RF to avoid long computation ?
        if prediction_type == DkuModelAccessorConstants.CLASSIFICATION_TYPE:
            self.clf = RandomForestClassifier(random_state=1407)
        else:
            self.clf = RandomForestRegressor(random_state=1407)

    def check(self, prediction_type):
        if prediction_type not in [DkuModelAccessorConstants.CLASSIFICATION_TYPE,
                                   DkuModelAccessorConstants.REGRRSSION_TYPE]:
            raise ValueError('Prediction type must either be CLASSIFICATION or REGRESSION.')

    def get_features(self):
        return self.feature_names

    def fit(self, df, target):
        preprocessor = Preprocessor(df, target)
        train, test = preprocessor.get_processed_train_test()
        train_X = train.drop(target, axis=1)
        train_Y = train[target]
        self.clf.fit(train_X, train_Y)
        self.feature_names = train_X.columns
