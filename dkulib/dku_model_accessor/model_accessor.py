# -*- coding: utf-8 -*-
import logging
import pandas as pd
from dku_model_accessor.constants import DkuModelAccessorConstants
from dku_model_accessor.surrogate_model import SurrogateModel
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, \
    GradientBoostingRegressor, ExtraTreesClassifier, ExtraTreesRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

logger = logging.getLogger(__name__)

ALGORITHMS_WITH_VARIABLE_IMPORTANCE = [RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier,
                                       GradientBoostingRegressor, ExtraTreesClassifier, ExtraTreesRegressor,
                                       DecisionTreeClassifier, DecisionTreeRegressor]


class ModelAccessor(object):
    """
    Wrapper for our internal object PredictionModelInformationHandler
    """
    def __init__(self, model_handler=None):
        """
        model_handler: PredictionModelInformationHandler object
        """
        self.model_handler = model_handler

    def get_prediction_type(self):
        """
        Wrap the prediction type accessor of the model
        """
        if self.model_handler.get_prediction_type() in [DkuModelAccessorConstants.DKU_BINARY_CLASSIF, DkuModelAccessorConstants.DKU_MULTICLASS_CLASSIF]:
            return DkuModelAccessorConstants.CLASSIFICATION_TYPE
        elif DkuModelAccessorConstants.REGRRSSION_TYPE == self.model_handler.get_prediction_type():
            return DkuModelAccessorConstants.REGRRSSION_TYPE
        else:
            return DkuModelAccessorConstants.CLUSTERING_TYPE

    def get_target_variable(self):
        """
        Return the name of the target variable
        """
        return self.model_handler.get_target_variable()

    def get_original_test_df(self, limit=DkuModelAccessorConstants.MAX_NUM_ROW):
        try:
            full_test_df = self.model_handler.get_test_df()[0]
            test_df = full_test_df[:limit]
            logger.info('Loading {}/{} rows of the original test set'.format(len(test_df), len(full_test_df)))
            return test_df
        except Exception as e:
            logger.warning('Can not retrieve original test set: {}. The plugin will take the whole original dataset.'.format(e))
            full_test_df = self.model_handler.get_full_df()[0]
            test_df = full_test_df[:limit]
            logger.info('Loading {}/{} rows of the whole original test set'.format(len(test_df), len(full_test_df)))
            return test_df

    def get_train_df(self, limit=DkuModelAccessorConstants.MAX_NUM_ROW):
        full_train_df = self.model_handler.get_train_df()[0]
        train_df = full_train_df[:limit]
        logger.info('Loading {}/{} rows of the original train set'.format(len(train_df), len(full_train_df)))
        return train_df

    def get_per_feature(self):
        return self.model_handler.get_per_feature()

    def get_predictor(self):
        return self.model_handler.get_predictor()

    def get_feature_importance(self,cumulative_percentage_threshold=DkuModelAccessorConstants.FEAT_IMP_CUMULATIVE_PERCENTAGE_THRESHOLD):
        """
        :param cumulative_percentage_threshold: only return the top n features whose sum of importance reaches this threshold
        :return:
        """
        if self._algorithm_is_tree_based():
            predictor = self.get_predictor()
            clf = predictor._clf
            feature_names = predictor.get_features()
            feature_importances = clf.feature_importances_

        else:  # use surrogate model
            logger.info('Fitting surrogate model ...')
            surrogate_model = SurrogateModel(self.get_prediction_type())
            original_test_df = self.get_original_test_df()
            predictions_on_original_test_df = self.get_predictor().predict(original_test_df)
            surrogate_df = original_test_df[self.get_selected_features()]
            surrogate_df[DkuModelAccessorConstants.SURROGATE_TARGET] = predictions_on_original_test_df['prediction']
            surrogate_model.fit(surrogate_df, DkuModelAccessorConstants.SURROGATE_TARGET)
            feature_names = surrogate_model.get_features()
            feature_importances = surrogate_model.clf.feature_importances_

        feature_importance = []
        for feature_name, feat_importance in zip(feature_names, feature_importances):
            feature_importance.append({
                DkuModelAccessorConstants.FEATURE: feature_name,
                DkuModelAccessorConstants.IMPORTANCE: 100 * feat_importance / sum(feature_importances)
            })

        dfx = pd.DataFrame(feature_importance).sort_values(by=DkuModelAccessorConstants.IMPORTANCE,
                                                           ascending=False).reset_index(drop=True)
        dfx[DkuModelAccessorConstants.CUMULATIVE_IMPORTANCE] = dfx[DkuModelAccessorConstants.IMPORTANCE].cumsum()
        dfx_top = dfx.loc[dfx[DkuModelAccessorConstants.CUMULATIVE_IMPORTANCE] <= cumulative_percentage_threshold]
        return dfx_top.rename_axis(DkuModelAccessorConstants.RANK).reset_index().set_index(
            DkuModelAccessorConstants.FEATURE)

    def get_selected_features(self):
        """
        Return only features used in the model
        """
        selected_features = []
        for feat, feat_info in self.get_per_feature().items():
            if feat_info.get('role') == 'INPUT':
                selected_features.append(feat)
        return selected_features

    def get_selected_and_rejected_features(self):
        """
        Return all features in the input dataset except the target
        """
        selected_features = []
        for feat, feat_info in self.get_per_feature().items():
            if feat_info.get('role') in ['INPUT', 'REJECT']:
                selected_features.append(feat)
        return selected_features

    def predict(self, df):
        return self.get_predictor().predict(df)

    def _algorithm_is_tree_based(self):
        predictor = self.get_predictor()
        algo = predictor._clf
        for algorithm in ALGORITHMS_WITH_VARIABLE_IMPORTANCE:
            if isinstance(algo, algorithm):
                return True
            elif predictor.params.modeling_params.get('algorithm') in [DkuModelAccessorConstants.DKU_XGBOOST_CLASSIF, DkuModelAccessorConstants.DKU_XGBOOST_REGRESSION]:
                return True
        return False
