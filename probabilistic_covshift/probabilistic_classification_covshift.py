from pyspark.sql import functions as F
from pyspark.sql.types import StringType

from probabilistic_covshift.constants.main_constants import OriginFeatures as OriginFeatures
from probabilistic_covshift.constants.automl_constants import AutoMLConfig as AutoMLConfig
from probabilistic_covshift.automl.predictor import AutoMLPredictor
from probabilistic_covshift.automl.trainer import AutoMLTrainer


class ProbabilisticClassification(object):
    def __init__(self, source_df, target_df, auto_ml_config):
        self.source_df = source_df
        self.target_df = target_df
        self.base_table_df = None
        self.auto_ml_config = auto_ml_config

        if auto_ml_config[AutoMLConfig.SERVER_CONN_INFO] is None:
            raise ValueError('H2O server info must be specified')

    def retrieve_continuous_predictors(self):
        cols_and_types = self.source_df.dtypes

        continuous_preds = []
        for col, type in cols_and_types:
            if type != 'string':
                continuous_preds.append(col)
        return continuous_preds

    def add_origin_feature(self):
        self.source_df = self.source_df.withColumn(self.auto_ml_config[AutoMLConfig.DATA][AutoMLConfig.LABEL_COL],
                                                   F.lit(OriginFeatures.SOURCE))
        self.target_df = self.target_df.withColumn(self.auto_ml_config[AutoMLConfig.DATA][AutoMLConfig.LABEL_COL],
                                                   F.lit(OriginFeatures.TARGET))

    def merge_source_and_target(self):
        self.base_table_df = self.source_df.unionByName(self.target_df)

    def set_origin_as_categorical(self):
        self.base_table_df = self.base_table_df.withColumn(OriginFeatures.ORIGIN,
                                                           F.col(OriginFeatures.ORIGIN).cast(StringType()))

    def save_base_table(self):
        self.base_table_df.write.parquet(
            self.auto_ml_config[AutoMLConfig.DATA][AutoMLConfig.BASE_TABLE_PATH],
            mode='overwrite')

    def estimate_density_ratio(self, predictions):
        self.source_df = self.source_df.withColumn(
            self.auto_ml_config[AutoMLConfig.DATA][AutoMLConfig.WEIGHT_COL],
            (predictions[OriginFeatures.TARGET] / predictions[OriginFeatures.SOURCE])
        )

    def run(self):
        continuous_predictors = self.retrieve_continuous_predictors()

        self.source_df = self.source_df.select(*continuous_predictors)
        self.target_df = self.target_df.select(*continuous_predictors)

        self.add_origin_feature()
        self.merge_source_and_target()
        self.set_origin_as_categorical()
        self.save_base_table()

        self.auto_ml_config[AutoMLConfig.DATA][AutoMLConfig.COL_NAMES] = continuous_predictors \
            + [self.auto_ml_config[AutoMLConfig.DATA][AutoMLConfig.LABEL_COL]]

        trainer = AutoMLTrainer(self.auto_ml_config)
        auto_ml_leader_path = trainer.run()
        print(auto_ml_leader_path)

        self.auto_ml_config[AutoMLConfig.MODEL][AutoMLConfig.BEST_MODEL_PATH] = auto_ml_leader_path

        predictor = AutoMLPredictor(self.auto_ml_config)
        predictions = predictor.run()

        self.estimate_density_ratio(predictions)