import h2o

from probabilistic_covshift import util as util
from probabilistic_covshift.automl import util as h2o_util
from probabilistic_covshift.constants.automl_constants import AutoMLConfig
from probabilistic_covshift.constants.main_constants import OriginFeatures

logger = util.create_logger(__name__)


class AutoMLPredictor(object):
    def __init__(self, auto_ml_config, num_source, num_target):
        self.auto_ml_config = auto_ml_config
        self.num_source = num_source
        self.num_target = num_target

    def load_model(self):
        model = h2o.load_model(self.auto_ml_config[AutoMLConfig.MODEL][AutoMLConfig.BEST_MODEL_PATH])
        return model

    def predict(self, auto_ml_leader, h2o_base_table):
        id_col = ['row_id']
        predictions = auto_ml_leader.predict(h2o_base_table)
        predictions = predictions.cbind(h2o_base_table[id_col])
        return predictions

    def compute_weight(self, predictions):
        predictions['weight'] = (self.num_source / self.num_target) \
                                * (predictions[OriginFeatures.TARGET] / predictions[OriginFeatures.SOURCE])
        return predictions.drop(['predict', 'source', 'target'])

    def export_predictions_frame(self, predictions):
        h2o.export_file(
            frame=predictions,
            path=self.auto_ml_config[AutoMLConfig.DATA][AutoMLConfig.WEIGHT_PATH],
            force=True)

    def run(self):
        h2o_base_table = h2o_util.convert_base_table_to_h2o_frame(
            self.auto_ml_config[AutoMLConfig.DATA][AutoMLConfig.BASE_TABLE_PATH],
            self.auto_ml_config[AutoMLConfig.DATA][AutoMLConfig.COL_NAMES])
    
        h2o_base_table = h2o_util.convert_label_to_enum_type(
            h2o_base_table,
            self.auto_ml_config[AutoMLConfig.DATA][AutoMLConfig.ORIGIN_COL])

        auto_ml_leader = self.load_model()

        predictions = self.predict(auto_ml_leader, h2o_base_table)

        weights_and_ids = self.compute_weight(predictions)

        self.export_predictions_frame(weights_and_ids)

        h2o.remove_all()
