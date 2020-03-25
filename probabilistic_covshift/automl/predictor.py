import h2o

from probabilistic_covshift.automl import util as h2o_util
from probabilistic_covshift import util as util
from probabilistic_covshift.constants.automl_constants import AutoMLConfig as AutoMLConfig

logger = util.create_logger(__name__)


class AutoMLPredictor(object):
    def __init__(self, auto_ml_config):
        self.auto_ml_config = auto_ml_config

    def load_model(self):
        logger.info('Load the model and leaderboard')

        model = h2o.load_model(self.auto_ml_config[AutoMLConfig.MODEL][AutoMLConfig.BEST_MODEL_PATH])

        logger.info('Type(model): {}'.format(type(model)))
        logger.info('Loaded model: {}'.format(model.summary()))
        return model

    def predict(self, auto_ml_leader, h2o_base_table):
        id_col = ['id']
        predictions = auto_ml_leader.predict(h2o_base_table)
        predictions = predictions.cbind(h2o_base_table[id_col])
        return predictions

    def run(self):
        h2o_base_table = h2o_util.convert_base_table_to_h2o_frame(
            self.auto_ml_config[AutoMLConfig.DATA][AutoMLConfig.BASE_TABLE_PATH],
            self.auto_ml_config[AutoMLConfig.DATA][AutoMLConfig.COL_NAMES])

        h2o_base_table = h2o_util.add_unique_row_id(h2o_base_table)

        h2o_base_table = h2o_util.convert_label_to_enum_type(
            h2o_base_table,
            self.auto_ml_config[AutoMLConfig.DATA][AutoMLConfig.LABEL_COL])

        auto_ml_leader = self.load_model()

        predictions = self.predict(auto_ml_leader, h2o_base_table)

        h2o.remove_all()

        return predictions
