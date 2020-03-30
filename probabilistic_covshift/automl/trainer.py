import h2o

from h2o.automl import get_leaderboard, H2OAutoML

from probabilistic_covshift import util as util
from probabilistic_covshift.automl import util as h2o_util
from probabilistic_covshift.constants.automl_constants import AutoMLConfig
from probabilistic_covshift.constants.automl_constants import H2OServerInfo

logger = util.create_logger(__name__)


class AutoMLTrainer(object):
    def __init__(self, auto_ml_config):
        self.auto_ml_config = auto_ml_config

    def view_leaderboard(self, auto_ml):
        leader_board = get_leaderboard(auto_ml, extra_columns='ALL')
        logger.info('Leaderboard: \n{}'.format(leader_board.head(rows=leader_board.nrows)))
        h2o_util.show_model_performance(auto_ml.leader)

    def save_auto_ml_leader(self, auto_ml):
        auto_ml_leader_path = h2o.save_model(
            auto_ml.leader,
            path=self.auto_ml_config[AutoMLConfig.MODEL][AutoMLConfig.BEST_MODEL_PATH])
        return auto_ml_leader_path
    
    def train(self, h2o_base_table: h2o.H2OFrame, h2o_base_table_predictors: [str]):
        auto_ml = H2OAutoML(
            nfolds=self.auto_ml_config[AutoMLConfig.CROSS_VAL][AutoMLConfig.NFOLDS],
            max_runtime_secs=self.auto_ml_config[AutoMLConfig.MODELING][AutoMLConfig.MAX_RUNTIME_SECS],
            max_models=self.auto_ml_config[AutoMLConfig.MODELING][AutoMLConfig.MAX_MODELS],
            stopping_metric=self.auto_ml_config[AutoMLConfig.MODELING][AutoMLConfig.STOPPING_METRIC],
            seed=self.auto_ml_config[AutoMLConfig.SEED],
            sort_metric=self.auto_ml_config[AutoMLConfig.MODELING][AutoMLConfig.SORT_METRIC],
            exclude_algos=self.auto_ml_config[AutoMLConfig.EXCLUDE_ALGOS])

        auto_ml.train(x=h2o_base_table_predictors, y=self.auto_ml_config[AutoMLConfig.DATA][AutoMLConfig.ORIGIN_COL],
                      training_frame=h2o_base_table,
                      fold_column=self.auto_ml_config[AutoMLConfig.CROSS_VAL][AutoMLConfig.FOLD_COL])

        self.view_leaderboard(auto_ml)

        auto_ml_leader_path = self.save_auto_ml_leader(auto_ml)

        h2o.remove_all()

        return auto_ml_leader_path

    def retrieve_h2o_base_table_predictors(self, h2o_base_table: h2o.H2OFrame):
        cols_to_drop = [
            'row_id',
            self.auto_ml_config[AutoMLConfig.DATA][AutoMLConfig.LABEL_COL],
            self.auto_ml_config[AutoMLConfig.DATA][AutoMLConfig.ORIGIN_COL]
        ] + self.auto_ml_config[AutoMLConfig.DATA][AutoMLConfig.CATEGORICAL_VARIABLES]
        return h2o_base_table.drop(cols_to_drop).col_names

    def run(self):
        h2o_util.init_server_connection(
            self.auto_ml_config[AutoMLConfig.SERVER_CONN_INFO][H2OServerInfo.IP],
            self.auto_ml_config[AutoMLConfig.SERVER_CONN_INFO][H2OServerInfo.PORT])

        h2o_base_table = h2o_util.convert_base_table_to_h2o_frame(
            self.auto_ml_config[AutoMLConfig.DATA][AutoMLConfig.BASE_TABLE_PATH],
            self.auto_ml_config[AutoMLConfig.DATA][AutoMLConfig.COL_NAMES])

        h2o_base_table = h2o_util.convert_label_to_enum_type(
            h2o_base_table,
            self.auto_ml_config[AutoMLConfig.DATA][AutoMLConfig.ORIGIN_COL])

        h2o_base_table_predictors = self.retrieve_h2o_base_table_predictors(h2o_base_table)

        logger.info(f'Base table inferred column types: {h2o_base_table.types}')

        h2o_base_table = h2o_util.create_fold_column_if_not_exist(
            h2o_base_table,
            self.auto_ml_config[AutoMLConfig.CROSS_VAL][AutoMLConfig.FOLD_COL],
            self.auto_ml_config[AutoMLConfig.CROSS_VAL][AutoMLConfig.NFOLDS])

        auto_ml_leader_path = self.train(h2o_base_table, h2o_base_table_predictors)

        return auto_ml_leader_path
