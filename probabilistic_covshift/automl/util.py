import h2o

from h2o.estimators.estimator_base import H2OEstimator

from probabilistic_covshift import util as util

logger = util.create_logger(__name__)


def init_server_connection(host_addr, port):
    h2o.init(ip=host_addr, port=port)


def convert_base_table_to_h2o_frame(base_table_path: str, col_names: [str]) -> h2o.H2OFrame:
    h2o_base_table = h2o.import_file(path=base_table_path, col_names=col_names)
    return h2o_base_table


def convert_label_to_enum_type(h2o_base_table: h2o.H2OFrame, label_col) -> h2o.H2OFrame:
    h2o_base_table[label_col] = h2o_base_table[label_col].asfactor()
    return h2o_base_table


def create_fold_column_if_not_exist(h2o_base_table: h2o.H2OFrame, fold_column: str, nfolds: int = None) -> h2o.H2OFrame:
    if fold_column and fold_column not in h2o_base_table.col_names:
        h2o_fold_col = h2o_base_table.kfold_column(n_folds=nfolds)
        h2o_fold_col.set_names([fold_column])
        h2o_base_table = h2o_base_table.cbind(h2o_fold_col)
    return h2o_base_table


def show_model_performance(model: H2OEstimator):
    cross_val_model_performance = model.model_performance(xval=True)
    logger.info('Cross validation model performance')
    logger.info(cross_val_model_performance)

    [[threshold, f1]] = cross_val_model_performance.F1()
    logger.info('Threshold = {} for maximum F1 = {}'.format(threshold, f1))


def add_unique_row_id(h2o_base_table: h2o.H2OFrame):
    num_rows = h2o_base_table.shape[0]

    ids = []
    for id in range(0, num_rows):
        ids.append(id)

    h2o_id_frame = h2o.H2OFrame(ids)
    return h2o_base_table.cbind(h2o_id_frame.set_names(['row_id']))
