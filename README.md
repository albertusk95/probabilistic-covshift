# Probabilistic Classification for Density Ratio Estimation

This repo provides a python package used for distribution density ratio estimation using probabilistic classification. It's one of the techniques to estimate the density ratio (the simple & classic one, though).

For more info on the approach, please visit this <a href="https://albertusk95.github.io/posts/2020/03/density-ratio-estimation-probabilistic-classification/">post</a>.

## Modules

The important modules:

<ul>
  <li><b>probabilistic_classification_covshift</b> - the main module</li>
  <li><b>automl.trainer</b> - leverages H2O AutoML to fit the classifiers</li>
  <li><b>automl.predictor</b> - the best classifier is used to compute the probability of each instance belongs to the source or target origin. The computed probabilities become the parameters of the weight calculation</li>
</ul>

## Tech stack

<ul>
  <li>PySpark</li>
  <li>H2O AutoML</li>
  <li>Python</li>
</ul>

## Quickstart

You might want to take a look at the <a href="https://github.com/albertusk95/probabilistic-covshift/tree/master/example">example</a>.

### A) Compute weight

Prepare the configuration for AutoML.

```
conf = {
    AutoMLConfig.DATA: {
        AutoMLConfig.LABEL_COL: OriginFeatures.ORIGIN,
        AutoMLConfig.WEIGHT_COL: WeightFeatures.WEIGHT,
        AutoMLConfig.BASE_TABLE_PATH: 'data/base_table.parquet',
        AutoMLConfig.WEIGHT_PATH: 'data/weight.csv'
    },
    AutoMLConfig.SERVER_CONN_INFO: {
        H2OServerInfo.IP: 'localhost',
        H2OServerInfo.PORT: '54321'
    },
    AutoMLConfig.CROSS_VAL: {
        AutoMLConfig.FOLD_COL: "fold",
        AutoMLConfig.NFOLDS: 8,
    },
    AutoMLConfig.MODELING: {
        AutoMLConfig.MAX_RUNTIME_SECS: 3600,
        AutoMLConfig.MAX_MODELS: 10,
        AutoMLConfig.STOPPING_METRIC: 'logloss',
        AutoMLConfig.SORT_METRIC: 'logloss'
    },
    AutoMLConfig.EXCLUDE_ALGOS: [
        "StackedEnsemble",
        "DeepLearning"
    ],
    AutoMLConfig.MODEL: {
        AutoMLConfig.BEST_MODEL_PATH: 'data/model/'
    },
    AutoMLConfig.SEED: 23
}
```

Run the probabilistic classification module.

```python
source_df = <spark_dataframe>
target_df = <spark_dataframe>

pc = ProbabilisticClassification(source_df, target_df, conf)
pc.run()
```

### B) Append the weights to the base table

We got the weights! They are stored as a csv file in a location specified by `conf[AutoMLConfig.DATA][AutoMLConfig.WEIGHT_PATH]`.

Now, we just need to append them to the base table. The base table could be the source data, target data, or merged data (source and target). Please adjust with your needs.

Suppose that we'd like to append the weights to the merged data.

```python
base_table_path = conf[AutoMLConfig.DATA][AutoMLConfig.BASE_TABLE_PATH]
weight_path = conf[AutoMLConfig.DATA][AutoMLConfig.WEIGHT_PATH]
label_col = conf[AutoMLConfig.DATA][AutoMLConfig.LABEL_COL]

base_frame_df = spark.read.parquet(base_table_path).drop(label_col)

weight_df = spark.read.csv(weight_path, header=True)

weighted_base_frame_df = base_frame_df.join(weight_df, how='left', on='row_id')
```

How about if we'd like to append the weights to the source data only?

```python
base_frame_df = spark.read.parquet(base_table_path)
source_df = base_frame_df.filter(F.col(label_col) == OriginFeatures.SOURCE)

weight_df = spark.read.csv(weight_path, header=True)

weighted_base_frame_df = source_df.join(weight_df, how='left', on='row_id')
```

Done.

## Contribute

All features requests or bugs fixes for future improvement are welcomed.

Simply do the followings:

<ul>
  <li>Fork this repo</li>
  <li>Create a local branch</li>
  <li>Develop your features on the branch</li>
  <li>Submit a pull request</li>
</ul>

## Author

Copyright Albertus Kelvin 2020
