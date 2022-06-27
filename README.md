# pipelines

A library for integrating data pipelines from different Python packages.

For example, `pipelines.pandas_to_sklearn` wraps pandas functions in Scikit-Learn transformers, providing a persistent state that remembers exactly how the training data was transformed. Then, the test data will be transformed in exactly the same way as the training data when fed through the pipeline.

Additionally, `pipelines.pandas_to_sklearn.ExtractBoolean`, `pipelines.pandas_to_sklearn.ExtractNumeric`, and `pipelines.pandas_to_sklearn.ExtractCategorical` provide a way to split a pandas DataFrame into boolean-, numeric-, and categorical-only DataFrames. These DataFrames can then be individually processed and rejoined with `sklearn.pipeline.FeatureUnion`.