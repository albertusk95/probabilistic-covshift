from setuptools import setup, find_packages

setuptools_kwargs = {
    "install_requires": [
        # H2O dependencies
        "requests == 2.22.0",
        "tabulate == 0.8.6",
        "colorama == 0.4.3",
        "future == 0.18.2",
        # h2o python driver
        "h2o == 3.28.0.2",
        # pyspark
        "pyspark == 2.4.0"
    ],
    "zip_safe": False,
}

setup(name='probabilistic-covshift',
      version='0.1',
      description='Probabilistic Classification for Density Ratio Estimation',
      url='https://github.com/albertusk95/probabilistic-covshift',
      author='Albertus Kelvin',
      license='Copyright (C) Albertus Kelvin - All Rights Reserved',
      python_requires=">=3.7.0",
      packages=find_packages(),
      **setuptools_kwargs)