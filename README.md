# Nearest Template Prediction for Python
This is a library of Nearest Template Prediction method for python

# Reference
[Nearest Template Prediction: A Single-Sample-Based Flexible Class Prediction with Confidence Assessment](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0015543)

# Example :
```Python
from algorithm.NTPrediction import NTP
import pandas as pd
import numpy as np

# Import class information for prediction with gene signature
class_information = pd.read_csv('input_example/input_example_class_information.csv', index_col=0)
class_information.index = class_information.index.astype(int).astype(str)

# Expression data
expr = pd.read_csv('input_example/input_example_expression.csv', index_col=0)
expr.index = expr.index.astype(int).astype(str)

ntp = NTP(template_df=class_information, input_df=expr)

# Nominal P-value and it does not have FDR
print ntp.predict(perm=1000)
```
