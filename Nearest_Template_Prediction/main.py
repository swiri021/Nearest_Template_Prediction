from algorithm.NTPrediction import NTP
import pandas as pd
import numpy as np

class_information = pd.read_csv('input_example/input_example_class_information.csv', index_col=0)
class_information.index = class_information.index.astype(int).astype(str)

expr = pd.read_csv('input_example/input_example_expression.csv', index_col=0)
expr.index = expr.index.astype(int).astype(str)

ntp = NTP(template_df=class_information, input_df=expr)
print ntp.predict(perm=1000)