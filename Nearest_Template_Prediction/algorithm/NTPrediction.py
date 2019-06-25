import pandas as pd
import numpy as np
from scipy import spatial

class NTP:

	def _make_template(self):
		template = self.template_df.copy()

		##### Multiclass case

		dict_arr = [] #####Dictionary to make template by using map function
		if len(self._label_info)>2:
			for x in self._label_info:
				template[x+'_Template']=0
		else:
			for x in self._label_info:
				template[x+'_Template']=-1

		template = template.drop('label', axis=1)
		for i,item in enumerate(self._label_info):
			template.loc[template.index.isin(self._sorted_features[i]), item+'_Template'] = 1

		##### Binary class case
		return template

	def _distance_test(self, X_test, template):

		##### Cosine distance between sample and template element
		d_arr = [spatial.distance.cosine(X_test, template[c].values) for c in template.columns.tolist()]

		##### Return predicted result and closest value
		return self._label_info[np.argmin(d_arr)], min(d_arr)

	def _running_test(self, X, perm=1000):

		##### Generated Template
		hs = self._make_template()

		##### Only test with intersected genes between expression and signatures
		inter_gene = list(set(X.index.tolist()).intersection(hs.index.tolist()))
		X_test = X.loc[inter_gene]
		hs = hs.loc[inter_gene]

		##### Permuted Gene list(Nominal P-value)
		null_dist = [X.sample(n=len(hs)).values for a in range(perm)]

		##### Null distribution
		null_dist_p = [self._distance_test(n, hs)[1] for n in null_dist]

		##### Test Value
		test_result, test_value = self._distance_test(X_test, hs)

		##### P-value of tested value
		pval = float(len(np.where( null_dist_p < test_value )[0]))/float(len(null_dist_p))
		return test_result, test_value, pval

	def predict(self, perm=1000):
		predicted_result = []
		for x in self.input_df.columns.tolist():
			cl, d, pval = self._running_test(self.input_df[x], perm)
			predicted_result.append([cl, pval])
		predicted_result_df = pd.DataFrame(data=predicted_result, index=self.input_df.columns.tolist(), columns=['Predicted_Class', 'Pval'])
		return predicted_result_df

	def __init__(self, template_df, input_df):
		assert 'label' in template_df.columns.tolist(), "DataFrame must contains 'label' for classfication in DataFrame columns"
		assert  len(template_df['label'].unique())>=2, "Label must have more than 2 classes (Binary or Multiple)"

		self._label_info = template_df['label'].unique()
		self._sorted_features = [template_df[template_df['label']==l].index.tolist() for l in self._label_info]
		self.template_df = template_df
		self.input_df = input_df
