# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import seaborn as sns
import datetime
import matplotlib.pyplot as plt
import matplotlib
# %matplotlib inline
myfont = matplotlib.font_manager.FontProperties(fname='./msyh.ttf')
import math
from pyecharts import Scatter, Bar, Overlap, HeatMap, Page, Timeline, Line, configure
configure(output_image='pdf')
from dateutil.relativedelta import relativedelta
from OrderedSet import OrderedSet
from bs4 import BeautifulSoup
import re
import scipy.stats.stats as stats
from sklearn.tree import DecisionTreeClassifier




class draw():
	def __init__(self):
		self = self
	
	def missing_check(df,columns,miss_rate=0.8,handling=None):
		temp = pd.DataFrame(df[columns].isnull().sum())
		temp = temp.reset_index().rename(columns={'index':'feature',0:'missing'})
		temp = temp.sort_values('missing',ascending=True)
		temp['missing_rate'] = np.round(temp['missing']/df.shape[0],2)*100
		temp = temp[temp['missing_rate']>miss_rate]
		attr = temp.feature.values.tolist()
		v1 = temp.missing.values.tolist()
		v2 = temp.missing_rate.values.tolist()
		bar = Bar()
		bar.add('缺失量',attr,v1,is_stack=True,is_datazoom_show=True,datazoom_type='both',datazoom_range=[0,100],
				label_color=['#0081FF','#FF007C'],is_visualmap=True, visual_type='size',visual_range=[0,100],
				visual_range_size=[10,10],is_yaxislabel_align=False,visual_dimension=1,tooltip_text_color='#000000',
				label_emphasis_textcolor='#000000',is_more_utils=True,yaxis_rotate=45,label_emphasis_pos='right',
				is_convert=True)
		line=Line()
		line.add('缺失率', attr, v2, yaxis_formatter="%", yaxis_min=0,yaxis_max=100,label_emphasis_textcolor='#000000',line_width=3,
				label_emphasis_pos='right',is_convert=True)
		overlap = Overlap(width=1000,height =np.round(temp.shape[0]/2)*30)
		overlap.add(bar)
		overlap.add(line, xaxis_index=1, is_add_xaxis=True)
		if handling == None:
			return overlap
		elif handling == 'drop':
			return temp['feature'].values.tolist()
	
	def numerical_binning_cut(X,Y,n=50,cut_way='auto',text_print=True):
		if X.value_counts().count()==1:
			return 'error'
		if X.value_counts().count()< n:
			num = X.value_counts().count()
		else:
			num = n
		r = 0
		cut_result=False
		best_r_list = []
		best_n_list = []
		if X.isnull().any()==True:
			Y = Y[X.notnull()]
			X = X[X.notnull()]
		if cut_way=='auto':
			while ((np.abs(r)<0.99) & (num>2)):
				cut_result = False
				while cut_result==False:
					try:
						d1 = pd.DataFrame({"X": X, "Y": Y, "Bucket": pd.cut(X, num)})
						cut_result=True
					except:
						num -= 1
				d2 = d1.groupby('Bucket', as_index = True)
				if (d2.size()>0).values.all():
					r, p = stats.spearmanr(d2.mean().X, d2.mean().Y)
					best_r_list.append(r)
					best_n_list.append(num)
				num -= 1
			if (num<5):
				try:
					m = max(np.abs(best_r_list))
					max_index = [i for i, j in enumerate(np.abs(best_r_list)) if j == m][0]
					num = best_n_list[max_index]
					d1 = pd.DataFrame({"X": X, "Y": Y, "Bucket": pd.cut(X, num)})
					d2 = d1.groupby('Bucket', as_index = True)
				except:
					numerical_binning_cut(X,Y,n=n)
		elif cut_way=='manual':
			if text_print == True:
				print('Start using manual cut method.')
				print('='*60)
			cut_result = False
			while cut_result==False:
				try:
					d1 = pd.DataFrame({"X": X, "Y": Y, "Bucket": pd.cut(X, num)})
					cut_result=True
				except:
					num -= 1
		return d1
	
	def numerical_binning_qcut(X,Y,n=50,cut_way='auto',text_print=True):
		# X = df_with_overdue[numerical_col[4]]
		# Y = df_with_overdue[target]
		# target = 'is_over'
		num=n
		r = 0
		cut_result=False
		best_r_qlist = []
		best_n_qlist = []
		if X.isnull().any()==True:
			Y = Y[X.notnull()]
			X = X[X.notnull()]
		if cut_way=='auto':
			while np.abs(r) < 0.99:
				cut_result = False
				while cut_result==False:
					try:
						d1 = pd.DataFrame({"X": X, "Y": Y, "Bucket": pd.qcut(X, num)})
						cut_result=True
					except:
						num -= 1
				d2 = d1.groupby('Bucket', as_index = True)
				r, p = stats.spearmanr(d2.mean().X, d2.mean().Y)
				best_r_qlist.append(r)
				best_n_qlist.append(num)
				num -= 1
			if ((num>2) & (num<4)):
				if text_print == True:
					print('Cannot find best quantile binning cut.')
					print('='*60)
					print('Select the best quantile binning cut from previous.\n')
				m = max(np.abs(best_r_qlist))
				max_index = [i for i, j in enumerate(np.abs(best_r_qlist)) if j == m][0]
				num = best_n_qlist[max_index]
				d1 = pd.DataFrame({"X": X, "Y": Y, "Bucket": pd.qcut(X, num)})
				d2 = d1.groupby('Bucket', as_index = True)
			if (num<=2):
				if text_print == True:
					print('Cannot find best quantile binning cut.')
					print('='*60)
					print('Start looking for equidistance binning cut.\n')
				return draw.numerical_binning_cut(X,Y,n=n)
		elif cut_way=='manual':
			if text_print == True:
					print('Start using manual qcut method.')
					print('='*60)
			cut_result = False
			while cut_result==False:
				try:
					d1 = pd.DataFrame({"X": X, "Y": Y, "Bucket": pd.qcut(X, num)})
					cut_result=True
					if num != n:
						print('Cannot cut with {} bins, cutting with {} bins.'.format(n,num))
				except:
					num -= 1
			if (num<2):
				if text_print == True:
					print('Cannot find any quantile binning cut.')
					print('='*60)
					print('Start looking for equidistance binning cut.\n')
				return draw.numerical_binning_cut(X,Y,n=n)
		return d1
	
	def numerical_binning_qcut_for_draw(df,X,Y,missing_exist=False):
		if type(df)==str:
			return 'error'
		d2 = df.groupby('Bucket', as_index = True)
		d2_index = d2.size().index.values
		attr = []
		for c in d2_index:
			attr.append(str(c))
		if missing_exist == True:
			attr.append('missing')
			missing_X = X[X.isnull()]
			missing_Y = Y[X.isnull()]
			missing_X.replace(to_replace=np.nan,value='missing',inplace=True)
			missing_d = pd.DataFrame({"X": missing_X, "Y": missing_Y, "Bucket": missing_X.values})
		d3 = pd.DataFrame(attr,columns = [X.name])
		if missing_exist == True:
			d3[Y.name] = np.array(d2.sum().Y.values.tolist()+missing_d.groupby('Bucket', as_index = True).sum().Y.values.tolist())
			d3['total'] = np.array(d2.count().Y.values.tolist()+missing_d.groupby('Bucket', as_index = True).count().Y.values.tolist())
			d3[Y.name + '_rate'] = np.array(d2.mean().Y.values.tolist()+missing_d.groupby('Bucket', as_index = True).mean().Y.values.tolist())
		else:
			d3[Y.name] = d2.sum().Y.values
			d3['total'] = d2.count().Y.values
			d3[Y.name + '_rate'] = d2.mean().Y.values
		return d3

	def DecisionTreeBinning(data,feature,target,max_depth=4,missing_exist=False):
		X = data[feature]
		Y = data[target]
		if X.isnull().any()==True:
			Y = Y[X.notnull()]
			X = X[X.notnull()]
		x = np.swapaxes(np.array([X.values]),0,1)
		y = Y.values
		clf_1 = DecisionTreeClassifier(criterion = 'entropy' , max_depth = max_depth)
		clf_1.fit(x, y)
		y_hat = clf_1.predict(x)
		thrs_out = np.unique( clf_1.tree_.threshold[clf_1.tree_.feature > -2] )
		thrs_out = np.sort(thrs_out)
		d1 = pd.DataFrame({"X": X, "Y": Y, "Bucket": pd.cut(X, bins = thrs_out)})
		d2 = d1.groupby('Bucket', as_index = True)
		d2.mean()
		d2_index = d2.size().index.values
		attr = []
		for c in d2_index:
			attr.append(str(c))
		d3 = pd.DataFrame(attr,columns = [X.name])
		if missing_exist == True:
			attr.append('missing')
			missing_X = X[X.isnull()]
			missing_Y = Y[X.isnull()]
			missing_X.replace(to_replace=np.nan,value='missing',inplace=True)
			missing_d = pd.DataFrame({"X": missing_X, "Y": missing_Y, "Bucket": missing_X.values})
			d3[Y.name] = np.array(d2.sum().Y.values.tolist()+missing_d.groupby('Bucket', as_index = True).sum().Y.values.tolist())
			d3['total'] = np.array(d2.count().Y.values.tolist()+missing_d.groupby('Bucket', as_index = True).count().Y.values.tolist())
			d3[Y.name + '_rate'] = np.array(d2.mean().Y.values.tolist()+missing_d.groupby('Bucket', as_index = True).mean().Y.values.tolist())
		else:
			d3[Y.name] = d2.sum().Y.values
			d3['total'] = d2.count().Y.values
			d3[Y.name + '_rate'] = d2.mean().Y.values
		return d3
	
	def woe_binning_cut_for_draw(X,Y,woe,missing_exist=False):
		woe = woe
		q = woe.fit(X,Y)
		tt = q.get_bins()
		tt = tt.append(pd.Series(np.inf),ignore_index=1)
		d1 = pd.DataFrame({"X": X, "Y": Y, "Bucket": pd.cut(X, bins=tt.values)})
		d2 = d1.groupby('Bucket', as_index = True)
		d2_index = d2.size().index.values
		attr = []
		for c in d2_index:
			attr.append(str(c))
		d3 = pd.DataFrame(attr,columns = [X.name])
		d3[Y.name] = d2.sum().Y.values
		d3['total'] = d2.count().Y.values
		d3[Y.name + '_rate'] = d2.mean().Y.values
		return d3
	
	def bivar(data,feature,t_type,target=None,draw_type='line',bins=20,tree_depth=4,woe = None,d_reorder=False,cut='none',cut_way='auto',\
          custom_bin=[],cut_num=None,cut_start=None,fill_na=-99999,text_print=True,save=None,path='./'):
		# data is dataframe
		# feature is feature name
		# target is your target name, should be binary integer
		# t_type has two selection, 'c' or 'd', 'c' means continue, 'd' means discreate
		# draw_type has line and scatter
		# bins means you can choose how many bins to cut by yourself
		# tree_depth is only working for decision tree binning
		# woe is only working for woe binning
		# cut has three selection, 'none','qcut','cut','F-D' only for numerical col
		# d_reorder will sort categorical values and reorder it
		# example bivar(df,'feature2','is_over','c')
		# make sure feature and target in your dataframe
		data=data;feature=feature;target=target;t_type=t_type
		X = data[feature]
		if target != None:
			Y = data[target]
		if fill_na != -99999:
			X = X.fillna(fill_na)
		missing_exist = X.isnull().any()
		#fillna value if it has.
		if cut == 'none':
			if (target != None) & (missing_exist == True):
				# Data preprocessing
				temp_df = data[[feature,target]].fillna('missing').copy()
				feature_a = temp_df[[feature,target]].groupby(feature).size()
				feature_b = temp_df[[feature,target]].groupby(feature).sum()
				feature_final = pd.concat([feature_a,feature_b],axis=1)
				feature_final.rename(columns = {0:'total'},inplace=True)
			elif (target != None) & (missing_exist == False):
				# Data preprocessing
				temp_df = data[[feature,target]].fillna(fill_na).copy()
				feature_a = temp_df[[feature,target]].groupby(feature).size()
				feature_b = temp_df[[feature,target]].groupby(feature).sum()
				feature_final = pd.concat([feature_a,feature_b],axis=1)
				feature_final.rename(columns = {0:'total'},inplace=True)
			elif (target == None) & (missing_exist == False):
				# Data preprocessing
				temp_df = data[[feature]].fillna(fill_na).copy()
				feature_a = temp_df[[feature]].groupby(feature).size()
				feature_b = temp_df[[feature]].groupby(feature).sum()
				feature_final = pd.concat([feature_a,feature_b],axis=1)
				feature_final.rename(columns = {0:'total'},inplace=True)
			else:
				# Data preprocessing
				temp_df = data[[feature]].fillna('missing').copy()
				feature_a = temp_df[[feature]].groupby(feature).size()
				feature_b = temp_df[[feature]].groupby(feature).sum()
				feature_final = pd.concat([feature_a,feature_b],axis=1)
				feature_final.rename(columns = {0:'total'},inplace=True)   
			# check if has missing value in dataframe
			# Data preprocessing with missing value
			if missing_exist:
				if (t_type == 'c'):
					missing_df = feature_final.loc['missing':'missing']
					feature_final = feature_final[feature_final.index != 'missing']
					index_list = list(pd.cut(feature_final.index, bins = bins))
					feature_final.index=index_list
					feature_final.index.name=feature
					feature_final = feature_final.groupby(feature_final.index).sum()
					feature_final = feature_final.reindex(list(OrderedSet(index_list)))
					if target != None:
						feature_final[target+'_rate'] = feature_final[target]/feature_final['total']
				elif target != None:
					feature_final[target+'_rate'] = feature_final[target]/feature_final['total']
			# Data preprocessing w/o missing value
			else:
				if (t_type == 'c'):
					index_list = list(pd.cut(feature_final.index, bins = bins))
					feature_final.index=index_list
					feature_final.index.name=feature
					feature_final = feature_final.groupby(feature_final.index).sum()
					feature_final = feature_final.reindex(list(OrderedSet(index_list)))
					feature_final.index = list(OrderedSet(index_list))
					if target != None:
						feature_final[target+'_rate'] = feature_final[target]/feature_final['total']
				elif target != None:
					feature_final[target+'_rate'] = feature_final[target]/feature_final['total']
			feature_final.reset_index(inplace=True)
			feature_final.rename(columns={'index':feature},inplace=True)
		if cut == 'custom':
			if (target != None) & (missing_exist == True):
				# Data preprocessing
				temp_df = data[[feature,target]].fillna('missing').copy()
				feature_a = temp_df[[feature,target]].groupby(feature).size()
				feature_b = temp_df[[feature,target]].groupby(feature).sum()
				feature_final = pd.concat([feature_a,feature_b],axis=1)
				feature_final.rename(columns = {0:'total'},inplace=True)
			elif (target != None) & (missing_exist == False):
				# Data preprocessing
				temp_df = data[[feature,target]].fillna(fill_na).copy()
				feature_a = temp_df[[feature,target]].groupby(feature).size()
				feature_b = temp_df[[feature,target]].groupby(feature).sum()
				feature_final = pd.concat([feature_a,feature_b],axis=1)
				feature_final.rename(columns = {0:'total'},inplace=True)
			elif (target == None) & (missing_exist == False):
				# Data preprocessing
				temp_df = data[[feature]].fillna(fill_na).copy()
				feature_a = temp_df[[feature]].groupby(feature).size()
				feature_b = temp_df[[feature]].groupby(feature).sum()
				feature_final = pd.concat([feature_a,feature_b],axis=1)
				feature_final.rename(columns = {0:'total'},inplace=True)
			else:
				# Data preprocessing
				temp_df = data[[feature]].fillna('missing').copy()
				feature_a = temp_df[[feature]].groupby(feature).size()
				feature_b = temp_df[[feature]].groupby(feature).sum()
				feature_final = pd.concat([feature_a,feature_b],axis=1)
				feature_final.rename(columns = {0:'total'},inplace=True)
			# check if has missing value in dataframe
			# Data preprocessing with missing value
			if missing_exist:
				if ((t_type == 'c') & (len(temp_df[feature].unique())>=bins)):
					missing_df = feature_final.loc['missing':'missing']
					feature_final = feature_final[feature_final.index != 'missing']
					index_list = list(pd.cut(feature_final.index, bins = custom_bin))
					feature_final.index=index_list
					feature_final.index.name=feature
					feature_final = feature_final.groupby(feature_final.index).sum()
					feature_final = feature_final.reindex(list(OrderedSet(index_list)))
					if target != None:
						feature_final[target+'_rate'] = feature_final[target]/feature_final['total']
				elif target != None:
					feature_final[target+'_rate'] = feature_final[target]/feature_final['total']
			# Data preprocessing w/o missing value
			else:
				if ((t_type == 'c') & (len(temp_df[feature].unique())>=bins)):
					index_list = list(pd.cut(feature_final.index, bins = custom_bin))
					feature_final.index=index_list
					feature_final.index.name=feature
					feature_final = feature_final.groupby(feature_final.index).sum()
					feature_final = feature_final.reindex(list(OrderedSet(index_list)))
					feature_final.index = list(OrderedSet(index_list))
					if target != None:
						feature_final[target+'_rate'] = feature_final[target]/feature_final['total']
				elif target != None:
					feature_final[target+'_rate'] = feature_final[target]/feature_final['total']
			feature_final.reset_index(inplace=True)
			feature_final.rename(columns={'index':feature},inplace=True)
		if ((t_type == 'c') & (cut == 'F-D') & (target != None)):
			if missing_exist == True:
				temp_df = data[[feature,target]].fillna('missing').copy()
			elif missing_exist != True:
				temp_df = data[[feature,target]].fillna(fill_na).copy()
			# Data preprocessing
			feature_a = temp_df[[feature,target]].groupby(feature).size()
			feature_b = temp_df[[feature,target]].groupby(feature).sum()
			feature_final = pd.concat([feature_a,feature_b],axis=1)
			feature_final.rename(columns = {0:'total'},inplace=True)
			# check if has missing value in dataframe
			# Data preprocessing with missing value
			if missing_exist:
				if ((t_type == 'c')):
					missing_df = feature_final.loc['missing':'missing']
					feature_final = feature_final[feature_final.index != 'missing']
					missing_temp_df = temp_df[temp_df[feature]=='missing']
					missing_temp_df = pd.DataFrame(missing_temp_df.groupby(feature).size())
					missing_temp_df.rename(columns={0:target},inplace=True)
					missing_over_size = temp_df[temp_df[feature]=='missing'][temp_df[target]==1].shape[0]
					missing_temp_df[target]['missing'] = missing_over_size
					non_missing_temp_df = temp_df[temp_df[feature]!='missing']
					non_missing_temp_df.index = non_missing_temp_df[feature]
					
					Q1 = non_missing_temp_df[feature].astype(float).describe()['25%']
					Q3 = non_missing_temp_df[feature].astype(float).describe()['75%']
					min_value = non_missing_temp_df.index.min()
					max_value = non_missing_temp_df.index.max()
					bins_list = [min_value-0.0001]
					temp_value = min_value
					F_D = 2*(Q3-Q1)/math.pow(non_missing_temp_df.shape[0],1/3)
					if F_D == 0:
						print('since too many values in one bin, so we start use cut method!')
						print('='*60)
						try:
							result = draw.bivar(data=data, feature=feature,target=target,bins=bins,cut='none',t_type=t_type,cut_num=cut_num,cut_start=cut_start)
							return result
						except:
							return draw.bivar(data=data, feature=feature,target=target,bins=bins,cut='none',t_type=t_type)
					while temp_value < max_value:
						temp_value += F_D
						bins_list.append(temp_value)
					index_list = list(pd.cut(feature_final.index, bins = bins_list))
					feature_final.index=index_list
					feature_final.index.name=feature
					feature_final = feature_final.groupby(feature_final.index).sum()
					feature_final = feature_final.reindex(list(OrderedSet(index_list)))
			# Data preprocessing w/o missing value
			else:
				if ((t_type == 'c')):
					non_missing_temp_df = temp_df.copy()
					non_missing_temp_df.index = non_missing_temp_df[feature]
					Q1 = non_missing_temp_df[feature].astype(float).describe()['25%']
					Q3 = non_missing_temp_df[feature].astype(float).describe()['75%']
					min_value = non_missing_temp_df.index.min()
					max_value = non_missing_temp_df.index.max()
					bins_list = [min_value-0.0001]
					temp_value = min_value
					F_D = 2*(Q3-Q1)/math.pow(non_missing_temp_df.shape[0],1/3)
					if F_D == 0:
						print('since too many values in one bin, so we start use cut method!')
						print('='*60)
						try:
							result = draw.bivar(data=data, feature=feature,target=target,bins=bins,cut='none',t_type=t_type,cut_num=cut_num,cut_start=cut_start)
							return result
						except:
							return draw.bivar(data=data, feature=feature,target=target,bins=bins,cut='none',t_type=t_type)
					while temp_value < max_value:
						temp_value += F_D
						bins_list.append(temp_value)
					index_list = list(pd.cut(feature_final.index, bins = bins_list))
					feature_final.index=index_list
					feature_final.index.name=feature
					feature_final = feature_final.groupby(feature_final.index).sum()
					feature_final = feature_final.reindex(list(OrderedSet(index_list)))
					feature_final[target+'_rate'] = feature_final[target]/feature_final['total']
			feature_final.reset_index(inplace=True)
			feature_final.rename(columns={'index':feature},inplace=True)
			feature_final[target+'_rate'] = feature_final[target]/feature_final['total']
		if ((t_type == 'c') & (cut == 'qcut') & (target != None)):
			try:
				feature_final = draw.numerical_binning_qcut_for_draw(draw.numerical_binning_qcut(X,Y,n=bins,cut_way=cut_way,text_print=text_print),\
                                                            X,Y,missing_exist=missing_exist)
			except:
				print('since too many values in one bin, so we start use cut method!')
				print('='*60)
				return draw.bivar(data=data, feature=feature,target=target,bins=bins,cut='none',t_type=t_type)
		elif ((t_type == 'c') & (cut == 'cut')& (target != None)):
			try:
				feature_final = draw.numerical_binning_qcut_for_draw(draw.numerical_binning_cut(X,Y,n=bins,cut_way=cut_way,text_print=True),\
                                                            X,Y,missing_exist=missing_exist)
			except:
				print('since too many values in one bin, so we start use cut method!')
				print('='*60)
				print('start using basic bins cut method.')
				return draw.bivar(data=data, feature=feature,target=target,bins=bins,cut='none',t_type=t_type)
		elif ((t_type == 'c') & (cut == 'dtree')& (target != None)):
			try:
				feature_final = draw.DecisionTreeBinning(data,feature,target,max_depth=tree_depth,missing_exist=False)
			except:
				print('since too many values in one bin, so we start use cut method!')
				print('='*60)
				print('start using basic bins cut method.')
				return draw.bivar(data=data, feature=feature,target=target,bins=bins,cut='none',t_type=t_type)
		elif((t_type == 'c') & (cut == 'woe')& (target != None)):
			try:
				feature_final = draw.woe_binning_cut_for_draw(X,Y,woe)
			except:
				print('since too many values in one bin, so we start use cut method!')
				print('='*60)
				print('start using basic bins cut method.')
				return draw.bivar(data=data, feature=feature,target=target,bins=bins,cut='none',t_type=t_type)
		if (cut_num != None) & (cut_start == 'start') & (t_type == 'c'):
			try:
				new_left = feature_final[feature_final['total']<cut_num][feature].iloc[0].left
				new_right = feature_final[feature_final['total']>cut_num][feature].iloc[0].left
			except:
				print('data cannot be cut from head.')
				print('='*60)
				print('start using basic bins cut method.')
				return draw.bivar(data=data, feature=feature,target=target,bins=bins,cut='none',t_type=t_type)
			new_total = feature_final.iloc[:feature_final[feature_final['total']>=cut_num].index[0]].total.sum()
			new_is_over = feature_final.iloc[:feature_final[feature_final['total']>=cut_num].index[0]][target].sum()
			new_tmp_df = pd.DataFrame({feature:[pd.Interval(left=new_left, right=new_right)],'total':[new_total],\
						 target:[new_is_over],target+'_rate':[new_is_over/new_total]})
			feature_final = pd.concat([new_tmp_df,feature_final.iloc[feature_final[feature_final['total']>cut_num].index[0]:]],\
									  axis=0).reset_index()
			feature_final = feature_final.drop('index',axis=1)
		elif (cut_num != None) & (cut_start == 'tail') & (t_type == 'c'):
			try:
				new_left = feature_final[feature_final['total']>=cut_num][feature].iloc[-1].right
				new_right = feature_final[feature_final['total']<cut_num][feature].iloc[-1].right
			except:
				print('data cannot be cut from tail.')
				print('='*60)
				print('start using basic bins cut method.')
				return draw.bivar(data=data, feature=feature,target=target,bins=bins,cut='none',t_type=t_type)
			new_total = feature_final.iloc[feature_final[feature_final['total']>=cut_num].index[-1]+1:].total.sum()
			new_is_over = feature_final.iloc[feature_final[feature_final['total']>=cut_num].index[-1]+1:][target].sum()
			new_tmp_df = pd.DataFrame({feature:[pd.Interval(left=new_left, right=new_right)],'total':[new_total],\
						 target:[new_is_over],target+'_rate':[new_is_over/new_total]})
			feature_final = pd.concat([feature_final.iloc[:feature_final[feature_final['total']>cut_num].index[-1]+1],new_tmp_df],\
									  axis=0).reset_index()
			feature_final = feature_final.drop('index',axis=1)
		elif (cut_num != None) & (cut_start == 'both') & (t_type == 'c'):
			try:
				new_left = feature_final[feature_final['total']<cut_num][feature].iloc[0].left
				new_right = feature_final[feature_final['total']>cut_num][feature].iloc[0].left
			except:
				print('data cannot be cut from head.')
				print('='*60)
				print('start using basic bins cut method.')
				return draw.bivar(data=data, feature=feature,target=target,bins=bins,cut='none',t_type=t_type)
			new_total = feature_final.iloc[:feature_final[feature_final['total']>=cut_num].index[0]].total.sum()
			new_is_over = feature_final.iloc[:feature_final[feature_final['total']>=cut_num].index[0]][target].sum()
			new_tmp_df = pd.DataFrame({feature:[pd.Interval(left=new_left, right=new_right)],'total':[new_total],\
						 target:[new_is_over],target+'_rate':[new_is_over/new_total]})
			feature_final = pd.concat([new_tmp_df,feature_final.iloc[feature_final[feature_final['total']>cut_num].index[0]:]],\
									  axis=0).reset_index()
			feature_final = feature_final.drop('index',axis=1)
			try:
				new_left = feature_final[feature_final['total']>=cut_num][feature].iloc[-1].right
				new_right = feature_final[feature_final['total']<cut_num][feature].iloc[-1].right
			except:
				print('data cannot be cut from head.')
				print('='*60)
				print('start using basic bins cut method.')
				return draw.bivar(data=data, feature=feature,target=target,bins=bins,cut='none',t_type=t_type)
			new_total = feature_final.iloc[feature_final[feature_final['total']>=cut_num].index[-1]+1:].total.sum()
			new_is_over = feature_final.iloc[feature_final[feature_final['total']>=cut_num].index[-1]+1:][target].sum()
			new_tmp_df = pd.DataFrame({feature:[pd.Interval(left=new_left, right=new_right)],'total':[new_total],\
						 target:[new_is_over],target+'_rate':[new_is_over/new_total]})
			feature_final = pd.concat([feature_final.iloc[:feature_final[feature_final['total']>cut_num].index[-1]+1],new_tmp_df],\
									  axis=0).reset_index()
			feature_final = feature_final.drop('index',axis=1)
		if (missing_exist) & (cut in ['none','F-D','custom']) & (t_type == 'c'):
			missing_df[target+'_rate'] = missing_df[target]/missing_df['total']
			missing_df = missing_df.reset_index()
			feature_final = pd.concat([feature_final,missing_df],axis=0)
			feature_final = feature_final.reset_index()
			feature_final = feature_final.drop('index',axis=1)
		if type(feature_final)!=pd.core.frame.DataFrame:
			return 'There are less than 1 values in {}.'.format(feature)
		if d_reorder:
			feature_final.sort_values(['total',target],ascending=[False,False],inplace=True)
		#Drawing bivar graph
		attr = []
		for c in feature_final[feature]:
			attr.append(str(c))
		if target != None:
			v1 = list(feature_final[target].values)
			v2 = [x - y for x, y in zip(feature_final['total'].tolist(), feature_final[target].tolist())]
			v3 = list(np.round(feature_final[target+'_rate'].values*100,2))
			yaxis_max = np.max(v3)
			bar = Bar(width=1200, height=600,title=feature,background_color='#ffffff')
			bar.add(target+"数量", attr, v1,is_stack=True,is_datazoom_show=True,datazoom_type='both',datazoom_range=[0,100],
					label_color=['#00FF7F','#0081FF','#FF007C'],is_visualmap=True, visual_type='size',visual_range=[0,yaxis_max],
					visual_range_size=[10,10],is_yaxislabel_align=False,visual_dimension=1,tooltip_text_color='#000000',
					label_emphasis_textcolor='#000000',is_more_utils=True)
			bar.add('non-'+target+"数量",attr,v2,is_stack=True,label_emphasis_textcolor='#000000')
			overlap = Overlap()
			overlap.add(bar)
			if draw_type=='line':
				line = Line()
				line.add(target, attr, v3, yaxis_formatter="%", yaxis_min=0,yaxis_max=1.2*yaxis_max,label_emphasis_textcolor='#000000',line_width=3)
				overlap.add(line, yaxis_index=1, is_add_yaxis=True)
			if draw_type=='scatter':
				scatter = Scatter()
				scatter.add(target, attr, v3, yaxis_formatter="%", yaxis_min=0,yaxis_max=1.2*yaxis_max,label_emphasis_textcolor='#000000')
				overlap.add(scatter, yaxis_index=1, is_add_yaxis=True)
		else:
			bar = Bar(width=1200, height=600,title=feature,background_color='#ffffff')
			v1 = list(feature_final['total'].values)
			bar.add("数量", attr, v1,is_stack=True,is_datazoom_show=True,datazoom_type='both',datazoom_range=[0,100],
					label_color=['#0081FF','#FF007C'],is_visualmap=True, visual_type='size',
					visual_range_size=[10,10],is_yaxislabel_align=False,visual_dimension=1,tooltip_text_color='#000000',
					label_emphasis_textcolor='#000000',is_more_utils=True)
			overlap = Overlap()
			overlap.add(bar)
		if save==None:
			return overlap
		elif save!= None:
			return overlap.render(path=path+'{}.{}'.format(feature,save))
	
	def page_all_bivar(data,num_fea,cat_fea,tar,draw_order=['none'],out=False,file_name='bivar.html',
						draw_type='line',bins=20,tree_depth=4,woe = None,d_reorder=False,
						custom_bin=[],cut_num=None,cut_start=None,fill_na=-99999):
		#print all bivar picture on one html page
		#page_all_bivar(data=df,num_fea=numerical_col,cat_fea=category_col,tar='pass',out=True)
		page = Page()
		if len(num_fea) != 0:
			for c in data[num_fea]:
				keep_draw = True
				for k in draw_order:
					while keep_draw == True:
						try:
							temp = draw.bivar(data=data,feature=c,target=tar,t_type='c',draw_type='line',
							bins=20,tree_depth=4,woe = None,d_reorder=False,custom_bin=[],cut_num=None,
							cut_start=None,fill_na=-99999)
							page.add(temp)
							keep_draw = False
						except:
							continue
		if len(cat_fea) != 0:
			for d in data[cat_fea]:
				try:
					temp = draw.bivar(data=data,feature=d,target=tar,t_type='d',d_reorder = d_reorder)
					page.add(temp)
				except:
					continue
		if len(num_fea+cat_fea) == 0:
			return 'no such feature in function'
		if out:
			return page.render(file_name)
		else:
			return page
	
	def draw_month_timeline(df,time_col,tgt,time_type=None,time_format='%Y-%m-%d %H:%M:%S',file_name='month_timeline.html',draw_type='timeline'):
		month_30=[4,6,9,11]
		month_31=[1,3,5,7,8,10,12]
		if time_type == 'timestamp':
			temp_time = df[time_col].map(lambda x: datetime.datetime.date(datetime.datetime.fromtimestamp(int(x))))
		elif time_type == 'str':
			temp_time = df[time_col].map(lambda x: datetime.datetime.date(datetime.datetime.strptime(x,time_format)))
		else:
			temp_time = df[time_col].map(lambda x: datetime.datetime.date(x))
		temp = pd.concat([temp_time,df[tgt]],axis=1)
		temp['day'] = temp[time_col].map(lambda x: x.day)
		min_time = temp[time_col].min()
		max_time = temp[time_col].max()
		month_interval = (max_time.year-min_time.year)*12+(max_time.month-min_time.month)+1
		color_list = draw.random_color_list(month_interval)
		start_time = datetime.datetime.date(datetime.datetime.\
											strptime(str(temp[time_col].min().year)+'-'+str(temp[time_col].min().month)+'-01',"%Y-%m-%d"))
		if draw_type == 'timeline':
			timeline = Timeline(timeline_bottom=0)
			while start_time<max_time:
				if (start_time.month in month_30):
					month_day = 30
				elif (start_time.month in month_31):
					month_day = 31
				elif (start_time.month==2&(((start_time.year%100 == 0)&((start_time.year/100%4)==0))|((start_time.year%100!=0)&((start_time.year%4)==0)))):
					month_day = 29
				else:
					month_day = 28
				temp_df = temp[((temp[time_col]>=start_time)&(temp[time_col]<start_time+relativedelta(months=1)))]
				temp_df_size = temp_df.groupby(['day']).size()
				temp_df_mean = temp_df.groupby(['day']).mean()
				time_df = pd.concat([temp_df_size,temp_df_mean],axis=1)
				time_df.rename(columns={0:'size'},inplace=True)
				idx = time_df.index.values[(np.abs(time_df[tgt].values-time_df[tgt].mean())).argmin()]
				mean_val = time_df[tgt].loc[idx]
				missing_day = list(set(range(1,month_day+1))-set(time_df.index.values))
				if len(missing_day)>0:
					missing_len = len(missing_day)
					missing_size = list(np.zeros(missing_len))
					missing_mean = list(np.zeros(missing_len))
					temp_missing_df = pd.DataFrame([missing_day,missing_size,missing_mean]).T
					temp_missing_df.rename(columns={0:'day',1:'size',2:tgt},inplace=True)
					temp_missing_df.set_index('day',inplace=True)
					time_df = pd.concat([time_df,temp_missing_df])
				time_df.sort_index(inplace=True)
				time_df.index = time_df.index.values.astype(int)
				x_axis_label = ['{}'.format(i) for i in time_df.index.values.tolist()]
				bar = Bar("{}年{}月份数据".format(start_time.year,start_time.month))
				bar.add("数量", x_axis_label, time_df['size'].values.tolist(),label_color=color_list,\
						label_emphasis_textcolor='#000000',is_more_utils=True)
				line = Line()
				line.add("{} rate".format(tgt), x_axis_label, round(time_df[tgt],2).values.tolist(),label_color=color_list,\
						 mark_point=[{"coord":[str(idx),round(mean_val,2)],"name":"mean value"}],\
						 label_emphasis_textcolor='#000000')
				overlap = Overlap()
				overlap.add(bar)
				overlap.add(line, yaxis_index=1, is_add_yaxis=True)
				timeline.add(overlap,"{}-{}".format(start_time.year,start_time.month))
				start_time += relativedelta(months=1)
			print('File rendered at {}'.format(os.path.join(ROOT_DIR,file_name)))
			return timeline.render(os.path.join(ROOT_DIR,file_name))
		elif draw_type == 'all_in_one':
			month_day = 31
			overlap = Overlap()
			while start_time<max_time:
				temp_df = temp[((temp[time_col]>=start_time)&(temp[time_col]<start_time+relativedelta(months=1)))]
				time_df = temp_df.groupby(['day']).mean()
				idx = time_df.index.values[(np.abs(time_df[tgt].values-time_df[tgt].mean())).argmin()]
				mean_val = time_df[tgt].loc[idx]
				missing_day = list(set(range(1,month_day+1))-set(time_df.index.values))
				time_df.sort_index(inplace=True)
				time_df.index = time_df.index.values.astype(int)
				x_axis_label = ['{}'.format(i) for i in time_df.index.values.tolist()]
				line = Line("{} rate数据".format(tgt))
				line.add("{}-{}".format(start_time.year,start_time.month), \
						 x_axis_label, round(time_df[tgt],2).values.tolist(),label_color=color_list,\
						 mark_point=[{"coord":[str(idx),round(mean_val,2)],"name":"mean value"}],\
						 label_emphasis_textcolor='#000000',is_more_utils=True)
				overlap.add(line)
				start_time += relativedelta(months=1)
			return overlap
	
	def box_or_violin_for_analysis(data,kind):
		# data is your cleaned dataframe
		# kind has two selection, 'box' and 'violin'
		# example:
		# pic_for_analysis(owd_before[numerical_col+categorical_col],['box','violin'])
		need_pic = len(kind)*data.shape[1]
		f, axes = plt.subplots((need_pic//4)+1, 4, figsize=(15, need_pic))
		count = 0;i=0;j=0
		while count < data.shape[1]:
			temp_data = data[[data.columns[count]]].reset_index().\
						drop('index',axis=1)[data.columns[count]]
			if (temp_data.dtype == 'O'):
				temp_data = temp_data.astype('category').cat.codes
			if 'violin' in kind:
				temp_pic_violin = sns.violinplot(data=temp_data,color='#604d9e',ax=axes[i, j])
				temp_pic_violin.set_title(data.columns[count],fontproperties = myfont)
				j += 1
			if 'box' in kind:
				temp_pic_box = sns.boxplot(data=temp_data,color='#ff6d38',ax=axes[i, j])
				temp_pic_box.set_title(data.columns[count],fontproperties = myfont)
				j += 1
			count += 1
			if j == 4:
				i += 1;j = 0
	
	def corr_heatmap(data,report=False,file_name='Pairwise_Heatmap.html'):
		# draing correlation heatmap for data
		Temp_pairwise_df = data.corr()
		Temp_pairwise_df = Temp_pairwise_df.fillna(0)
		Temp_pairwise_df = Temp_pairwise_df*100
		Temp_pairwise_df = np.round(Temp_pairwise_df,2)
		x_axis = list(Temp_pairwise_df.index)
		y_axis = list(Temp_pairwise_df.columns)
		data = [[x_axis[i],y_axis[j],Temp_pairwise_df[x_axis[i]][y_axis[j]]] for i in range(len(x_axis)) for j in range(len(y_axis))]
		heatmap = HeatMap()
		heatmap.add("热力图直角坐标系", x_axis, y_axis, data, is_visualmap=True,visual_range=[0,100],
					visual_text_color="#000", visual_orient='horizontal',is_label_emphasis=True,label_emphasis_pos='inside',
					label_emphasis_textcolor="#000",label_emphasis_textsize=15,tooltip_tragger='item',tooltip_formatter='{c}',
				   is_datazoom_show=True,datazoom_type='both',visual_top=0.1,is_more_utils=True)
		if report == False:
			return heatmap
		else:
			return heatmap.render(file_name)
	
	def corr_heatmap_with_dual_zoombar(data,file_name='Pairwise_Heatmap.html'):
		# add vertical zoom bar based on corr_heatmap fuction
		report=True
		corr_heatmap(data=data,report=report,file_name=file_name)
		soup = BeautifulSoup(open(file_name))
		p = re.compile('"dataZoom":')
		the_iter = p.finditer(str(soup.body()[1]))
		result = max(enumerate(the_iter))[1]
		idx = result.span()[1]
		s_new = str(soup.body()[1])[:idx+2] + \
		'\n        {\n            "show": true,\n            "type": "slider",\n            "start": 50,\n            "end": 100,\n            "orient": "horizontal",\n            "xAxisIndex": null,\n            "yAxisIndex": null\n        },' + \
		'\n        {\n            "show": true,\n            "type": "slider",\n            "start": 50,\n            "end": 100,\n            "orient": "vertical",\n            "xAxisIndex": null,\n            "yAxisIndex": null\n        },' + \
		'\n        {\n            "show": true,\n            "type": "inside",\n            "start": 50,\n            "end": 100,\n            "orient": "vertical",\n            "xAxisIndex": null,\n            "yAxisIndex": null\n        },' + str(soup.body()[1])[idx+227:]
		s_new = s_new[31:]
		s_new = s_new[:-9]
		soup.body()[1].string = s_new
		Html_file= open(file_name,"w")
		Html_file.write(soup.prettify())
		Html_file.close()
	
	def displot(data_list,feature,label=['overdue','non-overdue'],na=-5):
		#plot distribution picture for compare data distribute
		#na value will be default as -5
		#data_list = [data_with_overdue_0,data_with_early_repayment_0]
		#only for compare two diff data
		#feature = numerical_col+category_col
		#displot_1([data_with_overdue,data_with_early_repayment],numerical_col+category_col,label=['0+','0-'])
		lines_in_one_pic = len(data_list)
		need_pic = len(feature)
		label = label
		f, axes = plt.subplots((need_pic//4)+1, 4, figsize=(15, need_pic))
		count=0;i=0;j=0
		while count < len(feature):
			temp_data_0 = data_list[0][[feature[count]]].reset_index().\
			drop('index',axis=1)[feature[count]]
			temp_data_1 = data_list[1][[feature[count]]].reset_index().\
			drop('index',axis=1)[feature[count]]
			if (temp_data_1.dtype == 'O'):
				s = pd.concat([temp_data_1,temp_data_1.astype('category').cat.codes],axis=1)
				temp_data_1 = temp_data_1.astype('category').cat.codes
				s = s.drop_duplicates().set_index(feature[count]).to_dict()
				try:
					temp_data_0 = temp_data_0[feature[count]].map(s[0])
				except:
					temp_data_0 = temp_data_0.astype('category').cat.codes
			tt = sns.distplot(temp_data_0.fillna(na),color='r',kde_kws={"lw": 3, "label": label[0]},ax=axes[i, j],axlabel=False)
			tt = sns.distplot(temp_data_1.fillna(na),color="b",kde_kws={"lw": 3, "label": label[1]},ax=axes[i, j],axlabel=False)
			tt.set_title(feature[i*4+j],fontproperties = myfont)
			j += 1
			count += 1
			if j == 4:
				i += 1;j = 0
	
	def displot_single(df,feature,target,label=['overdue','non-overdue'],na=-5):
		#plot distribution picture for compare data distribute
		#na value will be default as -5
		#data_list = [data_with_overdue_0,data_with_early_repayment_0]
		#only for compare two diff data
		#feature = numerical_col+category_col
		#displot_1([data_with_overdue,data_with_early_repayment],numerical_col+category_col,label=['0+','0-'])
		need_pic = len(feature)
		label = label
		f, axes = plt.subplots((need_pic//3)+1, 3, figsize=(15, need_pic))
		plt.subplots_adjust(wspace=0.3, hspace = 0.5)
		count=0;i=0;j=0
		while count < need_pic:
			temp_data_0 = df[feature[count]][df[target]==1].reset_index().drop('index',axis=1)
			temp_data_1 = df[feature[count]][df[target]==0].reset_index().drop('index',axis=1)
			tt = sns.distplot(temp_data_0.fillna(na),color='g',kde_kws={"lw": 3, "label": label[0]},ax=axes[i,j],axlabel=False,hist=False)
			tt = sns.distplot(temp_data_1.fillna(na),color='b',kde_kws={"lw": 3, "label": label[1]},ax=axes[i,j],axlabel=False,hist=False)
			tt.set_title(feature[i*3+j],fontproperties = myfont)
			j += 1
			count += 1
			if j == 3:
				i += 1;j = 0
	
	def na_timeline(dff,column,time_col,time_type=None):
		# time_col should be format like 'yyyy-mm-dd', such as '2018-01-17'
		df = dff.copy()
		df['na']=df[df[column].columns.values.tolist()].T.count()
		df['na'] = df['na'].map(lambda x: 1 if x>0 else np.nan)
		if time_type == 'timestamp':
			df[time_col] = df[time_col].map(lambda x: datetime.datetime.fromtimestamp(x).strftime("%Y-%m-%d"))
		elif time_type == 'time_str':
			df[time_col] = df[time_col].map(lambda x: datetime.datetime.strptime(x, "%Y-%m-%d %H:%M:%S").strftime("%Y-%m-%d"))
		else:
			df[time_col] = df[time_col].map(lambda x: x.strftime("%Y-%m-%d"))
		attr = df[[time_col]+column].groupby(time_col).size().index.tolist()
		v1 = df[[time_col]+column+['na']].groupby(time_col).size().values.tolist()
		v2 = (np.round(100-df[[time_col]+['na']].groupby(time_col).count()['na'].values.tolist()/df[[time_col]+column+['na']].groupby(time_col).size()*100)).values.tolist()
		bar = Bar(width=1200, height=600,title='NA Timeline')
		bar.add("数量", attr, v1,is_stack=True,is_datazoom_show=True,datazoom_type='both',datazoom_range=[0,100],
					label_color=['#0081FF','#FF007C'],is_visualmap=True, visual_type='size',visual_range=[0,100],
					visual_range_size=[10,10],is_yaxislabel_align=False,visual_dimension=1,tooltip_text_color='#000000',
					label_emphasis_textcolor='#000000',is_more_utils=True)
		line = Line()
		line.add('缺失率', attr, v2, yaxis_formatter="%", yaxis_min=0,yaxis_max=100,label_emphasis_textcolor='#000000',
				 mark_line=['average'],line_width=3)
		overlap = Overlap()
		overlap.add(bar)
		overlap.add(line, yaxis_index=1, is_add_yaxis=True)
		return overlap

	def target_timeline(dff,target,time_col,time_type=None):
		df = dff.copy()
		if time_type == 'timestamp':
			df[time_col] = df[time_col].map(lambda x: datetime.datetime.fromtimestamp(x).strftime("%Y-%m-%d"))
		elif time_type == 'time_str':
			df[time_col] = df[time_col].map(lambda x: datetime.datetime.strptime(x, "%Y-%m-%d %H:%M:%S").strftime("%Y-%m-%d"))
		# time_col should be format like 'yyyy-mm-dd', such as '2018-01-17'
		attr = df[[time_col,target]].groupby(time_col).size().index.tolist()
		v1 = df[[time_col,target]].groupby(time_col).size().values.tolist()
		v2 = np.round(df[[time_col,target]].groupby(time_col).mean()*100)[target].values.tolist()
		y_max = max(v2)*1.2
		bar = Bar(width=1200, height=600,title='Target Timeline')
		bar.add("数量", attr, v1,is_stack=True,is_datazoom_show=True,datazoom_type='both',datazoom_range=[0,100],
					label_color=['#0081FF','#FF007C'],is_visualmap=True, visual_type='size',visual_range=[0,y_max],
					visual_range_size=[10,10],is_yaxislabel_align=False,visual_dimension=1,tooltip_text_color='#000000',
					label_emphasis_textcolor='#000000',is_more_utils=True)
		line = Line()
		line.add(target+'率', attr, v2, yaxis_formatter="%", yaxis_min=0,yaxis_max=y_max,label_emphasis_textcolor='#000000',
		mark_line=['average'],line_width=3)
		overlap = Overlap()
		overlap.add(bar)
		overlap.add(line, yaxis_index=1, is_add_yaxis=True)
		return overlap

	def mean_timeline(dff,column,time_col,time_type=None):
		# time_col should be format like 'yyyy-mm-dd', such as '2018-01-17'
		df = dff.copy()
		if time_type == 'timestamp':
			df[time_col] = df[time_col].map(lambda x: datetime.datetime.fromtimestamp(x).strftime("%Y-%m-%d"))
		elif time_type == 'time_str':
			df[time_col] = df[time_col].map(lambda x: datetime.datetime.strptime(x, "%Y-%m-%d %H:%M:%S").strftime("%Y-%m-%d"))
		else:
			df[time_col] = df[time_col].map(lambda x: x.strftime("%Y-%m-%d"))
		attr = df[[time_col,column]].groupby(time_col).size().index.tolist()
		v1 = df[[time_col,column]].groupby(time_col).size().values.tolist()
		v2 = np.round(df[[time_col,column]].groupby(time_col).mean()*100,2)[column].values.tolist()
		bar = Bar(width=1200, height=600,title='Mean Timeline')
		bar.add("数量", attr, v1,is_stack=True,is_datazoom_show=True,datazoom_type='both',datazoom_range=[0,100],
					label_color=['#0081FF','#FF007C'],is_visualmap=True, visual_type='size',visual_range=[min(v2),max(v2)],
					visual_range_size=[10,10],is_yaxislabel_align=False,visual_dimension=1,tooltip_text_color='#000000',
					label_emphasis_textcolor='#000000',is_more_utils=True)
		line = Line()
		line.add('平均值', attr, v2, yaxis_min=min(v2),yaxis_max=max(v2),label_emphasis_textcolor='#000000',
		mark_line=['average'],line_width=3)
		overlap = Overlap()
		overlap.add(bar)
		overlap.add(line, yaxis_index=1, is_add_yaxis=True)
		return overlap

	def random_color_list(n):
		cat_size=n
		color_list_ori = np.round(np.array(sns.husl_palette(cat_size, h=.3))*255).reshape(-1).astype(int)
		color_list_ori
		hex_list = np.array([hex(x)[2:] for x in color_list_ori.reshape(-1).astype(int)]).reshape(cat_size,3)
		color_list = []
		for c in hex_list:
			color_list.append("#"+''.join(c).upper())
		return color_list