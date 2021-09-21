# This file contains some simple but vry useful fuctions

#Function returns a correlation of all toher variables with specified target variable
def corr_matrix (df=None,target_var='',method=None):
    '''
    Function returns correlation of the target variable with all the remaining variables
    df : The dataframe with all variables
    target_var : Variable that you wanty to check of all other vars with passed as a string
    method : 'pearson','kendall','spearman'
    '''
    if method == None:
        method = 'pearson' 
    corr_name = method+'corr'
    df_corr = df.corr(method)
    var_corr = df_corr[[target_var]]
    var_corr.rename({target_var:corr_name},axis=1,inplace = True)
    var_corr['abs_'+corr_name] = var_corr[corr_name].abs()
    var_corr.drop(target_var,axis=0, inplace=True)
    var_corr.sort_values('abs_'+corr_name,ascending=False,inplace=True)
    return var_corr

############################################################################################################################

def dataframe_null_report(df=None,null_pct_threshold=0):
    '''
    This function returns the fraction of nulls in each column of the dataframe based on a threshold specified as a fraction
    Parameters:
    	df (Pandas dataframe) : The dataframe for which you want to check the null % for each of its columns
    	null_pct_threshold (numeric) : Between 0 to 1. This is the minimum threshold on fraction nulls that wil be returned.
    									If no threshold is specified then it shows all columns with even in a single null.
    									Example: If 0.1 is specified then the function will return all columns which have >10% nulls.
    Returns:
    	Dataframe with all columns as rows which have fraction of nulls above the 'null_pct_threshold' with their % nulls value
    '''
    null_pct_df = pd.DataFrame(df.isna().sum(axis = 0)).reset_index()
    null_pct_df.columns = ['Variable','NullCount']
    null_pct_df['NullCountPct'] = null_pct_df['NullCount']/df.shape[0]
    null_pct_df.sort_values('NullCountPct',ascending=False,inplace=True)
    total_columns = df.shape[1]
    null_pct_df = null_pct_df[null_pct_df['NullCountPct']>null_pct_threshold]
    columns_with_10pct_nulls = null_pct_df.shape[0]
    print('Total Columns in Dataframe: {}'.format(total_columns))
    print('Columns with >{}% Nulls: {}'.format(null_pct_threshold*100,columns_with_10pct_nulls))
    return null_pct_df

############################################################################################################################

from sklearn.metrics import mean_squared_error , mean_squared_log_error , mean_absolute_percentage_error , r2_score
def regression_eval(y_true,y_pred,thrs=0,predictors=0):
    '''
    This function evaluates the performance of a regression model and returns evaluation metrics
    
    Parameters:
        y_true (Numpy array) : The ground truth labels given in the dataset
        y_pred (Numpy array) : Our predictions
        thrs (numeric) : Threshold value for classifying as over-prediction/under-prediction
        predictors (int) : Number of predictors in our regression model
    
    Returns: Dataframe with below metrics to evaluate performance of the regression model
    	1. Number of Observations : Rows on which the model is being evaluated, length of the y_pred/y_true series
    	2. MAE : Mean Absolute Error
    	3. MAPE : Mean Absolute Percentage Error
    	4. RMSE : Root Mean Square Error
    	5. RMSLE : Root Mean Square Log Error
    	5. Coefficient of Determination - R2 : R Squared Value (Goodness of fit)
    	6. Adj. R2 : Adjusted R Squared Value (Takes into account number of predictors, observations, etc.)
    	7. Over-prediction Magnitude : Avg. Magnitude of over-prediction above a specified threshold
		8. Percentage of Over-prediction : %age Records where model over-predicts above a specified threshold compared to y_true
		9. Under-prediction Magnitude : Avg. Magnitude of under-prediction above a specified threshold
		10.Percentage of Under-prediction : %age Records where model under-predicts above a specified threshold compared to y_true
    
    '''
    if len(y_true) != len(y_true):
        print('Length of Actual and Predicted lists is unequal. Please check and correct the same!')
        return
    
    eval_metrics = dict()
    n = len(y_true)
    k=predictors
    
    #Absolute magnitude of error: 
    all_err=np.abs(y_pred-y_true)
    mae=all_err.mean()
    overpred_magnitude=(all_err[y_pred-y_true>thrs]).mean()
    underpred_magnitude=(all_err[y_true-y_pred>thrs]).mean()
    
    max_overpred=(all_err[y_pred-y_true>thrs]).max()
    max_underpred=(all_err[y_true-y_pred>thrs]).max()
    
    #Percentage Error
    mape=np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    overpred_pct = (y_pred-y_true>thrs).mean() * 100
    underpred_pct = (y_true-y_pred>thrs).mean() * 100
    
    #RMSE & RMSLE
    rmse = np.sqrt(mean_squared_error(y_true,y_pred))
    #combinig the 2 series to find min and max to then calculate the RMSLE
    Y = pd.concat({"y_true": y_true,"y_pred": y_pred},axis=1)
    Y_min = Y.min().min()
    Y_max = Y.max().max()
    normalized_y_true = (y_true - Y_min) / (Y_max - Y_min)
    normalized_y_pred = (y_pred - Y_min) / (Y_max - Y_min)
    try:
        rmsle = np.sqrt(mean_squared_log_error(normalized_y_true,normalized_y_pred))
    except:
        print("An exception occurred while calculating rmsle.")
        rmsle = np.NaN
    
    #R2 and Adj. R2
    R2 = r2_score(y_true, y_pred)
    if k==0:
        print('Number of predictors has not been specified, hence Adjusted R2 will not be calculated')
        adj_R2 = np.NaN
    else: 
        adj_R2 = 1-((1-R2)*(n-1)/(n-k-1))
    
    #Combinig all metrics in dictionary and then to Dataframe
    eval_metrics['Length of Data'] = len(y_true)
    eval_metrics['MAE'] = mae
    eval_metrics['MAPE'] = mape
    eval_metrics['RMSE'] = rmse
    eval_metrics['RMSLE'] = rmsle
    eval_metrics['Coefficient of Determination - R2'] = R2
    eval_metrics['Adj. R2'] = adj_R2
    
    eval_metrics['Over-prediction Magnitude'] = overpred_magnitude
    eval_metrics['Percentage of Over-prediction'] = overpred_pct
    eval_metrics['Under-prediction Magnitude'] = underpred_magnitude
    eval_metrics['Percentage of Under-prediction'] = underpred_pct
    
    #Converting to Dataframe
    eval_metrics = pd.DataFrame([eval_metrics],columns=eval_metrics.keys()).T
    eval_metrics.columns = ['Value']
    eval_metrics.index.set_names('Evaluation Metric',inplace=True)
    eval_metrics['Value'] = eval_metrics['Value'].apply(lambda x: '%.2f' % x)
    
    return eval_metrics

#regression_eval(y_true,y_pred,thrs = 1, predictors=X.shape[1])


############################################################################################################################

def classification_eval(y_true,y_pred,prob_thrs=0.5):
    '''
    This function evaluates the performance of a classification model and returns evaluation metrics
    
    Parameters:
        y_true (Numpy array) : The ground truth labels given in the dataset
        y_pred (Numpy array) : Our prediction probability
        prob_thrs (numeric) : Between 0 and 1; 0.5 by default. Predicted values >= threshold are considered as positive and below are negative.
    
    Returns: Dataframe with below metrics to evaluate performance of the classification model
    	1. Number of Observations : Rows on which the model is being evaluated, length of the y_pred/y_true series
	    	i.    Confusion Matrix :
	    	ii.   Accuracy : 
	    	iii.  Precision : 
	    	iv.   Recall : 
	    	v.    F1 Score : 
	    	vi.   AUC : 
	    	vii.  AUCPR : 
	    2. Log Loss / Binary Cross Entropy:
    	3. 
    	Categorical Cross Entropy : 
    
    '''
    if len(y_true) != len(y_true):
        print('Length of Actual and Predicted lists is unequal. Please check and correct the same!')
        return
    
    eval_metrics = dict()
    n = len(y_true)
