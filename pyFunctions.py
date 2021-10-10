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

def obj_cols(df):
    '''
    This function returns the object columns in a given dataframe
    Parameters:
    	df (Pandas dataframe) : The dataframe in which to search for object columns

    Returns:
    	List of Object columns
    '''
    object_cols = df.select_dtypes(include = ['object']).columns
    print('Count of object columns: {}'.format(object_cols.shape[0]))
    if object_cols.shape[0]>0 : 
    	return list(object_cols.values)
    else:
    	print('No object columns in the dataframe!')
    	return

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

from sklearn.metrics import mean_squared_error,mean_squared_log_error, mean_absolute_percentage_error, r2_score
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
    # mape = mean_absolute_percentage_error(y_true,y_pred) *100
    overpred_pct=(y_pred-y_true>thrs).mean() * 100
    underpred_pct=(y_true-y_pred>thrs).mean() * 100
    
    #RMSE & RMSLE
    rmse = np.sqrt(mean_squared_error(y_true,y_pred))
    #combinig the 2 series to find min and max to then calculate the RMSLE
    Y = pd.concat({"y_true": y_true,"y_pred": y_pred},axis=1)
    Y_min = Y.min().min()
    Y_max = Y.max().max()
    normalized_y_true = (y_true - Y_min) / (Y_max - Y_min)
    normalized_y_pred = (y_pred - Y_min) / (Y_max - Y_min)
    try:
        # rmsle = np.sqrt(mean_squared_log_error(normalized_y_true,normalized_y_pred))
        rmsle = np.sqrt(mean_squared_log_error(y_true,y_pred))
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

def show_distribution(df=None,variable_name='',
                      pctl = [0.01,  0.20 , 0.25, 0.30 , 0.33, 0.40 , 0.50 , 0.60 , 0.67, 0.75, 0.80 , 0.90 , 0.99], 
                      plot= True, bins = 50):
    '''
    This function shows the distribution (Histogram plot) of any variable and returns some important percentile values of the variable
    Parameters:
        df            : The pandas dataframe that contains this variable data
        varaible_name : String, Name of the variable for which the distribution should be shown
        pctl          : List of Percentile values that should be returned (0 to 1 in multiples of 0.01)
        plot          : Boolean, Default True; Will plot a histogram if set to True
        bins          : The number of bins into which the data should be plotted on the histogram
    '''
    percentil = df[variable_name].quantile(np.linspace(.01, 1, 99, 0), 'lower')
    op = percentil[pctl]
    op = op.to_frame().reset_index()
    op.rename(columns = {'index': 'Percentile'},inplace=True)
    op.loc[len(op.index)] = ['Average', df[variable_name].mean()]
    if plot == True:
        plt.figure(figsize = (20,8))
        sns.distplot(x=df[variable_name], bins = bins)
    return op


############################################################################################################################

# Language Translation
import re
import html
import urllib.request
import urllib.parse

def translate_function(to_translate, to_language="en", from_language="id"):
    """
    Returns the translation using google translate
    	You must shortcut the language you define (French = fr, English = en, Spanish = es, etc...).
    	If not defined it will detect it or use english by default
    Example:
    	print(translate("salut tu vas bien?", "en"))
    	hello you alright?
    """
    
    parser = html
    agent = {'User-Agent':
         "Mozilla/4.0 (\
compatible;\
MSIE 6.0;\
Windows NT 5.1;\
SV1;\
.NET CLR 1.1.4322;\
.NET CLR 2.0.50727;\
.NET CLR 3.0.04506.30\
)"}
	base_link = "http://translate.google.com/m?tl=%s&sl=%s&q=%s"
    to_translate = urllib.parse.quote(to_translate)
    link = base_link % (to_language, from_language, to_translate)
    request = urllib.request.Request(link, headers=agent)
    raw_data = urllib.request.urlopen(request).read()
    data = raw_data.decode("utf-8")
    expr = r'(?s)class="(?:t0|result-container)">(.*?)<'
    re_result = re.findall(expr, data)
    if (len(re_result) == 0):
        result = ""
    else:
        result = parser.unescape(re_result[0])
    return (result)

############################################################################################################################

# Changing skewed distribution to a normal distribution

def box_cox_transform(feature=None, lambda_ = 0.25 , reverse = False ):
    '''
    This function returns the normalized distribution of input array by applying a Box-Cox transformation
    OR returns the reverse transform of a box-cox transformation if reverse is set to True

    Parameters:
        feature : A numpy array of the feature that you want to transform
        lambda_ : Lambda value for the box-cox transformation (The lambda can be estimated by using maximum likelihood to optimize the normality of the model results
        reverse : Default False, if set to True, the function will return the revese transform of a box-cox transformation
    '''
    if type(feature) is np.ndarray:
        if reverse == False:
            print('Applying Box-Cox Transformation to the input array')
            y_ = (feature ** lambda_) - 1
            return y_
        else:
            print('Reversing Box-Cox Transformation to the input array')
            y_ = (feature + 1) ** (1/lambda_)
            return y_
    else:
        print('The input array is not a numpy array! Please provide a numpy array as input.')
        return


############################################################################################################################

from sklearn.model_selection import train_test_split
def get_samples(df, num_samples, sample_size = None, stratify = None, seed = 12345):
    '''
    This function splits a dataframe into a number of samples based on the sample_size or saples of the approximately same length
    Parameters:
        df          : Pandas Dataframe with the data, can also be an array
        num_samples : The number of samples to get from the data
        sample_size : Size of each sample required
        stratify    : List of columns on which the dataframe should be stratified
        seed        : Default 12345, For reproducing the same sample if you want to try again
    Returns: 
        A Dataframe with sample draw and labelled as 'Sample_Label'
    '''
    samples_dict = dict()
    # Getting the length of input data
    if 'pandas' not in str(type(df)) : l = len(df) 
    else: l = df.shape[0]
    # For determining the number of samples if not given
    given_ss = sample_size
    if sample_size == None : sample_size = int(np.floor(l/num_samples))
    #check for length
    if l < num_samples * sample_size:
        raise ValueError('The number of samples with required sample size is larger than the length of input dataframe/array. '
                         'Please reduce either the sample size OR number of samples required!')
    for x in range(0, num_samples):
        if x == 0:
            temp = train_test_split(df,test_size=sample_size, random_state=seed, stratify = df[stratify] )
            df, df2 = temp[0], temp[1]
        elif x == num_samples-1:
            if given_ss != None:
                temp = train_test_split(df,test_size=sample_size, random_state=seed, stratify = df[stratify] )
                df, df2 = temp[0], temp[1]
            else: df2 = df
        else:
            temp = train_test_split(df,test_size=sample_size, random_state=seed, stratify = df[stratify] )
            df, df2 = temp[0], temp[1]
        
        df2['Sample_Label'] = 'Sample_'+str(x+1)
        samples_dict['sample_'+str(x+1)] = df2
    
    samples_df = pd.concat([ v for k,v in samples_dict.items()] )
        
    return samples_df

############################################################################################################################

def one_hot_encoding(df,column_name,prefix=None):
    '''
    This function created one hot encoded dummy variable for any given column_name
    Parameters:
        df          : The input Pandas dataframe
        column_name : The column that has to be one hot encoded
        prefix      : The prefix to the new created column with the 1/0 flag for given column_name, If no prefix is provided, then column_name will be prefixed
    Returns:
        Dataframe with the added on-hot encoded columns
    '''
    if prefix == None: prefix = coulmn_name
    return pd.concat([df,pd.get_dummies(df['service_area_name'], prefix=prefix)],axis=1)

############################################################################################################################

def bin_function(feature=None,bins=10, return_bin_limits = False):
    '''
    This function takes a Numpy array or Pandas series as an input and divides it into bins with equal number or values in each bin
    Parameters : 
        feature           : Numpy array or Pandas series which will be divided into bins with equal number of values in them
        bins              : Number of bins
        return_bin_limits : This feature returns the bin limits in a dataframe if set to True
    Returns :
        Pandas dataframe with the original array along with the bin that it belongs to starting from 0 to bins-1;
        If return_bin_limits is set to True then it also return the bin limits as a dataframe
    '''
    print('NOTE: Lower Limit is exclusive (except minimum value) and Upper Limit is inclusive')
    feature_op = pd.DataFrame({'Input Values': feature })
    feature_op['Bin'] = pd.qcut(feature_op['Input Values'], bins, labels=False)
    if return_bin_limits != False :
        b = int(np.floor(100/bins)) 
        pct_lower = np.array([i/100 for i in range(0,100,b)])
        pct_upper = pct_lower + b/100
        pct_val = np.array([feature.quantile(i) for i in pct_upper])
        bins_op = pd.DataFrame({'Lower Limit':pct_lower*100,'Upper Limit':pct_upper*100, 'Bin Upper Limit Values': pct_val})
        return feature_op, bins_op
    else : return feature_op

############################################################################################################################

from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score, precision_recall_curve, auc
from sklearn.metrics import roc_curve
def classification_eval(y_true,y_pred,prob_thrs=0.5,return_conf_matrix = False,viz = True,):
    '''
    This function evaluates the performance of a classification model and returns evaluation metrics
    
    Parameters:
        y_true (Numpy array) : The ground truth labels given in the dataset
        y_pred (Numpy array) : Model prediction probability
        prob_thrs (numeric)  : Between 0 and 1; 0.5 by default. Predicted values >= threshold are considered as positive and below are negative.
        viz                  : Returns Visualization on the classifier performance along with the metrics
        return_conf_matrix   : Returns the confusion matrix as a dataframe, Default is False
    
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
        3. Categorical Cross Entropy : 
    
    '''
    if len(y_true) != len(y_true):
        print('Length of Actual and Predicted lists is unequal. Please check and correct the same!')
        return
    
    eval_metrics = dict()
    n = len(y_true)
    pred_df = pd.DataFrame({'y_true':y_true, 'y_pred':y_pred})
    pred_df['y_pred_r'] = np.vectorize(lambda x: 0 if (x < prob_thrs) else 1)(y_pred)
    cm = pd.crosstab(pred_df['y_pred_r'],pred_df['y_true'])
    cm.rename_axis('y_pred', inplace=True)
    normalized_cm=(cm-cm.min())/(cm.max()-cm.min())
    TP = cm[1][1]
    TN = cm[0][0]
    FP = cm[0][1]
    FN = cm[1][0]
    
    accuracy = (TP+TN)/(TP+TN+FP+FN)
    precision = TP / (TP+FP)
    recall = TP / (TP+FN)
    f1_score = 2 * (precision * recall) / (precision + recall)
    auc_score = roc_auc_score(y_true, y_pred)
    fpr, tpr, t1 = roc_curve(y_true, y_pred)
    p, r, t2 = precision_recall_curve(y_true, y_pred)
    aucpr = auc(r, p)
    
    eval_metrics['Length of Data'] = len(y_true)
    eval_metrics['True Positives'] = TP
    eval_metrics['True Negatives'] = TN
    eval_metrics['False Positives'] = FP
    eval_metrics['False Negatives'] = FN
    eval_metrics['Accuracy'] = accuracy
    eval_metrics['Precision'] = precision
    eval_metrics['Recall'] = recall
    eval_metrics['F1 Score'] = f1_score
    eval_metrics['AUC Score'] = auc_score
    eval_metrics['AUCPR Score'] = aucpr

    
    # Visualizations
    if viz == True:
        fig = plt.figure(figsize=(16,12))
        fig.suptitle('Classification Results')
        
        #Plotting confusion matrix
        ax1 = fig.add_subplot(2, 2, 1)
        ax1 = sns.heatmap(cm, annot=True, fmt='g', ax=ax1)
        # labels, title and ticks
        ax1.set_xlabel('True labels')
        ax1.set_ylabel('Predicted labels')
        ax1.set_title('Confusion Matrix')
        
        #Plotting the seperation between the predicted values
        ax2 = fig.add_subplot(2, 2, 2)
        ax2 = sns.kdeplot(pred_df[pred_df['y_true'] == 0]['y_pred'], shade = True, label = "0")
        ax2 = sns.kdeplot(pred_df[pred_df['y_true'] == 1]['y_pred'], shade = True, label = "1")
        ax2.set_title('Predictions KDE')
        ax2.set_xlabel('Predicted Probability')
        
        #Plotting the ROC-AUC Curve
        ax3 = fig.add_subplot(2, 2, 3)
        ax3 = sns.lineplot(x = fpr, y = tpr)
        ax3.set_title('ROC-AUC Curve')
        ax3.set_xlabel('FPR')
        ax3.set_ylabel('TPR')
        ax3.set_xlim([-0.05, 1.05])
        ax3.set_ylim([-0.05, 1.05])
        ax3.text(-0.02,1,'AUC score: {:0.3f}'.format(auc_score))
        
        #Plotting the Precision Recall Curve
        ax4 = fig.add_subplot(2, 2, 4)
        ax4 = sns.lineplot(x = r, y = p)
        ax4.set_title('Precision-Recall Curve')
        ax4.set_xlabel('Recall')
        ax4.set_ylabel('Precision')
        ax4.set_xlim([-0.05, 1.05])
        ax4.set_ylim([-0.05, 1.05])
        ax4.text(-0.02,1,'AUCPR score: {:0.3f}'.format(aucpr))
        
    #Converting to Dataframe
    eval_metrics = pd.DataFrame([eval_metrics],columns=eval_metrics.keys()).T
    eval_metrics.columns = ['Value']
    eval_metrics.index.set_names('Evaluation Metric',inplace=True)
    eval_metrics['Value'] = eval_metrics['Value'].apply(lambda x: '%.2f' % x)
    
    if return_conf_matrix == False: return eval_metrics
    else: return eval_metrics, cm

