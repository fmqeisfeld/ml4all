import numpy as np
import pandas as pd
import json 
from sklearn import preprocessing
from datetime import datetime, timedelta
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder
import pickle

## tree imports
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn import tree
from sklearn.ensemble import IsolationForest

## LR imports
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
#from sklearn.metrics import classification_report

## SHAP
from shap import TreeExplainer, summary_plot

## ASSOC
from mlxtend.frequent_patterns import apriori, fpgrowth, association_rules
from sklearn.preprocessing import OneHotEncoder

##ARIMA
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.stattools import adfuller # adf-test
from statsmodels.tsa.stattools import acf, pacf
from scipy.stats import norm
from pandas.core.tools.datetimes import _guess_datetime_format_for_array

### LIGHT GBM
import lightgbm as lgb 
from bayes_opt import BayesianOptimization
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error

## clustering 
import hdbscan

## SVC
from sklearn import svm
from sklearn.metrics import classification_report, multilabel_confusion_matrix
from sklearn.preprocessing import StandardScaler

## SVR
from sklearn import svm
from sklearn.metrics import r2_score

####################################################################################
# Verteiler
####################################################################################
def myMLmodel(infilepath, targetname,userinstance,typeof='0'):    
    if typeof =='FC' or typeof == 'FR':
        return myTree(infilepath,targetname,typeof,userinstance=userinstance)
    elif typeof=='LR':
        return myLR(infilepath,targetname,typeof,userinstance=userinstance)
    elif typeof=='AN':        
        return myAN(infilepath,targetname,typeof,userinstance=userinstance)
    elif typeof=='AS':        
        return myAS(infilepath,targetname,typeof,userinstance=userinstance)        
    elif typeof=='KO':
        return myKO(infilepath,targetname,typeof,userinstance=userinstance)        
    elif typeof=='KC':
        return myCluster(infilepath,targetname,typeof,userinstance=userinstance)
        #return {'success':False, 'msg': 'Die Clusteranalyse ist bisher noch nicht implementiert.' , 'outfile':None}
    elif typeof=='ARI':
        return myARIMA(infilepath,targetname,typeof,userinstance=userinstance)        
    elif typeof=='GBM':
        return myLightGBM(infilepath,targetname,typeof,userinstance=userinstance)
    elif typeof=='SVC':
        return mySVC(infilepath,targetname,typeof,userinstance=userinstance)        
    elif typeof=='SVR':
        return mySVR(infilepath,targetname,typeof,userinstance=userinstance)                

####################################################################################
# Cluster
####################################################################################
def myCluster(infilepath, targetname, typeof,userinstance):
    #df=pd.read_csv(infilepath,delimiter=',')

    now= datetime.now()
    fout = 'CLS' +'%s-%s-%s-%s-%s.json' % (now.day,now.month,now.year,now.hour,now.minute) 


    df = pd.read_pickle(infilepath,compression='gzip')

    # dont't use datetime columns here
    df = df.select_dtypes(exclude=['datetime'])

    df_num = df.select_dtypes(exclude=["object"])
    df_obj = df.select_dtypes(include=["object"])

    oe=OrdinalEncoder()
    oe.fit(df_obj)
    arr_obj_enc = oe.transform(df_obj)

    numeric_cols = df_num.shape[1]

    df_obj_enc = pd.DataFrame(data = arr_obj_enc, columns=df_obj.columns)
    df_merged = pd.concat([df_num.reset_index(drop=True),df_obj_enc.reset_index(drop=True)],axis=1)
    hdb = hdbscan.HDBSCAN(min_samples=10,).fit_predict(df_merged)

    clusters=dict() # enthält indices für jeden cluster
    for i in np.unique(hdb):
        indices = np.where(hdb == i)
        clusters[i] = indices
        #print(indices)

    df_merged_inv = pd.concat([df_num,df_obj],axis=1)
    summary = dict()

    for i in clusters:
        cur_cluster = dict()
        
        samples = len(clusters[i][0])    
        indices = clusters[i][0]
        rows = df_merged_inv.iloc[indices]
        
        cur_cluster['samples'] = samples
        
        for col in df_merged_inv:
            detail_dict=dict()
            if rows[col].dtype != 'object':
                minval = rows[col].min()
                maxval = rows[col].max()
                medval = rows[col].median()
                hist,bin_edges = np.histogram(rows[col],bins=5)
                bin_edges=bin_edges.round(decimals=2)
                
                detail_dict['type'] = 0
                detail_dict['min']=round(minval,2)
                detail_dict['med']=round(medval,2)
                detail_dict['max']=round(maxval,2)
                detail_dict['hist'] = hist.tolist()
                detail_dict['bin_edges'] = bin_edges.tolist()
                
            else:
                detail_dict['type'] = 1
                classes = rows[col].unique()            
                #hist = np.array(rows[col].value_counts().to_list())
                hist = rows[col].value_counts().to_list()
                detail_dict['classes'] = classes.tolist()
                detail_dict['hist'] = hist
            
            cur_cluster[col] = detail_dict
                
        summary[str(i)] = cur_cluster            

    with open(fout, 'w') as f:
        json_data = json.dump(summary, f, indent=4)

    return {'success':True, 'msg':None, 'outfile':fout}        

####################################################################################
# Entscheidungsbaum
####################################################################################
def myTree(infilepath, targetname, typeof,userinstance):
    #df=pd.read_csv(infilepath,delimiter=',')
    df = pd.read_pickle(infilepath,compression='gzip')

    # dont't use datetime columns here
    df = df.select_dtypes(exclude=['datetime'])

    Y = df[targetname]
    df.drop(columns=[targetname],inplace=True)
    df_dummy = pd.get_dummies(df,prefix_sep='_-_')


    
    ### validation ###
    if typeof == 'FC':  #classification
        if Y.dtype != 'object':
            return {'success':False, 'msg': 'Bei der Entscheidungsbaum-Klassifizierung muss die Zielgröße kategorisch sein.' , 'outfile':None}
    else: #regression
        if Y.dtype == 'object':
            return {'success':False, 'msg': 'Bei der Entscheidungsbaum-Regression  muss die Zielgröße numerisch sein.' , 'outfile':None}


    if Y.dtype=='object':
        #Y=pd.get_dummies(Y) #onehit macht alles kaputt. 
        le = preprocessing.LabelEncoder()
        Y = le.fit(Y).transform(Y)

    def generator(clf, features, labels,original_features, node_index=0,side=0,prev_index=0):

        node = {}
        if clf.tree_.children_left[node_index] == -1:  # indicates leaf
        
            #mod
            node['type']='leaf'
            #node['value'] = clf.tree_.value[node_index, 0].tolist()
            node['error'] = '%.1f' % np.float64(clf.tree_.impurity[node_index]).item()
            node['samples'] = str(clf.tree_.n_node_samples[node_index])
            ###
            
            count_labels = zip(clf.tree_.value[node_index, 0], labels)
                                        
            node['side'] = 'left' if side == 'l' else 'right'                              
            feature = features[clf.tree_.feature[prev_index]]
            threshold = clf.tree_.threshold[prev_index]
            
            node['label']='FOO'    
            node['value']='BAR'
            
            if node_index == 0:
                node["value"] = 'Root >'
                node["label"]=targetname
            elif ('_-_' in feature) and (feature not in original_features):
                
                node['label'] =  '{} = {}'.format(feature.split('_-_')[0], feature.split('_-_')[1] ) if side == 'r' else '{} != {}'.format(feature.split('_-_')[0], feature.split('_-_')[1] )  
                node['type'] = 'split' #'categorical'
            else:
                node['label'] = '{} > {}'.format(feature, round(threshold,2) ) if side == 'r' else '{} <= {}'.format(feature, round(threshold,2) ) 
                node['type'] = 'split'
            
            left_index = clf.tree_.children_left[node_index]
            right_index = clf.tree_.children_right[node_index]
            
            node['size'] = sum (clf.tree_.value[node_index, 0])

            #my
            if typeof=='FC':
                vals = clf.tree_.value[node_index]
                maxindex=np.argmax(vals)
                maxval = np.max(vals)
                conf = maxval/np.sum(vals)
                # node['class'] = '%s' % (str(maxindex))
                node['class'] = '%s' % (str(le.classes_[maxindex]))
                node['conf'] = '%.2f' % conf
            else: #zweckentfremdet bei regression. weniger änderungen im js nötig
                node['class'] = '%.2f' % (clf.tree_.value[node_index].squeeze())                
           
        else:

            count_labels = zip(clf.tree_.value[node_index, 0], labels)
            #node['pred'] = ', '.join(('{} of {}'.format(int(count), label)
            #                          for count, label in count_labels))
                                        
            node['side'] = 'left' if side == 'l' else 'right'                              
            feature = features[clf.tree_.feature[prev_index]]
            threshold = clf.tree_.threshold[prev_index]
            node['type']='split'
            node['error'] = '%.1f' % np.float64(clf.tree_.impurity[node_index]).item()
            node['samples'] = str(clf.tree_.n_node_samples[node_index])
            
            if typeof=='FC':
                vals = clf.tree_.value[node_index]
                maxindex=np.argmax(vals)
                maxval = np.max(vals)
                conf = maxval/np.sum(vals)
                # node['class'] = '%s' % (str(maxindex))
                node['class'] = '%s' % (str(le.classes_[maxindex]))
                node['conf'] = '%.2f' % conf
            else: #zweckentfremdet bei regression. weniger änderungen im js nötig
                node['class'] = '%.1f' % (clf.tree_.value[node_index].squeeze())        

            node['label']='FOO'    
            node['value']='BAR'
            
            if node_index == 0:
                node["value"] = 'Root >'
                node["label"]=targetname
            elif ('_-_' in feature) and (feature not in original_features):
                
                node['label'] =  '{} = {}'.format(feature.split('_-_')[0], feature.split('_-_')[1] ) if side == 'r' else '{} != {}'.format(feature.split('_-_')[0], feature.split('_-_')[1] )  
                #node['type'] = 'categorical'
            else:
                node['label'] = '{} > {}'.format(feature, round(threshold,2) ) if side == 'r' else '{} <= {}'.format(feature, round(threshold,2) ) 
                #node['type'] = 'numerical'
            
            left_index = clf.tree_.children_left[node_index]
            right_index = clf.tree_.children_right[node_index]
            
            node['size'] = sum (clf.tree_.value[node_index, 0])
            node['children'] = [generator(clf, features, labels, original_features, right_index,'r',node_index),
                                generator(clf, features, labels, original_features, left_index,'l',node_index)]
                            
        
        return node


    if typeof == 'FC':
        clf = DecisionTreeClassifier(max_leaf_nodes=12,criterion='entropy',max_depth=5)
    else: # FR
        clf =  DecisionTreeRegressor(max_depth=5, max_leaf_nodes=12)  # crit = mse per default, eigentlich reg 

    clf.fit(df_dummy, Y)

    features = df.columns.to_list()
    io=generator(clf, df_dummy.columns,np.unique(Y),features)
    
    now= datetime.now()
    fout = 'FC_' +'%s-%s-%s-%s-%s.json' % (now.day,now.month,now.year,now.hour,now.minute) 
    with open(fout, 'w') as outfile:
        json.dump(io, outfile, indent=4)

    return {'success':True, 'msg':None, 'outfile':fout}

####################################################################################
#   Logistische Regression
#####################################################################################
def myLR(infilepath, targetname, typeof,userinstance):    
    df = pd.read_pickle(infilepath, compression='gzip')

    # dont't use datetime columns here
    df = df.select_dtypes(exclude=['datetime'])

    target=targetname
    now= datetime.now()
    fout = 'LR_' +'%s-%s-%s-%s-%s.json' % (now.day,now.month,now.year,now.hour,now.minute) 
    
    ###############
    C= userinstance.logreg_C
    max_iter=userinstance.logreg_maxiter
    test_size = userinstance.logreg_testsize
    ##########################

    # after cleaning, collect target classes
    target_classes = df[target].unique()

    ### validation ###
    if df[target].dtype != 'object':
        return {'success':False, 'msg': 'Dieses Modell ist ein Klassifizierungs-Verfahren. Die Zielgröße muss kategorisch sein.' , 'outfile':None}

    if df[target].dtypes=='object':
        Y = df[target].astype('category').cat.codes
    else:
        Y=df[target]

    df2=df.drop(columns=[target]) 
    df_scaled = preprocessing.scale(df2.select_dtypes(exclude=["object"])) #,"int64"]))
    mean=df2.mean().to_list()
    std =df2.std().to_list()
    df_onehot = pd.get_dummies(df2.select_dtypes(include=["object"]),)
    X=np.concatenate((df_scaled,df_onehot.to_numpy()),axis=1)

    labels1=df2.select_dtypes(exclude=["object"]).columns.to_list()
    labels2=df2.select_dtypes(include=["object"]).columns.to_list()
    labels_all= labels1 + labels2    

    onehotcols = df_onehot.columns.to_list() 
    features_range={}
    features_type={}
    
    # collect labels
    for i in labels_all:
        #categorical or numeric?
        if df2[i].dtypes=='object': # or len(df3[i].unique())<=cat_thresh_upper:
            header=df_onehot.columns.to_list()                
            classes = [l.split('_')[1] for l in header if l.startswith(i)]    
            features_range[i]=classes
            features_type[i]='ABC'
        else:
            xmin = df2[i].min()
            xmax = df2[i].max()
            features_range[i]=[xmin,xmax] #np.linspace(xmin,xmax,10).tolist()
            if df2[i].dtypes=='int64':
                features_type[i]='int'
            else:
                features_type[i]='float'    

    # METRICS
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=0)    
    #unbedingt C mit rein sonst overfitting
    clf = LogisticRegression(solver='lbfgs',max_iter=max_iter,multi_class='multinomial',C=C)
    clf.fit(X_train,y_train,)
    pred = clf.predict(X_test)

    confmat = multilabel_confusion_matrix(y_test,pred).tolist()
    report=classification_report(y_test,pred,target_names=target_classes,output_dict=True,zero_division=0)         
    
    # PRODUCTioN
    clf.fit(X,Y)

    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, np.float32) or isinstance(obj, np.float64): 
                return float(obj)
            return json.JSONEncoder.default(self, obj)    

    if len(clf.coef_) > 1: #multiclass
        coeffs = clf.coef_
        intercepts = clf.intercept_
    else: #binary classification        
        coeffs = np.array([-clf.coef_[0], clf.coef_[0]])
        intercepts = np.array([-clf.intercept_, clf.intercept_]).squeeze()


    json_dump = json.dumps({'coeffs': coeffs, 'intercepts': intercepts,
                            'features_range':features_range, 'features_type':features_type,
                            'mean':mean, 'std':std,
                            'target_classes':target_classes,
                            'target_name': target,
                            'C':clf.C,
                            'max_iter':max_iter,
                            'optimloops':0,
                            'test_size':test_size,
                            'confmat':confmat,
                            'report':report,                            
                        }, cls=NumpyEncoder,indent=4)
    f = open(fout, "w")
    f.write(json_dump)
    f.close()                        

    return {'success':True, 'msg':None, 'outfile':fout}    

####################################################################################
#   Anomalie
####################################################################################
def myAN(infilepath, targetname, typeof,userinstance):    
    df = pd.read_pickle(infilepath, compression='gzip')

    # dont't use datetime columns here
    df = df.select_dtypes(exclude=['datetime'])

    target=targetname
    now= datetime.now()
    fout = 'AN_' +'%s-%s-%s-%s-%s.pkl' % (now.day,now.month,now.year,now.hour,now.minute) 

    df_num = df.select_dtypes(exclude=["object"])
    df_obj = df.select_dtypes(include=["object"])

    ordenc=OrdinalEncoder()
    ordenc.fit(df_obj)
    df_obj_enc = ordenc.transform(df_obj)

    headers = df.select_dtypes(exclude=["object"]).columns.to_list()
    headers += df_obj.columns.to_list()

    X=np.concatenate((df_num,df_obj_enc),axis=1)

    clf = IsolationForest(random_state=0,n_estimators=100, max_features=X.shape[1])
    clf.fit(X)

    shap_values = TreeExplainer(clf).shap_values(X)

    cols = shap_values.shape[1]
    shap_mean = list()

    for i in range(cols):
        shap_mean.append(np.mean(np.abs(shap_values[:,i])))

    top_anomalies = dict(zip(headers, shap_mean))    
    top_anomalies_sorted  = {k: v for k, v in sorted(top_anomalies.items(), key=lambda item: item[1],reverse=True)}

    top5 = list()
    offset = df_num.shape[1]

    for k in top_anomalies_sorted:
        index_in_header = headers.index(k)
        max_indices = np.abs(shap_values[:,index_in_header]).argsort()[-5:][::-1]
        #max_bins = min(50,len(np.unique(X[:,index_in_header])))        

        #invtrafo labels
        X_invtrafo =X[max_indices].copy().astype(np.str_)
        X_invtrafo[:,offset:] = ordenc.inverse_transform(X[max_indices,offset:])

        # wichtig: Ich transponiere die Feature-Zeilen, um sie später in django leichter auslesen zu können
        top5.append({'feature':k, 'score':top_anomalies[k], 'top5':X_invtrafo.T})

    top5.append(headers)
    with open(fout, 'wb') as f:
        pickle.dump(top5, f)

    return {'success':True, 'msg':None, 'outfile':fout}    

####################################################################################
#   Assoziation
####################################################################################
def myAS(infilepath, targetname, typeof,userinstance):
    df = pd.read_pickle(infilepath,compression='gzip')

    # dont't use datetime columns here
    df = df.select_dtypes(exclude=['datetime'])


    df_num = df.select_dtypes(exclude=["object"])
    df_obj = df.select_dtypes(include=["object"])

    now= datetime.now()
    fout = 'AS_' +'%s-%s-%s-%s-%s.json' % (now.day,now.month,now.year,now.hour,now.minute)

    nbins=10
    headers_num=list()
    vals_num=list()
    for curcol in df_num:
        dfmax = df_num[curcol].max()
        dfmin = df_num[curcol].min()
        dfstep = (dfmax-dfmin)/nbins
        dfrange = np.arange(dfmin,dfmax,dfstep)
        
        foo = np.round((df_num[curcol]-dfmin)/dfstep)
        foo[foo==nbins]=nbins-1
        vals_num.append(foo)
        
        for i in range(0,nbins):
            bar = '%.2f < %s <= %.2f' % (dfrange[i], curcol, dfrange[i]+dfstep)
            if (foo == i).any() :
                headers_num.append(bar)

    enc_num = OneHotEncoder(sparse=False)
    onehot_num = enc_num.fit(pd.concat(vals_num,axis=1)).transform(pd.concat(vals_num,axis=1))

    enc_obj = OneHotEncoder(sparse=False)
    enc_obj.fit(df_obj)
    onehot_obj=enc_obj.transform(df_obj)                

    headers_obj= list()
    counter=0
    for idx,i in enumerate(df_obj.columns):
        for n in range(0,len(enc_obj.categories_[idx])):        
            headers_obj.append('{0}={1}'.format(i,enc_obj.categories_[idx][n]))        
        counter += len(enc_obj.categories_[idx])  

    headers_all = headers_obj + headers_num
    df_merged = pd.DataFrame(data = np.concatenate((onehot_obj, onehot_num),axis=1),columns=headers_all)
    ##############################
    # APRIORI und ASSOC RULES
    ###########################

    frequent_items= apriori(df_merged, use_colnames=True, min_support=0.2)
    rules = association_rules(frequent_items, metric="confidence", min_threshold=0.3,)

    rules_dropped = rules.drop_duplicates(subset=['support'])

    rules_dropped['num_antecedents'] = rules_dropped['antecedents'].apply(lambda x: len(x))
    rules_dropped['num_consequents'] = rules_dropped['consequents'].apply(lambda x: len(x))
    rules_dropped.sort_values(by=['support'],ascending=False,inplace=True)

    antec_list = (rules_dropped["antecedents"].apply(lambda x: ', '.join(list(x))).astype("unicode")).to_list()

    conseq_list = (rules_dropped["consequents"].apply(lambda x: ', '.join(list(x))).astype("unicode")).to_list()

    support_list = (rules_dropped['support'].map('{:,.2f}'.format)).to_list()

    conf_list = (rules_dropped['confidence'].map('{:,.2f}'.format)).to_list()

    lift_list = (rules_dropped['lift'].map('{:,.2f}'.format)).to_list()

    lev_list = (rules_dropped['leverage'].map('{:,.2f}'.format)).to_list()

    num_antec_list = rules_dropped['num_antecedents'].to_list()

    num_conseq_list = rules_dropped['num_consequents'].to_list()    

    json_dict = {'antecedents': antec_list,
             'consequents': conseq_list,
             'support': support_list, 
             'confidence':conf_list,
             'lift': lift_list,
             'leverage': lev_list, 
             'num_antec':num_antec_list,
             'num_conseq':num_conseq_list}

             
    with open(fout, 'w') as f:
        json_data = json.dump(json_dict, f, indent=4)      

    return {'success':True, 'msg':None, 'outfile':fout}                
####################################################################################
#   Korrelation
####################################################################################
def myKO(infilepath, targetname, typeof,userinstance):
    df = pd.read_pickle(infilepath,compression='gzip')

    # dont't use datetime columns here
    df = df.select_dtypes(exclude=['datetime'])

    df_num = df.select_dtypes(exclude=["object"])
    df_obj = df.select_dtypes(include=["object"])

    now= datetime.now()
    fout = 'KO_' +'%s-%s-%s-%s-%s.csv' % (now.day,now.month,now.year,now.hour,now.minute)

    oe=preprocessing.OrdinalEncoder()
    oe.fit(df_obj)
    arr_obj_enc = oe.transform(df_obj)

    df_obj_enc = pd.DataFrame(data = arr_obj_enc, columns=df_obj.columns)

    df_merged = pd.concat([df_num.reset_index(drop=True),df_obj_enc.reset_index(drop=True)],axis=1)

    #corr_table = df_merged.astype(float).corr()
    ####################################################
    #   corr hat neben pearson auch weitere methoden
    ####################################################
    corr_table = df_merged.corr().fillna(0)
    for i in corr_table:
        corr_table[i] = corr_table[i].map('{:,.2f}'.format)

    out=list()
    out.append('group,variable,value')
    for col in corr_table:
        for row in corr_table[col].index:
            val = corr_table[col][row]
            tmp = '%s,%s,%s' % (col,row,val)
            #print(tmp)
            out.append(tmp)    

    with open(fout, 'w') as f:
        for item in out:
            f.write('%s\n' % item)            

    return {'success':True, 'msg':None, 'outfile':fout} 

####################################################################################
#   ARIMA
#####################################################################################
def myARIMA(infilepath, targetname, typeof,userinstance):    
    df = pd.read_pickle(infilepath, compression='gzip')

    target=targetname
    now= datetime.now()
    fout = 'ARI_' +'%s-%s-%s-%s-%s.json' % (now.day,now.month,now.year,now.hour,now.minute)     

    max_order =  userinstance.arima_maxpq # maximum p and q order
    optim_iter = userinstance.arima_optimloops # 2
    forecast_window = userinstance.arima_forecast
    maxdiff = 3
    test_size = userinstance.arima_testsize
    difforder=-1
    #test_size = 0.5
    

    warnings=list()

    if df[target].dtype =='object':
        return {'success':False, 'msg':'Das ARIMA-Modell akzeptiert nur numerische Zielgrößen','outfile':None}

    datetime_col = None
    guessed = None

    for i in df.columns:
        if 'datetime' in df[i].dtype.__str__():
            datetime_col=i
            #der rest ist nicht nötig (und führt zu fehler), da das datum bereits beim einlesen formatiert wurde
            #guessed = _guess_datetime_format_for_array(df[i].to_numpy())
            #if guessed != None:
            #    #print('%s is datetime' % (i))
            #    datetime_col=i
            #    break
        else:
            print('%s is not a valid datime-obj' % (i))

    if datetime_col==None:
        return {'success':False,
                'msg': 'Es konnte keine Zeitachse festgestellt werden. Bitte eine gültige Formatierung gewährleisten',
                'outfile':None}        
    else:
        #inferred_freq= pd.infer_freq(df[datetime_col])
        #df[datetime_col]=pd.to_datetime(df[datetime_col],format=guessed,unit=inferred_freq)
        df[datetime_col]=pd.to_datetime(df[datetime_col]) #,format=guessed)
        df.sort_values(by=datetime_col,inplace=True)
        df.set_index(datetime_col,inplace=True)
        dfs=pd.Series(df[target])

    ## check for missing dates
    inferred_freq = pd.infer_freq(dfs.index)
    if inferred_freq==None:
        warnmsg = "Es konnte keine Periodizität erkannt werden. Dies kann zu Folgefehlern, schlechten Prognosen und falschen Darstellungen führen"
        print(warnmsg)
        warnings.append(warnmsg)

    else:
        #print("Erkannte Periodizität: %s" % inferred_freq)
        full_range = pd.date_range(start=dfs.index[0],end=dfs.index[-1],freq=inferred_freq)        
        missing=full_range.difference(dfs.index)
        missing_dates= len(missing)
        if missing_dates > 0:
            return {'success':False,
                'msg': 'Der Datensatz enthältz fehlende Einträge auf der Zeitachse',
                'outfile':None}   

    # check stationarity
    for i in range(0,maxdiff+1):
        result = adfuller(np.diff(dfs.values,n=i))
    
        if result[0] < result[4]['1%']:
            difforder=i
            break
    
    # derive p-parameter
    if difforder > 0:
        ydiff = dfs
        for i in range(0,difforder):
            ydiff=ydiff.diff().dropna()

        #peaks_pacf,conf_pacf = pacf(dfs.diff(periods=difforder).dropna(),alpha=.05) # give also confidence intervals
        peaks_pacf,conf_pacf = pacf(ydiff,alpha=.05) # give also confidence intervals
    else:
        peaks_pacf,conf_pacf = pacf(dfs,alpha=.05) # give also confidence intervals

    
    nlags=len(peaks_pacf)    
    nobs = len(dfs)
    # compute conf. bounds
    varacf = np.ones(nlags)/nobs
    peaks = peaks_pacf
    varacf[0]=0
    varacf[1]=1/nobs
    varacf[2:] *= 1+2*np.cumsum(peaks[1:-1]**2)
    alpha = 0.05
    interval = norm.ppf(1 - alpha / 2.) * np.sqrt(varacf)    
    interval_pacf= interval.copy() # for output

    both = np.concatenate((np.abs(peaks),interval)).reshape(2,len(peaks)).T
    crit_indices_orig = np.where(both[:,0] > both[:,1])[0]
    if len(crit_indices_orig > 1): # nur dann ist es den aufwand wert. 0ter peak zählt nicht
        min_p_orig = max(crit_indices_orig) # diesen lag noch mitnehmen

        # jetzt vgl mit exp-fit (ab max-peak)
        all_indices = np.arange(1,len(peaks))
        y = np.abs(peaks[all_indices])
        # fitte ab dem max peak weil die vorherigen können auch kleiner sein!
        maxindex = np.argmax(y)
        all_indices = all_indices[maxindex:]
        y = np.abs(peaks[all_indices])
        weights=np.sqrt(y) # weil sonst bias
        pfit = np.polyfit(np.log(all_indices),y,1,w=weights)
        fitted_all = np.polyval(pfit,np.log(all_indices))        
        #combinded fitted and confidence bound
        both_fitted = np.concatenate((fitted_all,interval[maxindex+1:])).reshape(2,len(fitted_all)).T
        crit_indices_fitted = np.where(both_fitted[:,0] < both_fitted[:,1])[0]  # below crit line
        if len(crit_indices_fitted)==0:
            min_p_fitted = len(peaks)-1
        else:    
            crit_indices_fitted = crit_indices_fitted + maxindex + 1
            min_p_fitted = min(crit_indices_fitted)

        # das kleinere von beiden ansätzen wählen
        min_p = min(min_p_orig,min_p_fitted)

        if max_order < min_p:
            warnings.append('Der ermittlete Autoregressions-Paramter = {0} muss in der kostenfreien Nutzung auf {1} beschränkt werden.'.format(min_p,max_order))

        min_p = min(min_p,max_order)
        p_order = min_p
    else:        
        p_order = 0 # kann nur der 0te peak sein 
    
    print('derived p-param:%d' %  (p_order))


    # same for q-parameter    
    if difforder > 0:
        peaks_acf,conf_acf = acf(dfs.diff(periods=difforder).dropna(),alpha=.05) # give also confidence intervals
    else:
        peaks_acf,conf_acf = acf(dfs,alpha=.05) # give also confidence intervals    

    nlags=len(peaks_acf)
    nobs = len(dfs)
    # compute conf. bounds
    varacf = np.ones(nlags)/nobs
    peaks = peaks_acf
    varacf[0]=0
    varacf[1]=1/nobs
    varacf[2:] *= 1+2*np.cumsum(peaks[1:-1]**2)
    alpha = 0.05
    interval = norm.ppf(1 - alpha / 2.) * np.sqrt(varacf)
    interval_acf= interval.copy()

    both = np.concatenate((np.abs(peaks),interval)).reshape(2,len(peaks)).T
    crit_indices_orig = np.where(both[:,0] > both[:,1])[0]
    if len(crit_indices_orig > 1): # nur dann ist es den aufwand wert. 0ter peak zählt nicht
        min_q_orig = max(crit_indices_orig) # diesen lag noch mitnehmen

        # jetzt vgl mit exp-fit (ab max-peak)
        all_indices = np.arange(1,len(peaks))
        y = np.abs(peaks[all_indices])
        # fitte ab dem max peak weil die vorherigen können auch kleiner sein!
        maxindex = np.argmax(y)
        all_indices = all_indices[maxindex:]
        y = np.abs(peaks[all_indices])
        weights=np.sqrt(y) # weil sonst bias
        pfit = np.polyfit(np.log(all_indices),y,1,w=weights)
        fitted_all = np.polyval(pfit,np.log(all_indices))        
        #combinded fitted and confidence bound
        both_fitted = np.concatenate((fitted_all,interval[maxindex+1:])).reshape(2,len(fitted_all)).T
        crit_indices_fitted = np.where(both_fitted[:,0] < both_fitted[:,1])[0]  # below crit line
        if len(crit_indices_fitted)==0:
            min_q_fitted = len(peaks)-1
        else:    
            crit_indices_fitted = crit_indices_fitted + maxindex + 1
            min_q_fitted = min(crit_indices_fitted)


        # das kleinere von beiden ansätzen wählen
        min_q = min(min_q_orig,min_q_fitted)
        if max_order < min_q:
            warnings.append('Der ermittlete Parameter für den gleitenden Durschnitt =  {0} muss in der kostenfreien Nutzung auf {1} beschränkt werden.'.format(min_q,max_order))

        min_q = min(min_q,max_order)
        q_order = min_q
    else:        
        q_order = 0 # kann nur der 0te peak sein 

    p_order-=1 
    q_order-=1 

    p_order=max(0,p_order)
    q_order=max(0,q_order)        

    train,test = train_test_split(dfs,test_size=test_size,shuffle=False)

    # grid-search mit window = optim_iter nach unten

    results_list=list()
    dmin=max(-1,difforder-optim_iter)
    pmin=max(-1,p_order-optim_iter)
    qmin=max(-1,q_order-optim_iter)

    best_AIC=5e5
    best_model=None

    for d in range(difforder,dmin,-1):
        for p in range(p_order,pmin,-1):
            for q in range(q_order,qmin,-1):
                print(p,d,q)
                try:
                    #fitted = ARIMA(train,order=(p_order+1,0,q_order+1)).fit(start_ar_lags=12) 
                    fitted = ARIMA(train,order=(p,d,q)).fit(start_ar_lags=18) 
                    results_list.append({'AIC':fitted.aic, 'p':p, 'd':d,'q':q})        
                    if fitted.aic < best_AIC:
                        best_AIC = fitted.aic
                        best_model = fitted                        
                except Exception as e:
                    print(e)

    if best_model==None:
        return {'success':False,
                'msg': 'Die Optimierungsschleife konnte kein passendes ARIMA-Modell finden.',
                'outfile':None}                

    
    # brauch ich eigentlich nur für REPORT
    optim_params = sorted(results_list,key=lambda k:k['AIC'])[0]
    pred = best_model.predict(start = test.index[0], end= test.index[-1])
    
    rmse = mean_squared_error(test,pred)**0.5
    r2 = r2_score(test,pred)

    # save as json
    forecast_len = len(test)+int(round(len(dfs) * forecast_window))
    forecast=best_model.forecast(steps=forecast_len)[0]

    # x-labels for chartjs
    full_range = pd.date_range(start=dfs.index[0],periods=len(train)+forecast_len,freq=inferred_freq)
    full_range = full_range.strftime(guessed).to_list()    

    # prepare dataset for original input
    padding = ['null']*(int(round(len(dfs) * forecast_window)))
    y_original = dfs.round(decimals=2).to_list() + padding    

    # prepare dataset for predicted
    padding =['null']*len(train)
    y_forecast = padding + forecast.round(decimals=2).tolist()

    # extra plots : Acf und pacf
    y_pacf = peaks_pacf.round(decimals=2).tolist()
    y_acf = peaks_acf.round(decimals=2).tolist()
    xacf = np.arange(0,len(y_pacf)).tolist()

    interval_acf_pos =  interval_acf.round(decimals=2).tolist()
    interval_acf_neg = (-interval_acf.round(decimals=2)).tolist() 
    interval_pacf_pos = interval_pacf.round(decimals=2).tolist()
    interval_pacf_neg = (-interval_pacf.round(decimals=2)).tolist()     


    if len(warnings)==0:
        warnings.append('Keine.')
    json_dict = {'x': full_range,
                'y1': y_original,
                'y2': y_forecast, 
                'y_axis':target,
                'xacf':xacf,
                'acf':y_acf,
                'pacf':y_pacf,
                'iacf_pos':interval_acf_pos,
                'iacf_neg':interval_acf_neg,
                'ipacf_pos':interval_pacf_pos,
                'ipacf_neg':interval_pacf_neg,                
                'warnings':warnings,
                'rmse':np.round(rmse,decimals=3),
                'r2':np.round(r2,decimals=3),
                'max_order':max_order,
                'optim_iter':optim_iter,
                'forecast_window':forecast_window,
                'test_size':test_size,
                } 

    with open(fout, 'w') as f:
        json_data = json.dump(json_dict, f, indent=4)       

    return {'success':True, 'msg':None, 'outfile':fout} 


####################################################################################
#   Light
#####################################################################################
def myLightGBM(infilepath, targetname, typeof,userinstance):                 

    df = pd.read_pickle(infilepath, compression='gzip')

    target=targetname
    now= datetime.now()
    fout = 'LGBM' +'%s-%s-%s-%s-%s.json' % (now.day,now.month,now.year,now.hour,now.minute)     

    ##################
    #  MODEL PARAMS
    ##################
    forecast_window = userinstance.lgbm_forecast
    maxdiff = 3
    difforder=-1
    opt_round= userinstance.lgbm_optimloops
    test_size = userinstance.lgbm_testsize

    if df[target].dtype =='object':
        return {'success':False, 'msg':'Das LGBM-Modell akzeptiert nur numerische Zielgrößen','outfile':None}

    datetime_col = None
    guessed = None

    for i in df.columns:
        if 'datetime' in df[i].dtype.__str__():
            datetime_col=i
        else:
            print('%s is not a valid datime-obj' % (i))

    if datetime_col==None:
        return {'success':False,
                'msg': 'Es konnte keine Zeitachse festgestellt werden. Bitte eine gültige Formatierung gewährleisten',
                'outfile':None}        
    else:
        #inferred_freq= pd.infer_freq(df[datetime_col])
        #df[datetime_col]=pd.to_datetime(df[datetime_col],format=guessed,unit=inferred_freq)
        df[datetime_col]=pd.to_datetime(df[datetime_col]) #,format=guessed)
        df.sort_values(by=datetime_col,inplace=True)
        df.set_index(datetime_col,inplace=True)
        dfs=pd.Series(df[target])

    ## check for missing dates
    inferred_freq = pd.infer_freq(dfs.index)
    if inferred_freq==None:
        warnmsg = "Es konnte keine Periodizität erkannt werden. Dies kann zu Folgefehlern, schlechten Prognosen und falschen Darstellungen führen"
        print(warnmsg)
        warnings.append(warnmsg)

    else:
        #print("Erkannte Periodizität: %s" % inferred_freq)
        full_range = pd.date_range(start=dfs.index[0],end=dfs.index[-1],freq=inferred_freq)        
        missing=full_range.difference(dfs.index)
        missing_dates= len(missing)
        if missing_dates > 0:
            return {'success':False,
                'msg': 'Der Datensatz enthältz fehlende Einträge auf der Zeitachse',
                'outfile':None}   

    # ACHTUNG: Könnte auch keinen Sinn machen wenn z.B. die freq minuten oder sekunden etc. sind
    df['date'] = df.index
    df['hour'] = df['date'].dt.hour
    df['dayofweek'] = df['date'].dt.dayofweek
    df['quarter'] = df['date'].dt.quarter
    df['month'] = df['date'].dt.month
    df['year'] = df['date'].dt.year
    df['dayofyear'] = df['date'].dt.dayofyear
    df['dayofmonth'] = df['date'].dt.day
    df['weekofyear'] = df['date'].dt.weekofyear    

    X = df[['hour','dayofweek','quarter','month','year',
            'dayofyear','dayofmonth','weekofyear']]

    y=df[target]

    # check stationarity
    _, _, y_train, _ = train_test_split(X,y,test_size=test_size, shuffle=False)
    for i in range(0,maxdiff+1):
        result = adfuller(np.diff(y_train,n=i))
    
        if result[0] < result[4]['1%']: 
            difforder=i
            break

    y_original = y
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=test_size, shuffle=False)

    # decompose data into seasonality, trend, residuals, etc. 
    if difforder > 0:
        y_train_series = pd.Series(y_train,index=y_train.index)
        decomp = seasonal_decompose(y_train_series, model='additive') # falls die peaks anwachsen -> multiplikativ !
        decomp_trend = decomp.trend.dropna()
        polyparams = np.polyfit(np.arange(0,len(decomp_trend)),decomp_trend,difforder)

        # trend subtrahieren, später wieder hinzuaddieren
        y = y-np.polyval(polyparams,np.arange(0,len(y)))
        # nicht vergessen wieder train und test-set zu erstellen
        _, _, y_train, y_test = train_test_split(X,y,test_size=test_size, shuffle=False)        


    def bayes_parameter_opt_lgb(X, y, init_round=15, opt_round=25, 
                                n_folds=5, random_seed=6, 
                                n_estimators=10000, learning_rate=0.02, 
                                output_process=False):
        # prepare data
        train_data = lgb.Dataset(data=X, label=y)
        # parameters
        def lgb_eval(num_leaves, feature_fraction, 
                    bagging_fraction, max_depth, 
                    lambda_l1, lambda_l2, min_split_gain, 
                    min_child_weight):
            
            params = {#'application':'regression_l1',
                    'num_iterations': n_estimators, 
                    'learning_rate':learning_rate, 
                    'early_stopping_rounds':50, 
                    #'metric':'auc',
                    'metric':'rmse',
                    'objective':'regression_l1',
                    'verbose':-1,
                    'max_bin':200                
                    }
            
            params["num_leaves"] = int(round(num_leaves))
            params['feature_fraction'] = max(min(feature_fraction, 1), 0)
            params['bagging_fraction'] = max(min(bagging_fraction, 1), 0)
            params['max_depth'] = int(round(max_depth))
            params['lambda_l1'] = max(lambda_l1, 0)
            params['lambda_l2'] = max(lambda_l2, 0)
            params['min_split_gain'] = min_split_gain
            params['min_child_weight'] = min_child_weight
            
            cv_result = lgb.cv(params, train_data,  
                            nfold=n_folds,                            
                            seed=random_seed, 
                            stratified=False, 
                            verbose_eval =None, 
                            shuffle=False,
                            metrics=['rmse'])                
                        
            return 1.0/(max(cv_result['rmse-mean'])* max(cv_result['rmse-mean']) ) 
    
        # range  of params   
        lgbBO = BayesianOptimization(lgb_eval,
                                    {'num_leaves': (8, 35),
                                    'feature_fraction': (0.1, 0.9),
                                    'bagging_fraction': (0.25, 1.0),
                                    'max_depth': (5, 50),
                                    'lambda_l1': (0, 1),
                                    'lambda_l2': (0, 1),
                                    'min_split_gain': (0.001, 0.1),
                                    'min_child_weight': (5, 50)}, 
                                    random_state=0)
        # optimize
        lgbBO.maximize(init_points=init_round, n_iter=opt_round)    
    
        # return parameters
        return lgbBO.res #['max'] #['max_params']


    opt_params = bayes_parameter_opt_lgb(X_train, y_train, 
                                        init_round=5, 
                                        opt_round=opt_round,  # TODO: hier anpassen
                                        n_folds=3, 
                                        random_seed=6,                                      
                                        n_estimators=100, 
                                        learning_rate=0.05
                                        )

    best_params=sorted(opt_params,key=lambda k:k['target'],reverse=True )[0]['params']
    best_params['num_leaves'] = int(round(best_params['num_leaves']))
    best_params['max_depth'] = int(round(best_params['max_depth']))
    best_params['verbose']=-1                                        

    lgb_traindata = lgb.Dataset(data=X_train, label=y_train)    
    lgbm = lgb.train(params=best_params,
                    train_set=lgb_traindata,
                    num_boost_round=1000, 
                    #valid_sets=[lgb_traindata,lgb_testdata],
                    )    

    # save as json        
    # prepare dataframe for prediction
    forecast_len = len(X_test)+int(round(len(X) * forecast_window))
    pred_range = pd.date_range(start=X_test.index[0],periods=forecast_len,freq=inferred_freq,)
    X_pred = pd.DataFrame(index=pred_range)
    X_pred['date'] = X_pred.index
    X_pred['hour'] = X_pred['date'].dt.hour
    X_pred['dayofweek'] = X_pred['date'].dt.dayofweek
    X_pred['quarter'] = X_pred['date'].dt.quarter
    X_pred['month'] = X_pred['date'].dt.month
    X_pred['year'] = X_pred['date'].dt.year
    X_pred['dayofyear'] = X_pred['date'].dt.dayofyear
    X_pred['dayofmonth'] = X_pred['date'].dt.day
    X_pred['weekofyear'] = X_pred['date'].dt.weekofyear #deprecated      
    del X_pred['date']
    
    # calc forecast
    forecast=lgbm.predict(X_pred)

    # add trend to forecast
    if difforder>0:
        xtmp = np.arange(len(X_train),len(X_train)+forecast_len)
        ytmp = np.polyval(polyparams,xtmp)
        forecast += ytmp
    
    # x-labels for chartjs
    full_range = pd.date_range(start=X.index[0],periods=len(X_train)+forecast_len,freq=inferred_freq)
    full_range = full_range.strftime(guessed).to_list()    

    # prepare dataset for original input
    padding = ['null']*(int(round(len(y) * forecast_window)))
    y_original = y_original.round(decimals=2).to_list() + padding
    #y_original = y.round(decimals=2).to_list() + padding     


    # prepare dataset for predicted
    padding =['null']*len(X_train)
    y_forecast = padding + forecast.round(decimals=2).tolist()

    # feature importance 
    importance_dict={}
    feat_importance = lgbm.feature_importance(importance_type='split')
    cols = X.columns
    for i in range(0,len(cols)):
        importance_dict[cols[i]]=feat_importance[i]

    # confidence bounds
    gbr = GradientBoostingRegressor()
    pred = lgbm.predict(X_test)
    err = (y_test-pred)**2
    gbr.fit(X_test,err)

    st_dev = gbr.predict(X_pred)
    st_dev[st_dev<0]=0
    st_dev=st_dev**0.5
    rmse = mean_squared_error(y_test,pred)**0.5
    r2 = r2_score(y_test,pred)

    mu = rmse
    sigma = st_dev
    Z = 1.96 # bei 95% conf interval cf https://www.mathsisfun.com/data/confidence-interval.htmln
    n = len(err)
    conf = mu +  Z * sigma #/n

    y_upper = padding + (forecast+conf).round(decimals=2).tolist()
    y_lower = padding + (forecast-conf).round(decimals=2).tolist()

    for i in importance_dict:
        importance_dict[i]=int(importance_dict[i])

    json_dict = {'x': full_range,
                'y1': y_original,
                'y2': y_forecast,
                 'importance':importance_dict,
                 'y_lower':y_lower,
                 'y_upper':y_upper,
                 'rmse':np.round(rmse,decimals=3),
                 'r2':np.round(r2,decimals=3),
                 'opt_round':opt_round,
                 'test_size':test_size,
                 'forecast_window':forecast_window,
                 'target':target,                 
                } 

    with open(fout, 'w') as f:
        json_data = json.dump(json_dict, f, indent=4)       

    return {'success':True, 'msg':None, 'outfile':fout} 


####################################################################################
#   SVM Klassifizierung
#####################################################################################
def mySVC(infilepath, targetname, typeof,userinstance):    
    df = pd.read_pickle(infilepath, compression='gzip')

    # dont't use datetime columns here
    df = df.select_dtypes(exclude=['datetime'])

    target=targetname
    now= datetime.now()
    fout = 'SVC_' +'%s-%s-%s-%s-%s.json' % (now.day,now.month,now.year,now.hour,now.minute) 
    
    ### validation ###
    if df[target].dtype != 'object':
        return {'success':False, 'msg': 'Dies ist ein Klassifizierungs-Verfahren. Die Zielgröße muss kategorisch sein.' , 'outfile':None}

    ### MODeL PARAMS ######
    max_iter= userinstance.svc_maxiter # 1000
    C=userinstance.svc_C
    kernel = userinstance.svc_kernel
    order = userinstance.svc_degree
    test_size = userinstance.svc_testsize

    y=df[target]
    ency = OrdinalEncoder().fit(y.to_numpy().reshape(-1,1))
    y_enc = ency.transform(y.to_numpy().reshape(-1,1))
    df = df.drop(columns=[target])

    df_obj = df.select_dtypes(include=object)
    df_num = df.select_dtypes(exclude=object)

    enc = OrdinalEncoder().fit(df_obj)
    X_obj_enc = enc.transform(df_obj)

    df_concat = pd.concat([df_num.reset_index(drop=True),
                           pd.DataFrame(data=X_obj_enc,columns=df_obj.columns).reset_index(drop=True)],
                           axis=1)

    scalar = StandardScaler()
    scalar.fit(df_concat)

    X=scalar.transform(df_concat)

    # only training
    clf = svm.SVC(gamma='scale', C=C,probability=False, random_state=0,
                kernel=kernel,decision_function_shape='ovo',class_weight='balanced', max_iter=max_iter)

    X_train, X_test, y_train, y_test = train_test_split(X, y_enc.squeeze(), test_size=test_size, random_state=0)    
    result = clf.fit(X_train, y_train)
    pred = clf.predict(X_test)    

    # actual fittting with proba
    # diesmal mit proba = True
    clf = svm.SVC(gamma='scale', C=C,probability=True, random_state=0,
                kernel=kernel,decision_function_shape='ovo',
                class_weight='balanced',max_iter=max_iter)    

    result = clf.fit(X, y_enc.squeeze())

    # json output
    sup_vecs = clf.support_vectors_
    dual_coefs = clf.dual_coef_
    gamma = clf._gamma
    intercept = clf.intercept_
    coef0 = clf.coef0

    coeffs = clf.dual_coef_.round(decimals=3 ).tolist()   
    intercepts = clf.intercept_.round(decimals=3).tolist()
    n_vecs = clf.n_support_.tolist()
    indices = clf.support_.tolist()
    vecs = clf.support_vectors_.tolist()
    probA = clf.probA_.tolist()
    probB = clf.probB_.tolist()
    m = clf.support_vectors_.shape[0]

    #confmat = multilabel_confusion_matrix(y_test,pred,labels=ency.categories_[0].tolist()).tolist()
    confmat = multilabel_confusion_matrix(y_test,pred).tolist()
    report=classification_report(y_test,pred,target_names=ency.categories_[0].tolist(),output_dict=True)     

    labels1=df.select_dtypes(exclude=["object"]).columns.to_list()
    labels2=df.select_dtypes(include=["object"]).columns.to_list()
    labels_all= labels1 + labels2    

    mean = df_concat.mean().to_list()  #X.mean(axis=0) #.to_list()
    std =df_concat.std().to_list()                
    df_onehot = pd.get_dummies(df.select_dtypes(include=["object"]),)

    # prepare feat_range and feat_types
    onehotcols = df_onehot.columns.to_list() 
    features_range={}
    features_type={}
  
    # collect labels
    for i in labels_all:
        #categorical or numeric?
        if df[i].dtypes=='object': # or len(df3[i].unique())<=cat_thresh_upper:
            header=df_onehot.columns.to_list()                
            classes = [l.split('_')[1] for l in header if l.startswith(i)]    
            features_range[i]=classes
            features_type[i]='ABC'
        else:
            xmin = df[i].min()
            xmax = df[i].max()
            features_range[i]=[xmin,xmax] #np.linspace(xmin,xmax,10).tolist()
            if df[i].dtypes=='int64':
                features_type[i]='int'
            else:
                features_type[i]='float'   

    #target_classes = y.unique().tolist()    
    target_classes = ency.categories_[0].tolist()

    json_dump = json.dumps({'coeffs': coeffs, 'intercepts': intercepts,
                            'gamma':gamma, 'coef0':coef0,
                            'features_range':features_range, 'features_type':features_type,
                            'mean':mean, 'std':std,
                            'n_vecs':n_vecs,
                            'indices':indices,
                            'vecs':vecs,
                            'm':m,
                            'probA':probA,
                            'probB':probB,
                            'kernel':kernel,
                            'order':order,
                            'C':clf.C,
                            'max_iter':max_iter,
                            'optimloops':0,
                            'test_size':test_size,
                            'confmat':confmat,
                            'report':report,
                            'target_classes':target_classes,
                            'target_name': target,                        
                        },indent=4)   

    f = open(fout, "w")
    f.write(json_dump)
    f.close()    

    return {'success':True, 'msg':None, 'outfile':fout}                     

####################################################################################
#   SVR Regression
#####################################################################################
def mySVR(infilepath, targetname, typeof,userinstance):    
    df = pd.read_pickle(infilepath, compression='gzip')

    # dont't use datetime columns here
    df = df.select_dtypes(exclude=['datetime'])

    target=targetname
    now= datetime.now()
    fout = 'SVR_' +'%s-%s-%s-%s-%s.json' % (now.day,now.month,now.year,now.hour,now.minute) 

    ### validation ###
    if df[target].dtype == 'object':
        return {'success':False, 'msg': 'Dies ist ein Regressions-Verfahren. Die Zielgröße muss numerisch sein.' , 'outfile':None}    

    ##############################
    # model params
    ##############################
    max_iter = userinstance.svr_maxiter
    C= userinstance.svr_C
    kernel = userinstance.svr_kernel
    order = userinstance.svr_degree
    test_size = userinstance.svr_testsize
    ######################

    y=df[target]
    df = df.drop(columns=[target])
    df_obj = df.select_dtypes(include=object)
    df_num = df.select_dtypes(exclude=object)

    enc = OrdinalEncoder().fit(df_obj)
    X_obj_enc = enc.transform(df_obj)
    df_concat = pd.concat([df_num.reset_index(drop=True),
                        pd.DataFrame(data=X_obj_enc,columns=df_obj.columns).reset_index(drop=True)],
                        axis=1)   

    scalar = StandardScaler()
    scalar.fit(df_concat) 

    X=scalar.transform(df_concat)

    labels1=df_num.columns.to_list()
    labels2=df_obj.columns.to_list()
    labels_all= labels1 + labels2        

    clf = svm.SVR(gamma='scale', C=C,
                kernel=kernel, max_iter=max_iter)

    result= clf.fit(X, y)    

    pred = clf.predict(X)
    r2 = r2_score(y,pred)
    err = (y-pred)
    mean_err = np.mean(err)

    histo = np.histogram(err,20)

    counts = histo[0].round(decimals=2).tolist()
    edges = histo[1].round(decimals=2).tolist()    

    # confidence intervals
    alpha = 0.95
    p = ((1.0-alpha)/2.0)* 100
    lower = np.percentile(err, p)
    p = (alpha+((1.0-alpha)/2.0))* 100    
    upper = np.percentile(err, p)
    
    mean = df_concat.mean().to_list()  #X.mean(axis=0) #.to_list()
    std =df_concat.std().to_list()

    df_onehot = pd.get_dummies(df.select_dtypes(include=["object"]),)
    onehotcols = df_onehot.columns.to_list() 
    features_range={}
    features_type={}

    # collect labels
    for i in labels_all:
        #categorical or numeric?
        if df[i].dtypes=='object': # or len(df3[i].unique())<=cat_thresh_upper:
            header=df_onehot.columns.to_list()                
            classes = [l.split('_')[1] for l in header if l.startswith(i)]    
            features_range[i]=classes
            features_type[i]='ABC'
        else:
            xmin = df[i].min()
            xmax = df[i].max()
            features_range[i]=[xmin,xmax] #np.linspace(xmin,xmax,10).tolist()
            if df[i].dtypes=='int64':
                features_type[i]='int'
            else:
                features_type[i]='float'  

    kernel = clf.kernel
    order = clf.degree
    gamma =clf._gamma

    coeffs = clf.dual_coef_[0].round(decimals=3 ).tolist()    
    intercepts = clf.intercept_[0].round(decimals=3).tolist()
    vecs = clf.support_vectors_.tolist()
    m = clf.support_vectors_.shape[0]

    json_dump = json.dumps({'coeffs': coeffs, 'intercepts': intercepts,
                            'gamma':gamma,
                            'features_range':features_range, 'features_type':features_type,
                            'mean':mean, 'std':std,
                            'vecs':vecs,
                            'm':m,
                            'kernel':kernel,
                            'order':order,
                            'C':clf.C,
                            'max_iter':max_iter,
                            'counts':counts,
                            'edges':edges,
                            'upper_err':np.round(upper,decimals=3),
                            'lower_err':np.round(lower,decimals=3),
                            'mean_err':np.round(mean_err,decimals=3),
                            #'evs':evs,
                            'r2':np.round(r2,decimals=3),
                            'optimloops':0,
                            'test_size':test_size,
                            'target_name': target,                        
                        },indent=4)

    f = open(fout, "w")
    f.write(json_dump)
    f.close()    

    return {'success':True, 'msg':None, 'outfile':fout}
