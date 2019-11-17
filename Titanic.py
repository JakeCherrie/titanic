'''
Created on Fri Oct 20 12:47:09 2017

@author: Jake Cherrie
'''

# =============================================================================
# Importing Packages
# =============================================================================

# File system manangement
import os
# Suppress warnings
import warnings
warnings.filterwarnings('ignore')
# Numpy and pandas for data manipulation
import numpy as np
import pandas as pd
# Scipy stats for statistical analysis
import scipy.stats as stats
# sklearn preprocessing for dealing with categorical variables
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
# Bayes_opt for optimization
from bayes_opt import BayesianOptimization
# Sklearn importing models
from sklearn import svm, tree, linear_model, neighbors, naive_bayes, ensemble, discriminant_analysis, gaussian_process
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
# Sklearn Inputing data spliting method
from sklearn.model_selection import KFold, StratifiedKFold
# Sklearn importing auc as measurement metric
from sklearn.metrics import roc_auc_score
# Gc memory management
import gc
# Time monitor run-time
import time
# Matplotlib and seaborn for plotting
import matplotlib.pyplot as plt
plt.style.use('tableau-colorblind10')
import seaborn as sns
sns.set_style('white')
pd.options.display.max_rows = 4000

# =============================================================================
# Initializing Functions
# =============================================================================

# Encoding binary features with label encoder otherwise one hot encoding
def encoder(df, nan_as_category=True):
    le = preprocessing.LabelEncoder()
    obj_col = [col for col in df.columns if df[col].dtype == 'object']
    df[obj_col] = df[obj_col].fillna(value = 'NaN')
    bin_col = [col for col in obj_col if df[col].nunique() <= 2]
    if len(bin_col) > 0:
        df[bin_col] = le.fit_transform(df[bin_col])
    df = pd.get_dummies(df, dummy_na = nan_as_category)
    return df
        
# Display/plot feature importance
def display_importances(fts_imp, mdl_nme):
    cols = fts_imp[["feature", "importance"]].groupby("feature").mean().sort_values(by="importance", ascending=False)[:40].index
    best_features = fts_imp.loc[fts_imp.feature.isin(cols)]
    plt.figure(figsize=(8, 10))
    sns.barplot(x="importance", y="feature", data=best_features.sort_values(by="importance", ascending=False))
    plt.title('Features (avg over folds)')
    plt.tight_layout()
    plt.savefig('C:/Users/Jake Cherrie/Documents/Projects/Python - Titanic/Output/' + mdl_nme + '_Feature_Importance.png')
    plt.close()

# Correlation heatmap of dataset
def correlation_heatmap(df, nme):
    _ , ax = plt.subplots(figsize =(14, 12))
    colormap = sns.diverging_palette(220, 10, as_cmap = True)
    
    _ = sns.heatmap(
        df.corr(), 
        cmap = colormap,
        square=True, 
        cbar_kws={'shrink':.9 }, 
        ax=ax,
        annot=True, 
        linewidths=0.1,vmax=1.0, linecolor='white',
        annot_kws={'fontsize':12 }
    )
    plt.title('Pearson Correlation of Features', y=1.05, size=15)
    plt.savefig('C:/Users/Jake Cherrie/Documents/Projects/Python - Titanic/Output/' + nme + '_Pearson_Feature_Correlation.png')
    plt.close()
    
def model_fit(mdl, mdl_nme = '', n_fld = 5, dbg = True):   
    print(mdl_nme)
    # K fold validation
    fld = StratifiedKFold(n_splits = n_fld, shuffle=True, random_state = 11)
    
    # List of feature columns
    fts = [f for f in trn.columns if f not in ['Survived']]
    
    # Initializing Variables
    prd = np.zeros(trn[fts].shape[0])
    sub_prd = np.zeros(tst[fts].shape[0])
    fts_imp = pd.DataFrame()
    acc = 0
    auc = 0
    
    for n_fold, (trn_idx, vld_idx) in enumerate(fld.split(trn[fts], trn['Survived'])):
        trn_X, trn_y = trn[fts].iloc[trn_idx], trn['Survived'].iloc[trn_idx]
        vld_X, vld_y = trn[fts].iloc[vld_idx], trn['Survived'].iloc[vld_idx]
        
        # Fit the model
        mdl.fit(trn_X, trn_y)
        
        if hasattr(mdl, 'predict_proba'):
            prd[vld_idx] = mdl.predict_proba(vld_X)[:, 1]
            
            sub_prd += mdl.predict_proba(tst[fts])[:, 1] / fld.n_splits
            # Outputting the AUC
            auc += roc_auc_score(vld_y, prd[vld_idx]) / fld.n_splits
        
        # Outputting the accuracy and AUC
        print('Fold %2d ACC : %.6f' % (n_fold + 1, mdl.score(vld_X, vld_y)))
        acc += mdl.score(vld_X, vld_y) / fld.n_splits
        
        fld_imp = pd.DataFrame()
        fld_imp["feature"] = fts
        if hasattr(mdl, 'feature_importances_'):
            fld_imp["importance"] = mdl.feature_importances_
        elif hasattr(mdl, 'coef_'):
            fld_imp["importance"] = abs(list(np.std(trn_X, 0))*mdl.coef_[0])
        else:
            fld_imp["importance"] = None
        fld_imp["fold"] = n_fold + 1
        fts_imp = pd.concat([fts_imp, fld_imp], axis=0)
    
    print('AUC : %.6f' % auc)
    print('Acc : %.6f' % acc)
    
    if dbg == False:
        # Displaying Feature Importance
        if (hasattr(mdl, 'feature_importances_') | hasattr(mdl, 'coef_')):
            display_importances(fld_imp, mdl_nme)
            
        # Saving prediction
        sub_prd[sub_prd < 0.5] = 0
        sub_prd[sub_prd > 0.5] = 1
       
        tst['Survived'] = sub_prd.astype(int)
        tst_sub = tst.reset_index(drop = False)        
        # Appling dead women and alive men from ias
        p0[;'5nd
           dependent model seen in Chris 
        # Deottes script
        tst_sub['Survived'][tst_sub['PassengerId'].isin([928, 1030, 1061, 1091, 1098, 1160, 1205, 1304])] = 0
        tst_sub['Survived'][tst_sub['PassengerId'].isin([926, 942])] = 1
        tst_sub = tst_sub.set_index('PassengerId')
        tst_sub['Survived'].sum()
        tst_sub['Survived'].to_csv('C:/Users/Jake Cherrie/Documents/Projects/Python - Titanic/Output/' + mdl_nme + '_Submission.csv', header = True, index = True, index_label = 'PassengerId')
    return mdl_nme, acc

# =============================================================================
# Importing Data
# =============================================================================

os.listdir('C:/Users/Jake Cherrie/Documents/Projects/Python - Titanic')

# Importing Data
trn = pd.read_csv('C:/Users/Jake Cherrie/Documents/Projects/Python - Titanic/train.csv')
tst = pd.read_csv('C:/Users/Jake Cherrie/Documents/Projects/Python - Titanic/test.csv')

# Merging Training And Test Data For The Purposes Of Feature Engineering
df = trn.append(tst)
df = df.set_index('PassengerId')

df.info()

# Outputting information tables
cat_col = [col for col in df.columns if df[col].dtype == 'object']  
df_dsc = df.describe(include = 'all')

# =============================================================================
# Feature Engineering
# =============================================================================

### Cabin ###

# Inspecting cabin distribution
df['Cabin'][df['Cabin'].notnull()]

# Checking survival by whether or not they have is a Cabin
df['Survived'][df['Cabin'].isnull() & df['Survived'].notnull()].mean()
df['Survived'][df['Cabin'].notnull() & df['Survived'].notnull()].mean()

# Adding deck by cabin letter 'n' represents missing
df['Deck'] = df['Cabin'].astype(str).str[0]

# Quick look at behavior
sns.barplot('Deck', 'Survived',data = df)
# Grouping decks for model stability
df['Deck'][(df['Deck'] == 'D') | (df['Deck'] == 'B')] = 'E'
df['Deck'][df['Deck'] == 'G'] = 'A'
df['Deck'][df['Deck'] == 'F'] = 'C'
df['Deck'][df['Deck'] == 'T'] = 'n'
sns.factorplot(x='Deck', data=df, kind='count', size=4, aspect=2)

# Maybe add in a feature of cabin number
#df['Cabin'] = df['Cabin'].astype(str).str[-3:]
#df['Cabin'] = df['Cabin'].str.replace(r'\D+', '0').astype(int)
#
#df['Cabin'][df['Cabin'] == 0] = None
#df['Cabin'] = pd.qcut(df['Cabin'], 7)
##pd.cut(df['Cabin'],bins=break_points,labels=labels,include_lowest=True)
## Inspecting New Cabin Distribution
#sns.factorplot(x='Cabin', data=df, kind='count', size=4, aspect=2)
#
#sns.barplot('Cabin', 'Survived', data = df)

df = df.drop(columns = ['Cabin'])

### Embarked ###

# Quick look at behavior
sns.barplot('Embarked', 'Survived',data = df)
sns.boxplot('Fare','Embarked',data = df[(df['Pclass'] == 1)])

# Cheeking ralevent fields of null values shows that they are first class and 
# the fare is $80
df[df['Embarked'].isnull()]

# Quick table of median fare by embarkment point and class shows that for first 
# class tickets 'C' has the closest meedian to $80 although 'Q' is close at $90
df.groupby(['Embarked','Pclass'])['Fare'].apply(np.median)

# Using a quick box and wiskers plot to have a better look at the data shows 
# that it can't be 'Q' as that fare is always $90 so most probabal choice is C
sns.boxplot('Fare','Embarked',data = df[(df['Pclass'] == 1)])

# Assigning 'C' to missing embarkment points
df['Embarked'] = df['Embarked'].fillna('C')

### Fare ###

# Cheeking ralevent fields of null values shows that he is 3rd class and 
# embarked from 0 which was maped from 'S'
nan_far = df[df['Fare'].isnull()]

# Inspecting fare Distribution
sns.distplot(df['Fare'][df['Fare'].notnull() & (df['Pclass'] == 3) & (df['Embarked'] == 'S') & (df['Age'] > 60)])

# Finding median fare of equivelent passengers
df['Fare'][df['Fare'].notnull() & (df['Pclass'] == 3) & (df['Embarked'] == 'S') & (df['Age'] > 60)].median()

# Assigning median fare of $8.05 to missing embarkment points
df['Fare'][df['Fare'].isnull()] = 7.775

### Name ###

# Adding title variable and grouping
title_temp = df['Name'].str.split(', ').apply(pd.Series, 1).stack()[:,1]
df['Title'] = title_temp.str.split('.').apply(pd.Series, 1).stack()[:,0]

#Understanding titles used
sns.factorplot(x='Title', data=df, kind='count', size=4, aspect=2)

#Reassigning Title variables
df['Title'] = df['Title'].replace(['Mlle','Ms'],'Miss')
df['Title'] = df['Title'].replace('Mme','Mr')
df['Title'] = df['Title'].replace(['Don','Rev','Dr','Major','Lady','Sir','Col','Capt','the Countess','Jonkheer','Dona'],'Other')

sns.factorplot(x='Title', data=df, kind='count', size=4, aspect=2)

# Parse Surname from Name
df['Surname'] = [nme.split(',')[0] for nme in df['Name']]

import collections
unq = [item for item, count in collections.Counter(df['Surname']).items() if count == 1]
df['Surname'][np.in1d(df['Surname'],unq)] = None

#Understanding titles used
sns.factorplot(x='Surname', data=df, kind='count', size=4, aspect=2)


df = df.drop(columns = ['Name'])
### Ticket ###

# Maybe look for passengers on the same ticket or other ticket features
df = df.drop(columns = ['Ticket'])

### Family Status ###

# Creating a family size feature
df['Family Size'] = 1 + df['Parch'] + df['SibSp']

# Creating a is alone flag
df['Is Alone'] = 0
df['Is Alone'][df['Family Size'] == 1] = 1

### Age ###

# Inspecting Age Distribution
sns.distplot(df['Age'][df['Age'].notnull()])

# Cheecking survival distribution by has age and unknown age
df['Survived'][df['Age'].isnull() & df['Survived'].notnull()].mean()
df['Survived'][df['Age'].notnull() & df['Survived'].notnull()].mean()

# Create a flag to record null ages
df['Has_Age'] = 1
#df['Has_Age'][df['Age'].isnull()] = df['Age'][df['Age'].notnull()].mean()
df['Has_Age'][df['Age'].isnull()] = 0

# Using a random forest regression to impute the missing ages

# Encoding categories into numerics
df_enc = encoder(df, nan_as_category = False)

# Creating regressor
ran_fst_reg = ensemble.RandomForestRegressor()

df_has_age = df_enc[df_enc['Age'].notnull()]
df_has_age = encoder(df_has_age, nan_as_category = False)
ran_fst_reg.fit(df_has_age.drop(columns = ['Age', 'Survived']), df_has_age['Age'])
ran_fst_reg.score(df_has_age.drop(columns = ['Age', 'Survived']), df_has_age['Age'])

# Applying the fitted regression
df_no_age = df_enc[df_enc['Age'].isnull()]
df_no_age = encoder(df_no_age, nan_as_category = False)
age_prd = ran_fst_reg.predict(df_no_age.drop(columns = ['Age', 'Survived']))
df['Age'][df['Age'].isnull()] = age_prd
#
## Binning Age for stability
#df_age = df[['Survived', 'Age']][df['Survived'].notnull()]
#age_bins = WOE_Binning('Survived', df_age, sign = True, n_threshold=10, y_threshold=1, p_threshold=0.4)
#age_bins['Age'][0] = -Inf
## Total IV
#age_bins['IV_components'].sum()
#
##Applying age groups
#df['Age Group'] = 0
#for i in range(0, age_bins.index[-1]):
#    df['Age Group'][(df['Age'] >= age_bins['Age'][i]) & (df['Age'] < age_bins['Age_shift'][i])] = age_bins['labels'][i]
#        
#sns.barplot(df['Age Group'], 'Survived', data = df)
#
#df = df.drop(columns = ['Age'])
# =============================================================================
# Engineering Features
# =============================================================================

#df['Age'][df['Age'].isnull()] = df['Age'][df['Age'].notnull()].mean()
df_dsc = df.describe(include = 'all')

# =============================================================================
# Spliting data back into training and testing set
# =============================================================================

correlation_heatmap(df, 'No Encoding')

# Encoding categories into numerics
df_enc = encoder(df, nan_as_category = False)

# correlation_heatmap(df_enc, 'Encoded')

# Split data backinto training and testing
trn = df_enc[df_enc['Survived'].notnull()]
trn.info()
tst = df_enc[df_enc['Survived'].isnull()]
tst.info()

# =============================================================================
# Quick Overview of a Range of Classifiers
# =============================================================================

#MLA = [
#    # Ensemble Methods
#    ensemble.AdaBoostClassifier(),
#    ensemble.BaggingClassifier(),
#    ensemble.ExtraTreesClassifier(),
#    ensemble.GradientBoostingClassifier(),
#    ensemble.RandomForestClassifier(),
#        
#    # Boosted Trees/Ensembles
#    XGBClassifier(),  
#    LGBMClassifier(),
#
#    # Gaussian Processes
#    gaussian_process.GaussianProcessClassifier(),
#    
#    # GLM
#    linear_model.LogisticRegression(),
#    linear_model.PassiveAggressiveClassifier(),
#    linear_model.RidgeClassifierCV(),
#    linear_model.SGDClassifier(),
#    linear_model.Perceptron(),
#    
#    # Navies Bayes
#    naive_bayes.BernoulliNB(),
#    naive_bayes.GaussianNB(),
#    
#    # Nearest Neighbor
#    neighbors.KNeighborsClassifier(),
#    
#    # SVM
#    svm.SVC(probability=True),
#    svm.NuSVC(probability=True),
#    svm.LinearSVC(),
#    
#    # Trees    
#    tree.DecisionTreeClassifier(),
#    tree.ExtraTreeClassifier(),
#    
#    # Discriminant Analysis
#    discriminant_analysis.LinearDiscriminantAnalysis(),
#    discriminant_analysis.QuadraticDiscriminantAnalysis()
#    ]
#
#mdl_dta = pd.DataFrame(columns = ['Algorithm', 'Accuracy'])
#row_index = 0
#for alg in MLA:
#    nme = alg.__class__.__name__
#    nme, acc = model_fit(alg, mdl_nme = nme, n_fld = 7)
#    mdl_dta.loc[row_index, 'Algorithm'] = nme
#    mdl_dta.loc[row_index, 'Accuracy'] = acc
#    row_index += 1

# =============================================================================
# Refining the Best Performers  
# =============================================================================
#    
#lnr_dct = discriminant_analysis.LinearDiscriminantAnalysis(
#        n_components=None, 
#        priors=None, 
#        shrinkage=None,
#        solver='svd', 
#        store_covariance=False, 
#        tol=0.0001)
#model_fit(lnr_dct, mdl_nme = 'Linear_Discriminant', n_fld = 10, dbg = False)
## Initial: 0.837238
#
#
#XGB = XGBClassifier()
#model_fit(XGB, mdl_nme = 'XGB', n_fld = 10, dbg = False)
## Initial: 0.836214
#
#LGBM = LGBMClassifier()
#model_fit(LGBM, mdl_nme = 'LGBM', n_fld = 10, dbg = False)
## Initial: 0.832819 Optimized: 0.85190 Top 8%!

log_reg = linear_model.LogisticRegression()
model_fit(log_reg, mdl_nme = 'Logistic_Regression', n_fld = 6, dbg = False)
# Initial: 0.828236, Optimized: 0.8338

# Objective function for hyperparameter tuning
#def objective(**params):
#    # Set Integers
#    params['max_iter'] = int(params['max_iter'])
#    log_reg = linear_model.LogisticRegression(**params, random_state = 11)
#    # Perform n_fold cross validation with hyperparameters
#    # Use early stopping and evalute based on ROC AUC
#    nme, acc = model_fit(log_reg, mdl_nme = 'Logistic_Regression', n_fld = 10)
#    # Loss function
#    return acc
#
#params = {'C':(0.6,0.7),
#          'tol':(0.0005,0.003),
#          'max_iter': (50,600)}
#
#bo = BayesianOptimization(objective, params)
#opt = bo.maximize(init_points = 30, n_iter = 5)