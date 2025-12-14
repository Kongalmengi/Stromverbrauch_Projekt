import numpy as np
import pandas as pd

# PolynomialFeatures for Regression
from sklearn.preprocessing import PolynomialFeatures

# for VIF
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant

# for Reproduce prediction process for evaluation
from sklearn.model_selection import train_test_split
# for Evaluation
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_validate

from sklearn.linear_model import Ridge



# Year-based split (pre-2005 vs post-2005)
def divide_Jahr(Indust_df):
    indu_under_2005 = Indust_df[Indust_df['Jahr']<2006]
    indu_upper_2006 = Indust_df[Indust_df['Jahr']>2005]
    return indu_under_2005, indu_upper_2006


# ----(create correlation dataframe)----

# Correlation matrix for Umsatz data
def Gopp_corr_umsatz_dfs(indu_under_2005, indu_upper_2006):
    # Compute correlation of 'Gesamtumsatz' across time intervals
    # Rearrange data with pivot_table() to enable .corr() computation
    pivot_under_2005 = indu_under_2005.pivot_table(index='Jahr', columns='DName', values='Gesamtumsatz')
    pivot_upper_2006 = indu_upper_2006.pivot_table(index='Jahr', columns='DName', values='Gesamtumsatz')

    # Compute .corr()
    corr_under_2005 = pivot_under_2005.corr()
    corr_upper_2006 = pivot_upper_2006.corr()

    # Extract top 10 regions (year <= 2005)
    list_under_2005 = corr_under_2005['Göppingen'].drop(index='Göppingen').sort_values(ascending=False).head(10).index.tolist()
    # Extract top 10 regions (year >= 2006)
    list_upper_2006 = corr_upper_2006['Göppingen'].drop(index='Göppingen').sort_values(ascending=False).head(10).index.tolist()

    list_sum = list_under_2005 + list_upper_2006
    series_sum = pd.Series(list_sum)
    sum_counts = series_sum.value_counts()

    # Extract final candidate regions
    dup_only_name = sum_counts[sum_counts>1].index.tolist()

    return corr_under_2005, corr_upper_2006, dup_only_name


# Correlation matrix for Beschäftigte data
def Gopp_corr_bes_dfs(indu_under_2005, indu_upper_2006):
    # Compute correlation of 'Beschäftigte' across time intervals
    # Rearrange data with pivot_table() to enable .corr() computation
    pivot_under_2005_2 = indu_under_2005.pivot_table(index='Jahr', columns='DName', values='Beschäftigte')
    pivot_upper_2006_2 = indu_upper_2006.pivot_table(index='Jahr', columns='DName', values='Beschäftigte')

    # Compute .corr()
    corr_under_2005_2 = pivot_under_2005_2.corr()
    corr_upper_2006_2 = pivot_upper_2006_2.corr()

    return corr_under_2005_2, corr_upper_2006_2


# ----(DataFrames for Visualization)----
# DataFrame about 'Gesamtumsatz' for Visualization
def umsatz_corr_dfs_for_Vis(corr_under_2005, corr_upper_2006, dup_only_name):
    # Correlation mat.(1) ('Gesamtumsatz')
    Gopp_umsatz_corr_upper_2006_df = pd.DataFrame(corr_upper_2006['Göppingen'].drop(index='Göppingen').sort_values(ascending=False).head(10))

    Gopp_umsatz_corr_upper_2006_df['Kandidatenstatus'] = 'übrige Regionen'
    Gopp_umsatz_corr_upper_2006_df.loc[Gopp_umsatz_corr_upper_2006_df.index.isin(dup_only_name), 'Kandidatenstatus'] = 'Kandidatenregion'
    Gopp_umsatz_corr_upper_2006_df.columns = ['Korrelationskoeffizient', 'Kandidatenstatus']
    # Correlation mat.(2) ('Gesamtumsatz')
    Gopp_umsatz_corr_under_2005_df = pd.DataFrame(corr_under_2005['Göppingen'].drop(index='Göppingen').sort_values(ascending=False).head(10))
    
    Gopp_umsatz_corr_under_2005_df['Kandidatenstatus'] = 'übrige Regionen'
    Gopp_umsatz_corr_under_2005_df.loc[Gopp_umsatz_corr_under_2005_df.index.isin(dup_only_name), 'Kandidatenstatus'] = 'Kandidatenregion'
    Gopp_umsatz_corr_under_2005_df.columns = ['Korrelationskoeffizient', 'Kandidatenstatus']

    return Gopp_umsatz_corr_under_2005_df, Gopp_umsatz_corr_upper_2006_df

# DataFrame about 'Beschäftigte' for Visualization
def bes_corr_dfs_for_Vis(corr_under_2005_2, corr_upper_2006_2, dup_only_name):
    # Correlation mat.(1) ('Beschäftigte')
    Gopp_bes_corr_upper_2006_2_df = pd.DataFrame(corr_upper_2006_2['Göppingen'].drop(index='Göppingen').sort_values(ascending=False).head(10))

    Gopp_bes_corr_upper_2006_2_df['Kandidatenstatus'] = 'übrige Regionen'
    Gopp_bes_corr_upper_2006_2_df.loc[Gopp_bes_corr_upper_2006_2_df.index.isin(dup_only_name), 'Kandidatenstatus'] = 'Kandidatenregion'
    Gopp_bes_corr_upper_2006_2_df.columns = ['Korrelationskoeffizient', 'Kandidatenstatus']
    # Correlation mat.(2) ('Beschäftigte')
    Gopp_bes_corr_under_2005_2_df = pd.DataFrame(corr_under_2005_2['Göppingen'].drop(index='Göppingen').sort_values(ascending=False).head(10))

    Gopp_bes_corr_under_2005_2_df['Kandidatenstatus'] = 'übrige Regionen'
    Gopp_bes_corr_under_2005_2_df.loc[Gopp_bes_corr_under_2005_2_df.index.isin(dup_only_name), 'Kandidatenstatus'] = 'Kandidatenregion'
    Gopp_bes_corr_under_2005_2_df.columns = ['Korrelationskoeffizient', 'Kandidatenstatus']

    return Gopp_bes_corr_under_2005_2_df, Gopp_bes_corr_upper_2006_2_df


# for .corr Graph
# Add indicator column for highlighting candidate regions in red
def colormaps_for_Vis(Gopp_umsatz_corr_under_2005_df, Gopp_umsatz_corr_upper_2006_df, Gopp_bes_corr_under_2005_2_df, Gopp_bes_corr_upper_2006_2_df):
    # 'Kandidatenstatus' Column : .value_counts()
    umsatz_upper_2006_counts = Gopp_umsatz_corr_upper_2006_df['Kandidatenstatus'].value_counts()
    umsatz_under_2005_counts = Gopp_umsatz_corr_under_2005_df['Kandidatenstatus'].value_counts()
    bes_upper_2006_counts_2 = Gopp_bes_corr_upper_2006_2_df['Kandidatenstatus'].value_counts()
    bes_under_2005_counts_2 = Gopp_bes_corr_under_2005_2_df['Kandidatenstatus'].value_counts()

    # hilighting
    red_category_umsatz_2006 = umsatz_upper_2006_counts.idxmin()
    blue_category_umsatz_2006 = umsatz_upper_2006_counts.idxmax()

    red_category_umsatz_2005 = umsatz_under_2005_counts.idxmin()
    blue_category_umsatz_2005 = umsatz_under_2005_counts.idxmax()

    red_category_bes_2006 = bes_upper_2006_counts_2.idxmin()
    blue_category_bes_2006 = bes_upper_2006_counts_2.idxmax()

    red_category_bes_2005 = bes_under_2005_counts_2.idxmin()
    blue_category_bes_2005 = bes_under_2005_counts_2.idxmax()

    # color setting
    # 0th: #636EFA (blue)
    # 1st: #EF553B (red)
    color_map_umsatz_2006 = {
        red_category_umsatz_2006: '#EF553B',
        blue_category_umsatz_2006: '#636EFA',
    }

    color_map_umsatz_2005 = {
        red_category_umsatz_2005: '#EF553B',
        blue_category_umsatz_2005: '#636EFA',
    }

    color_map_bes_2006 = {
        red_category_bes_2006: '#EF553B',
        blue_category_bes_2006: '#636EFA',
    }

    color_map_bes_2005 = {
        red_category_bes_2005: '#EF553B',
        blue_category_bes_2005: '#636EFA',
    }

    return color_map_umsatz_2005, color_map_umsatz_2006, color_map_bes_2005, color_map_bes_2006


# subset df
def make_subset(Indust_df, Gopp_Bes_scaler, Gopp_umsatz_scaler, Ess_Bes_scaler, Ess_umsatz_scaler):
    subset_df = Indust_df[(Indust_df['DName']=='Göppingen')|(Indust_df['DName']=='Esslingen')].copy()

    # Apply StandardScaler
    scaled_Gopp_Bes = Gopp_Bes_scaler.transform(subset_df.loc[subset_df['DName']=='Göppingen', ['Beschäftigte']])
    scaled_Gopp_umsatz = Gopp_umsatz_scaler.transform(subset_df.loc[subset_df['DName']=='Göppingen', ['Gesamtumsatz']])
    scaled_Ess_Bes = Ess_Bes_scaler.transform(subset_df.loc[subset_df['DName']=='Esslingen', ['Beschäftigte']])
    scaled_Ess_umsatz = Ess_umsatz_scaler.transform(subset_df.loc[subset_df['DName']=='Esslingen', ['Gesamtumsatz']])

    # Add transformed feature columns to subset_df
    subset_df.loc[subset_df['DName']=='Göppingen', 'Beschäftigte_norm'] = scaled_Gopp_Bes
    subset_df.loc[subset_df['DName']=='Göppingen', 'Gesamtumsatz_norm'] = scaled_Gopp_umsatz
    subset_df.loc[subset_df['DName']=='Esslingen', 'Beschäftigte_norm'] = scaled_Ess_Bes
    subset_df.loc[subset_df['DName']=='Esslingen', 'Gesamtumsatz_norm'] = scaled_Ess_umsatz

    return subset_df


# Predict and inverse-transform missing values
def make_pred_subset(subset_df, lin_pre, lin_pan, lin_post, Gopp_umsatz_scaler):
    # Configuration for time segment(1)
    feat_cols_pre = ['Beschäftigte_norm', 'Jahr']
    target_col_pre = 'Gesamtumsatz_norm'
    # Set up training data
    Mask_Ess_pre = (subset_df['DName']=='Esslingen')&(subset_df['Jahr'].between(2009, 2017))
    train_pre_df = subset_df.loc[Mask_Ess_pre, feat_cols_pre+[target_col_pre]]
    X_train_pre = train_pre_df[feat_cols_pre].values
    # Create PolynomialFeatures
    poly = PolynomialFeatures(degree = 2, include_bias=False)
    X_Ess_pre_poly = poly.fit_transform(X_train_pre)


    # Configuration for time segment(2)
    feat_cols_pan = ['Beschäftigte_norm', 'Jahr']
    target_col_pan = 'Gesamtumsatz_norm'


    # Configuration for time segment(3)
    feat_cols_post = ['Beschäftigte_norm', 'Jahr']
    target_col_post = 'Gesamtumsatz_norm'
    # Set up training data
    Mask_Ess_post = (subset_df['DName']=='Esslingen')&(subset_df['Jahr']>=2021)
    train_post_df = subset_df.loc[Mask_Ess_post, feat_cols_post+[target_col_post]]
    X_train_post = train_post_df[feat_cols_post].values
    # Create PolynomialFeatures
    poly_post = PolynomialFeatures(degree = 2, include_bias=False)
    X_Ess_post_poly = poly_post.fit_transform(X_train_post)


    # Predict missing values for Göppingen (segment 1)
    # Create : Mask and Data segment1
    m_G_pre = (subset_df['DName']=='Göppingen')&(subset_df['Jahr'].between(2010, 2017))
    dfG_pre = subset_df.loc[m_G_pre, feat_cols_pre + [target_col_pre]].copy()

    # Apply PolynomialFeatures + LinearRegression
    XG_pre = poly.transform(dfG_pre[feat_cols_pre].values)
    yG_pred_z_pre = lin_pre.predict(XG_pre) # z-score

    # Inverse transform
    yG_pred_pre = Gopp_umsatz_scaler.inverse_transform(yG_pred_z_pre.reshape(-1, 1)).ravel()


    # Predict missing values for Göppingen (segment 2)
    # Create : Mask and Data segment2
    m_G_pan = (subset_df['DName']=='Göppingen')&(subset_df['Jahr'].between(2018, 2020))
    dfG_pan = subset_df.loc[m_G_pan, feat_cols_pan + [target_col_pan]].copy()

    # Apply LinearRegression
    XG_pan = dfG_pan[feat_cols_pan].values
    yG_pred_z_pan = lin_pan.predict(XG_pan) # z-score

    # Inverse transform
    yG_pred_pan = Gopp_umsatz_scaler.inverse_transform(yG_pred_z_pan.reshape(-1, 1)).ravel()


    # Predict missing values for Göppingen (segment 3)
    # Create : Mask and Data segment3
    m_G_post = (subset_df['DName']=='Göppingen')&(subset_df['Jahr']>=2021)
    dfG_post = subset_df.loc[m_G_post, feat_cols_post + [target_col_post]].copy()

    # Apply PolynomialFeatures + LinearRegression
    XG_post = poly_post.transform(dfG_post[feat_cols_post].values)
    yG_pred_z_post = lin_post.predict(XG_post) # z-score

    # Inverse transform
    yG_pred_post = Gopp_umsatz_scaler.inverse_transform(yG_pred_z_post.reshape(-1, 1)).ravel()


    # Fill missing values in subset_df (segment 1)
    subset_df.loc[m_G_pre, 'Gesamtumsatz_norm'] = yG_pred_z_pre
    subset_df.loc[m_G_pre, 'Gesamtumsatz'] = yG_pred_pre

    # Fill missing values in subset_df (segment 2)
    subset_df.loc[m_G_pan, 'Gesamtumsatz_norm'] = yG_pred_z_pan
    subset_df.loc[m_G_pan, 'Gesamtumsatz'] = yG_pred_pan

    # Fill missing values in subset_df (segment 3)
    subset_df.loc[m_G_post, 'Gesamtumsatz_norm'] = yG_pred_z_post
    subset_df.loc[m_G_post, 'Gesamtumsatz'] = yG_pred_post

    return subset_df


def make_subset_2(Indust_df):
    subset_df_2 = Indust_df[Indust_df['DName']=='Heidenheim'].copy()
    interpolated_s = subset_df_2.loc[subset_df_2['Jahr'].between(1998, 2003), 'Gesamtumsatz'].interpolate(method='polynomial', order=2).round().astype(int)
    # Fill missing values
    subset_df_2.loc[interpolated_s.index, 'Gesamtumsatz'] = interpolated_s
    return subset_df_2


def make_corr_mat(combined_df):
    combined_corr_mat = combined_df.corr(numeric_only=True)
    corr_mat = combined_corr_mat.round(4)
    return corr_mat


def make_vif(combined_df):
    # Choosing columns for VIF check
    features = ['Betriebe',
           'Beschäftigte', 'Gesamtumsatz', 'Investitionen']
    X = combined_df[features]

    # Add constant term (required for VIF computation)
    X_const = add_constant(X)

    vif_df = pd.DataFrame()
    vif_df["feature"] = X_const.columns
    vif_df["VIF"] = [variance_inflation_factor(X_const.values, i)
                     for i in range(X_const.shape[1])]
    return vif_df


# Reproduce prediction pipeline for evaluation

def test_gesamtmodell(combined_df, scaler_for_normal_reg, model):
    # Filter out non-target regions
    normal_combined_df = combined_df.copy()
    # first Filter out
    normal_combined_df = normal_combined_df[~normal_combined_df['DN_DT'].str.contains('Ortenaukreis|Mannheim|Rastatt|Waldshut|Alb-Donau-Kreis|Stuttgart|Böblingen')]
    # second Filter out
    normal_combined_df = normal_combined_df[~((normal_combined_df['DN_DT'].str.contains('Karlsruhe'))&(normal_combined_df['DN_DT'].str.contains('kreisfreie')))]

    # Split variables ; drop unnecessary columns
    X = normal_combined_df.drop(columns=['DN_DT', 'Jahr', 'Regionalverband', 'Bevölkerung insgesamt', 'Betriebe', 'Stromverbrauch(Industrie)', 'Stromverbrauch(Haushalt)', 'Beschäftigungsquote'])
    y = normal_combined_df['Stromverbrauch(Industrie)']

    # Split tarin-test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=39
    )

    # Apply StandardScaler
    X_train_scaled = scaler_for_normal_reg.transform(X_train)
    X_test_scaled = scaler_for_normal_reg.transform(X_test)

    # Compute importance
    coef = model.coef_
    features = X.columns

    importance_df = pd.DataFrame({
        'feature': features,
        'importance': coef
    }).sort_values(by='importance', key=np.abs, ascending=False)

    # Model Evaluation
    # Predict
    y_pred = model.predict(X_test_scaled)

    # Compute scores
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    # 3. Scores to df
    eval_results_df = pd.DataFrame({
        'Metric': ['R² Score', 'MAE', 'RMSE'],
        'Value': [round(r2, 3), round(mae, 2), round(rmse, 2)]
    })

    # Create residuals df for Visualization
    residuals = y_test - y_pred

    res_df = pd.DataFrame({
        'Predicted': y_pred,
        'Residual': residuals
    })

    return normal_combined_df, y_test, y_pred, residuals, importance_df, eval_results_df, res_df


def kfold_result(combined_df, scaler_for_special_reg):
    # Create special_combined_df
    temp_df1 = combined_df[combined_df['DN_DT'] == 'Karlsruhe (kreisfreie Stadt)']
    temp_df2 = combined_df[combined_df['DN_DT'].str.contains('Ortenaukreis|Mannheim|Rastatt|Waldshut|Alb-Donau-Kreis|Stuttgart|Böblingen')]
    special_combined_df = pd.concat([temp_df1, temp_df2], axis=0)

    # dict. for df.
    result_for_s_model_dict = {}

    for reg in special_combined_df['DN_DT'].drop_duplicates():
        # Split variables
        X_s = special_combined_df[special_combined_df['DN_DT']==reg].drop(columns=['DN_DT', 'Jahr', 'Regionalverband', 'Bevölkerung insgesamt', 'Betriebe', 'Stromverbrauch(Industrie)', 'Stromverbrauch(Haushalt)', 'Beschäftigungsquote'])
        y_s = special_combined_df[special_combined_df['DN_DT']==reg]['Stromverbrauch(Industrie)']

        # Apply Scaler
        X_s_scaled = scaler_for_special_reg.transform(X_s)

        # Set alpha parameter for Ridge model
        if reg=='Karlsruhe (kreisfreie Stadt)':
            alpha_value = 0.01
        elif reg=='Alb-Donau-Kreis (Landkreis)':
            alpha_value = 0.07
        elif reg=='Böblingen (Landkreis)':
            alpha_value = 2.9
        elif reg=='Mannheim (kreisfreie Stadt)':
            alpha_value = 0.04
        elif reg=='Ortenaukreis (Landkreis)':
            alpha_value = 1
        elif reg=='Rastatt (Landkreis)':
            alpha_value = 3.79
        elif reg=='Stuttgart (kreisfreie Stadt)':
            alpha_value = 0.15
        elif reg=='Waldshut (Landkreis)':
            alpha_value = 0.02
    
        # Create Ridge model
        model_s = Ridge(alpha=alpha_value)

        # Setting; k-fold CV
        kfold = KFold(n_splits=4, shuffle=True, random_state=52)

        # Evaluation
        scoring = {
            'r2': 'r2',
            'mae': 'neg_mean_absolute_error',
            'rmse': 'neg_root_mean_squared_error'
        }

        results = cross_validate(
            model_s,
            X_s_scaled, y_s,
            cv=kfold,
            scoring=scoring,
            return_train_score=True
        )


        # Results to df
        result_df = pd.DataFrame([{
            'Kreis': reg,
            'alpha': alpha_value,
            'Mean R²': results['test_r2'].mean(),
            'Mean MAE': -results['test_mae'].mean(),
            'Mean RMSE': -results['test_rmse'].mean(),
            'Mean Target Value': y_s.mean()
        }])
        # results_df to dict.
        result_for_s_model_dict[reg] = result_df

    # concat DataFrames
    result_for_s_model_df = pd.concat(result_for_s_model_dict.values(), axis=0).reset_index(drop=True)

    return result_for_s_model_df


# for IndustEV scale Visualization
def groupby_IndustEV(normal_combined_df):
    grouped_normal_combined = normal_combined_df.groupby('DN_DT')
    groupby_IndustEV_series = grouped_normal_combined['Stromverbrauch(Industrie)'].mean()
    groupby_IndustEV_df = pd.DataFrame(groupby_IndustEV_series)
    return groupby_IndustEV_df


# Kreis Evaluation : Gesamtmodell
def kreis_eval(normal_combined_df, y_test, scaler_for_normal_reg, model):
    sample_test_df = normal_combined_df.loc[y_test.index]
    sampled_results = []

    for kreis in sample_test_df['DN_DT'].unique():
        sampled_y_test = sample_test_df.loc[sample_test_df['DN_DT']==kreis, 'Stromverbrauch(Industrie)']
        sampled_X_test = sample_test_df.loc[sample_test_df['DN_DT']==kreis].drop(columns=['DN_DT', 'Jahr', 'Regionalverband', 'Bevölkerung insgesamt', 'Betriebe', 'Stromverbrauch(Industrie)', 'Stromverbrauch(Haushalt)', 'Beschäftigungsquote'])

        # 예측을 위한 스케일링 : scaler_for_normal_reg 이용.
        sampled_X_test_scaled = scaler_for_normal_reg.transform(sampled_X_test)

        # 모델을 이용한 예측 : 모델명 : model
        sampled_y_pred = model.predict(sampled_X_test_scaled)

        # 평가 : 지역별 MAE와 RMSE
        sampled_mae = mean_absolute_error(sampled_y_test, sampled_y_pred)
        sampled_rmse = np.sqrt(mean_squared_error(sampled_y_test, sampled_y_pred))

        # 지역별 MAE/평균, RMAE/표준편차 구하기
        # 관련지역 전체 산업 소비전력의 평균과 표준편차 구하기
        # 평균
        sampled_mean_y = normal_combined_df.loc[normal_combined_df['DN_DT']==kreis, 'Stromverbrauch(Industrie)'].mean()
        # 표준편차
        sampled_std_y = normal_combined_df.loc[normal_combined_df['DN_DT']==kreis, 'Stromverbrauch(Industrie)'].std()

        sampled_results.append({
            'Kreis': kreis,
            'MAE': sampled_mae,
            'RMSE': sampled_rmse,
            'Mean_y': sampled_mean_y,
            'Std_y': sampled_std_y,
            'MAE/Mean_y': sampled_mae/sampled_mean_y,
            'RMSE/Std_y': sampled_rmse/sampled_std_y
        })

    df_all_results = pd.DataFrame(sampled_results)

    return df_all_results