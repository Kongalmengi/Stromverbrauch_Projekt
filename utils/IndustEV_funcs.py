import numpy as np
import pandas as pd
import geopandas as gpd


# ---(직원수 데이터 계산)---

# pred_Bev_for_empl_df - step1(End)
def pred_Bev_for_empl(Bev_for_Vis_df, TBD_df, pred_empl_r_df, Grp_1, Grp_2, Grp_3, empl_r_inc_1, empl_r_inc_2, empl_r_inc_3, empl_r_inc):
    # Art컬럼이 predict이면서 연도가 2023이 아닌 경우를 추출
    pred_Bev_for_empl_df = Bev_for_Vis_df[(Bev_for_Vis_df['Art']=='predict')&~(Bev_for_Vis_df['Jahr']==2023)].copy()

    # 직원수 컬럼 만들기 : 이게 없으면 그룹1, 2, 3에 아무 지역도 선택되지 않았을 때 그룹0의 루프 기준에 오류가 발생한다. 그룹1, 2, 3에서 for루프가 돌지 않아 직원수 컬럼이 만들어지지 않기 때문!
    # 이에 따라 아래와 같이 직원수 컬럼을 미리 만들어둬야 한다.
    if 'Beschäftigte' not in pred_Bev_for_empl_df.columns:
        pred_Bev_for_empl_df['Beschäftigte'] = pd.NA

    # 그룹1
    # pred_Bev_for_empl_df에서 group_1에 속하는 지역들 중 DN_DT들을 뽑아보자. 근데 이건 TBD에서도 가능한데?
    for reg in TBD_df.loc[TBD_df['Regionalverband'].isin(Grp_1), 'DN_DT']:
        for year in pred_empl_r_df['Jahr'].drop_duplicates():
            pred_Bev_for_empl_df.loc[(pred_Bev_for_empl_df['DN_DT']==reg)&(pred_Bev_for_empl_df['Jahr']==year), 'Beschäftigte'] = (pred_empl_r_df.loc[(pred_empl_r_df['DN_DT']==reg)&(pred_empl_r_df['Jahr']==year), f'rate_{empl_r_inc_1}'].iloc[0] * pred_Bev_for_empl_df.loc[(pred_Bev_for_empl_df['DN_DT']==reg)&(pred_Bev_for_empl_df['Jahr']==year), 'Bevölkerung insgesamt'].iloc[0]).round()

    # 그룹2의 pred_Bev_for_empl_df 직원수 컬럼 채우기
    for reg in TBD_df.loc[TBD_df['Regionalverband'].isin(Grp_2), 'DN_DT']:
        for year in pred_empl_r_df['Jahr'].drop_duplicates():
            pred_Bev_for_empl_df.loc[(pred_Bev_for_empl_df['DN_DT']==reg)&(pred_Bev_for_empl_df['Jahr']==year), 'Beschäftigte'] = (pred_empl_r_df.loc[(pred_empl_r_df['DN_DT']==reg)&(pred_empl_r_df['Jahr']==year), f'rate_{empl_r_inc_2}'].iloc[0] * pred_Bev_for_empl_df.loc[(pred_Bev_for_empl_df['DN_DT']==reg)&(pred_Bev_for_empl_df['Jahr']==year), 'Bevölkerung insgesamt'].iloc[0]).round()

    # 그룹3의 pred_Bev_for_empl_df 직원수 컬럼 채우기
    for reg in TBD_df.loc[TBD_df['Regionalverband'].isin(Grp_3), 'DN_DT']:
        for year in pred_empl_r_df['Jahr'].drop_duplicates():
            pred_Bev_for_empl_df.loc[(pred_Bev_for_empl_df['DN_DT']==reg)&(pred_Bev_for_empl_df['Jahr']==year), 'Beschäftigte'] = (pred_empl_r_df.loc[(pred_empl_r_df['DN_DT']==reg)&(pred_empl_r_df['Jahr']==year), f'rate_{empl_r_inc_3}'].iloc[0] * pred_Bev_for_empl_df.loc[(pred_Bev_for_empl_df['DN_DT']==reg)&(pred_Bev_for_empl_df['Jahr']==year), 'Bevölkerung insgesamt'].iloc[0]).round()

    # 그룹0(나머지 지역)의 pred_Bev_for_empl_df 직원수 컬럼 채우기
    for reg in pred_Bev_for_empl_df.loc[pred_Bev_for_empl_df['Beschäftigte'].isna(), 'DN_DT'].drop_duplicates():
        for year in pred_empl_r_df['Jahr'].drop_duplicates():
            pred_Bev_for_empl_df.loc[(pred_Bev_for_empl_df['DN_DT']==reg)&(pred_Bev_for_empl_df['Jahr']==year), 'Beschäftigte'] = (pred_empl_r_df.loc[(pred_empl_r_df['DN_DT']==reg)&(pred_empl_r_df['Jahr']==year), f'rate_{empl_r_inc}'].iloc[0] * pred_Bev_for_empl_df.loc[(pred_Bev_for_empl_df['DN_DT']==reg)&(pred_Bev_for_empl_df['Jahr']==year), 'Bevölkerung insgesamt'].iloc[0]).round()

    # 인구 컬럼을 제거하고, 직원수 컬럼의 값을 int로 바꿔준다.
    pred_Bev_for_empl_df = pred_Bev_for_empl_df.drop(columns=['Bevölkerung insgesamt'])
    pred_Bev_for_empl_df['Beschäftigte'] = pred_Bev_for_empl_df['Beschäftigte'].astype(int)

    return pred_Bev_for_empl_df


# act_Bev_for_empl_df - Step1(End)
def act_Bev_for_empl(BW_Industrie_df, groups, group_vals, Grp_val_0):
    act_Bev_for_empl_df = BW_Industrie_df.drop(columns=['DName', 'DType', 'Betriebe', 'Gesamtumsatz']).copy()

    # 그룹1, 2, 3
    for grp, val in zip(groups, group_vals):
        act_Bev_for_empl_df.loc[
            act_Bev_for_empl_df['Regionalverband'].isin(grp),
            'Group'
        ] = val

    # 그룹0
    act_Bev_for_empl_df.loc[~((act_Bev_for_empl_df['Regionalverband'].isin(groups[0]))|(act_Bev_for_empl_df['Regionalverband'].isin(groups[1]))|(act_Bev_for_empl_df['Regionalverband'].isin(groups[2]))), 'Group'] = Grp_val_0

    # Art컬럼 추가
    act_Bev_for_empl_df['Art'] = 'actual'

    return act_Bev_for_empl_df


# middle_Bev_for_empl_df
def middle_Bev_for_empl(act_Bev_for_empl_df):
    middle_Bev_for_empl_df = act_Bev_for_empl_df[act_Bev_for_empl_df['Jahr']==2023].copy()
    middle_Bev_for_empl_df['Art'] = 'predict'

    return middle_Bev_for_empl_df


# 데이터 합치기
# 데이터를 모두 합쳐 empl_for_Vis_df를 만든다.
# empl_for_Vis_df
def empl_for_Vis(act_Bev_for_empl_df, middle_Bev_for_empl_df, pred_Bev_for_empl_df):
    empl_for_Vis_df = pd.concat([act_Bev_for_empl_df, middle_Bev_for_empl_df], axis=0)
    empl_for_Vis_df = pd.concat([empl_for_Vis_df, pred_Bev_for_empl_df], axis=0).reset_index(drop=True)

    return empl_for_Vis_df



# ---(매출 데이터 관련)---
# temp_pred_Umsatz_df - Step1
def temp_pred_Umsatz_base(BW_pred_Umsatz_df, Grp_1, Grp_2, Grp_3, Grp_val_0, umsatz_inc):
    # 과정1-1. 어느 그룹에도 속하지 않은 지역 매출 예측
    temp_pred_Umsatz_base_df = BW_pred_Umsatz_df.loc[~((BW_pred_Umsatz_df['Regionalverband'].isin(Grp_1))|(BW_pred_Umsatz_df['Regionalverband'].isin(Grp_2))|(BW_pred_Umsatz_df['Regionalverband'].isin(Grp_3))), ['DN_DT', 'Regionalverband', 'Jahr', f'rate_{umsatz_inc}']]
    # Group 컬럼 추가 : 값 = Grp_val_0
    temp_pred_Umsatz_base_df['Group'] = Grp_val_0
    # 컬럼명 변경 : concat을 할 때 용이하도록 컬럼명을 Bevölkerung insgesamt로 통일
    temp_pred_Umsatz_base_df = temp_pred_Umsatz_base_df.rename(columns={f'rate_{umsatz_inc}':'Gesamtumsatz'})

    return temp_pred_Umsatz_base_df

# temp_pred_Umsatz_df - Step2
def temp_pred_Umsatz_Grp(BW_pred_Umsatz_df, groups, group_vals, umsatz_inc_rates):
    # 과정1-2. 그룹에 속한 지역 매출 예측
    # 데이터프레임을 담을 빈 리스트
    temp_pred_Umsatz_grp_list = []

    # for루프
    for grp, grp_val, umsatz_rate in zip(groups, group_vals, umsatz_inc_rates):
        rate_col = f"rate_{umsatz_rate}"

        temp_df = BW_pred_Umsatz_df.loc[
            BW_pred_Umsatz_df['Regionalverband'].isin(grp),
            ['DN_DT', 'Regionalverband', 'Jahr', rate_col]
        ].copy()

        # Group 컬럼 추가
        temp_df['Group'] = grp_val

        # 컬럼명 변경
        temp_df = temp_df.rename(columns={rate_col: 'Gesamtumsatz'})

        temp_pred_Umsatz_grp_list.append(temp_df)

    # 그룹에 속한 데이터 합치기
    temp_pred_Umsatz_Grp_df = pd.concat(temp_pred_Umsatz_grp_list, ignore_index=True)

    return temp_pred_Umsatz_Grp_df


# temp_pred_Umsatz_df - Step3(End)
def temp_pred_Umsatz(temp_pred_Umsatz_base_df, temp_pred_Umsatz_Grp_df):
    # 과정1-3. 그룹에 속하지 않은 데이터 + 그룹에 속한 데이터 합치기
    temp_pred_Umsatz_df = pd.concat([temp_pred_Umsatz_base_df, temp_pred_Umsatz_Grp_df], ignore_index=True).reset_index(drop=True)

    # Art 컬럼을 추가하여 'predict'를 할당.
    temp_pred_Umsatz_df['Art'] = 'predict'

    return temp_pred_Umsatz_df


# temp_act_Umsatz_df - Step1(End)
def temp_act_Umsatz(BW_Industrie_df, Grp_1, Grp_2, Grp_3, Grp_val_1, Grp_val_2, Grp_val_3, Grp_val_0):
    # 과정2-1. 실제 데이터를 시각화 틀에 맞춰 가공
    temp_act_Umsatz_df = BW_Industrie_df[['DN_DT', 'Regionalverband', 'Jahr', 'Gesamtumsatz']].copy()

    # 과정2-2. 시각화 틀에 맞춰 가공2 : Group, Art컬럼을 추가한다.
    # Group 컬럼 추가
    # 그룹1
    temp_act_Umsatz_df.loc[temp_act_Umsatz_df['Regionalverband'].isin(Grp_1), 'Group'] = Grp_val_1

    # 그룹2
    temp_act_Umsatz_df.loc[temp_act_Umsatz_df['Regionalverband'].isin(Grp_2), 'Group'] = Grp_val_2

    # 그룹3
    temp_act_Umsatz_df.loc[temp_act_Umsatz_df['Regionalverband'].isin(Grp_3), 'Group'] = Grp_val_3

    # 그룹0
    temp_act_Umsatz_df.loc[temp_act_Umsatz_df['Group'].isna(), 'Group'] = Grp_val_0

    # Art 컬럼 추가
    temp_act_Umsatz_df['Art'] = 'actual'

    return temp_act_Umsatz_df


# temp_middle_Umsatz_df - Step1(End)
def temp_middle_Umsatz(temp_act_Umsatz_df):
    # 과정2-3. 2023년값을 한 번 더 추가 : Art = 'predict'로.
    # 이는 그래프 시각화를 유연하게 하기 위함.
    # 먼저 실제 값에서 2023년의 데이터를 추출.
    temp_middle_Umsatz_df = temp_act_Umsatz_df[temp_act_Umsatz_df['Jahr']==2023].copy()

    # Art컬럼을 predict로 변경
    temp_middle_Umsatz_df['Art'] = 'predict'

    return temp_middle_Umsatz_df


# Umsatz_for_Vis_df - Step1(End)
def Umsatz_for_Vis(temp_act_Umsatz_df, temp_middle_Umsatz_df, temp_pred_Umsatz_df):
    # 과정2-4. 준비된 데이터프레임을 아래의 순으로 차례로 합친다.
    # temp_act_Umsatz_df, temp_middle_Umsatz_df, temp_pred_Umsatz_df
    Umsatz_for_Vis_df = pd.concat([temp_act_Umsatz_df, temp_middle_Umsatz_df], axis=0)
    Umsatz_for_Vis_df = pd.concat([Umsatz_for_Vis_df, temp_pred_Umsatz_df], axis=0).reset_index(drop=True)

    return Umsatz_for_Vis_df


# ---(산업 투자금액 관련 : 시각화는 하지 않음. 소비전력 예측에 필요한 데이터일 뿐.)---
# temp_2023_Invest_df
def temp_2023_Invest(BW_Invest_df, Grp_1, Grp_2, Grp_3, Grp_val_1, Grp_val_2, Grp_val_3, Grp_val_0):
    # 2023년도 데이터에 대해 필요한 컬럼만 가져온다.
    temp_2023_Invest_df = BW_Invest_df[BW_Invest_df['Jahr']==2023].copy()
    temp_2023_Invest_df = temp_2023_Invest_df[['DN_DT', 'Regionalverband', 'Jahr', 'Investitionen']]

    # Group 컬럼 만들어주기
    # 그룹1
    temp_2023_Invest_df.loc[temp_2023_Invest_df['Regionalverband'].isin(Grp_1), 'Group'] = Grp_val_1

    # 그룹2
    temp_2023_Invest_df.loc[temp_2023_Invest_df['Regionalverband'].isin(Grp_2), 'Group'] = Grp_val_2

    # 그룹3
    temp_2023_Invest_df.loc[temp_2023_Invest_df['Regionalverband'].isin(Grp_3), 'Group'] = Grp_val_3

    # 그룹0
    temp_2023_Invest_df.loc[temp_2023_Invest_df['Group'].isna(), 'Group'] = Grp_val_0

    return temp_2023_Invest_df


# temp_fut_Invest_df - Step1
def temp_fut_Invest_1(TBD_df, Grp_1, Grp_2, Grp_3, Grp_val_1, Grp_val_2, Grp_val_3, Grp_val_0):
    # 과정2-1.
    frame_fut_Invest_df = pd.DataFrame(columns=['DN_DT', 'Regionalverband', 'Jahr', 'Investitionen'])

    # 연도값 입력
    frame_fut_Invest_df['Jahr'] = [2030, 2040, 2050, 2060, 2070]

    # 데이터프레임 모음 딕셔너리 생성
    temp_dict = {}
    for reg in TBD_df['DN_DT'].drop_duplicates():
        # Regierungsverband 지정
        rv = TBD_df.loc[TBD_df['DN_DT'] == reg, 'Regionalverband'].iloc[0]
        # 지역명 채우기
        temp_df = frame_fut_Invest_df.copy()
        temp_df['DN_DT'] = reg
        temp_df['Regionalverband'] = rv

        # 딕셔너리에 넣기
        temp_dict[reg] = temp_df

    # 딕셔너리 내의 데이터프레임 합치기
    temp_fut_Invest_df = pd.concat(temp_dict.values(), axis=0).reset_index(drop=True)

    # 과정2-2. Group 컬럼 만들어주기.
    # 그룹1
    temp_fut_Invest_df.loc[temp_fut_Invest_df['Regionalverband'].isin(Grp_1), 'Group'] = Grp_val_1

    # 그룹2
    temp_fut_Invest_df.loc[temp_fut_Invest_df['Regionalverband'].isin(Grp_2), 'Group'] = Grp_val_2

    # 그룹3
    temp_fut_Invest_df.loc[temp_fut_Invest_df['Regionalverband'].isin(Grp_3), 'Group'] = Grp_val_3

    # 그룹0
    temp_fut_Invest_df.loc[temp_fut_Invest_df['Group'].isna(), 'Group'] = Grp_val_0

    return temp_fut_Invest_df


# temp_fut_Invest_df - Step2(End)
def temp_fut_Invest_2(temp_fut_Invest_df, temp_2023_Invest_df, Grp_val_1, Grp_val_2, Grp_val_3, Grp_val_0, invest_inc_1, invest_inc_2, invest_inc_3, invest_inc):
    # Investitionen컬럼에 계산값 넣기.
    # 그룹별로 값을 계산해서 넣어야 한다.
    # 증가율 변수 : invest_inc, invest_inc_1, invest_inc_2, invest_inc_3

    # 그룹1
    for reg in temp_fut_Invest_df.loc[temp_fut_Invest_df['Group']==Grp_val_1, 'DN_DT'].drop_duplicates():
        for year in temp_fut_Invest_df['Jahr'].drop_duplicates():
            temp_fut_Invest_df.loc[(temp_fut_Invest_df['DN_DT']==reg)&(temp_fut_Invest_df['Jahr']==year), 'Investitionen'] = (
                (temp_2023_Invest_df.loc[temp_2023_Invest_df['DN_DT']==reg, 'Investitionen'].iloc[0] + (temp_2023_Invest_df.loc[temp_2023_Invest_df['DN_DT']==reg, 'Investitionen'].iloc[0])*(invest_inc_1/100)).round()
            )

    # 그룹2
    for reg in temp_fut_Invest_df.loc[temp_fut_Invest_df['Group']==Grp_val_2, 'DN_DT'].drop_duplicates():
        for year in temp_fut_Invest_df['Jahr'].drop_duplicates():
            temp_fut_Invest_df.loc[(temp_fut_Invest_df['DN_DT']==reg)&(temp_fut_Invest_df['Jahr']==year), 'Investitionen'] = (
                (temp_2023_Invest_df.loc[temp_2023_Invest_df['DN_DT']==reg, 'Investitionen'].iloc[0] + (temp_2023_Invest_df.loc[temp_2023_Invest_df['DN_DT']==reg, 'Investitionen'].iloc[0])*(invest_inc_2/100)).round()
            )

    # 그룹3
    for reg in temp_fut_Invest_df.loc[temp_fut_Invest_df['Group']==Grp_val_3, 'DN_DT'].drop_duplicates():
        for year in temp_fut_Invest_df['Jahr'].drop_duplicates():
            temp_fut_Invest_df.loc[(temp_fut_Invest_df['DN_DT']==reg)&(temp_fut_Invest_df['Jahr']==year), 'Investitionen'] = (
                (temp_2023_Invest_df.loc[temp_2023_Invest_df['DN_DT']==reg, 'Investitionen'].iloc[0] + (temp_2023_Invest_df.loc[temp_2023_Invest_df['DN_DT']==reg, 'Investitionen'].iloc[0])*(invest_inc_3/100)).round()
            )

    # 그룹0
    for reg in temp_fut_Invest_df.loc[temp_fut_Invest_df['Group']==Grp_val_0, 'DN_DT'].drop_duplicates():
        for year in temp_fut_Invest_df['Jahr'].drop_duplicates():
            temp_fut_Invest_df.loc[(temp_fut_Invest_df['DN_DT']==reg)&(temp_fut_Invest_df['Jahr']==year), 'Investitionen'] = (
                (temp_2023_Invest_df.loc[temp_2023_Invest_df['DN_DT']==reg, 'Investitionen'].iloc[0] + (temp_2023_Invest_df.loc[temp_2023_Invest_df['DN_DT']==reg, 'Investitionen'].iloc[0])*(invest_inc/100)).round()
            )

    return temp_fut_Invest_df


# fut_Invest_df
def fut_Invest(temp_2023_Invest_df, temp_fut_Invest_df):
    fut_Invest_df = pd.concat([temp_2023_Invest_df, temp_fut_Invest_df], axis=0).reset_index(drop=True)

    return fut_Invest_df



# ---(산업 전기 예측 관련)---

# temp_pred_empl_for_Vis_df : 직원수 예측값
def temp_pred_empl_for_Vis(empl_for_Vis_df):
    temp_pred_empl_for_Vis_df = empl_for_Vis_df[empl_for_Vis_df['Art']=='predict'].copy()
    return temp_pred_empl_for_Vis_df

# temp_pred_Umsatz_for_Vis_df : 매출 예측값
def temp_pred_Umsatz_for_Vis(Umsatz_for_Vis_df):
    temp_pred_Umsatz_for_Vis_df = Umsatz_for_Vis_df[Umsatz_for_Vis_df['Art']=='predict'].copy()
    return temp_pred_Umsatz_for_Vis_df

# fut_Invest_df : 이 함수는 위에 있다.

# IndustEV_for_Vismap_df
def IndustEV_for_Vismap(temp_pred_empl_for_Vis_df, temp_pred_Umsatz_for_Vis_df, fut_Invest_df):
    IndustEV_for_Vismap_df = temp_pred_empl_for_Vis_df.merge(temp_pred_Umsatz_for_Vis_df, on=['DN_DT', 'Regionalverband', 'Jahr', 'Group', 'Art'])
    IndustEV_for_Vismap_df = IndustEV_for_Vismap_df.merge(fut_Invest_df, on=['DN_DT', 'Regionalverband', 'Jahr', 'Group'])

    # 매출과 투자금액 데이터를 정수화
    IndustEV_for_Vismap_df['Gesamtumsatz'] = IndustEV_for_Vismap_df['Gesamtumsatz'].astype(int)
    IndustEV_for_Vismap_df['Investitionen'] = IndustEV_for_Vismap_df['Investitionen'].astype(int)

    return IndustEV_for_Vismap_df


# 스케일링 + 소비전력 예측 : IndustEV_for_Vismap_df
def scal_pred_IndustEV_for_Vismap(IndustEV_for_Vismap_df, normal_reg_scaler, special_reg_scaler, normal_ridge_model, Alb_model, Boeblingen_model, Mannheim_model, Ortenaukreis_model, Rastatt_model, Stuttgart_model, Waldshut_model, Karlsruhe_model):
    # ---(스케일링 파트)---
    # 일반지역 변수 : X_nr, 특별지역 변수 : X_sr
    # 편의를 위해 마스크를 설정한다.
    mask = (IndustEV_for_Vismap_df['DN_DT'].str.contains('Ortenaukreis|Mannheim|Rastatt|Waldshut|Alb-Donau-Kreis|Stuttgart|Böblingen')|((IndustEV_for_Vismap_df['DN_DT'].str.contains('Karlsruhe'))&(IndustEV_for_Vismap_df['DN_DT'].str.contains('kreisfreie'))))

    # 일반지역 마스크 : ~mask
    # 특별지역 마스크 : mask

    # 여기가 변수 설정부분
    X_nr = IndustEV_for_Vismap_df.loc[~mask, ['Beschäftigte', 'Gesamtumsatz', 'Investitionen']]
    X_sr = IndustEV_for_Vismap_df.loc[mask, ['Beschäftigte', 'Gesamtumsatz', 'Investitionen']]

    # 일반지역 스케일링
    X_scaled_nr = normal_reg_scaler.transform(X_nr)
    # 특별지역 스케일링
    X_scaled_sr = special_reg_scaler.transform(X_sr)

    # 스케일 결과를 데이터프레임에 할당.
    # 일반지역
    IndustEV_for_Vismap_df.loc[~mask, ['scaled_Beschäftigte', 'scaled_Gesamtumsatz', 'scaled_Investitionen']] = X_scaled_nr
    # 특별지역
    IndustEV_for_Vismap_df.loc[mask, ['scaled_Beschäftigte', 'scaled_Gesamtumsatz', 'scaled_Investitionen']] = X_scaled_sr

    # ---(예측 파트)---
    # 일반 지역 예측 : normal_ridge_model
    y_nr_pred = normal_ridge_model.predict(X_scaled_nr)
    # 데이터프레임에 값을 할당.
    IndustEV_for_Vismap_df.loc[~mask, 'Stromverbrauch(Industrie)'] = y_nr_pred.round()

    # 특별지역 예측 : 8개 모델
    # Karlsruhe (kreisfreie Stadt)를 제외한 7개 지역 먼저 진행.
    for reg, mod in {'Alb' : Alb_model, 'Böblingen' : Boeblingen_model, 'Mannheim' : Mannheim_model, 'Ortenaukreis' : Ortenaukreis_model, 'Rastatt' : Rastatt_model, 'Stuttgart' : Stuttgart_model, 'Waldshut' : Waldshut_model}.items():
        # 예측
        y_pred = mod.predict(IndustEV_for_Vismap_df.loc[IndustEV_for_Vismap_df['DN_DT'].str.contains(reg), ['scaled_Beschäftigte', 'scaled_Gesamtumsatz', 'scaled_Investitionen']].values)
        # 예측 결과를 IndustEV_for_Vismap_df에 할당.
        IndustEV_for_Vismap_df.loc[IndustEV_for_Vismap_df['DN_DT'].str.contains(reg), 'Stromverbrauch(Industrie)'] = y_pred.round()

    # Karlsruhe (kreisfreie Stadt) 예측.
    y_sr_Karl_pred = Karlsruhe_model.predict(IndustEV_for_Vismap_df.loc[(IndustEV_for_Vismap_df['DN_DT'].str.contains('Karlsruhe'))&(IndustEV_for_Vismap_df['DN_DT'].str.contains('kreisfreie')), ['scaled_Beschäftigte', 'scaled_Gesamtumsatz', 'scaled_Investitionen']].values)
    # 예측 결과를 IndustEV_for_Vismap_df에 할당.
    IndustEV_for_Vismap_df.loc[(IndustEV_for_Vismap_df['DN_DT'].str.contains('Karlsruhe'))&(IndustEV_for_Vismap_df['DN_DT'].str.contains('kreisfreie')), 'Stromverbrauch(Industrie)'] = y_sr_Karl_pred.round()

    return IndustEV_for_Vismap_df


# IndustEV_for_Graph_df - Step1 : 시각화용 데이터프레임
def temp_act_IndEV(BW_IndustEV_df, Grp_1, Grp_2, Grp_3, Grp_val_1, Grp_val_2, Grp_val_3, Grp_val_0):
    temp_act_IndEV_df = BW_IndustEV_df[['DN_DT', 'Regionalverband', 'Jahr', 'Stromverbrauch(Industrie)']].copy()
    # Group 컬럼 추가
    # 그룹1
    temp_act_IndEV_df.loc[temp_act_IndEV_df['Regionalverband'].isin(Grp_1), 'Group'] = Grp_val_1

    # 그룹2
    temp_act_IndEV_df.loc[temp_act_IndEV_df['Regionalverband'].isin(Grp_2), 'Group'] = Grp_val_2

    # 그룹3
    temp_act_IndEV_df.loc[temp_act_IndEV_df['Regionalverband'].isin(Grp_3), 'Group'] = Grp_val_3

    # 그룹0
    temp_act_IndEV_df.loc[temp_act_IndEV_df['Group'].isna(), 'Group'] = Grp_val_0

    # Art 컬럼 추가
    temp_act_IndEV_df['Art'] = 'actual'

    return temp_act_IndEV_df


# IndustEV_for_Graph_df - Step2
def temp_middle_IndEV(temp_act_IndEV_df):
    # 먼저 실제 값에서 2022년의 데이터를 추출.
    temp_middle_IndEV_df = temp_act_IndEV_df[temp_act_IndEV_df['Jahr']==2022].copy()

    # Art컬럼을 predict로 변경
    temp_middle_IndEV_df['Art'] = 'predict'

    return temp_middle_IndEV_df


# IndustEV_for_Graph_df - Step3
def temp_pred_IndEV(IndustEV_for_Vismap_df):
    temp_pred_IndEV_df = IndustEV_for_Vismap_df[['DN_DT', 'Regionalverband', 'Jahr', 'Stromverbrauch(Industrie)', 'Group', 'Art']].copy()
    return temp_pred_IndEV_df


# IndustEV_for_Graph_df - Step4(End)
def IndustEV_for_Graph(temp_act_IndEV_df, temp_middle_IndEV_df, temp_pred_IndEV_df):
    # 준비된 데이터프레임을 아래의 순으로 차례로 합친다.
    # temp_act_IndEV_df, temp_middle_IndEV_df, temp_pred_IndEV_df
    IndustEV_for_Graph_df = pd.concat([temp_act_IndEV_df, temp_middle_IndEV_df], axis=0)
    IndustEV_for_Graph_df = pd.concat([IndustEV_for_Graph_df, temp_pred_IndEV_df], axis=0).reset_index(drop=True)
    return IndustEV_for_Graph_df

# 산업 소비전력 지도시각화 관련
# IndustEV_for_Map_df
def IndustEV_for_Map(IndustEV_for_Graph_df):
    IndustEV_for_Map_df = IndustEV_for_Graph_df[IndustEV_for_Graph_df['Jahr']>=2023].copy()
    # 증가량 계산
    # 지역별 2023년 기준값 뽑기 (index: DN_DT, value: 2023년 소비량)
    baseline_2023_2 = (
        IndustEV_for_Map_df
            .loc[IndustEV_for_Map_df['Jahr'] == 2023]
            .set_index('DN_DT')['Stromverbrauch(Industrie)']
    )

    # DN_DT를 기준으로 2023년 값을 매핑해서 빼기
    IndustEV_for_Map_df['Zunahme(Industrie)'] = (
        IndustEV_for_Map_df['Stromverbrauch(Industrie)']
        - IndustEV_for_Map_df['DN_DT'].map(baseline_2023_2)
    )

    # 증가율 계산
    IndustEV_for_Map_df['Zunahmequote(Industrie)'] = (
        IndustEV_for_Map_df['Zunahme(Industrie)'] /
        IndustEV_for_Map_df['DN_DT'].map(baseline_2023_2) * 100
    )

    return IndustEV_for_Map_df



# BW_Summe_Vis_gdf - Step1 : 최종 종합 gdf
# 가정 소비전력, 산업 소비전력도 이것으로 지도 시각화를 할 수 있다.
def Summe(HausEV_for_Vismap_df, IndustEV_for_Map_df):
    Summe_df = HausEV_for_Vismap_df.merge(IndustEV_for_Map_df, on=['DN_DT', 'Regionalverband', 'Jahr', 'Group', 'Art'])
    # 계산
    Summe_df['Stromverbrauch(Summe)'] = Summe_df['Stromverbrauch(Haushalt)'] + Summe_df['Stromverbrauch(Industrie)']
    # 증가량 계산
    # 지역별 2023년 기준값 뽑기 (index: DN_DT, value: 2023년 소비량)
    baseline_2023_3 = (
        Summe_df
            .loc[Summe_df['Jahr'] == 2023]
            .set_index('DN_DT')['Stromverbrauch(Summe)']
    )

    # DN_DT를 기준으로 2023년 값을 매핑해서 빼기
    Summe_df['Zunahme(Summe)'] = (
        Summe_df['Stromverbrauch(Summe)']
        - Summe_df['DN_DT'].map(baseline_2023_3)
    )

    # 증가율 계산
    Summe_df['Zunahmequote(Summe)'] = (
        Summe_df['Zunahme(Summe)'] /
        Summe_df['DN_DT'].map(baseline_2023_3) * 100
    )
    return Summe_df

# BW_Summe_Vis_gdf - Step2(End) : 최종 종합 gdf
def BW_Summe_Vis(BW_gdf, Summe_df):
    BW_Summe_Vis_gdf = BW_gdf.merge(Summe_df, on=['DN_DT', 'Regionalverband'], how='right')
    # 데이터 형변환1
    BW_Summe_Vis_gdf['Stromverbrauch(Summe)'] = pd.to_numeric(BW_Summe_Vis_gdf['Stromverbrauch(Summe)'], errors='coerce')
    BW_Summe_Vis_gdf['Zunahme(Summe)'] = pd.to_numeric(BW_Summe_Vis_gdf['Zunahme(Summe)'], errors='coerce')
    BW_Summe_Vis_gdf['Zunahmequote(Summe)'] = pd.to_numeric(BW_Summe_Vis_gdf['Zunahmequote(Summe)'], errors='coerce')
    # 데이터 형변환2
    BW_Summe_Vis_gdf['Stromverbrauch(Haushalt)'] = pd.to_numeric(BW_Summe_Vis_gdf['Stromverbrauch(Haushalt)'], errors='coerce')
    BW_Summe_Vis_gdf['Zunahme(Haushalt)'] = pd.to_numeric(BW_Summe_Vis_gdf['Zunahme(Haushalt)'], errors='coerce')
    BW_Summe_Vis_gdf['Zunahmequote(Haushalt)'] = pd.to_numeric(BW_Summe_Vis_gdf['Zunahmequote(Haushalt)'], errors='coerce')

    return BW_Summe_Vis_gdf
