import numpy as np
import pandas as pd
import geopandas as gpd


# ---(DataFrame : Bevölkerung)---

# temp_pred_Bev_df - Step1
def temp_pred_Bev_base(BW_pred_Bev_df, Grp_1, Grp_2, Grp_3, EW_inc_rate, Grp_val_0):
    # 과정1
    # 과정1-1. 어느 그룹에도 속하지 않은 지역 인구 예측
    temp_pred_Bev_base_df = BW_pred_Bev_df.loc[~((BW_pred_Bev_df['Regionalverband'].isin(Grp_1))|(BW_pred_Bev_df['Regionalverband'].isin(Grp_2))|(BW_pred_Bev_df['Regionalverband'].isin(Grp_3))), ['DN_DT', 'Regionalverband', 'Jahr', f'rate_{EW_inc_rate}']]
    # Group 컬럼 추가 : 값 = Grp_val_0
    temp_pred_Bev_base_df['Group'] = Grp_val_0
    # 컬럼명 변경 : concat을 할 때 용이하도록 컬럼명을 Bevölkerung insgesamt로 통일
    temp_pred_Bev_base_df = temp_pred_Bev_base_df.rename(columns={f'rate_{EW_inc_rate}':'Bevölkerung insgesamt'})
    return temp_pred_Bev_base_df


# temp_pred_Bev_df - Step2
def temp_pred_Bev_Grp(BW_pred_Bev_df, groups, group_vals, ew_inc_rates):
    # 데이터프레임을 담을 빈 리스트
    temp_pred_Bev_grp_list = []

    # for루프
    for grp, grp_val, ew_rate in zip(groups, group_vals, ew_inc_rates):
        rate_col = f"rate_{ew_rate}"

        temp_df = BW_pred_Bev_df.loc[
            BW_pred_Bev_df['Regionalverband'].isin(grp),
            ['DN_DT', 'Regionalverband', 'Jahr', rate_col]
        ].copy()

        # Group 컬럼 추가
        temp_df['Group'] = grp_val

        # 컬럼명 변경
        temp_df = temp_df.rename(columns={rate_col: 'Bevölkerung insgesamt'})

        temp_pred_Bev_grp_list.append(temp_df)

    # 그룹에 속한 데이터 합치기
    temp_pred_Bev_Grp_df = pd.concat(temp_pred_Bev_grp_list, ignore_index=True)

    return temp_pred_Bev_Grp_df


# temp_pred_Bev_df - Step3(End)
def temp_pred_Bev(temp_pred_Bev_base_df, temp_pred_Bev_Grp_df):
    # 그룹에 속하지 않은 데이터 + 그룹에 속한 데이터
    temp_pred_Bev_df = pd.concat([temp_pred_Bev_base_df, temp_pred_Bev_Grp_df], ignore_index=True).reset_index(drop=True)

    # 과정2-1.
    # 과정1에서 얻은 temp_pred_Bev_df에 대해 Art컬럼을 추가하여 'predict'값을 할당해준다.
    temp_pred_Bev_df['Art'] = 'predict'
    return temp_pred_Bev_df


# temp_act_Bev_df - Step1(End)
def temp_act_Bev(BW_Bev_df, Grp_1, Grp_2, Grp_3, Grp_val_1, Grp_val_2, Grp_val_3, Grp_val_0):
    # 과정2-2.
    # 2023년 이전의 실제 데이터에 대해 시각화 틀을 만들어준다.
    temp_act_Bev_df = BW_Bev_df[['DN_DT', 'Regionalverband', 'Jahr', 'Bevölkerung insgesamt']].copy()

    # 과정2-3. Group 컬럼 만들어주기.
    # 크게 의미는 없을 수 있지만 그래도 기존에 구분된 그룹대로 지역 그룹을 나누어준다.

    # 그룹1
    temp_act_Bev_df.loc[temp_act_Bev_df['Regionalverband'].isin(Grp_1), 'Group'] = Grp_val_1

    # 그룹2
    temp_act_Bev_df.loc[temp_act_Bev_df['Regionalverband'].isin(Grp_2), 'Group'] = Grp_val_2

    # 그룹3
    temp_act_Bev_df.loc[temp_act_Bev_df['Regionalverband'].isin(Grp_3), 'Group'] = Grp_val_3

    # 그룹0
    temp_act_Bev_df.loc[temp_act_Bev_df['Group'].isna(), 'Group'] = Grp_val_0

    # 과정2-4. Art컬럼 추가
    temp_act_Bev_df['Art'] = 'actual'
    return temp_act_Bev_df


# temp_middle_Bev_df - Step1(End)
def temp_middle_Bev(temp_act_Bev_df):
    # 과정2-5. 2023년값을 한 번 더 추가 : Art = 'predict'로.
    # 이는 그래프 시각화를 유연하게 하기 위함.

    # 먼저 실제 값에서 2023년의 데이터를 추출.
    temp_middle_Bev_df = temp_act_Bev_df[temp_act_Bev_df['Jahr']==2023].copy()

    # Art컬럼을 predict로 변경
    temp_middle_Bev_df['Art'] = 'predict'

    return temp_middle_Bev_df


# Bev_for_Vis_df - Step1(End)
def Bev_for_Vis(temp_pred_Bev_df, temp_middle_Bev_df, temp_act_Bev_df):
    # 과정2-6. 준비된 데이터프레임을 아래의 순으로 차례로 합친다.
    # temp_act_Bev_df, temp_middle_Bev_df, temp_pred_Bev_df
    Bev_for_Vis_df = pd.concat([temp_act_Bev_df, temp_middle_Bev_df], axis=0)
    Bev_for_Vis_df = pd.concat([Bev_for_Vis_df, temp_pred_Bev_df], axis=0).reset_index(drop=True)
    return Bev_for_Vis_df



# ---가정 소비전력 데이터프레임(그래프 시각화용)---

# temp_pred_HausEV_df - Step1
def frame_pred_HausEV():
    # 과정3-1.
    # 시각화용 템플릿 만들기

    frame_pred_HausEV_df = pd.DataFrame(columns=['DN_DT', 'Regionalverband', 'Jahr', 'Stromverbrauch(Haushalt)'])

    # 연도값 입력
    frame_pred_HausEV_df['Jahr'] = [2030, 2040, 2050, 2060, 2070]

    # Art컬럼을 추가하여 'predict'값을 할당
    frame_pred_HausEV_df['Art'] = 'predict'
    return frame_pred_HausEV_df

# temp_pred_HausEV_df - Step2
def temp_pred_HausEV_1(frame_pred_HausEV_df, TBD_df):
    # 데이터프레임 모음 딕셔너리 생성
    temp_dict = {}
    for reg in TBD_df['DN_DT'].drop_duplicates():
        # Regierungsverband 지정
        rv = TBD_df.loc[TBD_df['DN_DT'] == reg, 'Regionalverband'].iloc[0]
        # 지역명 채우기
        temp_df = frame_pred_HausEV_df.copy()
        temp_df['DN_DT'] = reg
        temp_df['Regionalverband'] = rv

        # 딕셔너리에 넣기
        temp_dict[reg] = temp_df

    # 딕셔너리 내의 데이터프레임 합치기
    temp_pred_HausEV_df = pd.concat(temp_dict.values(), axis=0).reset_index(drop=True)
    return temp_pred_HausEV_df

# temp_pred_HausEV_df - Step3
def temp_pred_HausEV_2(temp_pred_HausEV_df, Grp_1, Grp_2, Grp_3, Grp_val_1, Grp_val_2, Grp_val_3, Grp_val_0):
    # 과정3-2. Group 컬럼 만들어주기.

    # 그룹1
    temp_pred_HausEV_df.loc[temp_pred_HausEV_df['Regionalverband'].isin(Grp_1), 'Group'] = Grp_val_1

    # 그룹2
    temp_pred_HausEV_df.loc[temp_pred_HausEV_df['Regionalverband'].isin(Grp_2), 'Group'] = Grp_val_2

    # 그룹3
    temp_pred_HausEV_df.loc[temp_pred_HausEV_df['Regionalverband'].isin(Grp_3), 'Group'] = Grp_val_3

    # 그룹0
    temp_pred_HausEV_df.loc[temp_pred_HausEV_df['Group'].isna(), 'Group'] = Grp_val_0

    return temp_pred_HausEV_df


# temp_pred_HausEV_df - Step4(End)
def temp_pred_HausEV_3(temp_pred_HausEV_df, Bev_for_Vis_df, Grp_val_1, Grp_val_2, Grp_val_3, Grp_val_0, EW_GJ_1, EW_GJ_2, EW_GJ_3, EW_GJ):
    # 과정3-3. Stromverbrauch(Haushalt)컬럼에 계산값 넣기.
    # 그룹별로 값을 계산해서 넣어야 한다.

    # 그룹1
    for reg in temp_pred_HausEV_df.loc[temp_pred_HausEV_df['Group']==Grp_val_1, 'DN_DT'].drop_duplicates():
        for year in temp_pred_HausEV_df['Jahr'].drop_duplicates():
            temp_pred_HausEV_df.loc[(temp_pred_HausEV_df['DN_DT']==reg)&(temp_pred_HausEV_df['Jahr']==year), 'Stromverbrauch(Haushalt)'] = (
                ((Bev_for_Vis_df.loc[(Bev_for_Vis_df['DN_DT']==reg)&(Bev_for_Vis_df['Jahr']==year), 'Bevölkerung insgesamt'] * EW_GJ_1) / 1000).iloc[0].round()
            )

    # 그룹2
    for reg in temp_pred_HausEV_df.loc[temp_pred_HausEV_df['Group']==Grp_val_2, 'DN_DT'].drop_duplicates():
        for year in temp_pred_HausEV_df['Jahr'].drop_duplicates():
            temp_pred_HausEV_df.loc[(temp_pred_HausEV_df['DN_DT']==reg)&(temp_pred_HausEV_df['Jahr']==year), 'Stromverbrauch(Haushalt)'] = (
                ((Bev_for_Vis_df.loc[(Bev_for_Vis_df['DN_DT']==reg)&(Bev_for_Vis_df['Jahr']==year), 'Bevölkerung insgesamt'] * EW_GJ_2) / 1000).iloc[0].round()
            )

    # 그룹3
    for reg in temp_pred_HausEV_df.loc[temp_pred_HausEV_df['Group']==Grp_val_3, 'DN_DT'].drop_duplicates():
        for year in temp_pred_HausEV_df['Jahr'].drop_duplicates():
            temp_pred_HausEV_df.loc[(temp_pred_HausEV_df['DN_DT']==reg)&(temp_pred_HausEV_df['Jahr']==year), 'Stromverbrauch(Haushalt)'] = (
                ((Bev_for_Vis_df.loc[(Bev_for_Vis_df['DN_DT']==reg)&(Bev_for_Vis_df['Jahr']==year), 'Bevölkerung insgesamt'] * EW_GJ_3) / 1000).iloc[0].round()
            )

    # 그룹0
    for reg in temp_pred_HausEV_df.loc[temp_pred_HausEV_df['Group']==Grp_val_0, 'DN_DT'].drop_duplicates():
        for year in temp_pred_HausEV_df['Jahr'].drop_duplicates():
            temp_pred_HausEV_df.loc[(temp_pred_HausEV_df['DN_DT']==reg)&(temp_pred_HausEV_df['Jahr']==year), 'Stromverbrauch(Haushalt)'] = (
                ((Bev_for_Vis_df.loc[(Bev_for_Vis_df['DN_DT']==reg)&(Bev_for_Vis_df['Jahr']==year), 'Bevölkerung insgesamt'] * EW_GJ) / 1000).iloc[0].round()
            )

    return temp_pred_HausEV_df



# temp_act_HausEV_df - Step1(End)
def temp_act_HausEV(BW_HausEV_Kreis_df, Grp_1, Grp_2, Grp_3, Grp_val_1, Grp_val_2, Grp_val_3, Grp_val_0):
    # 과정4-1. 시각화 템플릿에 맞게 가공
    temp_act_HausEV_df = BW_HausEV_Kreis_df[['DN_DT', 'Jahr', 'Regionalverband', 'Stromverbrauch(Haushalt)']].copy()

    # 과정4-2. Art 컬럼 추가 : 'actual'
    temp_act_HausEV_df['Art'] = 'actual'

    # Group 컬럼 추가
    # 그룹1
    temp_act_HausEV_df.loc[temp_act_HausEV_df['Regionalverband'].isin(Grp_1), 'Group'] = Grp_val_1

    # 그룹2
    temp_act_HausEV_df.loc[temp_act_HausEV_df['Regionalverband'].isin(Grp_2), 'Group'] = Grp_val_2

    # 그룹3
    temp_act_HausEV_df.loc[temp_act_HausEV_df['Regionalverband'].isin(Grp_3), 'Group'] = Grp_val_3

    # 그룹0
    temp_act_HausEV_df.loc[temp_act_HausEV_df['Group'].isna(), 'Group'] = Grp_val_0

    return temp_act_HausEV_df

# temp_middle_HausEV_df - Step1(End)
def temp_middle_HausEV(temp_act_HausEV_df, Grp_1, Grp_2, Grp_3, Grp_val_1, Grp_val_2, Grp_val_3, Grp_val_0):
    # 과정4-3. 2023년값을 한 번 더 추가 : Art = 'predict'로.
    # 이는 그래프 시각화를 유연하게 하기 위함.
    temp_middle_HausEV_df = temp_act_HausEV_df[temp_act_HausEV_df['Jahr']==2023].copy()
    temp_middle_HausEV_df['Art'] = 'predict'

    # 과정4-4. Group 컬럼 만들어주기.
    # 그룹1
    temp_middle_HausEV_df.loc[temp_middle_HausEV_df['Regionalverband'].isin(Grp_1), 'Group'] = Grp_val_1

    # 그룹2
    temp_middle_HausEV_df.loc[temp_middle_HausEV_df['Regionalverband'].isin(Grp_2), 'Group'] = Grp_val_2

    # 그룹3
    temp_middle_HausEV_df.loc[temp_middle_HausEV_df['Regionalverband'].isin(Grp_3), 'Group'] = Grp_val_3

    # 그룹0
    temp_middle_HausEV_df.loc[temp_middle_HausEV_df['Group'].isna(), 'Group'] = Grp_val_0

    return temp_middle_HausEV_df

# HausEV_for_Vis_df - Step1(End)
def HausEV_for_Vis(temp_act_HausEV_df, temp_middle_HausEV_df, temp_pred_HausEV_df):
    # 과정4-4. 준비된 데이터프레임을 아래의 순으로 차례로 합친다.
    # temp_act_HausEV_df, temp_middle_HausEV_df, temp_pred_HausEV_df
    HausEV_for_Vis_df = pd.concat([temp_act_HausEV_df, temp_middle_HausEV_df], axis=0)
    HausEV_for_Vis_df = pd.concat([HausEV_for_Vis_df, temp_pred_HausEV_df], axis=0).reset_index(drop=True)
    return HausEV_for_Vis_df



# ---가정 소비전력 데이터프레임(지도 시각화용)---

# HausEV_for_Vismap_df - Step1(End)
def HausEV_for_Vismap(HausEV_for_Vis_df):
    # Art컬럼이 predict인 데이터 추출
    HausEV_for_Vismap_df = HausEV_for_Vis_df.loc[HausEV_for_Vis_df['Art']=='predict'].copy()


    # 증가량 계산
    # 지역별 2023년 기준값 뽑기 (index: DN_DT, value: 2023년 소비량)
    baseline_2023 = (
        HausEV_for_Vismap_df
            .loc[HausEV_for_Vismap_df['Jahr'] == 2023]
            .set_index('DN_DT')['Stromverbrauch(Haushalt)']
    )

    # DN_DT를 기준으로 2023년 값을 매핑해서 빼기
    HausEV_for_Vismap_df['Zunahme(Haushalt)'] = (
        HausEV_for_Vismap_df['Stromverbrauch(Haushalt)']
        - HausEV_for_Vismap_df['DN_DT'].map(baseline_2023)
    )

    # 증가율 계산 : Zunahmequote
    HausEV_for_Vismap_df['Zunahmequote(Haushalt)'] = (
        HausEV_for_Vismap_df['Zunahme(Haushalt)'] /
        HausEV_for_Vismap_df['DN_DT'].map(baseline_2023) * 100
    )
    return HausEV_for_Vismap_df
