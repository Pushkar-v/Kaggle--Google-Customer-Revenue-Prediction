# -*- coding: utf-8 -*-
"""
@author: Pushkar Vengurlekar
"""

import os
import json
import swifter
import numpy as np

from pandas.io.json import json_normalize
from ast import literal_eval
import time
import datetime

from sklearn.preprocessing import OneHotEncoder


JSON_COLUMNS = ['device', 'geoNetwork', 'totals', 'trafficSource']
COLUMNS = ['fullVisitorId', 'date', 'totals']
data = pd.read_csv(r'raw_data/train_v2.csv', converters={column: json.loads for column in JSON_COLUMNS},dtype={'fullVisitorId': 'str'}, usecols = COLUMNS, parse_dates = ['date'])
data.head()
training_from = [20160801,20160901, 20161001, 20161101, 20161201, 20170101, 20170201, 20170301, 20170401, 20170501, 20170601, 20170701, 20170801]
training_to = [20170115,20170215,  20170315,  20170415,  20170515,  20170615,  20170715,  20170815,  20170915,  20171015,  20171115,  20171215,  20180115]

target_from = [20170301,20170401,20170501,20170601,20170701,20170801,20170901,20171001,20171101,20171201,20180101,20180201,20180301]
target_to = [20170501,20170601,20170701,20170801,20170901,20171001,20171101,20171201,20180101,20180201,20180301,20180401,20180501]


def target_variable(filename, from_date, to_date):
    df = data[(data['date']>=from_date) & (data['date']<to_date)]
    print("date range",min(df['date']), max(df['date']))

    df = load_df(df)
    df_target = df[['fullVisitorId', 'totals.transactionRevenue']]
    df_target['totals.transactionRevenue'] = df_target['totals.transactionRevenue'].astype(float)
    df_target = df_target.fillna(0)
    df_target = df_target.groupby('fullVisitorId').agg({'totals.transactionRevenue':'sum'})
    df_target['LogRevenue'] = np.log(df_target['totals.transactionRevenue'].astype(float) + 1)
    df_target['Purchase'] = df_target['LogRevenue'].swifter.apply(flag)
    df_target = df_target[['LogRevenue', 'Purchase']].reset_index()
    df_target.columns = ['fullVisitorId', 'LogRevenue_y', 'Purchase_y']

    current_df = pd.read_csv('final_data/'+filename, sep = ',')

    final = pd.merge(current_df, df_target, on = 'fullVisitorId', how = 'left')
#     final['LogRevenue_y'] = np.where(final['LogRevenue_y'].isnull(), 0, final['LogRevenue_y'])
#     final['Purchase_y'] = np.where(final['Purchase_y'] == 1, 1, 0)
    final= final.fillna(0)
#     final['LogRevenue_y'] = final['LogRevenue_y'].fillna(0)
    final.to_csv('final/'+filename, sep = ',', index = False , float_format='%f')
    print('final file characteristics')
    print(final.shape)
    print(final[['Purchase_y', 'LogRevenue_y']].sum())

    return final

def data_munge(df):
#     a = time.time()

    df_target = df[['fullVisitorId', 'totals.transactionRevenue']]
    df_target['totals.transactionRevenue'] = df_target['totals.transactionRevenue'].astype(float)
    df_target = df_target.fillna(0)
    df_target = df_target.groupby('fullVisitorId').agg({'totals.transactionRevenue':'sum'})
    df_target['LogRevenue'] = np.log(df_target['totals.transactionRevenue'].astype(float) + 1)
    df_target['Purchase'] = df_target['LogRevenue'].swifter.apply(flag)
    df_target = df_target[['LogRevenue', 'Purchase']].reset_index()

    df_channel = training[['fullVisitorId', 'channelGrouping']]
    df_channel = pd.concat((df_channel,pd.get_dummies(df_channel.channelGrouping)),1)
    df_channel = df_channel.drop('channelGrouping', axis = 1)
    df_channel = df_channel.groupby('fullVisitorId').sum().reset_index()

    df_visits = df[['fullVisitorId','visitNumber']]
    df_visits = df_visits.groupby('fullVisitorId').size().reset_index()
    df_visits.columns = ['fullVisitorId', 'visits']

    df['totals.bounces'] = df['totals.bounces'].swifter.apply(convert_float)
    df['totals.hits'] = df['totals.hits'].swifter.apply(convert_float)
    df['totals.pageviews'] = df['totals.pageviews'].swifter.apply(convert_float)
#     df['totals.sessionQualityDim'] = df['totals.sessionQualityDim'].swifter.apply(convert_float)
    df['totals.newVisits'] = df['totals.newVisits'].swifter.apply(convert_float)
    df['totals.timeOnSite'] = df['totals.timeOnSite'].swifter.apply(convert_float)
    df['totals.transactions'] = df['totals.transactions'].swifter.apply(convert_float)

    df_boyang = df.fillna(-1).groupby("fullVisitorId").agg({"totals.bounces":"sum","totals.hits":"sum","totals.newVisits":"sum",\
                                                            "totals.pageviews":"sum", \
#                                                             "totals.sessionQualityDim":"sum",\
                                                            "totals.timeOnSite":"sum","totals.transactions":"sum"})
    df_boyang.reset_index(level=0, inplace=True)



    df['trafSrc_adCont_isNULL'] = np.where(df['trafficSource.adContent'].isnull(), 1, 0)
    df['trafSrc_adCont_Others'] = np.where(df['trafficSource.adContent'].isin(["First Full Auto Template Test Ad","Ad from 11/7/16",\
                                                                               "{KeyWord:Google Brand Items}","Full auto ad TEXT ONLY",\
                                                                               "LeEco_1a","JD_5a_v1","{KeyWord:Google Men's T-Shirts}",\
                                                                               "Full auto ad TEXT/NATIVE","Official Google Merchandise - Fast Shipping",\
                                                                               "free shipping","Full auto ad with Primary Color","Free Shipping!",\
                                                                               "GA Help Center","{KeyWord:Google Branded Outerwear}","Full auto ad NATIVE ONLY",\
                                                                               "Swag w/ Google Logos"]), 1, 0)
    df['trafSrc_adCont_20disc'] = np.where(df['trafficSource.adContent'] == '20% discount', 1, 0)
    df['trafSrc_adCont_GoogleMerc'] = np.where(df['trafficSource.adContent'] == 'Google Merchandise', 1, 0)
    df['trafSrc_adCont_DispAd15'] = np.where(df['trafficSource.adContent'] == 'Display Ad created 3/11/15', 1, 0)
    df['trafSrc_adCont_Ad121316'] = np.where(df['trafficSource.adContent'] == 'Ad from 12/13/16', 1, 0)
    df['trafSrc_adCont_DispAd14'] = np.where(df['trafficSource.adContent'] == 'Display Ad created 3/11/14', 1, 0)
    df['trafSrc_adCont_FullAutoAd'] = np.where(df['trafficSource.adContent'] == 'Full auto ad IMAGE ONLY', 1, 0)
    df['trafSrc_NetType_GSearch'] = np.where(df['trafficSource.adwordsClickInfo.adNetworkType'] == 'Google Search', 1, 0)
    df['trafSrc_NetType_SrchPartners'] = np.where(df['trafficSource.adwordsClickInfo.adNetworkType'] == 'Search Partners', 1, 0)
    df['trafSrc_gcl_isNULL'] = np.where(df['trafficSource.adwordsClickInfo.gclId'].isnull(), 1, 0)
    df['trafSrc_isVidAd'] = np.where(df['trafficSource.adwordsClickInfo.isVideoAd'].isnull(), 1, 0)
    df['trafSrc_page_isNULL'] = np.where(df['trafficSource.adwordsClickInfo.page'].isnull(), 1, 0)
    df['trafSrc_page_1'] = np.where(df['trafficSource.adwordsClickInfo.page'] == '1', 1, 0)
    df['trafSrc_page_others'] = np.where(df['trafficSource.adwordsClickInfo.page'].isin(['2','3','4','5','7','9']), 1, 0)

    df_samira = df.groupby('fullVisitorId').agg({'trafSrc_adCont_isNULL': 'sum', 'trafSrc_adCont_Others': 'sum', \
                                                 'trafSrc_adCont_20disc': 'sum', 'trafSrc_adCont_GoogleMerc': 'sum', \
                                                 'trafSrc_adCont_DispAd15': 'sum', 'trafSrc_adCont_Ad121316': 'sum', \
                                                 'trafSrc_adCont_DispAd14': 'sum', 'trafSrc_adCont_FullAutoAd': 'sum', \
                                                 'trafSrc_NetType_GSearch': 'sum', 'trafSrc_NetType_SrchPartners': 'sum', \
                                                 'trafSrc_gcl_isNULL': 'sum', 'trafSrc_isVidAd': 'sum', 'trafSrc_page_isNULL': 'sum', \
                                                 'trafSrc_page_1': 'sum', 'trafSrc_page_others': 'sum'}).reset_index()

    df_samira.reset_index(level=0, inplace=True)


    df['trafSrc_page2_isNULL'] = np.where(df['trafficSource.adwordsClickInfo.slot'].isnull(), 1, 0)
    df['trafSrc_page2'] = np.where(df['trafficSource.adwordsClickInfo.slot'] == 'Top', 1, 0)
    df['trafSrc_page2_others'] = np.where(df['trafficSource.adwordsClickInfo.slot'] == 'RHS', 1, 0)
    df['trafSrc_camp'] = np.where(df['trafficSource.campaign'].isin(['Data Share Promo','AW - Dynamic Search Ads Whole Site',\
                                                                     'AW - Electronics','AW - Accessories','All Products']),1,0)
    df['trafSrc_isTrueDir_isNULL'] = np.where(df['trafficSource.isTrueDirect'].isnull(), 1, 0)
    df['trafSrc_isTrueDir_True'] = np.where(df['trafficSource.isTrueDirect'] == 'True', 1, 0)

    df['trafSrc_kw_others'] = np.where(df['trafficSource.keyword'].isin(['water bottle','(not provided)',
           '(Remarketing/Content targeting)', '6qEhsCssdK0z36ri',
           '(automatic matching)', 'Google men', '1hZbAqLCbjwfgOH7',
           '1X4Me6ZKNV0zg-jV', 'google online merchandise',
           'google water bottle', '(User vertical targeting)',
           'google company store', 'https://www.googlemerchandisestore.com/',
           'letterman jacket ebay', 'google canada',
           'Google where is gabika clothing', 'google online shops',
           'google Merchandising kosten', 'www.google.com bag',
           'youtube youtube', 'youtube www youtube com', 'merchdise',
           'yputube', 'youtube', 'www you tope', 'youtubeshop',
           'googel store', 'google shirts buy', 'Google mens']), 1, 0)

    df['trafSrc_kw_isNULL'] = np.where(df['trafficSource.keyword'].isnull(),1,0)

    df['trafSrc_medium_org'] = np.where(df['trafficSource.medium'] == 'organic', 1, 0)
    df['trafSrc_medium_ref'] = np.where(df['trafficSource.medium'] == 'referral', 1, 0)
    df['trafSrc_medium_isNULL'] = np.where(df['trafficSource.medium'] == '(none)', 1, 0)
    df['trafSrc_medium_cpc'] = np.where(df['trafficSource.medium'] == 'cpc', 1, 0)
    df['trafSrc_medium_aff'] = np.where(df['trafficSource.medium'] == 'affiliate', 1, 0)
    df['trafSrc_medium_cpm'] = np.where(df['trafficSource.medium'] == 'cpm', 1, 0)


    df_shobhit = df.groupby('fullVisitorId').agg({'trafSrc_page2_isNULL': 'sum', 'trafSrc_page2': 'sum', 'trafSrc_page2_others': 'sum', \
                                                  'trafSrc_camp': 'sum', 'trafSrc_isTrueDir_isNULL': 'sum', 'trafSrc_isTrueDir_True': 'sum', \
                                                  'trafSrc_kw_others': 'sum', 'trafSrc_kw_isNULL': 'sum', 'trafSrc_medium_org': 'sum', \
                                                  'trafSrc_medium_ref': 'sum', 'trafSrc_medium_isNULL': 'sum', 'trafSrc_medium_cpc': 'sum', \
                                                  'trafSrc_medium_aff': 'sum','trafSrc_medium_cpm': 'sum'})


    df_shobhit.reset_index(level=0, inplace=True)
#     print(df_target.shape)
    df_final = pd.merge(df_target, df_visits, how = 'left', on = 'fullVisitorId')
#     print(df_final.shape)
    df_final = pd.merge(df_final, df_channel, how = 'left', on = 'fullVisitorId')
#     print(df_final.shape)
    df_final = pd.merge(df_final, df_boyang, how = 'left', on = 'fullVisitorId')
#     print(df_final.shape)
    df_final = pd.merge(df_final, df_samira, how = 'left', on = 'fullVisitorId')
#     print(df_final.shape)
    df_final = pd.merge(df_final, df_shobhit, how = 'left', on = 'fullVisitorId')
#     print(df_final.shape)

    print('data munge done!!')

    return df_final

# Splitting The Dataaframes
def mungemunge(from_date, to_date):
    train = data[(data['date']>=from_date) & (data['date']<=to_date)]
    print("Date Range",min(train['date']), max(train['date']))
    print("# of Visitors",len(train.fullVisitorId.unique()))
    print("shape of training",train.shape)

#     print("Load Dataframe to expand JSON")
    training = load_df(train)
    print("Final Shape",training.shape)
    print("# of Visitors",len(training.fullVisitorId.unique()))
#     print('done!!!')
    return training

# Getting Jsons
def load_df(df1):
#     a = time.time()
    print('input shape:', df1.shape)
    JSON_COLUMNS = ['device', 'geoNetwork', 'totals', 'trafficSource']
    for column in JSON_COLUMNS:
#         print(column)
        column_as_df = json_normalize(df1[column])
        column_as_df.columns = [f"{column}.{subcolumn}" for subcolumn in column_as_df.columns]
        df1 = df1.drop(column, axis=1).merge(column_as_df, right_index=True, left_index=True, how = 'left')
#     print(time.time() -a)
    print('Splitting Json Done!! final shape:', df1.shape)
    return df1

# Changing Datatypes
def convert_float(x):
    try:
        return float(x)
    except ValueError:
        return np.nan

def flag(x):
    if float(x) == 0:
        return 0
    else:
        return 1

for i in range(13):
    x = time.time()
    print('Iteration:',i)
    training = mungemunge(training_from[i], training_to[i])
    train1 = data_munge(training)
    print(train1.groupby('Purchase').size())
    train1.to_csv('final_data/train' + str(i) + '.csv', index = False, float_format='%f')
    print('\n')
    final1 = target_variable('train' + str(i) + '.csv', target_from[i], target_to[i])

    print(time.time() - a)
    print('Iteration Done')


for i in range(13):
    target_variable(data, str(training_from[i]), str(training_to[i]), str(i))


