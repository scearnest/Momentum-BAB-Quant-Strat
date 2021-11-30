import modin.pandas as md
import pandas as pd
from pandas.tseries.offsets import *
import numpy as np
import matplotlib.pyplot as plt
from matplotlib_inline.backend_inline import set_matplotlib_formats
from copy import deepcopy
set_matplotlib_formats('svg')

def import_clean_data(path):
    '''
    import and clean up the crsp_monthly csv file
    '''
    print('importing and cleaning data...')
    df = pd.read_csv(path)
    # convert dates to integers
    df['date'] = df['date'].astype(int)
    # convert dates to datetime
    df['date'] = pd.to_datetime(df['date'],format='%Y%m%d') 
    # convert rets to floats, set non-numeric rets to nans                     
    df['RET'] = pd.to_numeric(df['RET'], errors='coerce')  
    # compute market capitalization                                                                  
    df['MKTCAP'] = np.abs(df['PRC']) * df['SHROUT'] * 1000
    # sort data by stock, date
    df = df.sort_values(by=['PERMNO','date'],ascending=True)
    # set tickers to strings
    df['TICKER'] = df['TICKER'].astype(str)
    return df

def compute_price_volume(df):
    '''
    compute average daily trading volume
    '''
    print('computing average daily volumes...')
    df['ADV'] = (np.abs(df['PRC']) * df['VOL'])/21
    return df

def compute_mkt_stock_volatilities(df):
    '''
    compute rolling volatilities
    '''
    print('computing market/stock volatilities...')
    df['STD'] = df.groupby('PERMNO')[['RET']].apply(lambda x: x.rolling(window=24,min_periods=24).std())
    df['STD_M'] = df.groupby('PERMNO')[['vwretd']].apply(lambda x: x.rolling(window=24,min_periods=24).std())
    return df

def compute_rolling_correlations(df):
    '''
    compute rolling correlations between each stock and the value-weighted market
    '''
    r = df['RET']
    m = df['vwretd']
    correl = pd.DataFrame(r.rolling(window=60,min_periods=60).corr(m))
    return correl

def compute_betas(df):
    '''
    compute rolling betas of each stock relative to the market
    '''
    corr = df['COR']
    std_r = df['STD']
    std_m = df['STD_M']
    beta = pd.DataFrame(corr * (std_r / std_m))
    return beta

def compute_adjusted_betas(df):
    '''
    shrink betas toward one
    '''
    print('computing adjusted betas...')
    df['BETA'] = (df['BETA'] * (0.6) + (0.4) * 1)
    return df

def compute_adjusted_betas_alt(df):
    '''
    shrink betas toward cross-sectional mean
    '''
    print('computing adjusted betas...')
    frames = []
    for name, frame in df.groupby('date'):
        mean_beta = frame['BETA'].mean()
        frame['BETA'] = (frame['BETA'] * (0.6) + (0.4) * mean_beta)
        frames.append(frame)
    df = pd.concat(frames)
    return df

def compute_reduced_dataset(df,quantiles,top_mktcap=True,no_companies_per_quantile=500):
    frames = []
    if top_mktcap == True:
        print(f'reducing data to just top and bottom quantiles, taking top {no_companies_per_quantile} companies per quantile...')
    else:
        print(f'reducing data to just top and bottom quantiles, taking all companies...')
    for name, frame in df.groupby(['date','QTLS']):
        if name[1] == 1 or name[1] == quantiles:
            if top_mktcap == True:
                frame = frame.sort_values(by=['MKTCAP']).head(no_companies_per_quantile)
                frames.append(frame)
            else:
                frames.append(frame)
        else:
            continue
    df = pd.concat(frames)
    return df
            
def compute_holding_periods(df):
    print('computing one-month holding periods...')
    df = df.set_index(['PERMNO','date'])
    df['FRM_DATE'] = df.index.get_level_values(level='date')
    df['HLD_BEGIN'] = df.index.get_level_values(level='date') + MonthEnd(0) + MonthBegin(1)
    df['HLD_END'] = df.index.get_level_values(level='date') + MonthEnd(0) + MonthEnd(1)
    df = deepcopy(df[['FRM_DATE','QTLS','HLD_BEGIN','HLD_END','MKTCAP','BETA']])
    return df

def merge_holding_periods_on_returns(data,holding_periods):
    print('merging return and holding period data...')
    portfolio = pd.merge(left=(data[['PERMNO','date','RET']]),right=(holding_periods),on=['PERMNO'],how='inner')
    portfolio = portfolio[(portfolio['HLD_BEGIN'] <= portfolio['date']) & (portfolio['date'] <= portfolio['HLD_END'])]
    portfolio = portfolio[['PERMNO','FRM_DATE','QTLS','HLD_BEGIN','HLD_END','date','RET','MKTCAP','BETA']]

    print('scaling weights according to betas...')
    frames = []
    for name, frame in portfolio.groupby(['date','QTLS','FRM_DATE']):
        # compute equal weights for each monthly quantile portfolio
        frame['EQ_WTS'] = np.ones(len(frame)) / frame['BETA'].count()
        # compute quantile portfolio beta
        port_beta = frame['EQ_WTS'] @ frame['BETA']
        # scale weights by quantile portfolio beta
        frame['EQ_WTS_SCLD'] = frame['EQ_WTS'] / port_beta
        # compute stock level returns using scaled weights
        frame['RET'] = frame['EQ_WTS_SCLD'] * frame['RET']
        # don't allow more than 2x leverage in the portfolio
        if abs(frame['EQ_WTS_SCLD'].sum()) > 2:
            frame['EQ_WTS_SCLD'] = frame['EQ_WTS_SCLD']/(abs(frame['EQ_WTS_SCLD'].sum())/2)
        frames.append(frame)
    # merge computations into one dataframe
    portfolio = pd.concat(frames)
    # sum up weighted-returns to get the portfolio return for each month
    port = portfolio.groupby(['date','QTLS','FRM_DATE'])[['RET']].sum()
    return port.reset_index(), portfolio

def pivot_and_plot_returns(df,quantiles):
    # pivot returns
    eq_weighted_strategy_pivot = df.pivot(index='date', columns='QTLS', values='RET').add_prefix('quantile ').rename(columns={'quantile 1': 'low',f'quantile {quantiles}': 'high'}).fillna(0)
    eq_weighted_strategy_pivot['low-high'] = eq_weighted_strategy_pivot.low - eq_weighted_strategy_pivot.high
    # compute long_short cumulative returns
    eq_weighted_strategy_pivot['long_only_lo'] = np.log(1 + eq_weighted_strategy_pivot.low).cumsum()
    eq_weighted_strategy_pivot['long_only_hi'] = np.log(1 + eq_weighted_strategy_pivot.high).cumsum()
    eq_weighted_strategy_pivot['long_short'] = np.log(1 + (eq_weighted_strategy_pivot.low - eq_weighted_strategy_pivot.high)).cumsum()
    # compute plots for each strategy
    plt.figure(figsize=(10,4))
    eq_weighted_strategy_pivot['long_only_lo'].plot()
    eq_weighted_strategy_pivot['long_only_hi'].plot()
    eq_weighted_strategy_pivot['long_short'].plot()
    plt.title('Cumulative Returns Before Leverage')
    plt.legend()
    plt.show()