#!/usr/bin/env python3
"""
Credit Suisse KMV Model - Final Data Integration
Merge stock prices, volatility, risk-free rates and calculate market cap
"""

import pandas as pd
import numpy as np

# === 1. Load data ===
stock_prices = pd.read_csv('/Users/huzhichao/Desktop/金融风险管理/数据搜集/CS_Stock_Price_Monthly.csv')
volatility = pd.read_csv('/Users/huzhichao/Desktop/金融风险管理/数据搜集/CS_Equity_Volatility_Monthly.csv')
risk_free = pd.read_csv('/Users/huzhichao/Desktop/金融风险管理/数据搜集/Risk_Free_Rate_US1Y_FRED.csv')

# === 2. Merge monthly data ===
monthly = stock_prices.merge(volatility, on='Date', how='left')
monthly = monthly.merge(risk_free[['Date', 'GS1_Rate_Decimal']], on='Date', how='left')

# === 3. Calculate shares outstanding ===
# 2020: ~2,460 Mn shares (from 20-F 2020)
# 2021: ~4,459 Mn shares (from 6-K Q1 2021)
monthly['Date_dt'] = pd.to_datetime(monthly['Date'])
monthly['Year'] = monthly['Date_dt'].dt.year
monthly['Shares_Outstanding_Mn'] = np.where(
    monthly['Year'] == 2020, 2460, 
    np.where(monthly['Year'] == 2021, 4459, np.nan)
)

# === 4. Calculate Market Cap ===
monthly['Market_Cap_USD_Mn'] = monthly['Stock_Price_USD'] * monthly['Shares_Outstanding_Mn']

# === 5. Save merged monthly data ===
output = monthly[['Date', 'Stock_Price_USD', 'Shares_Outstanding_Mn', 'Market_Cap_USD_Mn', 
                  'Equity_Vol_Annual', 'GS1_Rate_Decimal']].copy()
output.rename(columns={'GS1_Rate_Decimal': 'Risk_Free_Rate'}, inplace=True)
output['Type'] = 'Monthly'
output = output[['Date', 'Stock_Price_USD', 'Shares_Outstanding_Mn', 'Market_Cap_USD_Mn',
                 'Equity_Vol_Annual', 'Risk_Free_Rate', 'Type']]
output.to_csv('/Users/huzhichao/Desktop/金融风险管理/数据搜集/CS_KMV_Monthly_Data.csv', index=False)
print("✓ Saved: CS_KMV_Monthly_Data.csv")

# === 6. Calculate quarterly summary for KMV inputs ===
quarterly_dates = ['2020-12-01', '2021-03-01', '2021-06-01', '2021-09-01', '2021-12-01']
quarterly_labels = ['2020-12-31', '2021-03-31', '2021-06-30', '2021-09-30', '2021-12-31']

quarterly_df = pd.DataFrame()
for q_date, q_label in zip(quarterly_dates, quarterly_labels):
    row = monthly[monthly['Date'] == q_date]
    if len(row) > 0:
        tmp = pd.DataFrame([{
            'Date': q_label,
            'Stock_Price_USD': row['Stock_Price_USD'].values[0],
            'Shares_Outstanding_Mn': row['Shares_Outstanding_Mn'].values[0],
            'Market_Cap_USD_Mn': row['Market_Cap_USD_Mn'].values[0],
            'Equity_Vol_Annual': row['Equity_Vol_Annual'].values[0],
            'Risk_Free_Rate': row['GS1_Rate_Decimal'].values[0],
            'Type': 'Quarterly'
        }])
        quarterly_df = pd.concat([quarterly_df, tmp], ignore_index=True)

print("✓ Quarterly data prepared:")
for _, row in quarterly_df.iterrows():
    print(f"  {row['Date']}: Price=${row['Stock_Price_USD']:.2f}, MktCap=${row['Market_Cap_USD_Mn']:,.0f}Mn, Vol={row['Equity_Vol_Annual']:.2f}, RF={row['Risk_Free_Rate']:.4f}")

# === 7. Load existing financials and update ===
financials = pd.read_csv('/Users/huzhichao/Desktop/金融风险管理/数据搜集/Credit_Suisse_KMV_Financials.csv')

# Create updated version with market data
financials_updated = financials[['Date', 'Type', 'Total_Assets_CHF_Mn', 'Total_Liabilities_CHF_Mn', 
                                  'Total_Equity_CHF_Mn', 'Short_Term_Debt_CHF_Mn', 'Long_Term_Debt_CHF_Mn',
                                  'Shares_Outstanding_Mn']].copy()

# Update shares outstanding from quarterly data
shares_map = dict(zip(quarterly_df['Date'], quarterly_df['Shares_Outstanding_Mn']))
financials_updated['Shares_Outstanding_Mn'] = financials_updated['Date'].map(shares_map)

# Add market data columns
financials_updated['Stock_Price_USD'] = financials_updated['Date'].map(dict(zip(quarterly_df['Date'], quarterly_df['Stock_Price_USD'])))
financials_updated['Market_Cap_USD_Mn'] = financials_updated['Date'].map(dict(zip(quarterly_df['Date'], quarterly_df['Market_Cap_USD_Mn'])))
financials_updated['Equity_Vol_Annual'] = financials_updated['Date'].map(dict(zip(quarterly_df['Date'], quarterly_df['Equity_Vol_Annual'])))
financials_updated['Risk_Free_Rate_Annual'] = financials_updated['Date'].map(dict(zip(quarterly_df['Date'], quarterly_df['Risk_Free_Rate'])))

# Source column
financials_updated['Source'] = 'Reconstructed / FRED GS1'

# Save
financials_updated.to_csv('/Users/huzhichao/Desktop/金融风险管理/数据搜集/Credit_Suisse_KMV_Financials_Updated.csv', index=False)
print("✓ Saved: Credit_Suisse_KMV_Financials_Updated.csv")

# === 8. Summary ===
print("\n" + "="*80)
print("CREDIT SUISSE KMV MODEL - DATA INTEGRATION COMPLETE")
print("="*80)
print(f"\nMonthly records: {len(monthly)}")
print(f"Quarterly records: {len(quarterly_df)}")
print(f"\nDate range: {monthly['Date'].min()} to {monthly['Date'].max()}")
print(f"\nStock price range: ${monthly['Stock_Price_USD'].min():.2f} - ${monthly['Stock_Price_USD'].max():.2f}")
print(f"Volatility range: {monthly['Equity_Vol_Annual'].min():.2f} - {monthly['Equity_Vol_Annual'].max():.2f}")
print(f"Market cap range: ${monthly['Market_Cap_USD_Mn'].min():,.0f}Mn - ${monthly['Market_Cap_USD_Mn'].max():,.0f}Mn")

print("\n" + "-"*80)
print("FILES GENERATED:")
print("-"*80)
print("1. CS_KMV_Monthly_Data.csv (complete monthly series)")
print("2. Credit_Suisse_KMV_Financials_Updated.csv (quarterly KMV inputs)")
print("="*80)
