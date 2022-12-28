import pandas as pd
import requests
import io
import os
import zipfile
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import numpy as np

import matplotlib as mpl

pd.options.display.max_rows = 40

# % matplotlib
# inline
# % config
# InlineBackend.figure_format = 'retina'
plt.style.use('seaborn-colorblind')
plt.rcParams["patch.force_edgecolor"] = True

# You may want to adjust the figure size if the figures appear too big or too small on your screen.
mpl.rcParams['figure.figsize'] = (10.0, 6.0)

# Download
# Path
Inputs = "./"
Outputs = "./"
Download = "https://freddiemac.embs.com/FLoan/Data/"

# Assign column names and extract shorter list of columns from original files
orig_columns = ["Credit_Score", "First_Payment_Date", "First_Time_Home_Buyer", "Maturity_Date", "MSA", "Insurance_pct",
                "Units", "Occupancy", "Combined_LTV", "DTI", "UPB", "LTV", "Interest", "Channel", "PPM", "Product_Type",
                "State", "Property_Type", "ZIP", "ID", "Loan_Purpose", "Term", "Borrowers", "Seller", "Servicer",
                "Super_Confirming", "Program"]
orig_columns_extract = ["ID", "Credit_Score", "MSA", "Combined_LTV", "LTV", "DTI", "State", "ZIP"]
perf_columns = ["ID", "Date", "C_UPB", "Del_Status", 'Loan_Age', "Remain_Age", "Repurchase", "Modification", "Zero_UPB",
                "Zero_UPB_Date", "C_IR", "C_deferred_UPB", "DDLPI", "MI_Rec", "Net_Sales", "Non_MI_Rec", "Expenses",
                "Legal_Costs", "Maintain_Cost", "Taxes", "Other_Expenses", "Actual_Loss", "Modification_Cost",
                "Step_Modification", "Deferred_PP", "ELTV", "Zero_UPB_Removal", "Del_Accrued_Interest", "Del_Disaster",
                "Borrower_Assistance"]
perf_columns_extract = ["ID", "Date", "C_UPB", "Del_Status", 'Loan_Age', "Remain_Age", "Zero_UPB", "Zero_UPB_Date",
                        "Net_Sales", "Actual_Loss", "ELTV", "Del_Disaster"]

# Specify login credentials
login = {
    "username": "adeshp27@uic.edu",
    "password": "NU;quUwO"
}
# Define the vintage data to download (for this article, we will look into vintage from Q1 2014 to Q2 2019.
Period = ["Q12014", "Q22014", "Q32014", "Q42014", "Q12015", "Q22015", "Q32015", "Q42015", "Q12016", "Q22016",
          "Q32016", "Q42016", "Q12017", "Q22017", "Q32017", "Q42017", "Q12018", "Q22018", "Q32018", "Q42018",
          "Q12019", "Q22019"]


def Vintage_data(perf_file, orig_file, export_file):
    Perf = pd.read_csv(os.path.join(Inputs, perf_file), sep=',', delimiter="|", header=None)
    Orig = pd.read_csv(os.path.join(Inputs, orig_file), sep=',', delimiter="|", header=None)
    Orig.columns = orig_columns
    Perf.columns = perf_columns
    Orig = Orig[orig_columns_extract]
    Perf = Perf[perf_columns_extract]
    print("Origination File Unique ID #:%s" % (Orig['ID'].nunique()))
    print("Performance File Unique ID #:%s" % (Perf['ID'].nunique()))
    Vintage = pd.merge(Perf, Orig, how='left', on=['ID'])
    Vintage.to_csv(os.path.join(Outputs, export_file))


with requests.Session() as s:
    p = s.post('https://freddiemac.embs.com/FLoan/Data/download2.php')
    for Qtr in Period:
        # File Names
        zip_file = "historical_data1_" + Qtr + ".zip"
        perf_file = "historical_data1_time_" + Qtr + ".txt"
        orig_file = "historical_data1_" + Qtr + ".txt"
        export_file = Qtr + ".csv"
        r = s.get(os.path.join(Download, zip_file))
        z = zipfile.ZipFile(io.BytesIO(r.content))
        z.extractall(Input_Data_Path)
        Vintage_data(perf_file, orig_file, export_file)

data_all = pd.DataFrame()
for cohort in Period:
    cohort_data = cohort + ".csv"
    cohort = pd.read_csv(os.path.join(Outputs, cohort_data), index_col=0, dtype={'Del_Status': object},
                         low_memory=False)
    cohort['Date'] = pd.to_datetime(cohort['Date'], format='%Y%m')
    cohort['Orig_Date'] = cohort['Date'] - cohort['Loan_Age'].values.astype("timedelta64[M]") - pd.DateOffset(months=1)
    cohort['Orig_Date'] = pd.to_datetime(cohort.Orig_Date) + pd.offsets.MonthBegin(0)
    cohort = cohort.sort_values(by=['ID', 'Date'])
    orig_qtr = cohort.groupby("ID").first().reset_index()
    orig_qtr['Orig_Date'] = orig_qtr['Orig_Date'].dt.date
    data_all = data_all.append(orig_qtr)

# Originations count over time
Orig_count = data_all.groupby("Orig_Date")['ID'].count().reset_index()
Orig_count['ID'] = Orig_count['ID'] / 1000
Orig_count = Orig_count[:-1]

# Originations UPB over time
Orig_UPB = data_all.groupby("Orig_Date")['C_UPB'].sum().reset_index()
Orig_UPB['C_UPB'] = Orig_UPB['C_UPB'] / 1000000000
Orig_UPB = Orig_UPB[:-1]

Orig_count['Orig_Date'] = pd.to_datetime(Orig_count['Orig_Date'], 
                                         format='%Y-%m-%d')

x = Orig_count['Orig_Date'].tolist()
y1 = Orig_count['ID']
y2 = Orig_UPB['C_UPB']

fig = plt.figure()
fig.suptitle("Freddie Mac Mortgage Origination: Q1 2014 to Q2 2019 \n left - by count; right - by volume", fontsize=15)
ax1 = fig.add_subplot(111)
ax1.plot(x, y1)
ax1.set_ylabel('Count in Thousands')

ax2 = ax1.twinx()
ax2.plot(x, y2, 'r-')
ax2.set_ylabel('Volumn in Billion', color='r')
for tl in ax2.get_yticklabels():
    tl.set_color('r')

ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
ax1.xaxis.set_major_locator(mdates.MonthLocator((1, 4, 7, 10)))

plt.setp(ax1.get_xticklabels(), rotation=70, horizontalalignment='right')
plt.show()

data_all['Orig_Year'] = pd.DatetimeIndex(data_all["Orig_Date"]).year

# Distribution of Original Loan to Value: value  = 999 are used for missing data - 155 accounts
data_all.LTV.plot(kind='hist', bins=np.linspace(data_all.LTV.min(), 110, 100))

# Distribution of current credit score: value  = 9999 are used for missing data - 650 accounts
data_all.Credit_Score.plot(kind='hist', bins=np.linspace(data_all.Credit_Score.min(), 850, 100))

# Distribution of DTI: value  = 999 are used for missing data - 328 accounts)
data_all.DTI.plot(kind='hist', bins=np.linspace(data_all.DTI.min(), 60, 100))


# Default Flag
def del_analysis(vintage):
    # Create Default flag due to 180day delinquency
    vintage.loc[(~vintage["Del_Status"].isin(('0', '1', '2', '3', '4', '5'))), "default_flag1"] = 1

    # Create Default flag due to zero_balances. Not using actual loss here because in some cases there will be "loss" due to reperforming sale
    # for now including zero balance code of 02, 03, and 09 as default - TBD
    vintage.loc[(vintage["Zero_UPB"].isin(('2.', '9.', '3.'))), "default_flag2"] = 1
    vintage.loc[(vintage["default_flag1"] == 1) | (vintage["default_flag2"] == 1), "default_flag"] = 1

    # To create first default date, First sort by ID and Date, then group by ID take first date
    default = vintage[vintage["default_flag"] == 1][["ID", "Date", "default_flag"]]
    default = default.sort_values(by=['ID', 'Date'])
    default = default.rename(columns={'Date': 'Default_Date'})
    default = default.groupby("ID").first().reset_index()

    # Merge first default date and finalize default flag
    vintage_tag = pd.merge(vintage, default[["ID", "Default_Date"]], how='left', on=['ID'])
    conditions = [
        (vintage_tag['Default_Date'].isnull()),
        (vintage_tag['Date'] < vintage_tag['Default_Date']),
        (vintage_tag['Date'] == vintage_tag['Default_Date']),
        (vintage_tag['Date'] > vintage_tag['Default_Date'])
    ]
    values = [0, 0, 1, 2]
    vintage_tag['default_flag'] = np.select(conditions, values)

    # Create PD dataset (all default_flag = 0 and first time default_flag = 1)
    PD_non_default = vintage_tag[vintage_tag['default_flag'] == 0]
    PD_default = vintage_tag[vintage_tag['default_flag'] == 1]
    df_PD = PD_non_default.append(PD_default)

    conditions_1 = [
        (df_PD['default_flag'] == 1),
        (df_PD['Del_Status'] == '1'),
        (df_PD['Del_Status'] == '2'),
        (df_PD['Del_Status'] == '3'),
        (df_PD['Del_Status'] == '4'),
        (df_PD['Del_Status'] == '5'),
        (df_PD['Del_Status'] == '0')
    ]
    values_1 = ["Default", "DPD_30", "DPD_60", "DPD_90", "DPD_120", "DPD_150", "DPD_0"]
    df_PD['DPD_Bucket'] = np.select(conditions_1, values_1)

    TM_PD = df_PD[["Date", "ID", "DPD_Bucket"]]
    TM_count = Transition_Analysis(TM_PD)

    # Delinquency / Default status count over time
    # Del_count = df_PD.groupby(("Date","DPD_Bucket"))["ID"].count().reset_index(name ='Count')
    # Del_count = Del_count.pivot(index='Date', columns='DPD_Bucket', values='Count').reset_index()

    # return Del_count
    return TM_count


for cohort in Period:
    print(cohort)
    cohort_data = cohort + ".csv"
    cohort = pd.read_csv(os.path.join(Outputs, cohort_data), index_col=0, dtype={'Del_Status': object},
                         low_memory=False)
    cohort['Date'] = pd.to_datetime(cohort['Date'], format='%Y%m')
    # Del_count = del_analysis(cohort)
    TM_count = del_analysis(cohort)
    # df_PD_all = df_PD_all.append(df_PD)
    # Del_count_all = Del_count_all.append(Del_count)
    TM_count_all = TM_count_all.append(TM_count)

# Plot Delinquencies and Defaults over time
Del_count_all = Del_count_all.groupby(("Date"))[
    ["Default", "DPD_30", "DPD_60", "DPD_90", "DPD_120", "DPD_150", "DPD_0"]].sum().reset_index()
Del_count_all['total'] = y1 = Del_count_all["DPD_0"] + Del_count_all["DPD_30"] + Del_count_all["DPD_60"] + \
                              Del_count_all["DPD_90"] + Del_count_all["DPD_120"] + Del_count_all["DPD_150"] + \
                              Del_count_all["Default"]

fig, axs = plt.subplots(2, 4)

fig.set_size_inches(20, 8)
x = Del_count_all['Date']
y1 = Del_count_all["DPD_0"] / Del_count_all['total']
y2 = Del_count_all["DPD_30"] / Del_count_all['total']
y3 = Del_count_all["DPD_60"] / Del_count_all['total']
y4 = Del_count_all["DPD_90"] / Del_count_all['total']
y5 = Del_count_all["DPD_120"] / Del_count_all['total']
y6 = Del_count_all["DPD_150"] / Del_count_all['total']
y7 = Del_count_all["Default"] / Del_count_all['total']

axs[0, 0].plot(x, y1)
axs[0, 0].set_title('Current')
axs[0, 1].plot(x, y2, 'tab:orange')
axs[0, 1].set_title('30 Day Delinquent')
axs[0, 2].plot(x, y3, 'tab:green')
axs[0, 2].set_title('60 Day Delinquent')
axs[0, 3].plot(x, y4, 'tab:red')
axs[0, 3].set_title('90 Day Delinquent')
axs[1, 0].plot(x, y5)
axs[1, 0].set_title('120 Day Delinquent')
axs[1, 1].plot(x, y6, 'tab:orange')
axs[1, 1].set_title('150 Day Delinquent')
axs[1, 2].plot(x, y7, 'tab:green')
axs[1, 2].set_title('Default')

fig.suptitle("Freddie Mac Delinquency Status: \n Q1 2014 onwards", fontsize=15)

fig.delaxes(axs[1][3])


# Transition Analysis
# from df_PD <- PD datasets by vintage creating a file with Date, account, previous status, current status by merging
def Transition_Analysis(TM_PD):
    TM_Current = TM_PD.rename(columns={"DPD_Bucket": "Current"})

    # create date forward column used for merge
    TM_Previous = TM_PD.rename(columns={"DPD_Bucket": "Previous"})
    TM_Previous["Date_forward"] = TM_Previous["Date"] + pd.DateOffset(months=1)

    # Merge - left join previous states
    TM_merged = pd.merge(TM_Previous[["Date_forward", "ID", "Previous"]], TM_Current, left_on=["Date_forward", "ID"],
                         right_on=["Date", "ID"], how='left')
    TM_merged = TM_merged.dropna()
    TM_merged["from_to_freq"] = TM_merged["Previous"] + "_to_" + TM_merged["Current"]

    # create from-to-freq count
    TM_count = TM_merged.groupby(("Date", "from_to_freq"))["ID"].count().reset_index(name='Count')
    TM_count = TM_count.pivot(index='Date', columns='from_to_freq', values='Count').reset_index()
    TM_count = TM_count.fillna(0)

    return TM_count


TM_count_all = TM_count_all.groupby("Date").sum().reset_index()
Del_count_all["Date_forward"] = Del_count_all["Date"] + pd.DateOffset(months=1)
Del_count_all = Del_count_all.drop(columns=['Date'])

TM_all = pd.merge(TM_count_all, Del_count_all, left_on=["Date"], right_on=["Date_forward"], how='left')

# Current-to-30DPD vs PD
x = TM_all['Date'].tolist()
y1 = TM_all["DPD_0_to_DPD_30"] / TM_all["DPD_0"]
y2 = Del_count_all["Default"][1:] / Del_count_all['total'][1:]

fig = plt.figure()
fig.suptitle("From Current to 30day Delinquency vs. Default", fontsize=15)
ax1 = fig.add_subplot(111)
ax1.plot(x, y1)
ax1.set_ylabel('Transition Probability')

ax2 = ax1.twinx()
ax2.plot(x, y2, 'r-')
ax2.set_ylabel('Default Probability', color='r')
for tl in ax2.get_yticklabels():
    tl.set_color('r')

ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
ax1.xaxis.set_major_locator(mdates.MonthLocator((1, 4, 7, 10)))

plt.setp(ax1.get_xticklabels(), rotation=70, horizontalalignment='right')
plt.show()
