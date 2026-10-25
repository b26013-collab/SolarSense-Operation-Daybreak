from pathlib import Path
import os, json, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import seaborn as sns
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.feature_selection import mutual_info_regression
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, PageBreak

# Operation Daybreak — STAC WARS PS2. Outputs are saved; no plot windows open.
PROJECT_DIR = Path(__file__).resolve().parent
CSV = Path(os.environ.get('SOLARSENSE_CSV', str(PROJECT_DIR / 'data' / 'SolarPrediction.csv'))).expanduser()
OUT = Path(os.environ.get('SOLARSENSE_OUTPUT', str(PROJECT_DIR / 'solar_analysis_output'))).expanduser()
PLOTS, DATA, REPORTS = OUT/'graphs', OUT/'data', OUT/'reports'
for d in (OUT, PLOTS, DATA, REPORTS): d.mkdir(parents=True, exist_ok=True)
RANDOM_STATE = 42
sns.set_theme(style='whitegrid')
plt.rcParams['savefig.dpi'] = 150
warnings.filterwarnings('ignore', category=FutureWarning)

def col(df, *names):
    lookup = {str(c).strip().lower(): c for c in df.columns}
    return next((lookup[n.strip().lower()] for n in names if n.strip().lower() in lookup), None)
def savefig(name):
    plt.tight_layout(); plt.savefig(PLOTS/name, bbox_inches='tight'); plt.close()
def write(lines, name): (REPORTS/name).write_text('\n'.join(map(str, lines))+'\n', encoding='utf-8')
def fmt(x): return 'not available' if pd.isna(x) else f'{x:.3f}' if isinstance(x, (float, np.floating, int, np.integer)) else str(x)

if not CSV.exists():
    raise FileNotFoundError(f'CSV not found: {CSV}\nSet SOLARSENSE_CSV or edit CSV near the top of this script.')
raw = pd.read_csv(CSV); raw.columns = raw.columns.astype(str).str.strip(); df = raw.copy()
date_c=col(df,'Data','Date'); time_c=col(df,'Time'); dt_c=col(df,'Datetime','DateTime','Timestamp')
rad_c=col(df,'Radiation','SolarRadiation'); temp_c=col(df,'Temperature','Temp'); hum_c=col(df,'Humidity')
press_c=col(df,'Pressure'); speed_c=col(df,'Speed','WindSpeed','Wind Speed')
wind_c=col(df,'WindDirection(Degrees)','WindDirection','Wind Direction','WindDirectionDegrees')
sunrise_c=col(df,'TimeSunRise','Sunrise','TimeSunrise'); sunset_c=col(df,'TimeSunSet','Sunset','TimeSunset')
if rad_c is None: raise ValueError(f'Radiation column not found. Columns: {list(df.columns)}')
if dt_c: df['Datetime']=pd.to_datetime(df[dt_c], errors='coerce')
elif date_c and time_c:
    dts=pd.to_datetime(df[date_c], errors='coerce').dt.strftime('%Y-%m-%d')
    df['Datetime']=pd.to_datetime(dts.fillna(df[date_c].astype(str))+' '+df[time_c].astype(str).str.strip(), errors='coerce')
elif date_c: df['Datetime']=pd.to_datetime(df[date_c], errors='coerce')
else: raise ValueError('Need Datetime/Timestamp or Data and Time columns.')
invalid_ts=int(df.Datetime.isna().sum()); df=df.dropna(subset=['Datetime']).sort_values('Datetime').reset_index(drop=True)
measure=[x for x in [rad_c,temp_c,hum_c,press_c,speed_c,wind_c] if x]
for c in measure: df[c]=pd.to_numeric(df[c], errors='coerce')
df['Date']=df.Datetime.dt.date; df['Hour']=df.Datetime.dt.hour; df['Minute']=df.Datetime.dt.minute
df['MinuteOfDay']=df.Hour*60+df.Minute; df['DayOfYear']=df.Datetime.dt.dayofyear; df['Month']=df.Datetime.dt.month
df['HourSin']=np.sin(2*np.pi*df.Hour/24); df['HourCos']=np.cos(2*np.pi*df.Hour/24)
df['DayOfYearSin']=np.sin(2*np.pi*df.DayOfYear/365.25); df['DayOfYearCos']=np.cos(2*np.pi*df.DayOfYear/365.25)
for source, target in [(sunrise_c,'SunriseDatetime'),(sunset_c,'SunsetDatetime')]:
    if source:
        df[target]=pd.to_datetime(df.Datetime.dt.strftime('%Y-%m-%d')+' '+df[source].astype(str).str.strip(), errors='coerce')
if 'SunriseDatetime' in df: df['MinutesSinceSunrise']=(df.Datetime-df.SunriseDatetime).dt.total_seconds()/60
if 'SunriseDatetime' in df and 'SunsetDatetime' in df:
    df['DayLengthMinutes']=(df.SunsetDatetime-df.SunriseDatetime).dt.total_seconds()/60

# Phase 0 novelty: solar clock, clear-sky proxy, and data-quality audit
if 'SunriseDatetime' in df and 'SunsetDatetime' in df:
    valid_day = df.SunriseDatetime.notna() & df.SunsetDatetime.notna() & (df.SunsetDatetime > df.SunriseDatetime)
    df['DayLength_h'] = (df.SunsetDatetime-df.SunriseDatetime).dt.total_seconds()/3600
    df['MinSinceSunrise'] = (df.Datetime-df.SunriseDatetime).dt.total_seconds()/60
    df['SolarFrac'] = (df.MinSinceSunrise/(df.DayLength_h*60)).where(df.DayLength_h > 0)
    df['IsDaytime'] = valid_day & df.Datetime.ge(df.SunriseDatetime) & df.Datetime.le(df.SunsetDatetime)
else:
    df['DayLength_h'] = np.nan
    df['MinSinceSunrise'] = np.nan
    df['SolarFrac'] = np.nan
    df['IsDaytime'] = df[rad_c].gt(0)

# Idealised half-sine envelope: illustrative proxy, not a measured clear-sky model.
positive_daily_peaks = df.loc[df[rad_c] > 0].groupby('Date')[rad_c].max()
peak_clear = float(positive_daily_peaks.quantile(.95)) if len(positive_daily_peaks) else 0.0
solar_fraction = df.SolarFrac.clip(0, 1)
df['ClearSky'] = np.where(df.IsDaytime & solar_fraction.notna(), peak_clear*np.sin(np.pi*solar_fraction.fillna(0)), 0.0)
df['ClearnessIdx'] = (df[rad_c]/df.ClearSky).where(df.ClearSky > 50)

# Data quality audit: document issues and the handling decision.
raw_rad = pd.to_numeric(raw[rad_c], errors='coerce')
negative_radiation = int((df[rad_c] < 0).sum())
duplicate_timestamps = int(df.Datetime.duplicated(keep=False).sum())
gaps = df.Datetime.sort_values().diff().dt.total_seconds().dropna()
median_gap = float(gaps.median()) if len(gaps) else np.nan
large_gap_count = int((gaps > 1.5*median_gap).sum()) if pd.notna(median_gap) and median_gap > 0 else 0
audit_rows = [
    {'issue':'Missing values','count':int(df.isna().sum().sum()),'handling':'Numeric gaps retained for audit; model pipelines impute features; target-missing rows excluded from modelling.'},
    {'issue':'Negative radiation readings','count':negative_radiation,'handling':'Flagged for review; retained rather than silently clipping because they may be sensor/quality artefacts.'},
    {'issue':'Duplicate timestamps','count':duplicate_timestamps,'handling':'Reported; duplicate records retained in raw analysis. Consider aggregation if duplicates are confirmed accidental.'},
    {'issue':'Large sampling gaps (>1.5× median interval)','count':large_gap_count,'handling':'Reported as gaps; no synthetic readings inserted.'},
    {'issue':'Invalid timestamps removed','count':invalid_ts,'handling':'Rows without a parseable Datetime were removed before time-series analysis.'},
]
pd.DataFrame(audit_rows).to_csv(DATA/'data_quality_audit.csv',index=False)

# Dataset inspection
q1,q3=df[rad_c].quantile([.25,.75]); iqr=q3-q1; low_fence=q1-1.5*iqr; high_fence=q3+1.5*iqr
outliers=df[(df[rad_c]<low_fence)|(df[rad_c]>high_fence)].copy()
inspection={'rows_original':len(raw),'columns_original':list(raw.columns),'rows_analyzed':len(df),'invalid_timestamps_removed':invalid_ts,'date_start':str(df.Datetime.min()),'date_end':str(df.Datetime.max()),'unique_dates':int(df.Date.nunique()),'missing_by_column':df.isna().sum().to_dict(),'datetime_dtype':str(df.Datetime.dtype),'radiation_iqr_lower_fence':float(low_fence),'radiation_iqr_upper_fence':float(high_fence),'iqr_outliers':len(outliers)}
(REPORTS/'data_inspection.json').write_text(json.dumps(inspection,indent=2,default=str),encoding='utf-8')
(REPORTS/'column_dtypes.txt').write_text(df.dtypes.to_string(),encoding='utf-8')
df.describe(include='all').T.to_csv(DATA/'descriptive_statistics.csv'); df.isna().sum().to_csv(DATA/'missing_values.csv')
outliers[[c for c in ['Datetime','Date','Hour',rad_c,temp_c,hum_c,press_c,speed_c] if c]].to_csv(DATA/'radiation_iqr_outliers.csv',index=False)

# Part A: figures
fig,ax=plt.subplots(figsize=(13,4)); ax.plot(df.Datetime,df[rad_c],lw=.65); ax.set(title='A1. Solar Radiation Over Time',xlabel='Datetime',ylabel='Radiation (dataset units)'); savefig('A1_radiation_over_time.png')
hourly=df.groupby('Hour')[rad_c].mean()
fig,ax=plt.subplots(figsize=(9,4)); ax.plot(hourly.index,hourly.values,marker='o'); ax.set(title='A2. Mean Radiation by Hour',xlabel='Hour of day',ylabel='Mean radiation'); savefig('A2_mean_radiation_by_hour.png')
fig,ax=plt.subplots(figsize=(8,4)); ax.hist(df[rad_c].dropna(),bins=30); ax.set(title='A3. Radiation Distribution',xlabel='Radiation',ylabel='Count'); savefig('A3_radiation_distribution.png')
num=df.select_dtypes(include=np.number)
fig,ax=plt.subplots(figsize=(10,7)); sns.heatmap(num.corr(),cmap='coolwarm',center=0,ax=ax); ax.set(title='A4. Numeric Correlation Heatmap'); savefig('A4_correlation_heatmap.png')
for i,c in enumerate([temp_c,hum_c,press_c,speed_c]):
    if c:
        fig,ax=plt.subplots(figsize=(7,4)); ax.scatter(df[c],df[rad_c],s=10,alpha=.35); ax.set(title=f'A{i+5}. Radiation vs {c}',xlabel=c,ylabel='Radiation'); savefig(f'A{i+5}_radiation_vs_{c.lower().replace(" ","_")}.png')
date_hour=df.pivot_table(index='Date',columns='Hour',values=rad_c,aggfunc='mean')
fig,ax=plt.subplots(figsize=(12,max(4,min(15,len(date_hour)*.18)))); sns.heatmap(date_hour,cmap='YlOrRd',ax=ax,cbar_kws={'label':'Mean radiation'}); ax.set(title='A9. Daybreak Heatmap: Date by Hour',xlabel='Hour',ylabel='Date'); savefig('A9_daybreak_heatmap.png')
daily=df.groupby('Date')[rad_c].agg(DailyMax='max',DailyMean='mean',Observations='count').reset_index()
daily['Date']=daily['Date'].astype(str)
daily.to_csv(DATA/'daily_report.csv',index=False)
fig,ax=plt.subplots(figsize=(11,4)); ax.plot(daily.Date.astype(str),daily.DailyMax,marker='o',ms=3); ax.set(title='A10. Daily Maximum Radiation',xlabel='Date',ylabel='Daily maximum'); ax.tick_params(axis='x',rotation=60); savefig('A10_daily_maximum.png')
fig,ax=plt.subplots(figsize=(8,4)); sns.boxplot(x=df[rad_c],ax=ax); ax.set(title='A11. Radiation Outlier Check',xlabel='Radiation'); savefig('A11_radiation_outliers.png')

# Part B: Pearson/Spearman correlations, all vs positive radiation
corr_rows=[]
for subset_name,subset in [('all_records',df),('positive_radiation',df[df[rad_c]>0])]:
    for c in [temp_c,hum_c,press_c,speed_c,wind_c]:
        if not c: continue
        pair=subset[[c,rad_c]].dropna()
        if len(pair)>=3 and pair[c].nunique()>1 and pair[rad_c].nunique()>1:
            corr_rows.append({'subset':subset_name,'parameter':c,'n':len(pair),'pearson_r':pair[c].corr(pair[rad_c]),'spearman_rho':pair[c].corr(pair[rad_c],method='spearman')})
corr=pd.DataFrame(corr_rows)
if not corr.empty: corr['abs_r']=corr.pearson_r.abs(); corr=corr.sort_values(['subset','abs_r'],ascending=[True,False])
corr.to_csv(DATA/'radiation_parameter_correlations.csv',index=False)

# Novel comparison: all hours vs daytime-only and mutual information.
mi_rows=[]
for subset_name, subset in [('all_records', df), ('daytime_only', df[df.IsDaytime])]:
    for feature in [temp_c, hum_c, press_c, speed_c, wind_c, 'Hour', 'SolarFrac', 'ClearnessIdx']:
        if not feature or feature not in subset.columns: continue
        pair=subset[[feature, rad_c]].replace([np.inf,-np.inf],np.nan).dropna()
        if len(pair) >= 10 and pair[feature].nunique() > 1 and pair[rad_c].nunique() > 1:
            mi=mutual_info_regression(pair[[feature]], pair[rad_c], random_state=RANDOM_STATE)[0]
            mi_rows.append({'subset':subset_name,'feature':feature,'n':len(pair),'mutual_information':float(mi)})
mi_df=pd.DataFrame(mi_rows)
if not mi_df.empty: mi_df=mi_df.sort_values(['subset','mutual_information'],ascending=[True,False])
mi_df.to_csv(DATA/'radiation_mutual_information.csv',index=False)

# Classify flagged readings conservatively; these are candidate labels, not definitive diagnoses.
classified=outliers.copy()
classified['ClearSky']=df.loc[classified.index,'ClearSky']
classified['outlier_class']='genuine low-light anomaly or statistical outlier'
classified.loc[classified[rad_c] < 0,'outlier_class']='impossible/quality-control candidate (negative radiation)'
if 'ClearSky' in classified:
    classified.loc[(classified[rad_c] > classified.ClearSky*1.15) & (classified[rad_c] > 50),'outlier_class']='possible cloud-edge enhancement or envelope-model mismatch'
classified.to_csv(DATA/'classified_radiation_outliers.csv',index=False)

# Solar-noon-relative peak timing and thermal lag cross-correlation.
if 'SunriseDatetime' in df and 'SunsetDatetime' in df:
    df['SolarNoon'] = df.SunriseDatetime + (df.SunsetDatetime-df.SunriseDatetime)/2
    df['MinFromSolarNoon'] = (df.Datetime-df.SolarNoon).dt.total_seconds()/60
else:
    df['SolarNoon'] = pd.NaT
    df['MinFromSolarNoon'] = np.nan
lag_lines=['ADDITIONAL PART B EVIDENCE']
if not hourly.dropna().empty:
    peak_hr=int(hourly.idxmax())
    peak_records=df[df.Hour==peak_hr]
    if 'MinFromSolarNoon' in peak_records and peak_records.MinFromSolarNoon.notna().any():
        lag_lines.append(f'Hourly radiation peak is {peak_hr:02d}:00; mean offset from solar noon among readings in that hour is {peak_records.MinFromSolarNoon.mean():+.1f} minutes.')
if temp_c:
    hourly_rt=df.groupby('Hour')[[rad_c,temp_c]].mean().dropna()
    lags=list(range(-3,4))
    lag_corr=[]
    for lag in lags:
        # Positive lag means temperature series is shifted later relative to radiation.
        lag_corr.append(hourly_rt[rad_c].corr(hourly_rt[temp_c].shift(-lag)))
    if any(pd.notna(x) for x in lag_corr):
        best_lag=lags[int(np.nanargmax(lag_corr))]
        lag_lines.append(f'Hourly cross-correlation is highest at lag {best_lag:+d} hour(s) (correlation={np.nanmax(lag_corr):.3f}); this is an exploratory diurnal pattern, not proof of causal delay.')
if hum_c and 'ClearnessIdx' in df:
    pair=df[[hum_c,'ClearnessIdx']].replace([np.inf,-np.inf],np.nan).dropna()
    if len(pair)>=3 and pair[hum_c].nunique()>1 and pair.ClearnessIdx.nunique()>1:
        lag_lines.append(f'Humidity–clearness-index Pearson r={pair[hum_c].corr(pair.ClearnessIdx):.3f} (n={len(pair)}); clearness index is only a proxy because the clear-sky envelope is idealised.')
if not mi_df.empty:
    for sn in ['all_records','daytime_only']:
        sm=mi_df[mi_df.subset==sn]
        if len(sm): lag_lines.append(f'Top mutual-information feature for {sn}: {sm.iloc[0].feature}, MI={sm.iloc[0].mutual_information:.3f}.')
write(lag_lines,'part_B_novel_evidence.txt')

# Extra visualizations: clear-sky envelope on a representative bright day, hysteresis, single-day zoom.
valid_dates=df.groupby('Date')[rad_c].mean().sort_values(ascending=False)
if len(valid_dates):
    chosen_date=valid_dates.index[min(0, len(valid_dates)-1)]
    # Use the sunniest observed day for illustrative overlay.
    chosen_date=valid_dates.index[0]
    one=df[df.Date==chosen_date].sort_values('Datetime')
    fig,ax=plt.subplots(figsize=(10,4)); ax.plot(one.Datetime,one[rad_c],label='Observed radiation'); ax.plot(one.Datetime,one.ClearSky,label='Idealised half-sine envelope',ls='--'); ax.set(title=f'Clear-sky envelope comparison: {chosen_date}',xlabel='Time',ylabel='Radiation'); ax.legend(); savefig('A12_clear_sky_envelope.png')
    fig,ax=plt.subplots(figsize=(8,5))
    if temp_c:
        xy=one[[rad_c,temp_c]].dropna().to_numpy()
        if len(xy)>1:
            segments=np.stack([xy[:-1],xy[1:]],axis=1)
            lc=LineCollection(segments,cmap='viridis',norm=plt.Normalize(one.Datetime.dt.hour.min(),max(1,one.Datetime.dt.hour.max())))
            lc.set_array(one.Datetime.dt.hour.iloc[:len(xy)-1].to_numpy()); lc.set_linewidth(2)
            ax.add_collection(lc); ax.scatter(xy[:,0],xy[:,1],s=12,c=one.Datetime.dt.hour.iloc[:len(xy)].to_numpy(),cmap='viridis')
            ax.autoscale(); fig.colorbar(lc,ax=ax,label='Hour of day')
        ax.set(ylabel=temp_c)
    else:
        ax.text(.5,.5,'Temperature column unavailable',ha='center',va='center',transform=ax.transAxes)
        ax.set_ylabel('Temperature unavailable')
    ax.set(title=f'Temperature–radiation path: {chosen_date}',xlabel='Radiation'); savefig('A13_hysteresis_path.png')
# Single-day zoom: day with greatest mean radiation.
if len(valid_dates):
    one=df[df.Date==valid_dates.index[0]].sort_values('Datetime')
    fig,ax=plt.subplots(figsize=(10,4)); ax.plot(one.Datetime,one[rad_c],marker='.',ms=3); ax.set(title=f'Single-day radiation zoom: {valid_dates.index[0]}',xlabel='Datetime',ylabel='Radiation'); savefig('A14_single_day_zoom.png')
lines=['PART B — DRAWING INFERENCES','Evidence values are calculated from the supplied CSV. Correlation is not causation.']
if not hourly.dropna().empty:
    ph=int(hourly.idxmax()); lines.append(f'1. Highest mean radiation hour: {ph:02d}:00, hourly mean {hourly.loc[ph]:.3f}. See graphs/A2_mean_radiation_by_hour.png.')
for subset in ['all_records','positive_radiation']:
    sub=corr[corr.subset==subset] if not corr.empty else pd.DataFrame()
    if not sub.empty:
        b=sub.iloc[0]; lines.append(f'2. Strongest absolute Pearson relationship ({subset}): {b.parameter}, r={b.pearson_r:.3f}, Spearman rho={b.spearman_rho:.3f}, n={int(b.n)}. Full ranking: data/radiation_parameter_correlations.csv.')
if temp_c and hum_c:
    tri=df[[rad_c,temp_c,hum_c]].dropna()
    if len(tri)>=3: lines.append(f'3. Radiation-temperature r={tri[rad_c].corr(tri[temp_c]):.3f}; radiation-humidity r={tri[rad_c].corr(tri[hum_c]):.3f}; temperature-humidity r={tri[temp_c].corr(tri[hum_c]):.3f}. See the relevant scatterplots. Temperature may partly be a response to solar heating.')
lines.append(f'4. IQR outlier rule flags {len(outliers)} of {df[rad_c].notna().sum()} non-missing radiation readings; fences={low_fence:.3f} to {high_fence:.3f}. Review data/radiation_iqr_outliers.csv for timestamps and context. Flags do not prove sensor failure.')
if temp_c:
    th=df.groupby('Hour')[temp_c].mean().dropna()
    if len(th) and len(hourly.dropna()): lines.append(f'Bonus thermal-lag check: hourly mean radiation peaks at {int(hourly.idxmax()):02d}:00 and temperature at {int(th.idxmax()):02d}:00; difference {int(th.idxmax())-int(hourly.idxmax()):+d} hours. Verify against plots; aggregation and season coverage matter.')
write(lines,'part_B_inferences.txt')

# Part C: single peak details, sunrise/sunset lag, top five daily means
peak=df.loc[df[rad_c].idxmax()] if df[rad_c].notna().any() else None
extra=[]
for day,frame in df.groupby('Date'):
    valid=frame.dropna(subset=[rad_c]); pr=valid.loc[valid[rad_c].idxmax()] if len(valid) else None
    sr=frame.SunriseDatetime.dropna().iloc[0] if 'SunriseDatetime' in frame and frame.SunriseDatetime.notna().any() else pd.NaT
    ss=frame.SunsetDatetime.dropna().iloc[0] if 'SunsetDatetime' in frame and frame.SunsetDatetime.notna().any() else pd.NaT
    extra.append({'Date':str(day),'daily_peak_time':pr.Datetime if pr is not None else pd.NaT,'sunrise':sr,'sunset':ss,'day_length_hours':(ss-sr).total_seconds()/3600 if pd.notna(sr) and pd.notna(ss) and ss>sr else np.nan})
daily=daily.merge(pd.DataFrame(extra),on='Date',how='left')
# Rebellion Daily Briefing: integrated radiation (assuming Radiation is W/m²).
def daily_energy(g):
    g=g.dropna(subset=[rad_c]).sort_values('Datetime')
    if len(g)<2: return np.nan
    t=(g.Datetime-g.Datetime.iloc[0]).dt.total_seconds().to_numpy()/3600
    return float((np.trapezoid if hasattr(np, 'trapezoid') else np.trapz)(g[rad_c].to_numpy(),t)/1000)
energy=df.groupby('Date',group_keys=False).apply(daily_energy)
energy.index=energy.index.astype(str)
daily['DailyEnergy_kWh_m2']=daily['Date'].map(energy)
def minmax(series):
    series=pd.to_numeric(series,errors='coerce'); lo=series.min(); hi=series.max()
    if pd.isna(lo) or pd.isna(hi) or hi==lo: return pd.Series(0.5,index=series.index)
    return (series-lo)/(hi-lo)
daily['PeakTime']=daily['daily_peak_time']
daily['DayLength_h']=daily['day_length_hours']
daily['MinSincePeak_from_SolarNoon']=np.nan
daily['MeanClearness']=daily['Date'].map(df.groupby('Date')['ClearnessIdx'].mean())
for ix,row in daily.iterrows():
    date_value=row['Date']
    frame=df[df.Date.astype(str)==str(date_value)]
    if len(frame) and frame.MinFromSolarNoon.notna().any():
        peak_idx=frame[rad_c].idxmax()
        daily.loc[ix,'MinSincePeak_from_SolarNoon']=df.loc[peak_idx,'MinFromSolarNoon']
# Transparent, heuristic day buckets; thresholds are assumptions, not standards.
def day_type(v):
    if pd.isna(v): return 'Unknown'
    if v >= 0.65: return 'Clear'
    if v >= 0.35: return 'Partly cloudy'
    return 'Overcast'
daily['DayType']=daily['MeanClearness'].apply(day_type)
daily['ReadinessScore']=100*minmax(daily['DailyEnergy_kWh_m2'])
daily.to_csv(DATA/'rebellion_daily_briefing.csv',index=False)
daily.to_csv(DATA/'daily_report.csv',index=False)
top5_energy=daily.nlargest(5,'DailyEnergy_kWh_m2')
top5_energy.to_csv(DATA/'top5_days_by_energy.csv',index=False)
top5=daily.nlargest(5,'DailyMean'); top5.to_csv(DATA/'top5_days_by_average_radiation.csv',index=False)

# Base Power Planning: bottom-20% energy risk, day-type reliability, operating window, calendar readiness.
energy_threshold=float(daily.DailyEnergy_kWh_m2.quantile(.20)) if daily.DailyEnergy_kWh_m2.notna().any() else np.nan
daily['LowEnergyRiskDay']=daily.DailyEnergy_kWh_m2.le(energy_threshold) if pd.notna(energy_threshold) else False
daily.to_csv(DATA/'rebellion_daily_briefing.csv',index=False); daily.to_csv(DATA/'daily_report.csv',index=False)
risk_days=daily[daily.LowEnergyRiskDay].copy(); risk_days.to_csv(DATA/'low_energy_risk_days.csv',index=False)
# Consecutive low-energy risk clusters based on consecutive calendar dates.
risk_dates=pd.to_datetime(risk_days.Date,errors='coerce').dropna().sort_values(); clusters=[]
if len(risk_dates):
    current=[risk_dates.iloc[0]]
    for d in risk_dates.iloc[1:]:
        if (d-current[-1]).days==1: current.append(d)
        else: clusters.append(current); current=[d]
    clusters.append(current)
cluster_rows=[{'start':str(g[0].date()),'end':str(g[-1].date()),'consecutive_days':len(g)} for g in clusters]
pd.DataFrame(cluster_rows).to_csv(DATA/'low_energy_risk_clusters.csv',index=False)
daytype_pct=(daily.DayType.value_counts(normalize=True)*100).rename_axis('DayType').reset_index(name='PercentOfDays')
daytype_pct.to_csv(DATA/'day_type_reliability.csv',index=False)
# Typical operating window: average by clock hour, relative to 80% of each day's maximum.
per_day_peak=df.groupby('Date')[rad_c].transform('max')
operating=df.assign(Above80PctPeak=df[rad_c]>=0.8*per_day_peak).groupby('Hour').Above80PctPeak.mean().rename('FractionOfReadingsAbove80PctDailyPeak').reset_index()
operating.to_csv(DATA/'best_operating_window_by_hour.csv',index=False)
fig,ax=plt.subplots(figsize=(9,4)); ax.plot(operating.Hour,operating.FractionOfReadingsAbove80PctDailyPeak,marker='o'); ax.set(title='Base Power Planning: Fraction Above 80% of Daily Peak',xlabel='Hour of day',ylabel='Fraction of readings'); savefig('N3_best_operating_window.png')
# Calendar-style readiness heatmap.
cal=daily.copy(); cal['DateParsed']=pd.to_datetime(cal.Date,errors='coerce'); cal=cal.dropna(subset=['DateParsed']); cal['Week']=cal.DateParsed.dt.isocalendar().week.astype(int); cal['Weekday']=cal.DateParsed.dt.dayofweek
if len(cal):
    pivot=cal.pivot_table(index='Weekday',columns='Week',values='ReadinessScore',aggfunc='mean')
    fig,ax=plt.subplots(figsize=(max(8,min(14,pivot.shape[1]*.35)),4)); sns.heatmap(pivot,cmap='YlGn',vmin=0,vmax=100,ax=ax,cbar_kws={'label':'Readiness score (0–100)'})
    ax.set(title='N4. Rebellion Readiness Calendar Heatmap',xlabel='ISO week number',ylabel='Weekday (0=Monday)'); savefig('N4_readiness_calendar_heatmap.png')
base_lines=['BASE POWER PLANNING — OPERATIONAL INDICATORS',f'Low-energy threshold is the 20th percentile of observed daily integrated radiation energy: {energy_threshold:.4f} kWh/m² (assuming radiation is W/m² and timestamps are representative).',f'Low-energy days: {len(risk_days)} of {len(daily)} days. This is a relative risk flag, not blackout probability.']
if cluster_rows:
    base_lines.append('Longest consecutive low-energy cluster: '+str(max(cluster_rows,key=lambda x:x['consecutive_days'])))
base_lines.append('Day-type proportions: '+(daytype_pct.to_string(index=False) if len(daytype_pct) else 'unavailable'))
if len(operating):
    best_hours=operating.nlargest(3,'FractionOfReadingsAbove80PctDailyPeak')
    base_lines.append('Hours most often at/above 80% of each day peak: '+', '.join(f"{int(r.Hour):02d}:00 ({r.FractionOfReadingsAbove80PctDailyPeak:.1%})" for r in best_hours.itertuples()))
base_lines.append('ReadinessScore is min-max-normalized daily energy; it ranks days within this dataset and is not engineering-grade system sizing.')
write(base_lines,'base_power_planning.txt')
fig,ax=plt.subplots(figsize=(8,4)); t=top5.sort_values('DailyMean'); ax.bar(t.Date.astype(str),t.DailyMean); ax.set(title='C1. Top 5 Days by Average Radiation',xlabel='Date',ylabel='Daily mean radiation'); ax.tick_params(axis='x',rotation=30); savefig('C1_top5_daily_average.png')
pc=['PART C — SOLAR DAY ANALYSIS','Datetime dtype: '+str(df.Datetime.dtype),'Daily max/mean and peak times: data/daily_report.csv','Top five days by average radiation:']
pc += [f'{i}. {r.Date}: mean={r.DailyMean:.3f}, max={r.DailyMax:.3f}, n={int(r.Observations)}' for i,r in enumerate(top5.itertuples(),1)]
if peak is not None:
    pc += [f'Highest single reading: date={peak.Datetime.date()}, time={peak.Datetime.time()}, radiation={peak[rad_c]:.3f}.',f'Temperature={fmt(peak[temp_c]) if temp_c else "column unavailable"}; humidity={fmt(peak[hum_c]) if hum_c else "column unavailable"}; wind speed={fmt(peak[speed_c]) if speed_c else "column unavailable"}.']
    day=df[df.Date==peak.Datetime.date()]; sr=day.SunriseDatetime.dropna().iloc[0] if 'SunriseDatetime' in day and day.SunriseDatetime.notna().any() else pd.NaT; ss=day.SunsetDatetime.dropna().iloc[0] if 'SunsetDatetime' in day and day.SunsetDatetime.notna().any() else pd.NaT
    if pd.notna(sr) and pd.notna(ss) and ss>sr:
        elapsed=(peak.Datetime-sr).total_seconds()/60; length=(ss-sr).total_seconds()/60; noon=sr+(ss-sr)/2
        pc += [f'Sunrise={sr.time()}, sunset={ss.time()}, day length={length/60:.2f} hours.',f'Peak after sunrise={int(elapsed//60)} hours {int(round(elapsed%60))} minutes ({elapsed:.1f} minutes); daylight fraction elapsed={elapsed/length:.3f}; offset from solar-noon midpoint={(peak.Datetime-noon).total_seconds()/60:+.1f} minutes.']
    else: pc.append('Sunrise/sunset unavailable or invalid for the peak date; sunrise-lag calculations skipped.')
write(pc,'part_C_solar_day_analysis.txt')

# Bonus prediction — leak-aware validation, ablation, model ladder, uncertainty, explainability.
# Random split is included for the assignment, but adjacent 5-minute rows can leak
# temporal patterns. Chronological and grouped-by-date scores are more conservative.
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.inspection import permutation_importance, PartialDependenceDisplay
from sklearn.model_selection import GroupKFold

weather_features=[c for c in [temp_c,hum_c,press_c,speed_c,wind_c] if c and c in df and df[c].nunique(dropna=True)>1]
time_features=[c for c in ['SolarFrac','Hour','DayLength_h','DayOfYear','HourSin','HourCos','DayOfYearSin','DayOfYearCos','MinutesSinceSunrise','DayLengthMinutes'] if c in df and df[c].nunique(dropna=True)>1]
# User-facing ablation names are preserved even if the input CSV lacks some columns.
feature_sets={
    'Time-only':time_features,
    'Weather-only':weather_features,
    'Combined':list(dict.fromkeys(time_features+weather_features)),
}
feature_sets={k:v for k,v in feature_sets.items() if v}
base_features=list(dict.fromkeys(time_features+weather_features))
ml=['BONUS — LEAK-AWARE SOLAR RADIATION PREDICTION',
    'Random 80/20 split is the assignment baseline. Chronological 80/20 and GroupKFold by date reduce neighbouring-row/date leakage.',
    'Temperature is partly caused by radiation and is suitable for nowcasting, not a clean forecast feature. Time-only is the forecast-style ablation.',
    'Model ladder: idealised ClearSky baseline, Linear Regression, Random Forest, Gradient Boosting.']
metric_rows=[]; pred_df=pd.DataFrame(); importance=pd.DataFrame(); perm_df=pd.DataFrame(); ablation_gap=[]
md=df.replace([np.inf,-np.inf],np.nan).copy()
md=md[md[rad_c].notna()].copy()
# A clear-sky baseline has no fitted ML model; it is only meaningful where the proxy exists.
if 'ClearSky' in md and md.ClearSky.notna().any():
    for split_name, idx in [('all_available_proxy',md.index)]:
        pp=md.loc[idx,['ClearSky',rad_c]].dropna()
        if len(pp)>=3:
            metric_rows.append({'split':'clear_sky_baseline','model':'Idealised ClearSky','features':'ClearSky (proxy)','train_n':len(pp),'test_n':len(pp),'train_RMSE':np.sqrt(mean_squared_error(pp[rad_c],pp.ClearSky)),'train_R2':r2_score(pp[rad_c],pp.ClearSky),'test_RMSE':np.sqrt(mean_squared_error(pp[rad_c],pp.ClearSky)),'test_R2':r2_score(pp[rad_c],pp.ClearSky)})

if len(md)>=30 and feature_sets:
    split_point=max(1,min(len(md)-1,int(len(md)*.8)))
    # Evaluate every feature ablation and model on both the standard random and chronological splits.
    for fs_name, fs in feature_sets.items():
        data=md[fs+[rad_c,'Date']].copy()
        data=data.replace([np.inf,-np.inf],np.nan).dropna(subset=[rad_c])
        if len(data)<20: continue
        X=data[fs]; y=data[rad_c]
        # align date groups with rows; median imputation is fitted inside each pipeline.
        random_ix=np.arange(len(data)); rng=np.random.RandomState(RANDOM_STATE); rng.shuffle(random_ix)
        cut=max(1,min(len(data)-1,int(len(data)*.8)))
        random_train,random_test=random_ix[:cut],random_ix[cut:]
        chrono_cut=max(1,min(len(data)-1,int(len(data)*.8)))
        split_defs=[('random_80_20',random_train,random_test),('chronological_80_20',np.arange(chrono_cut),np.arange(chrono_cut,len(data)))]
        models=[('LinearRegression',LinearRegression()),('RandomForest',RandomForestRegressor(n_estimators=160,min_samples_leaf=2,random_state=RANDOM_STATE,n_jobs=-1)),('GradientBoosting',GradientBoostingRegressor(n_estimators=120,max_depth=2,learning_rate=.05,random_state=RANDOM_STATE))]
        for split_name,tr_i,te_i in split_defs:
            if len(tr_i)<2 or len(te_i)<1: continue
            for model_name,est in models:
                pipe=Pipeline([('imputer',SimpleImputer(strategy='median')),('model',est)])
                pipe.fit(X.iloc[tr_i],y.iloc[tr_i]); trp=pipe.predict(X.iloc[tr_i]); tep=pipe.predict(X.iloc[te_i])
                metric_rows.append({'split':split_name,'model':model_name,'features':fs_name,'feature_count':len(fs),'train_n':len(tr_i),'test_n':len(te_i),'train_RMSE':np.sqrt(mean_squared_error(y.iloc[tr_i],trp)),'train_R2':r2_score(y.iloc[tr_i],trp),'test_RMSE':np.sqrt(mean_squared_error(y.iloc[te_i],tep)),'test_R2':r2_score(y.iloc[te_i],tep)})
                # Chronological Combined Random Forest is the main diagnostic model.
                if fs_name=='Combined' and split_name=='chronological_80_20' and model_name=='RandomForest':
                    test_rows=data.iloc[te_i]
                    pred_df=pd.DataFrame({'Datetime':df.loc[test_rows.index,'Datetime'].values,'Date':test_rows['Date'].astype(str).values,'Hour':df.loc[test_rows.index,'Hour'].values,'actual':y.iloc[te_i].values,'predicted':tep,'residual':y.iloc[te_i].values-tep})
                    try:
                        perm=permutation_importance(pipe,X.iloc[te_i],y.iloc[te_i],n_repeats=5,random_state=RANDOM_STATE,scoring='neg_root_mean_squared_error',n_jobs=-1)
                        perm_df=pd.DataFrame({'feature':fs,'permutation_importance_mean':perm.importances_mean,'permutation_importance_std':perm.importances_std}).sort_values('permutation_importance_mean',ascending=False)
                    except Exception as exc: ml.append(f'Permutation importance unavailable: {exc}')
                    try:
                        importance=pd.DataFrame({'feature':fs,'importance':pipe.named_steps['model'].feature_importances_}).sort_values('importance',ascending=False)
                    except Exception: importance=pd.DataFrame()
                    # Partial dependence for up to three features, only when supported by the fitted model.
                    try:
                        top_feats=(perm_df.feature.head(3).tolist() if len(perm_df) else fs[:3])
                        fig,ax=plt.subplots(figsize=(10,3.5*len(top_feats)))
                        axes=np.atleast_1d(ax)
                        PartialDependenceDisplay.from_estimator(pipe,X.iloc[tr_i],features=top_feats,ax=axes[:len(top_feats)],kind='average')
                        fig.suptitle('ML4. Partial Dependence — Combined Random Forest',y=1.01); savefig('ML4_partial_dependence.png')
                    except Exception as exc: ml.append(f'Partial dependence plot unavailable: {exc}')

    # GroupKFold by date for Combined features: all rows from a date stay in one fold.
    if 'Combined' in feature_sets and len(feature_sets['Combined']):
        fs=feature_sets['Combined']; data=md[fs+[rad_c,'Date']].replace([np.inf,-np.inf],np.nan).dropna(subset=[rad_c]); X=data[fs]; y=data[rad_c]; groups=data['Date'].astype(str)
        n_groups=groups.nunique(); n_splits=min(5,n_groups)
        if n_splits>=2:
            gkf=GroupKFold(n_splits=n_splits)
            for model_name,est in [('LinearRegression',LinearRegression()),('RandomForest',RandomForestRegressor(n_estimators=160,min_samples_leaf=2,random_state=RANDOM_STATE,n_jobs=-1)),('GradientBoosting',GradientBoostingRegressor(n_estimators=120,max_depth=2,learning_rate=.05,random_state=RANDOM_STATE))]:
                fold_y=[]; fold_p=[]; fold_train_rmse=[]; fold_train_r2=[]
                for tr_i,te_i in gkf.split(X,y,groups):
                    pipe=Pipeline([('imputer',SimpleImputer(strategy='median')),('model',est.__class__(**est.get_params()))])
                    pipe.fit(X.iloc[tr_i],y.iloc[tr_i]); p=pipe.predict(X.iloc[te_i]); fold_y.extend(y.iloc[te_i].tolist()); fold_p.extend(p.tolist())
                    trp=pipe.predict(X.iloc[tr_i]); fold_train_rmse.append(np.sqrt(mean_squared_error(y.iloc[tr_i],trp))); fold_train_r2.append(r2_score(y.iloc[tr_i],trp))
                metric_rows.append({'split':f'GroupKFold_by_date_{n_splits}fold','model':model_name,'features':'Combined','feature_count':len(fs),'train_n':len(data)-len(fold_y),'test_n':len(fold_y),'train_RMSE':float(np.mean(fold_train_rmse)),'train_R2':float(np.mean(fold_train_r2)),'test_RMSE':np.sqrt(mean_squared_error(fold_y,fold_p)),'test_R2':r2_score(fold_y,fold_p)})

    # Physics-informed prediction: predict clearness index, then reconstruct radiation = index × ClearSky.
    # This is evaluated only where ClearSky is positive and ClearnessIdx is defined.
    phys=md[(md.ClearSky>50)&md.ClearnessIdx.notna()].copy() if {'ClearSky','ClearnessIdx'}.issubset(md.columns) else pd.DataFrame()
    phys_features=list(dict.fromkeys(time_features+weather_features))
    if len(phys)>=30 and phys_features:
        Xp=phys[phys_features].replace([np.inf,-np.inf],np.nan); yp=phys.ClearnessIdx; splitp=max(1,min(len(phys)-1,int(len(phys)*.8)))
        for split_name, tr_i, te_i in [('random_80_20',np.random.RandomState(RANDOM_STATE).permutation(len(phys))[:int(len(phys)*.8)],np.random.RandomState(RANDOM_STATE).permutation(len(phys))[int(len(phys)*.8):]),('chronological_80_20',np.arange(splitp),np.arange(splitp,len(phys)))]:
            if not len(te_i) or not len(tr_i): continue
            pipe=Pipeline([('imputer',SimpleImputer(strategy='median')),('model',RandomForestRegressor(n_estimators=160,min_samples_leaf=2,random_state=RANDOM_STATE,n_jobs=-1))])
            pipe.fit(Xp.iloc[tr_i],yp.iloc[tr_i]); clearness_pred=pipe.predict(Xp.iloc[te_i]); radiation_pred=clearness_pred*phys.ClearSky.iloc[te_i].to_numpy(); actual=phys[rad_c].iloc[te_i].to_numpy()
            metric_rows.append({'split':split_name,'model':'Physics-informed RF (ClearnessIdx×ClearSky)','features':'Combined','feature_count':len(phys_features),'train_n':len(tr_i),'test_n':len(te_i),'train_RMSE':np.nan,'train_R2':np.nan,'test_RMSE':np.sqrt(mean_squared_error(actual,radiation_pred)),'test_R2':r2_score(actual,radiation_pred)})

    # Quantile prediction interval for the last/most recent sample day; intervals are empirical model ranges.
    try:
        fs=feature_sets.get('Time-only',base_features)
        qdata=md[fs+[rad_c,'Date']].replace([np.inf,-np.inf],np.nan).dropna(subset=[rad_c])
        if len(qdata)>=30:
            splitq=max(1,min(len(qdata)-1,int(len(qdata)*.8))); Xq=qdata[fs]; yq=qdata[rad_c]
            qlo=Pipeline([('imputer',SimpleImputer(strategy='median')),('model',GradientBoostingRegressor(loss='quantile',alpha=.1,n_estimators=120,max_depth=2,learning_rate=.05,random_state=RANDOM_STATE))])
            qhi=Pipeline([('imputer',SimpleImputer(strategy='median')),('model',GradientBoostingRegressor(loss='quantile',alpha=.9,n_estimators=120,max_depth=2,learning_rate=.05,random_state=RANDOM_STATE))])
            qlo.fit(Xq.iloc[:splitq],yq.iloc[:splitq]); qhi.fit(Xq.iloc[:splitq],yq.iloc[:splitq]); latest_date=qdata.Date.max(); sample=qdata[qdata.Date.astype(str)==str(latest_date)]
            if len(sample):
                lowp=qlo.predict(sample[fs]); highp=qhi.predict(sample[fs]); interval=pd.DataFrame({'Date':sample.Date.astype(str).values,'actual_radiation':sample[rad_c].values,'q10_prediction':lowp,'q90_prediction':highp,'interval_width':highp-lowp})
                interval.to_csv(DATA/'quantile_prediction_intervals_sample_day.csv',index=False)
                fig,ax=plt.subplots(figsize=(10,4)); x=np.arange(len(interval)); ax.plot(x,interval.actual_radiation,label='Actual'); ax.plot(x,interval.q10_prediction,label='10th percentile'); ax.plot(x,interval.q90_prediction,label='90th percentile'); ax.fill_between(x,interval.q10_prediction,interval.q90_prediction,alpha=.2); ax.set(title=f'ML5. Quantile Prediction Range — {latest_date}',xlabel='Sample-day observations (time order)',ylabel='Radiation'); ax.legend(); savefig('ML5_quantile_interval.png')
                ml.append(f'Quantile interval sample day: {latest_date}; interval coverage on that sample day is computed in the CSV only after comparing actual readings with q10/q90; this interval is not guaranteed calibrated.')
    except Exception as exc: ml.append(f'Quantile interval generation skipped: {exc}')

    # Error breakdown by hour and heuristic day type for chronological combined RF predictions.
    if len(pred_df):
        pred_df['absolute_error']=pred_df.residual.abs(); pred_df['squared_error']=pred_df.residual**2
        hour_err=pred_df.groupby('Hour').agg(n=('residual','size'),RMSE=('squared_error',lambda z:np.sqrt(z.mean())),MAE=('absolute_error','mean')).reset_index(); hour_err.to_csv(DATA/'ml_error_by_hour.csv',index=False)
        if 'DayType' in daily:
            pred_df=pred_df.merge(daily[['Date','DayType']].assign(Date=lambda z:z.Date.astype(str)),on='Date',how='left')
            type_err=pred_df.groupby('DayType').agg(n=('residual','size'),RMSE=('squared_error',lambda z:np.sqrt(z.mean())),MAE=('absolute_error','mean')).reset_index(); type_err.to_csv(DATA/'ml_error_by_day_type.csv',index=False)
        fig,ax=plt.subplots(figsize=(8,4)); ax.bar(hour_err.Hour,hour_err.RMSE); ax.set(title='ML6. Chronological RF RMSE by Hour',xlabel='Hour of day',ylabel='RMSE'); savefig('ML6_rmse_by_hour.png')
        if 'DayType' in pred_df and pred_df.DayType.notna().any():
            type_err=pred_df.groupby('DayType').residual.apply(lambda z:np.sqrt(np.mean(z**2))).sort_values(); fig,ax=plt.subplots(figsize=(7,4)); ax.bar(type_err.index.astype(str),type_err.values); ax.set(title='ML7. Chronological RF RMSE by Day Type',xlabel='Heuristic day type',ylabel='RMSE'); savefig('ML7_rmse_by_day_type.png')

metrics=pd.DataFrame(metric_rows)
if len(metrics):
    metrics.to_csv(DATA/'model_comparison_metrics.csv',index=False)
    # Quantify random-versus-chronological generalisation gap for matching model/feature rows.
    for model_name in metrics.model.unique():
        for feat_name in metrics.features.dropna().unique():
            r=metrics[(metrics.model==model_name)&(metrics.features==feat_name)&(metrics.split=='random_80_20')]
            c=metrics[(metrics.model==model_name)&(metrics.features==feat_name)&(metrics.split=='chronological_80_20')]
            if len(r) and len(c): ablation_gap.append(f'{model_name} / {feat_name}: chronological RMSE minus random RMSE = {c.iloc[0].test_RMSE-r.iloc[0].test_RMSE:+.3f}; random RMSE={r.iloc[0].test_RMSE:.3f}, chronological RMSE={c.iloc[0].test_RMSE:.3f}.')
else:
    pd.DataFrame().to_csv(DATA/'model_comparison_metrics.csv',index=False)
pred_df.to_csv(DATA/'chronological_test_predictions.csv',index=False)
importance.to_csv(DATA/'feature_importance.csv',index=False); perm_df.to_csv(DATA/'permutation_importance.csv',index=False)
ml += ['Results summary table (model × features × split × RMSE × R²):',metrics[['model','features','split','test_RMSE','test_R2']].to_string(index=False) if len(metrics) else 'No valid model metrics available.', 'Validation gap:'] + (ablation_gap if ablation_gap else ['Random/chronological matching results unavailable; check data size and columns.'])
if len(importance): ml += ['Random Forest impurity importance (predictive association, not causation):']+[f'{r.feature}: {r.importance:.4f}' for r in importance.head(10).itertuples()]
if len(perm_df): ml += ['Permutation importance on chronological test rows (RMSE scoring; larger positive value indicates greater predictive contribution):']+[f'{r.feature}: {r.permutation_importance_mean:.4f} ± {r.permutation_importance_std:.4f}' for r in perm_df.head(10).itertuples()]
ml += ['Interpretation: time-only is the forecast-style model; weather-only shows weather information alone; combined estimates the benefit of both. Temperature is partly downstream of radiation and makes the combined model a nowcast if temperature is not known at forecast time.', 'ClearSky is an idealised half-sine envelope based on a dataset percentile, not a validated atmospheric clear-sky model. ClearnessIdx and physics-informed predictions inherit that limitation.', 'Error may be higher around sunrise/sunset and on cloud-variable days; inspect ml_error_by_hour.csv and ml_error_by_day_type.csv rather than assuming this pattern.']
write(ml,'bonus_prediction_results.txt')

# Novelty: relative Solar Reliability Score and sustained low-radiation episodes
positive=df.loc[df[rad_c]>0,rad_c].dropna(); threshold=float(positive.quantile(.10)) if len(positive) else 0.0
if 'SunriseDatetime' in df and 'SunsetDatetime' in df:
    day_df=df[df.Datetime.ge(df.SunriseDatetime)&df.Datetime.le(df.SunsetDatetime)&df.SunriseDatetime.notna()&df.SunsetDatetime.notna()].copy()
else: day_df=df[df[rad_c]>0].copy()
rr=[]
for date,frame in day_df.groupby('Date'):
    v=frame[rad_c].dropna()
    if len(v): rr.append({'Date':str(date),'daylight_n':len(v),'daylight_mean':v.mean(),'cv':v.std(ddof=0)/abs(v.mean()) if v.mean()!=0 else np.nan,'low_fraction':(v<=threshold).mean()})
rel=pd.DataFrame(rr)
def mm(s):
    s=pd.to_numeric(s,errors='coerce'); lo=s.min(); hi=s.max()
    return pd.Series(.5,index=s.index) if pd.isna(lo) or pd.isna(hi) or lo==hi else (s-lo)/(hi-lo)
if len(rel):
    rel['score_0_100']=(100*(.5*mm(rel.daylight_mean)+.25*(1-mm(rel.cv.replace([np.inf,-np.inf],np.nan).fillna(0)))+.25*(1-mm(rel.low_fraction)))).round(2)
    rel=rel.sort_values('score_0_100',ascending=False)
rel.to_csv(DATA/'solar_reliability_score.csv',index=False)
all_diff=df.Datetime.diff().dt.total_seconds().dropna(); med=all_diff.median() if len(all_diff) else 600; gap=1.5*med if pd.notna(med) and med>0 else 900
low=day_df[day_df[rad_c].notna()&(day_df[rad_c]<=threshold)].sort_values('Datetime').copy(); low['gap']=low.Datetime.diff().dt.total_seconds(); low['episode']=((low.gap.isna())|(low.gap>gap)).cumsum(); eps=[]
for eid,g in low.groupby('episode'):
    eps.append({'episode_id':int(eid),'start':g.Datetime.min(),'end':g.Datetime.max(),'duration_minutes_est':((g.Datetime.max()-g.Datetime.min()).total_seconds()+med)/60,'observations':len(g),'mean_radiation':g[rad_c].mean(),'threshold':threshold})
episodes=pd.DataFrame(eps)
if len(episodes): episodes=episodes.sort_values('duration_minutes_est',ascending=False)
episodes.to_csv(DATA/'low_radiation_episodes.csv',index=False)
nov=['OPERATION DAYBREAK NOVELTY','Solar Reliability Score is a relative ranking, not actual energy yield or blackout probability.',f'Low-radiation threshold: {threshold:.3f} (10th percentile of positive radiation).', 'Daylight is defined by sunrise/sunset when available; otherwise positive readings are used. Episodes split at gaps greater than 1.5 times median sampling interval.']
if len(rel): nov.append(f"Top reliability day: {rel.iloc[0]['Date']}, score {rel.iloc[0]['score_0_100']:.2f}/100, mean daylight radiation {rel.iloc[0]['daylight_mean']:.3f}.")
if len(episodes): nov.append(f"Longest low-radiation episode: {episodes.iloc[0]['start']} to {episodes.iloc[0]['end']}, estimated {episodes.iloc[0]['duration_minutes_est']:.1f} minutes.")
else: nov.append('No low-radiation episodes identified under the chosen threshold.')
nov.append('Actual blackout risk requires panel capacity, battery state of charge, and load data.')
write(nov,'operation_daybreak_novelty.txt')
if len(rel):
    t=rel.head(10).sort_values('score_0_100'); fig,ax=plt.subplots(figsize=(8,max(4,len(t)*.35))); ax.barh(t.Date,t.score_0_100); ax.set(title='N1. Relative Solar Reliability Score',xlabel='Score (0–100)',ylabel='Date'); savefig('N1_reliability_score.png')
if len(episodes):
    t=episodes.head(12).sort_values('duration_minutes_est'); fig,ax=plt.subplots(figsize=(8,max(4,len(t)*.35))); ax.barh(t.start.astype(str),t.duration_minutes_est); ax.set(title='N2. Longest Low-Radiation Episodes',xlabel='Estimated minutes',ylabel='Episode start'); savefig('N2_low_radiation_episodes.png')

df.to_csv(DATA/'SolarPrediction_cleaned.csv',index=False)

# Generate a populated PDF from actual outputs.
styles=getSampleStyleSheet(); styles.add(ParagraphStyle(name='TitleX',parent=styles['Title'],fontSize=23,leading=28,textColor=colors.HexColor('#243B53'),spaceAfter=12)); styles.add(ParagraphStyle(name='H1X',parent=styles['Heading1'],fontSize=15,leading=18,textColor=colors.HexColor('#243B53'),spaceBefore=8,spaceAfter=6))
story=[Spacer(1,.8*inch),Paragraph('STAC WARS PS2',styles['TitleX']),Paragraph('OPERATION DAYBREAK',styles['TitleX']),Paragraph('Can the Rebellion’s base trust the Sun?',styles['Heading2']),Spacer(1,.2*inch),Paragraph(f'Dataset: {CSV.name} | Rows analyzed: {len(df)} | Dates: {df.Date.nunique()}',styles['BodyText']),Spacer(1,.2*inch),Paragraph('Solar radiation analysis, evidence-based inference, daily solar report, prediction, and relative solar-reliability novelty.',styles['BodyText']),PageBreak()]
# Mission Brief — calculated from the real dataset rather than filled with guessed headline findings.
mission=['MISSION BRIEF']
if not hourly.dropna().empty:
    mission.append(f'Headline finding 1 — Mean radiation peaks at {int(hourly.idxmax()):02d}:00 ({hourly.max():.3f} dataset units). This is the best observed clock-hour average, not a guaranteed forecast.')
if not corr.empty:
    c0=corr[corr.subset=='positive_radiation']
    if c0.empty: c0=corr
    if len(c0):
        r0=c0.iloc[0]; mission.append(f'Headline finding 2 — Strongest absolute Pearson relationship in the selected comparison is {r0.parameter} (r={r0.pearson_r:.3f}, n={int(r0.n)}). Correlation does not establish causation.')
if len(daily) and daily.DailyEnergy_kWh_m2.notna().any():
    best=daily.loc[daily.DailyEnergy_kWh_m2.idxmax()]
    mission.append(f'Headline finding 3 — Highest integrated daily radiation energy occurs on {best.Date} ({best.DailyEnergy_kWh_m2:.3f} kWh/m² under the W/m² assumption). This is radiation energy per area, not electrical output.')
if pd.notna(energy_threshold): mission.append(f'Recommendation for the base — treat days at or below the observed 20th-percentile daily energy ({energy_threshold:.3f} kWh/m²) as relative low-resource planning flags; preserve storage margin and verify against battery, load, and panel data before making operational decisions.')
mission.append('These are dataset-specific results. The idealised clear-sky envelope and day-type labels are heuristic proxies; they are not weather forecasts or engineering guarantees.')
for line in mission: story.append(Paragraph(line.replace('&','&amp;').replace('<','&lt;').replace('>','&gt;'),styles['BodyText'])); story.append(Spacer(1,5))
story.append(PageBreak())
def add_txt(path,limit=60):
    if path.exists():
        for line in path.read_text(encoding='utf-8',errors='replace').splitlines()[:limit]:
            if line.strip(): story.append(Paragraph(line.replace('&','&amp;').replace('<','&lt;').replace('>','&gt;'),styles['BodyText']))
            else: story.append(Spacer(1,4))
story += [Paragraph('1. Dataset and Cleaning',styles['H1X'])]
add_txt(REPORTS/'column_dtypes.txt',30); add_txt(REPORTS/'data_inspection.json',30)
story.append(Paragraph('Data quality audit table',styles['Heading2']))
audit_data=[['Issue','Count','Handling decision']]
for ar in audit_rows:
    audit_data.append([str(ar['issue']),str(ar['count']),str(ar['handling'])])
audit_table=Table(audit_data,colWidths=[1.65*inch,.55*inch,4.45*inch],repeatRows=1)
audit_table.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#243B53')),('TEXTCOLOR',(0,0),(-1,0),colors.white),('FONTNAME',(0,0),(-1,0),'Helvetica-Bold'),('FONTSIZE',(0,0),(-1,-1),7),('LEADING',(0,0),(-1,-1),8),('VALIGN',(0,0),(-1,-1),'TOP'),('GRID',(0,0),(-1,-1),.35,colors.grey),('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.white,colors.HexColor('#F0F4F8')]),('LEFTPADDING',(0,0),(-1,-1),4),('RIGHTPADDING',(0,0),(-1,-1),4),('TOPPADDING',(0,0),(-1,-1),4),('BOTTOMPADDING',(0,0),(-1,-1),4)]))
story.append(audit_table); story.append(Spacer(1,8))
story += [Paragraph('2. Part A — Visualizations',styles['H1X'])]
for name,caption in [('A1_radiation_over_time.png','Radiation over time'),('A2_mean_radiation_by_hour.png','Mean radiation by hour'),('A3_radiation_distribution.png','Radiation distribution'),('A4_correlation_heatmap.png','Numeric correlation heatmap'),('A9_daybreak_heatmap.png','Date-hour heatmap'),('A10_daily_maximum.png','Daily maximum radiation'),('A11_radiation_outliers.png','Radiation outlier check'),('A12_clear_sky_envelope.png','Idealised clear-sky envelope comparison'),('A13_hysteresis_path.png','Temperature-radiation path coloured by hour'),('A14_single_day_zoom.png','Single-day radiation zoom')]:
    p=PLOTS/name
    if p.exists():
        story.append(Paragraph(caption,styles['BodyText'])); im=Image(str(p)); im._restrictSize(6.7*inch,3.1*inch); story.append(im); story.append(Spacer(1,5))
story.append(PageBreak()); story.append(Paragraph('3. Part B — Drawing Inferences',styles['H1X'])); add_txt(REPORTS/'part_B_inferences.txt',40)
story.append(Paragraph('4. Part C — Solar Day Analysis',styles['H1X'])); add_txt(REPORTS/'part_C_solar_day_analysis.txt',45)
story.append(Paragraph('5. Bonus — Prediction',styles['H1X'])); add_txt(REPORTS/'bonus_prediction_results.txt',35)
for name,caption in [('ML1_actual_vs_predicted.png','Actual vs predicted'),('ML2_residuals.png','Residual diagnostics'),('ML3_feature_importance.png','Random Forest feature importance'),('ML4_partial_dependence.png','Partial dependence plots'),('ML5_quantile_interval.png','Quantile prediction interval for a sample day'),('ML6_rmse_by_hour.png','Chronological test RMSE by hour'),('ML7_rmse_by_day_type.png','Chronological test RMSE by heuristic day type')]:
    p=PLOTS/name
    if p.exists(): story.append(Paragraph(caption,styles['BodyText'])); im=Image(str(p)); im._restrictSize(6.5*inch,3*inch); story.append(im)
story.append(Paragraph('6. Novelty — Operation Daybreak',styles['H1X'])); add_txt(REPORTS/'operation_daybreak_novelty.txt',20)
story.append(Paragraph('7. Data Quality Audit',styles['H1X'])); add_txt(REPORTS/'data_inspection.json',35)
add_txt(REPORTS/'part_B_novel_evidence.txt',30)
story.append(Paragraph('8. Rebellion Daily Briefing and Base Power Planning',styles['H1X'])); add_txt(REPORTS/'base_power_planning.txt',30)
for name,caption in [('N1_reliability_score.png','Relative reliability ranking'),('N2_low_radiation_episodes.png','Sustained low-radiation episodes'),('N3_best_operating_window.png','Best operating window'),('N4_readiness_calendar_heatmap.png','Readiness calendar heatmap')]:
    p=PLOTS/name
    if p.exists(): story.append(Paragraph(caption,styles['BodyText'])); im=Image(str(p)); im._restrictSize(6.5*inch,3*inch); story.append(im)
story += [Spacer(1,8),Paragraph('9. Assumptions and Limitations',styles['H1X'])]
assumptions=[
'Only the dates and season represented in the supplied file are analysed; one season or a short date range may not generalise to other seasons or years.',
'No cloud-cover variable is assumed to exist. Clear/Partly cloudy/Overcast labels are heuristic bins of an idealised clearness proxy, not verified meteorological labels.',
'Sensor noise, timestamp errors, duplicated records, and missing intervals can affect summaries. Outliers are candidates for review, not automatic proof of sensor failure.',
'Flare caveat: readings above the idealised half-sine envelope may be cloud-edge enhancement, sensor noise, timing mismatch, or envelope-model error. They are not automatically “solar flares.”',
'DailyEnergy_kWh_m2 assumes Radiation is W/m² and uses trapezoidal integration over observed timestamps; irregular gaps can make this estimate incomplete. It is not panel electrical yield.',
'Temperature may partly be caused by solar radiation. Weather-feature models are nowcasts if temperature is measured at prediction time; time-only features are closer to forecast-style prediction.',
'Quantile intervals are model estimates and are not guaranteed to have calibrated 80% coverage. The clear-sky proxy is not a validated atmospheric clear-sky model.',
'Blackout-risk flags are relative low-resource indicators only. Actual outage risk requires load demand, panel efficiency/capacity, battery state of charge, storage losses, and system constraints.'
]
for item in assumptions: story.append(Paragraph('• '+item.replace('&','&amp;').replace('<','&lt;').replace('>','&gt;'),styles['BodyText'])); story.append(Spacer(1,4))
import platform
requirements='pandas numpy matplotlib seaborn scikit-learn reportlab'
(OUT/'requirements.txt').write_text('\n'.join(['pandas','numpy','matplotlib','seaborn','scikit-learn','reportlab'])+'\n',encoding='utf-8')
story.append(Paragraph(f'Reproducibility: Python {platform.python_version()}; dependencies listed in {OUT / "requirements.txt"}; this single Python script was syntax-checked and run as a script on a synthetic smoke-test dataset during development. “Restart &amp; Run All” is a notebook workflow and is not applicable to this standalone .py file. Re-run this script on the original CSV to reproduce the actual report.',styles['BodyText']))
story += [Spacer(1,8),Paragraph('What this means for the base: use these indicators to prepare and prioritise monitoring, not to promise solar availability. Confirm the site’s electrical system and weather forecasts before committing critical loads.',styles['BodyText'])]
def page_num(canvas,doc):
    canvas.saveState(); canvas.setFont('Helvetica',8); canvas.setFillColor(colors.grey); canvas.drawString(.6*inch,.35*inch,'STAC WARS PS2 • Operation Daybreak'); canvas.drawRightString(7.9*inch,.35*inch,f'Page {doc.page}'); canvas.restoreState()
pdf=REPORTS/'Operation_Daybreak_SolarSense_Report.pdf'
SimpleDocTemplate(str(pdf),pagesize=letter,rightMargin=.6*inch,leftMargin=.6*inch,topMargin=.55*inch,bottomMargin=.6*inch,title='Operation Daybreak SolarSense Report').build(story,onFirstPage=page_num,onLaterPages=page_num)
print('Operation Daybreak complete.')
print(f'Analyzed {len(df)} rows across {df.Date.nunique()} dates.')
print(f'Plots: {PLOTS}\nTables: {DATA}\nReports/PDF: {REPORTS}')
print(f'Populated PDF: {pdf}')
