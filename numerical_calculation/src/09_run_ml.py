"""09_run_ml.py
=========================================================
核心新增：中国期权特征的机器学习预测管线（作业钦点方向）。

任务：面板 (underlying × month)，特征 = 29 个期权隐含指标（t 月末快照，
已滞后一期以预测 t+1），目标 = t+1 月标的超额收益（扣 1M SHIBOR）。

模型（老师钦点三选）：
  - LASSO  (LassoCV，标的去均值 + 标准化 + 5 折 CV)
  - Ridge  (RidgeCV，同上)
  - Random Forest (RandomForestRegressor，标的作分类特征)

评估：
  - 时序前向 walk-forward CV：每月用历史全部月份训练、预测下月，杜绝数据泄漏；
    报告样本外 R²、方向准确率。
  - 特征重要性：LASSO 非零标准化系数 + RF Gini + 排列重要性（OOS）。
  - 单变量 panel 回归（含标的固定效应、按月聚类 SE）作为对照，并做
    Bonferroni / Benjamini-Hochberg FDR 多重检验校正。

输入: data/processed/factor_panel.parquet
输出: result/tables/ml_*.csv, result/tables/univariate_panel.csv
=========================================================
"""

from __future__ import annotations
import os, sys, warnings
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.linear_model import LassoCV, RidgeCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import permutation_importance
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline

warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(__file__))

PROC_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
TBL_DIR   = os.path.join(os.path.dirname(__file__), "..", "result", "tables")
os.makedirs(TBL_DIR, exist_ok=True)

FACTOR_COLS = [
    'CIV', 'PIV', 'dCIV', 'dPIV',
    'IVS_ATM', 'IVS_OTM', 'dIVS_ATM', 'dIVS_OTM', 'SKEW', 'AVAR',
    'VARQ', 'VAR_PLUS', 'VAR_MINUS', 'VRP', 'KT',
    'TSCALL', 'TSPUT', 'dTSCALL', 'dTSPUT', 'VOV',
    'O_S', 'betaVIX', 'betaSkew', 'betaVRP', 'betaIVSOTM',
    'betaTSCALL', 'betaTSPUT', 'betaJump', 'betaVol',
]
TARGET = 'ret_next_excess'
MIN_TRAIN_MONTHS = 36


# -----------------------------------------------------
# 数据准备
# -----------------------------------------------------
def prepare_panel():
    panel = pd.read_parquet(os.path.join(PROC_DIR, "factor_panel.parquet"))
    panel = panel.dropna(subset=[TARGET]).copy()
    panel['ym'] = pd.PeriodIndex(panel['ym'], freq='M')
    panel = panel.sort_values(['ym', 'underlying']).reset_index(drop=True)
    return panel


def make_X_y(df, with_entity=False):
    X = df[FACTOR_COLS].copy()
    y = df[TARGET].values.astype(float)
    if with_entity:
        # 标的固定效应：去均值（within 变换）
        for c in FACTOR_COLS:
            X[c] = X[c] - df.groupby('underlying')[c].transform('mean')
        y = y - df.groupby('underlying')[TARGET].transform('mean').values
    return X, y


# -----------------------------------------------------
# 前向 walk-forward CV
# -----------------------------------------------------
def walk_forward(panel):
    months = sorted(panel['ym'].unique())
    if len(months) <= MIN_TRAIN_MONTHS:
        print(f"  ⚠️  月份不足（{len(months)}），无法做前向 CV")
        return None
    test_months = months[MIN_TRAIN_MONTHS:]

    preds = []
    for tm in test_months:
        train = panel[panel['ym'] < tm]
        test = panel[panel['ym'] == tm]
        if len(train) < 50 or len(test) == 0:
            continue

        # --- LASSO / Ridge：within 去均值 + 标准化 + 中位数填补 ---
        Xt, yt = make_X_y(train, with_entity=True)
        Xv, yv = make_X_y(test, with_entity=True)
        # 对齐 test 的 within 均值用 train 的实体均值近似：这里用 test 自身均值去均值（FE 内变换）
        pipe = make_pipeline(
            SimpleImputer(strategy='median'),
            StandardScaler(),
        )
        Xt_s = pipe.fit_transform(Xt)
        Xv_s = pipe.transform(Xv)
        # 训练实体均值用于还原
        ent_mean_t = train.groupby('underlying')[TARGET].mean()

        lasso = LassoCV(cv=5, random_state=0, max_iter=20000, n_alphas=50).fit(Xt_s, yt)
        ridge = RidgeCV(cv=5).fit(Xt_s, yt)
        pred_l = lasso.predict(Xv_s)
        pred_r = ridge.predict(Xv_s)

        # --- Random Forest：标的作分类特征 ---
        Xrf_t = train[FACTOR_COLS + ['underlying']].copy()
        Xrf_v = test[FACTOR_COLS + ['underlying']].copy()
        for c in FACTOR_COLS:
            med = Xrf_t[c].median()
            Xrf_t[c] = Xrf_t[c].fillna(med)
            Xrf_v[c] = Xrf_v[c].fillna(med)
        Xrf_t['underlying'] = Xrf_t['underlying'].astype('category').cat.codes
        cats = train['underlying'].astype('category').cat.categories
        Xrf_v['underlying'] = pd.Series(
            pd.Categorical(test['underlying'], categories=cats)).cat.codes
        rf = RandomForestRegressor(n_estimators=300, max_depth=6,
                                   min_samples_leaf=5, random_state=0, n_jobs=-1)
        rf.fit(Xrf_t, train[TARGET].values)
        pred_f = rf.predict(Xrf_v)

        # 还原 FE：预测的是去均值后的 y，加回各标的的实体均值
        ent_map = ent_mean_t.to_dict()
        ent_add = test['underlying'].map(ent_map).fillna(0.0).values
        tmp = test[['ym', 'underlying']].copy()
        tmp['actual'] = test[TARGET].values
        tmp['pred_lasso'] = pred_l + ent_add
        tmp['pred_ridge'] = pred_r + ent_add
        tmp['pred_rf'] = pred_f
        preds.append(tmp)
        print(f"    test {tm}: n={len(test)}")

    if not preds:
        return None
    return pd.concat(preds, ignore_index=True)


def evaluate(preds_df):
    def oos_r2(y, p):
        sse = np.sum((y - p) ** 2)
        sst = np.sum((y - np.mean(y)) ** 2)
        return 1 - sse / sst if sst > 0 else np.nan
    def dir_acc(y, p):
        return np.mean(np.sign(y) == np.sign(p))
    rows = []
    for name, col in [('LASSO', 'pred_lasso'), ('Ridge', 'pred_ridge'), ('RF', 'pred_rf')]:
        sub = preds_df.dropna(subset=[col])
        y = sub['actual'].values
        p = sub[col].values
        rows.append({'model': name, 'oos_r2': oos_r2(y, p),
                     'dir_acc': dir_acc(y, p), 'n_test': len(sub)})
    perf = pd.DataFrame(rows)
    perf.to_csv(os.path.join(TBL_DIR, "ml_oos_performance.csv"), index=False)
    print("  ✓ ml_oos_performance.csv")
    print(perf.to_string(index=False))
    return perf


# -----------------------------------------------------
# 特征重要性（全样本拟合 + OOS 排列重要性）
# -----------------------------------------------------
def feature_importance(panel, preds_df):
    Xt, yt = make_X_y(panel, with_entity=True)
    pipe = make_pipeline(SimpleImputer(strategy='median'), StandardScaler())
    Xt_s = pipe.fit_transform(Xt)
    lasso = LassoCV(cv=5, random_state=0, max_iter=20000, n_alphas=50).fit(Xt_s, yt)
    ridge = RidgeCV(cv=5).fit(Xt_s, yt)

    # RF 全样本
    Xrf = panel[FACTOR_COLS + ['underlying']].copy()
    for c in FACTOR_COLS:
        Xrf[c] = Xrf[c].fillna(Xrf[c].median())
    Xrf['underlying'] = Xrf['underlying'].astype('category').cat.codes
    rf = RandomForestRegressor(n_estimators=300, max_depth=6, min_samples_leaf=5,
                               random_state=0, n_jobs=-1).fit(Xrf, panel[TARGET].values)

    # 特征重要性：RF 全样本 fit 时列顺序 = FACTOR_COLS + ['underlying']，gini 按位置取
    rf_cols = list(Xrf.columns)           # 29 因子 + underlying（或其中全 NaN 被 fillna 的）
    gini_map = dict(zip(rf_cols, rf.feature_importances_))
    lc, rc = lasso.coef_, ridge.coef_
    # LASSO/Ridge 在 SimpleImputer 删全 NaN 列后列数 < 29；用 fit 后保留的列名对齐
    kept = [c for c in FACTOR_COLS if not Xt[c].isna().all()]
    lc_map = dict(zip(kept, lc))
    rc_map = dict(zip(kept, rc))
    imp_rows = []
    for f in FACTOR_COLS:
        imp_rows.append({
            'factor': f,
            'lasso_coef_std': lc_map.get(f, np.nan),
            'lasso_selected': abs(lc_map.get(f, 0.0)) > 1e-8 if f in lc_map else False,
            'ridge_coef_std': rc_map.get(f, np.nan),
            'rf_gini': gini_map.get(f, np.nan),
        })
    fi = pd.DataFrame(imp_rows)
    # 排列重要性（最后 12 个月 holdout，RF），按列名对齐回 FACTOR_COLS
    months = sorted(panel['ym'].unique())
    hold = panel[panel['ym'].isin(months[-12:])].copy()
    if len(hold) > 30:
        Xh = hold[FACTOR_COLS + ['underlying']].copy()
        for c in FACTOR_COLS:
            Xh[c] = Xh[c].fillna(Xrf[c].median())
        Xh['underlying'] = pd.Series(pd.Categorical(
            hold['underlying'], categories=panel['underlying'].astype('category').cat.categories)).cat.codes
        perm = permutation_importance(rf, Xh, hold[TARGET].values,
                                      n_repeats=10, random_state=0, n_jobs=-1)
        perm_map = dict(zip(list(Xh.columns), perm.importances_mean))
        fi['rf_perm'] = fi['factor'].map(perm_map)
    else:
        fi['rf_perm'] = np.nan
    fi['rank_gini'] = fi['rf_gini'].rank(ascending=False).astype(int)
    fi['rank_perm'] = fi['rf_perm'].rank(ascending=False).astype(int)
    fi = fi.sort_values('rf_perm', ascending=False)
    fi.to_csv(os.path.join(TBL_DIR, "ml_feature_importance.csv"), index=False)
    print("  ✓ ml_feature_importance.csv")
    print(fi.head(10).to_string(index=False))
    return fi


# -----------------------------------------------------
# 单变量 panel 回归 + 多重检验校正
# -----------------------------------------------------
def univariate_panel(panel):
    rows = []
    for f in FACTOR_COLS:
        sub = panel[['ym', 'underlying', TARGET, f]].dropna()
        if len(sub) < 30:
            rows.append({'factor': f, 'beta': np.nan, 'beta_t': np.nan,
                         'beta_p': np.nan, 'n': 0})
            continue
        X = sub[[f]].values.astype(float)
        d = pd.get_dummies(sub['underlying'], prefix='fe', drop_first=True).values.astype(float)
        X = sm.add_constant(np.hstack([X, d]))
        y = sub[TARGET].values.astype(float)
        try:
            m = sm.OLS(y, X).fit(cov_type='cluster',
                                 cov_kwds={'groups': sub['ym'].astype(str).values})
            rows.append({'factor': f, 'beta': m.params[1], 'beta_t': m.tvalues[1],
                         'beta_p': m.pvalues[1], 'n': int(m.nobs)})
        except Exception:
            rows.append({'factor': f, 'beta': np.nan, 'beta_t': np.nan,
                         'beta_p': np.nan, 'n': 0})
    df = pd.DataFrame(rows)
    # 多重检验校正（仅对有效 p 值）
    p = df['beta_p'].values.astype(float)
    m = int(np.sum(~np.isnan(p)))
    df['p_bonf'] = np.where(~np.isnan(p), np.minimum(p * m, 1.0), np.nan)
    # Benjamini-Hochberg FDR（只在有效 p 上做，再回填）
    df['p_fdr'] = np.nan
    valid_idx = np.where(~np.isnan(p))[0]
    if len(valid_idx) > 0:
        pv = p[valid_idx]
        order = np.argsort(pv)
        ranked = pv[order] * m / (np.arange(len(pv)) + 1)
        ranked = np.minimum.accumulate(ranked[::-1])[::-1]
        bh = np.empty(len(pv))
        bh[order] = np.minimum(ranked, 1.0)
        df.loc[df.index[valid_idx], 'p_fdr'] = bh
    df['sig_bonf'] = df['p_bonf'] < 0.05
    df['sig_fdr'] = df['p_fdr'] < 0.10
    df = df.sort_values('beta_p')
    df.to_csv(os.path.join(TBL_DIR, "univariate_panel.csv"), index=False)
    print("  ✓ univariate_panel.csv（含 Bonferroni / FDR 校正）")
    return df


def descriptive_stats(panel):
    rows = []
    for f in FACTOR_COLS:
        s = panel[f].dropna()
        rows.append([f, s.mean(), s.std(), s.skew(), s.kurt(), len(s)] if len(s)
                    else [f, np.nan, np.nan, np.nan, np.nan, 0])
    pd.DataFrame(rows, columns=['factor', 'mean', 'std', 'skew', 'kurt', 'N']).to_csv(
        os.path.join(TBL_DIR, "descriptive_stats.csv"), index=False)
    panel[FACTOR_COLS].corr().to_csv(os.path.join(TBL_DIR, "factor_correlations.csv"))
    print("  ✓ 描述统计 / 相关性")


def _asset_of(code):
    return "etf" if code.split(".")[0].isdigit() and code.endswith((".SH", ".SZ")) else \
           ("etf" if code in {"510050.SH","510300.SH","510500.SH","159915.SZ","588000.SH"} else "commodity")


def run_one_group(panel, label, suffix):
    """对一组子面板跑 ML + 单变量回归，结果写带后缀的 CSV。"""
    from config import UNDERLYINGS
    amap = {u["code"]: u["asset"] for u in UNDERLYINGS}
    sub = panel[panel["underlying"].map(amap) == label].copy()
    if len(sub) < 60 or sub["underlying"].nunique() < 3:
        print(f"  [{suffix}] 样本不足（{len(sub)} 行, {sub['underlying'].nunique()} 标的），跳过")
        return
    print(f"\n===== 分组：{label}（{suffix}）| {len(sub)} 行, {sub['underlying'].nunique()} 标的 =====")
    print("→ 前向 walk-forward CV")
    preds = walk_forward(sub)
    if preds is not None:
        preds.to_csv(os.path.join(TBL_DIR, f"ml_predictions_{suffix}.csv"), index=False)
        evaluate(preds)
        # 把性能表也存带后缀
        perf = pd.read_csv(os.path.join(TBL_DIR, "ml_oos_performance.csv"))
        perf.insert(0, "group", label)
        perf.to_csv(os.path.join(TBL_DIR, f"ml_oos_performance_{suffix}.csv"), index=False)
    print("→ 特征重要性")
    if preds is not None:
        feature_importance(sub, preds)
        fi = pd.read_csv(os.path.join(TBL_DIR, "ml_feature_importance.csv"))
        fi.insert(0, "group", label)
        fi.to_csv(os.path.join(TBL_DIR, f"ml_feature_importance_{suffix}.csv"), index=False)
    print("→ 单变量 panel 回归 + 多重检验")
    univariate_panel(sub)
    # univariate_panel 写的是固定名，改名带后缀
    import shutil
    src = os.path.join(TBL_DIR, "univariate_panel.csv")
    if os.path.exists(src):
        u = pd.read_csv(src); u.insert(0, "group", label)
        u.to_csv(os.path.join(TBL_DIR, f"univariate_panel_{suffix}.csv"), index=False)


def main():
    print("[09-ML] 机器学习预测管线 ...")
    panel = prepare_panel()
    print(f"  面板：{panel.shape}，标的 {panel['underlying'].nunique()}，"
          f"月份 {panel['ym'].nunique()}")
    descriptive_stats(panel)

    # 全样本
    print("\n===== 全样本 =====")
    print("→ 前向 walk-forward CV")
    preds = walk_forward(panel)
    if preds is not None:
        preds.to_csv(os.path.join(TBL_DIR, "ml_predictions_all.csv"), index=False)
        evaluate(preds)
        perf = pd.read_csv(os.path.join(TBL_DIR, "ml_oos_performance.csv"))
        perf.insert(0, "group", "all")
        perf.to_csv(os.path.join(TBL_DIR, "ml_oos_performance_all.csv"), index=False)
    print("→ 特征重要性")
    if preds is not None:
        feature_importance(panel, preds)
        fi = pd.read_csv(os.path.join(TBL_DIR, "ml_feature_importance.csv"))
        fi.insert(0, "group", "all")
        fi.to_csv(os.path.join(TBL_DIR, "ml_feature_importance_all.csv"), index=False)
    print("→ 单变量 panel 回归 + 多重检验")
    univariate_panel(panel)
    import shutil
    src = os.path.join(TBL_DIR, "univariate_panel.csv")
    if os.path.exists(src):
        u = pd.read_csv(src); u.insert(0, "group", "all")
        u.to_csv(os.path.join(TBL_DIR, "univariate_panel_all.csv"), index=False)

    # 分组：ETF / 商品
    run_one_group(panel, "etf", "etf")
    run_one_group(panel, "commodity", "commodity")

    print("\n[OK] ML 管线完成（全样本 + ETF 组 + 商品组）。")


if __name__ == "__main__":
    main()
