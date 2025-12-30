# ==============================================
#   今彩 539 自動化預測系統（多模型 + 特徵工程）
#   Walk-forward 真實策略回測版（每 K 期重訓一次）
# ==============================================
#
# 訓練模型：
#   1. XGBoost
#   2. LightGBM
#   3. CatBoost
#   4. CRF（Conditional Random Field）
#   5. Logistic Regression
#   6. RandomForest
#
# 核心功能：
#   1) 特徵只建一次（包含數值統計、群組特徵、熱度、miss、pair 共現等）
#   2) 樹模型＋Logistic＋RF 使用 MultiOutputClassifier，一次學 39 標籤
#   3) CRF 使用「每期一條序列（1~39 號）」進行序列預測
#   4) FAST_MODE + 滑動訓練窗（可調），提供加速模式
#   5) 完整回測 & 預測流程：命中、標準差、彩金盈虧、錯題本、票選整合
#   6) 回測與預測 TOP_N 可分開設定
#   7) 「跨期（1~MAX_LAG）版路拖牌分析」，輸出 4 份 txt 筆記本
#
# 重要修正（本版本）：
#   A) 回測改成 Walk-forward（每 K 期重訓一次，貼近真實買法）
#   B) 可固定 seed 完全重現（建議先開）
#   C) 對獎改用「期數」優先，避免日期推算誤差
#   D) 錯題本懲罰只讀「真實預測對獎」結果，不再用回測內生污染
#
# 顯示/輸出修正（本次修改）：
#   E) tqdm 進度條不中斷：不再每 6 期 print 換行
#   F) Logistic 的 ConvergenceWarning / stderr 雜訊不再洗版終端機（並擴充到所有模型）
#   G) 回測照跑 backtest_N，但終端機只顯示最近 DISPLAY_LOGS_N 期命中紀錄（其餘輸出到檔案）
#
# ✅ 本次關鍵修正（你指出的問題）：
#   H)「版路拖牌」改成：以最新開獎為起點，推測下一期（或 lag 期後）可能拖出的號碼
#      （不再是拖出“目前最新開獎號碼”）
#
# By Party87 + ChatGPT
# 版本日期：2025/12/30

import warnings
import pandas as pd
import numpy as np
import requests
import os
import time
from io import StringIO
from tqdm import tqdm
from colorama import Fore, Style
import multiprocessing
from datetime import timedelta, datetime
from math import comb
import csv
import pickle
from collections import Counter, defaultdict
import random
import sys
import contextlib

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
import sklearn_crfsuite
from sklearn.multioutput import MultiOutputClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# ==============================
# ========== 輸出/雜訊控制 ==========
# ==============================

# ✅ 讓「子行程/worker」也直接忽略 warnings（最穩）
os.environ.setdefault("PYTHONWARNINGS", "ignore")

# 關閉 Logistic 的收斂警告（Python warnings 層）
warnings.simplefilter("ignore")  # 全域更乾淨：所有 warning 都不顯示
warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", message=".*lbfgs failed to converge.*", module=r"sklearn\.linear_model\._logistic")

# 額外關掉常見第三方模型的 warning
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

@contextlib.contextmanager
def silence_warnings_and_stderr():
    """
    ✅ 終極消音（只在需要時用）
    - 直接把 warnings.showwarning 改成 no-op（避免任何 warning 印出）
    - 同時 redirect stderr 到 devnull（避免有些庫直接寫 stderr）
    """
    old_showwarning = warnings.showwarning
    warnings.showwarning = lambda *args, **kwargs: None  # 完全不顯示 warnings

    with open(os.devnull, "w") as devnull:
        with contextlib.redirect_stderr(devnull):
            try:
                yield
            finally:
                warnings.showwarning = old_showwarning

@contextlib.contextmanager
def suppress_stdout():
    """
    ✅ 關掉 stdout（例如某些模型/套件會直接 print）
    """
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout

@contextlib.contextmanager
def suppress_all_output(enable=True):
    """
    enable=True 時：
    - stdout / stderr 全部導向 os.devnull
    - warnings.showwarning 改成 no-op（避免任何 warning 印到終端）
    """
    if not enable:
        yield
        return

    old_showwarning = warnings.showwarning
    warnings.showwarning = lambda *args, **kwargs: None

    with open(os.devnull, "w") as devnull:
        with contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
            try:
                yield
            finally:
                warnings.showwarning = old_showwarning

# -------- 多執行緒環境變數（搭配 hybrid CPU，建議 12~16 threads）--------
os.environ["OMP_NUM_THREADS"] = "16"
os.environ["MKL_NUM_THREADS"] = "16"
os.environ["OPENBLAS_NUM_THREADS"] = "16"
os.environ["NUMEXPR_NUM_THREADS"] = "16"
os.environ["VECLIB_MAXIMUM_THREADS"] = "16"
os.environ["JOBLIB_MULTIPROCESSING"] = "1"

# 統一控管各模型 n_jobs / thread_count
N_JOBS = 16  # 20 核心（6P+8E），16 threads 較適合 hybrid 結構

# ---------------- 可調參數 ----------------
FAST_MODE = True
TRAIN_WINDOW = 500 if FAST_MODE else None
DOWNLOAD_RETRY = 3

# 錯題本懲罰強度（建議先開小一點）
MISTAKE_ALPHA = 0.05

SAVE_PROBA_CSV = True

# ✅ 建議先開 True 做驗證，確認不是隨機 seed 造成你「回測漂亮、預測差」的錯覺
USE_DETERMINISTIC = True
GLOBAL_SEED = 42

# ✅ Walk-forward 回測：每 K 期才重訓一次
WALK_FORWARD_RETRAIN_INTERVAL = 20  # 建議 5 / 10 / 20 自己測

# 版路拖牌相關參數
DRAG_ENABLE = True
MAX_LAG = 10
MIN_CHAIN_RUN = 2
DRAG_MAX_ROWS = 365

# ✅ 拖牌預測目標：一次預測 1~10 期
DRAG_PREDICT_LAGS = list(range(1, 11))  # [1,2,3,4,5,6,7,8,9,10]


# ========== CatBoost 訓練資料目錄 ==========
def ensure_catboost_train_dir():
    abs_path = os.path.abspath("catboost_tmp")
    os.makedirs(abs_path, exist_ok=True)
    return abs_path

catboost_train_dir = ensure_catboost_train_dir()
os.environ["CATBOOST_TRAIN_DIR"] = catboost_train_dir

def set_global_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

if USE_DETERMINISTIC:
    set_global_seed(GLOBAL_SEED)


# ==============================
# ========== 第一步：下載或快取官方今彩539資料 + 數據清洗 ==========
# ==============================
def load_tw539(force_download=False):
    DATA_CACHE = "tw539_latest.csv"
    DATA_URL = "https://biga.com.tw/HISTORYDATA/tw539.csv"

    if (not force_download) and os.path.exists(DATA_CACHE):
        df = pd.read_csv(DATA_CACHE)
        if "日期" in df.columns:
            df["日期"] = pd.to_datetime(df["日期"], errors="coerce")
        return df

    last_err = None
    for _ in range(DOWNLOAD_RETRY):
        try:
            r = requests.get(DATA_URL, timeout=15)
            r.raise_for_status()
            csv_str = r.content.decode("utf-8-sig")
            df = pd.read_csv(StringIO(csv_str), header=None)
            df.columns = [
                "日期", "期數",
                "落序號碼1", "落序號碼2", "落序號碼3", "落序號碼4", "落序號碼5",
                "",
                "正序號碼1", "正序號碼2", "正序號碼3", "正序號碼4", "正序號碼5"
            ]
            df["日期"] = pd.to_datetime(df["日期"], errors="coerce")

            num_cols = [
                "落序號碼1", "落序號碼2", "落序號碼3", "落序號碼4", "落序號碼5",
                "正序號碼1", "正序號碼2", "正序號碼3", "正序號碼4", "正序號碼5"
            ]
            for c in num_cols:
                df[c] = pd.to_numeric(df[c], errors="coerce")

            df["期數"] = pd.to_numeric(df["期數"], errors="coerce")

            df = df.dropna(subset=["日期", "期數"] + num_cols).reset_index(drop=True)
            df["期數"] = df["期數"].astype(int)
            df[num_cols] = df[num_cols].astype(int)

            if df["日期"].iloc[0] > df["日期"].iloc[-1]:
                df = df.iloc[::-1].reset_index(drop=True)

            df = df.drop(columns=[""])
            df.to_csv(DATA_CACHE, index=False)
            return df
        except Exception as e:
            last_err = e
            time.sleep(1.0)

    raise RuntimeError(f"下載官方資料失敗：{last_err}")


# ==============================
# ========== 對獎：優先用期數，避免日期推算對不上 ==========
# ==============================
def try_save_latest_prediction_result(df, nums):
    pred_file = "latest_predict.csv"
    result_file = "latest_predict_result.csv"
    if not os.path.exists(pred_file):
        return

    pred_df = pd.read_csv(pred_file, dtype=str)
    df2 = df.copy()
    df2["日期"] = pd.to_datetime(df2["日期"], errors="coerce")

    if os.path.exists(result_file):
        result_df = pd.read_csv(result_file, dtype=str)
        recorded_keys = set(result_df.apply(lambda r: f"{r.get('預測期數','')}_{r.get('模型','')}", axis=1))
    else:
        result_df = pd.DataFrame()
        recorded_keys = set()

    new_records = []
    for _, row in pred_df.iterrows():
        model = str(row.get("模型", "")).strip()
        pred_issue = str(row.get("預測期數", "")).strip()
        pred_date = str(row.get("預測日期", "")).strip()
        pred_nums_str = str(row.get("預測號碼", "")).strip()

        key = f"{pred_issue}_{model}"
        if key in recorded_keys:
            continue

        if not pred_nums_str:
            continue

        pred_numbers = [int(n) for n in pred_nums_str.split(",") if str(n).strip().isdigit()]

        real_row = None
        if pred_issue.isdigit():
            real_row = df2[df2["期數"] == int(pred_issue)]
        if (real_row is None) or real_row.empty:
            if pred_date:
                real_row = df2[df2["日期"].dt.strftime("%Y-%m-%d") == pred_date]

        if real_row is None or real_row.empty:
            continue

        real_numbers = [int(n) for n in real_row.iloc[0][nums].values]
        real_date = real_row.iloc[0]["日期"].strftime("%Y-%m-%d")
        real_issue = int(real_row.iloc[0]["期數"])

        hit_set = sorted(set(pred_numbers) & set(real_numbers))
        new_records.append({
            "預測日期": pred_date if pred_date else "",
            "預測期數": str(pred_issue) if pred_issue else "",
            "實際開獎日期": real_date,
            "實際開獎期數": str(real_issue),
            "模型": model,
            "預測號碼": ",".join(f"{n:02d}" for n in sorted(pred_numbers)),
            "開獎號碼": ",".join(f"{n:02d}" for n in sorted(real_numbers)),
            "命中數": str(len(hit_set)),
            "命中號碼": ",".join(f"{n:02d}" for n in hit_set)
        })

    if new_records:
        df_new = pd.DataFrame(new_records)
        if os.path.exists(result_file):
            old = pd.read_csv(result_file, dtype=str)
            result_df = pd.concat([old, df_new], ignore_index=True)
        else:
            result_df = df_new

        result_df.to_csv(result_file, index=False, encoding="utf-8-sig")
        print(f"\n自動完成對獎結果存檔：{result_file}（本次新增 {len(df_new)} 筆）")


# ==============================
# ========== 第二步：特徵工程（一次建滿，後續用切片） ==========
# ==============================
FEATURE_CACHE = {}

def feature_engineering(df, nums, features):
    cache_key = (df.index[0] if len(df) else -1, df.index[-1] if len(df) else -1)
    if cache_key in FEATURE_CACHE:
        return FEATURE_CACHE[cache_key].copy()

    X = df[features].copy()

    if "日期" in df.columns:
        X["weekday"] = df["日期"].dt.weekday.astype(np.int8)
        X["month"] = df["日期"].dt.month.astype(np.int8)
        X["is_weekend"] = (X["weekday"] >= 5).astype(np.int8)

    X["max_num"] = df[nums].max(axis=1)
    X["min_num"] = df[nums].min(axis=1)
    X["sum_num"] = df[nums].sum(axis=1)
    X["mean_num"] = df[nums].mean(axis=1)
    X["std_num"] = df[nums].std(axis=1)
    X["range_num"] = X["max_num"] - X["min_num"]
    X["odd_count"] = df[nums].apply(lambda r: sum(n % 2 for n in r), axis=1)
    X["even_count"] = 5 - X["odd_count"]
    X["odd_even_ratio"] = X["odd_count"] / (X["even_count"] + 1e-6)
    X["num_gap"] = df[nums].apply(lambda r: np.mean(np.diff(sorted(r))), axis=1)

    def count_serials(arr):
        s = sorted(arr)
        return sum((s[i+1] - s[i]) == 1 for i in range(4))
    X["serial_count"] = df[nums].apply(count_serials, axis=1)

    def repeat_count(cur, prev):
        return len(set(cur) & set(prev))
    X["repeat_prev"] = [0] + [repeat_count(df.loc[i, nums], df.loc[i-1, nums]) for i in range(1, len(df))]

    X["big_small_score"] = df[nums].sum(axis=1) / 195
    X["is_big"] = (X["big_small_score"] > 0.5).astype(np.int8)

    for i in range(1, 6):
        X[f"num{i}_尾數"] = (df[f"正序號碼{i}"] % 10).astype(np.int8)
        X[f"num{i}_合數"] = ((df[f"正序號碼{i}"] // 10 + df[f"正序號碼{i}"] % 10) % 10).astype(np.int8)

    hot_frames = []
    for n in range(1, 40):
        num_mask = (df[nums] == n).sum(axis=1).astype(np.int16)
        col_all = pd.DataFrame({f"hot_{n:02d}_all": num_mask.cumsum().astype(np.int32)})
        col_windows = pd.DataFrame({
            f"hot_{n:02d}_{w}": num_mask.rolling(w, min_periods=1).sum().astype(np.float32)
            for w in [5, 10, 20, 30]
        })
        hot_frames.append(pd.concat([col_all, col_windows], axis=1))
    X = pd.concat([X.reset_index(drop=True), pd.concat(hot_frames, axis=1).reset_index(drop=True)], axis=1)

    miss_features = pd.DataFrame(index=df.index)
    last_pos = {n: -1 for n in range(1, 40)}
    miss_counts = {n: [] for n in range(1, 40)}
    for i in range(len(df)):
        cur_set = set(df.loc[i, nums])
        for n in range(1, 40):
            if n in cur_set:
                miss_counts[n].append(0)
                last_pos[n] = i
            else:
                miss_counts[n].append(i - last_pos[n] if last_pos[n] != -1 else i+1)
    for n in range(1, 40):
        miss_features[f"miss_{n:02d}"] = pd.Series(miss_counts[n], dtype=np.int16)
    X = pd.concat([X.reset_index(drop=True), miss_features.reset_index(drop=True)], axis=1)

    pair_mean_list = []
    pair_max_list = []
    pair_counts = defaultdict(int)

    for i in range(len(df)):
        row_nums = sorted(int(df.loc[i, col]) for col in nums)
        pair_scores = []
        for a in range(4):
            for b in range(a + 1, 5):
                p = (row_nums[a], row_nums[b])
                pair_scores.append(pair_counts[p])
        if pair_scores:
            pair_mean_list.append(float(np.mean(pair_scores)))
            pair_max_list.append(float(np.max(pair_scores)))
        else:
            pair_mean_list.append(0.0)
            pair_max_list.append(0.0)
        for a in range(4):
            for b in range(a + 1, 5):
                p = (row_nums[a], row_nums[b])
                pair_counts[p] += 1

    X["pair_cofreq_mean"] = pd.Series(pair_mean_list, index=df.index).astype(np.float32)
    X["pair_cofreq_max"] = pd.Series(pair_max_list, index=df.index).astype(np.float32)

    def apply_group_feats(df, nums, name, mapping, X):
        dummy_frames = []
        for i in range(1, 6):
            cat_col = df[f"正序號碼{i}"].map(mapping).astype("category")
            dummies = pd.get_dummies(cat_col, prefix=f"{name}_num{i}")
            dummy_frames.append(dummies)
        if dummy_frames:
            X = pd.concat([X.reset_index(drop=True), pd.concat(dummy_frames, axis=1).reset_index(drop=True)], axis=1)
        all_groups = sorted(set(mapping.values()))
        for k in all_groups:
            X[f"{name}_{k}_count"] = df[nums].apply(lambda row: sum(mapping.get(n, "") == k for n in row), axis=1)
        return X

    mean_group_map = {**{i:'A' for i in range(1,5)}, **{i:'B' for i in range(5,9)}, **{i:'C' for i in range(9,13)},
                      **{i:'D' for i in range(13,17)}, **{i:'E' for i in range(17,21)}, **{i:'F' for i in range(21,25)},
                      **{i:'G' for i in range(25,29)}, **{i:'H' for i in range(29,33)}, **{i:'I' for i in range(33,37)},
                      **{i:'J' for i in range(37,40)}}
    bagua_map = {1:'乾',16:'乾',17:'乾',32:'乾',8:'坤',9:'坤',24:'坤',25:'坤',6:'坎',11:'坎',22:'坎',27:'坎',
                 3:'離',14:'離',19:'離',30:'離',38:'離',5:'震',12:'震',21:'震',28:'震',36:'震',4:'巽',13:'巽',20:'巽',
                 29:'巽',37:'巽',2:'艮',15:'艮',18:'艮',31:'艮',39:'艮',7:'兌',10:'兌',23:'兌',26:'兌',34:'兌'}
    shengxiao_map = {11:'羊',23:'羊',35:'羊',10:'猴',22:'猴',34:'猴',9:'雞',21:'雞',33:'雞',8:'狗',20:'狗',32:'狗',
                     7:'豬',19:'豬',31:'豬',6:'鼠',18:'鼠',30:'鼠',5:'牛',17:'牛',29:'牛',4:'虎',16:'虎',28:'虎',
                     3:'兔',15:'兔',27:'兔',39:'兔',2:'龍',14:'龍',26:'龍',38:'龍',1:'蛇',13:'蛇',25:'蛇',37:'蛇',
                     12:'馬',24:'馬',36:'馬'}
    wuxing_map = {13:'金',14:'金',21:'金',22:'金',29:'金',30:'金',1:'木',2:'木',9:'木',10:'木',17:'木',18:'木',31:'木',
                  32:'木',39:'木',3:'水',4:'水',11:'水',12:'水',25:'水',26:'水',33:'水',34:'水',7:'火',8:'火',15:'火',
                  16:'火',23:'火',24:'火',37:'火',38:'火',5:'土',6:'土',19:'土',20:'土',27:'土',28:'土',35:'土',36:'土'}
    liuchong_map = {1:'子午',7:'子午',13:'子午',19:'子午',25:'子午',31:'子午',37:'子午',2:'丑未',8:'丑未',14:'丑未',
                    20:'丑未',26:'丑未',32:'丑未',38:'丑未',3:'寅申',9:'寅申',15:'寅申',21:'寅申',27:'寅申',33:'寅申',
                    39:'寅申',4:'卯酉',10:'卯酉',16:'卯酉',22:'卯酉',28:'卯酉',34:'卯酉',5:'辰戌',11:'辰戌',17:'辰戌',
                    23:'辰戌',29:'辰戌',35:'辰戌',6:'巳亥',12:'巳亥',18:'巳亥',24:'巳亥',30:'巳亥',36:'巳亥'}
    qizheng_map = {1:'日',8:'日',15:'日',22:'日',29:'日',36:'日',2:'月',9:'月',16:'月',23:'月',30:'月',37:'月',
                   3:'火',10:'火',17:'火',24:'火',31:'火',38:'火',4:'水',11:'水',18:'水',25:'水',32:'水',39:'水',
                   5:'木',12:'木',19:'木',26:'木',33:'木',6:'金',13:'金',20:'金',27:'金',34:'金',7:'土',14:'土',
                   21:'土',28:'土',35:'土'}

    X = apply_group_feats(df, nums, "bagua", bagua_map, X)
    X = apply_group_feats(df, nums, "shengxiao", shengxiao_map, X)
    X = apply_group_feats(df, nums, "mean_group", mean_group_map, X)
    X = apply_group_feats(df, nums, "wuxing", wuxing_map, X)
    X = apply_group_feats(df, nums, "liuchong", liuchong_map, X)
    X = apply_group_feats(df, nums, "qizheng", qizheng_map, X)

    for col in X.select_dtypes(include=["float64"]).columns:
        X[col] = X[col].astype(np.float32)
    for col in X.select_dtypes(include=(["int64"])).columns:
        X[col] = X[col].astype(np.int32)

    FEATURE_CACHE[cache_key] = X.copy()
    return X

def build_full_features_once(df):
    nums = ["正序號碼1", "正序號碼2", "正序號碼3", "正序號碼4", "正序號碼5"]
    features = nums + ["落序號碼1", "落序號碼2", "落序號碼3", "落序號碼4", "落序號碼5"]
    X_full = feature_engineering(df, nums, features)
    base_columns = list(X_full.columns)
    return X_full, base_columns, nums, features

def fix_columns(df, columns):
    return df.reindex(columns=columns, fill_value=0)


# ==============================
# ========== 彩金盈虧計算 ==========
# ==============================
def calc_prize(hits_list, top_n):
    JACKPOTS = {
        "2星": {"cost": 10, "prize": 72},
        "3星": {"cost": 1, "prize": 883},
        "4星": {"cost": 1, "prize": 15300},
    }
    total = {k: {'win': 0, 'bets': 0, 'cost': 0, 'prize': 0} for k in JACKPOTS}
    for hit in hits_list:
        comb2 = comb(hit, 2) if hit >= 2 else 0
        total['2星']['win'] += comb2
        total['2星']['bets'] += comb(top_n, 2)

        comb3 = comb(hit, 3) if hit >= 3 else 0
        total['3星']['win'] += comb3
        total['3星']['bets'] += comb(top_n, 3)

        comb4 = comb(hit, 4) if hit >= 4 else 0
        total['4星']['win'] += comb4

    for k in JACKPOTS:
        total[k]['cost'] = total[k]['bets'] * JACKPOTS[k]['cost']
        total[k]['prize'] = total[k]['win'] * JACKPOTS[k]['cost'] * JACKPOTS[k]['prize']
        total[k]['profit'] = total[k]['prize'] - total[k]['cost']
    return total


# ==============================
# ========== 錯題本（只讀真實預測對獎，不用回測污染） ==========
# ==============================
def load_penalty_from_real_results(result_file="latest_predict_result.csv"):
    penalty = np.zeros(39, dtype=np.float32)
    if not os.path.exists(result_file):
        return penalty

    try:
        df = pd.read_csv(result_file, dtype=str)
    except Exception:
        return penalty

    for _, r in df.iterrows():
        pred = str(r.get("預測號碼", "")).strip()
        ans = str(r.get("開獎號碼", "")).strip()
        hit_s = str(r.get("命中數", "")).strip()

        if not pred or not ans:
            continue

        try:
            pred_nums = [int(x) for x in pred.split(",") if x.strip().isdigit()]
            ans_nums = [int(x) for x in ans.split(",") if x.strip().isdigit()]
            hit = int(hit_s) if hit_s.isdigit() else len(set(pred_nums) & set(ans_nums))
        except Exception:
            continue

        if hit >= 2:
            continue

        w = 3.0 if hit == 0 else 2.0
        miss_nums = sorted(set(pred_nums) - set(ans_nums))
        for m in miss_nums:
            if 1 <= m <= 39:
                penalty[m-1] += w

    return penalty


# ==============================
# ========== 多模型（XGB / LGBM / CatBoost / Logistic / RF） + CRF ==========
# ==============================
def _seed(seed_offset=0):
    if USE_DETERMINISTIC:
        return int((GLOBAL_SEED + seed_offset) % (2**32 - 1))
    raw = int(time.time_ns()) + random.randint(0, 1_000_000)
    return int(raw % (2**32 - 1))

def make_base_estimator(model_name, seed_offset=0):
    n_cpu = max(1, multiprocessing.cpu_count() - 1)

    if model_name == "XGBoost":
        params_fast = dict(n_estimators=60, max_depth=3, learning_rate=0.12, subsample=0.8, colsample_bytree=0.8)
        params_full = dict(n_estimators=160, max_depth=4, learning_rate=0.10, subsample=0.9, colsample_bytree=0.9)
        params = params_fast if FAST_MODE else params_full
        return XGBClassifier(
            tree_method="hist",
            use_label_encoder=False,
            eval_metric='logloss',
            n_jobs=n_cpu,
            verbosity=0,
            random_state=_seed(seed_offset),
            **params
        )

    if model_name == "LightGBM":
        params_fast = dict(n_estimators=120, num_leaves=31, learning_rate=0.08, subsample=0.9, colsample_bytree=0.9)
        params_full = dict(n_estimators=300, num_leaves=63, learning_rate=0.06, subsample=0.9, colsample_bytree=0.9)
        params = params_fast if FAST_MODE else params_full
        return LGBMClassifier(**params, n_jobs=n_cpu, random_state=_seed(seed_offset), verbose=-1)

    if model_name == "CatBoost":
        params_fast = dict(iterations=120, depth=4, learning_rate=0.12)
        params_full = dict(iterations=500, depth=5, learning_rate=0.10)
        params = params_fast if FAST_MODE else params_full
        return CatBoostClassifier(
            task_type="CPU",
            thread_count=1,
            verbose=0,
            train_dir=catboost_train_dir,
            random_seed=_seed(seed_offset),
            **params
        )

    if model_name == "Logistic":
        # ✅ 根治：先做標準化再丟 Logistic（大幅降低 lbfgs 不收斂）
        # ✅ solver 改 saga 更穩（MultiOutput 的 39 個二元分類都適用）
        max_iter = 3000 if FAST_MODE else 8000
        tol = 1e-3 if FAST_MODE else 5e-4

        base_lr = LogisticRegression(
            C=1.0,
            max_iter=max_iter,
            tol=tol,
            solver="saga",
            penalty="l2",
            n_jobs=1  # ✅ 避免子 worker 亂噴；外層 MultiOutput 再控制
        )

        return Pipeline([
            ("scaler", StandardScaler()),
            ("clf", base_lr)
        ])

    if model_name == "RandomForest":
        params_fast = dict(n_estimators=80, max_depth=7)
        params_full = dict(n_estimators=200, max_depth=None)
        params = params_fast if FAST_MODE else params_full
        return RandomForestClassifier(
            **params,
            n_jobs=n_cpu,
            random_state=_seed(seed_offset)
        )

    raise ValueError("Unknown model")

def make_multi_model(model_name, seed_offset=0):
    n_cpu = max(1, multiprocessing.cpu_count() - 1)
    base = make_base_estimator(model_name, seed_offset=seed_offset)

    # ✅ 關鍵：Logistic 會在 joblib worker 內噴 ConvergenceWarning
    # 把 MultiOutputClassifier 改成單線程 fit，就能讓外層 suppression 100% 生效
    if model_name == "Logistic":
        return MultiOutputClassifier(base, n_jobs=1)

    return MultiOutputClassifier(base, n_jobs=n_cpu)

def proba_from_multioutput(mo_clf, X_pred_row):
    probas = []
    for est in mo_clf.estimators_:
        p = est.predict_proba(X_pred_row)
        if p.shape[1] == 2:
            probas.append(float(p[0, 1]))
        else:
            probas.append(float(p[0, 0]))
    return np.array(probas, dtype=np.float32)


# ==============================
# ========== CRF ==========
# ==============================
def crf_token_features_from_row(row, n):
    fmt = f"{n:02d}"
    feat = {
        "n": n,
        "n_mod10": n % 10,
        "n_bin": n // 10,
        "hot_all": float(row.get(f"hot_{fmt}_all", 0.0)),
        "hot_5": float(row.get(f"hot_{fmt}_5", 0.0)),
        "hot_10": float(row.get(f"hot_{fmt}_10", 0.0)),
        "hot_20": float(row.get(f"hot_{fmt}_20", 0.0)),
        "hot_30": float(row.get(f"hot_{fmt}_30", 0.0)),
        "miss": float(row.get(f"miss_{fmt}", 0.0)),
        "odd_cnt": int(row.get("odd_count", 0)),
        "even_cnt": int(row.get("even_count", 0)),
        "serial_cnt": int(row.get("serial_count", 0)),
        "repeat_prev": int(row.get("repeat_prev", 0)),
        "big_small": float(row.get("big_small_score", 0.0)),
    }
    return feat

def build_crf_seq_features(X_bt, idx):
    row = X_bt.iloc[idx]
    return [crf_token_features_from_row(row, n) for n in range(1, 40)]

def build_crf_labels(y_all, idx):
    row = y_all.iloc[idx]
    return ['1' if int(row[f"y_{n:02d}"]) == 1 else '0' for n in range(1, 40)]

def fit_crf(X_bt, y_all, start_idx, end_idx, seed_offset=0):
    X_seqs = [build_crf_seq_features(X_bt, t) for t in range(start_idx, end_idx)]
    y_seqs = [build_crf_labels(y_all, t) for t in range(start_idx, end_idx)]
    max_iter = 60 if FAST_MODE else 150
    crf = sklearn_crfsuite.CRF(
        algorithm='lbfgs',
        c1=0.1, c2=0.1,
        max_iterations=max_iter,
        all_possible_transitions=True
    )
    # ✅ CRF 有時也會有輸出：包起來
    with suppress_all_output(True):
        crf.fit(X_seqs, y_seqs)
    return crf

def crf_predict_proba(crf, X_bt, idx):
    feats = build_crf_seq_features(X_bt, idx)
    marginals = crf.predict_marginals_single(feats)
    probs = [float(m.get('1', 0.0)) for m in marginals]
    return np.array(probs, dtype=np.float32)


# ==============================
# ========== 版路拖牌（✅改成：預測下一期/lag期後） ==========
# ==============================
def _build_rows_for_mode(df_bt_use, nums, mode):
    if mode == "size":
        seqs = []
        for _, row in df_bt_use.iterrows():
            arr = sorted(int(row[c]) for c in nums)
            seqs.append(arr)
        return seqs
    elif mode == "order":
        cols = [f"落序號碼{i}" for i in range(1, 6)]
        seqs = df_bt_use[cols].astype(int).values.tolist()
        return seqs
    else:
        raise ValueError("mode must be 'size' or 'order'")

def _analyze_pattern_unpos_next(seqs, dates, A, B, lag, last_base_idx):
    """
    ✅ 新版：以最新開獎( last_base_idx )含 A 做起點，預測 lag 期後可能出 B
    作法：
      - 收集歷史上所有 base_idx（含 A）且 base_idx+lag < N 的事件
      - success = B 在 base_idx+lag
      - chain_run = 從「最接近現在的歷史事件」往前連續成功次數（不包含 last_base_idx 這個 pending）
      - 要求 chain_run >= MIN_CHAIN_RUN
    """
    N = len(seqs)
    if N <= lag or last_base_idx < 0 or last_base_idx >= N:
        return None

    # 起點必須包含 A（最新一期）
    if A not in set(seqs[last_base_idx]):
        return None

    events = []
    base_indices = []
    for base_idx in range(0, N - lag):
        if A not in set(seqs[base_idx]):
            continue
        drag_idx = base_idx + lag
        is_success = (B in set(seqs[drag_idx]))
        events.append((base_idx, drag_idx, is_success))
        base_indices.append(base_idx)

    if not events:
        return None

    success_count = sum(1 for e in events if e[2])
    fail_count = len(events) - success_count
    if success_count < MIN_CHAIN_RUN:
        return None

    # 找最靠近現在但仍可驗證的 base（必須 < last_base_idx 且 base+lag < N）
    valid_events = [e for e in events if e[0] < last_base_idx]
    if not valid_events:
        return None

    valid_events.sort(key=lambda x: x[0], reverse=True)

    chain_run = 0
    tail_events = []
    for base_idx, drag_idx, is_success in valid_events:
        tail_events.append((base_idx, drag_idx, is_success))
        if is_success:
            chain_run += 1
        else:
            break

    if chain_run < MIN_CHAIN_RUN:
        return None

    total_events = success_count + fail_count
    success_rate = success_count / total_events if total_events > 0 else 0.0

    # 最近一次成功事件的拖出日期（方便展示）
    last_success_drag_date = None
    for base_idx, drag_idx, is_success in valid_events:
        if is_success:
            last_success_drag_date = dates[drag_idx]
            break

    return {
        "A": A,
        "B": B,
        "lag": lag,
        "chain_run": chain_run,
        "last_base_date": dates[last_base_idx],
        "last_success_drag_date": last_success_drag_date,
        "success_count": success_count,
        "fail_count": fail_count,
        "success_rate": success_rate,
        "tail_events": tail_events[:50],  # 避免太長
        "predict_target": "NEXT"
    }

def _analyze_pattern_pos_next(seqs, dates, A, B, lag, pos_a, pos_b, last_base_idx):
    N = len(seqs)
    if N <= lag or last_base_idx < 0 or last_base_idx >= N:
        return None

    if seqs[last_base_idx][pos_a] != A:
        return None

    events = []
    for base_idx in range(0, N - lag):
        if seqs[base_idx][pos_a] != A:
            continue
        drag_idx = base_idx + lag
        is_success = (seqs[drag_idx][pos_b] == B)
        events.append((base_idx, drag_idx, is_success))

    if not events:
        return None

    success_count = sum(1 for e in events if e[2])
    fail_count = len(events) - success_count
    if success_count < MIN_CHAIN_RUN:
        return None

    valid_events = [e for e in events if e[0] < last_base_idx]
    if not valid_events:
        return None

    valid_events.sort(key=lambda x: x[0], reverse=True)

    chain_run = 0
    tail_events = []
    for base_idx, drag_idx, is_success in valid_events:
        tail_events.append((base_idx, drag_idx, is_success))
        if is_success:
            chain_run += 1
        else:
            break

    if chain_run < MIN_CHAIN_RUN:
        return None

    total_events = success_count + fail_count
    success_rate = success_count / total_events if total_events > 0 else 0.0

    last_success_drag_date = None
    for base_idx, drag_idx, is_success in valid_events:
        if is_success:
            last_success_drag_date = dates[drag_idx]
            break

    return {
        "A": A,
        "B": B,
        "lag": lag,
        "pos_a": pos_a,
        "pos_b": pos_b,
        "chain_run": chain_run,
        "last_base_date": dates[last_base_idx],
        "last_success_drag_date": last_success_drag_date,
        "success_count": success_count,
        "fail_count": fail_count,
        "success_rate": success_rate,
        "tail_events": tail_events[:50],
        "predict_target": "NEXT"
    }

def _format_unpos_block_next(label, res, dates):
    A = res["A"]; B = res["B"]; lag = res["lag"]
    chain_run = res["chain_run"]
    base_date = res["last_base_date"]
    last_success_drag_date = res["last_success_drag_date"]
    sc = res["success_count"]; fc = res["fail_count"]
    total = sc + fc
    rate = sc / total if total > 0 else 0.0

    lines = []
    lines.append(label)
    lines.append(f"【預測目標】以最新開獎日 {base_date} 為起點 → 預測隔 {lag} 期可能拖出")
    lines.append(f"起始牌(A)：{A:02d}")
    lines.append(f"拖牌(B)：{B:02d}")
    lines.append("")
    lines.append("--- 近期連版（由近到遠，最後一次為“可驗證的歷史”）---")

    tail = res["tail_events"]
    for base_idx, drag_idx, is_success in tail:
        base_d = dates[base_idx]
        drag_d = dates[drag_idx]
        mark = "✔" if is_success else "✖"
        extra = "   ← 斷點" if (not is_success) else ""
        lines.append(
            f"{base_d} 出現A {A:02d} → {drag_d} "
            f"{'出現' if is_success else '未出現'}B {B:02d} {mark}{extra}"
        )

    lines.append("")
    lines.append(f"近期連版：{chain_run} 次（代表最近連續 {chain_run} 次 A 出現後，隔 {lag} 期都有出 B）")
    if last_success_drag_date is not None:
        lines.append(f"最近一次成功拖出日期：{last_success_drag_date}")
    lines.append(f"成功紀錄共：{sc} 次")
    lines.append(f"失敗紀錄共：{fc} 次")
    lines.append(f"成功率：{sc} / {total} = {rate * 100:.1f}%")
    lines.append("--------------------------------")
    return "\n".join(lines)

def _format_pos_block_next(label, res, dates):
    A = res["A"]; B = res["B"]; lag = res["lag"]
    chain_run = res["chain_run"]
    base_date = res["last_base_date"]
    last_success_drag_date = res["last_success_drag_date"]
    sc = res["success_count"]; fc = res["fail_count"]
    total = sc + fc
    rate = sc / total if total > 0 else 0.0
    pos_a = res["pos_a"] + 1
    pos_b = res["pos_b"] + 1

    lines = []
    lines.append(label)
    lines.append(f"【預測目標】以最新開獎日 {base_date} 為起點 → 預測隔 {lag} 期可能拖出")
    lines.append(f"起始牌(A)：{A:02d}")
    lines.append(f"拖牌(B)：{B:02d}")
    lines.append(f"定位：起牌第 {pos_a} 球 → 拖牌第 {pos_b} 球")
    lines.append("")
    lines.append("--- 近期連版（由近到遠）---")

    tail = res["tail_events"]
    for base_idx, drag_idx, is_success in tail:
        base_d = dates[base_idx]
        drag_d = dates[drag_idx]
        mark = "✔" if is_success else "✖"
        extra = "   ← 斷點" if (not is_success) else ""
        lines.append(
            f"{base_d} (第{pos_a}球)=A {A:02d} → {drag_d} "
            f"{'出現' if is_success else '未出現'}(第{pos_b}球)=B {B:02d} {mark}{extra}"
        )

    lines.append("")
    lines.append(f"近期連版：{chain_run} 次")
    if last_success_drag_date is not None:
        lines.append(f"最近一次成功拖出日期：{last_success_drag_date}")
    lines.append(f"成功紀錄共：{sc} 次")
    lines.append(f"失敗紀錄共：{fc} 次")
    lines.append(f"成功率：{sc} / {total} = {rate * 100:.1f}%")
    lines.append("--------------------------------")
    return "\n".join(lines)

def compute_drag_notebooks(df_bt, nums, predict_lags=1):
    """
    ✅ 修正版：以最新開獎為起點，預測多個 lag（例如 1~10）期後可能拖出的號碼
    - predict_lags 可傳 int 或 list/tuple/set
    - ✅ 保證任何情況都回傳 5 個值（避免 None 導致 unpack 失敗）
    - ✅ 若找不到拖牌資訊，txt 至少會有「搜尋不到拖牌資訊」
    """
    # ---- 不啟用就直接回傳（仍回傳 5 個）----
    if not DRAG_ENABLE:
        msg = "（未啟用版路拖牌功能 DRAG_ENABLE=False）\n"
        return msg, msg, msg, msg, set()

    # ✅ 統一：避免 txt 空白
    def _ensure_not_empty(txt, title, lag_list):
        if txt is None or str(txt).strip() == "":
            return (
                f"{title}\n"
                f"搜尋不到拖牌資訊（條件過嚴或樣本不足）。\n"
                f"（目前設定：預測隔 {lag_list} 期、MIN_CHAIN_RUN={MIN_CHAIN_RUN}）\n"
            )
        return txt

    try:
        # ---- 裁切資料列 ----
        if DRAG_MAX_ROWS is not None and len(df_bt) > DRAG_MAX_ROWS:
            df_bt_use = df_bt.tail(DRAG_MAX_ROWS).reset_index(drop=True)
        else:
            df_bt_use = df_bt.copy().reset_index(drop=True)

        if len(df_bt_use) < 2:
            msg = "資料不足，無法計算拖牌（至少需要 2 期以上）。\n"
            return msg, msg, msg, msg, set()

        dates = df_bt_use["日期"].dt.date.tolist()
        last_base_idx = len(dates) - 1
        last_base_date = dates[last_base_idx]

        # ✅ 支援 int 或 iterable
        if isinstance(predict_lags, (list, tuple, set, np.ndarray)):
            lag_list = sorted({int(x) for x in predict_lags if str(x).strip().isdigit() and int(x) >= 1})
        else:
            lag_list = [int(predict_lags)]

        # ✅ 兜底：避免空集合
        if not lag_list:
            lag_list = [1]

        print(f"\n==== 開始計算版路拖牌（✅以最新開獎日為起點：{last_base_date}，預測隔 {lag_list} 期）====")
        t0 = time.time()

        size_seqs = _build_rows_for_mode(df_bt_use, nums, mode="size")
        order_seqs = _build_rows_for_mode(df_bt_use, nums, mode="order")

        drag_nums_size_unpos = set()
        drag_nums_size_pos = set()
        drag_nums_order_unpos = set()
        drag_nums_order_pos = set()

        def build_unpos_notebook_next(seqs, label_header, collect_set):
            results = []
            base_row = seqs[last_base_idx]
            base_set = set(base_row)

            for lag in lag_list:
                for A in sorted(base_set):
                    for B in range(1, 40):
                        res = _analyze_pattern_unpos_next(seqs, dates, A, B, lag, last_base_idx)
                        if res is not None:
                            results.append(res)
                            collect_set.add(B)

            results.sort(key=lambda r: (r["lag"], -r["chain_run"], -r["success_rate"], r["A"], r["B"]))
            blocks = []
            for r in results:
                blocks.append(_format_unpos_block_next(label_header, r, dates))
            return "\n\n".join(blocks)

        def build_pos_notebook_next(seqs, label_header, collect_set):
            results = []
            base_row = seqs[last_base_idx]

            for lag in lag_list:
                for pos_a, A in enumerate(base_row):
                    for pos_b in range(5):
                        for B in range(1, 40):
                            res = _analyze_pattern_pos_next(seqs, dates, A, B, lag, pos_a, pos_b, last_base_idx)
                            if res is not None:
                                results.append(res)
                                collect_set.add(B)

            results.sort(key=lambda r: (r["lag"], -r["chain_run"], -r["success_rate"], r["A"], r["B"]))
            blocks = []
            for r in results:
                blocks.append(_format_pos_block_next(label_header, r, dates))
            return "\n\n".join(blocks)

        txt_size_unpos  = build_unpos_notebook_next(size_seqs,  "【大小序 / 不定位】", drag_nums_size_unpos)
        txt_size_pos    = build_pos_notebook_next(size_seqs,    "【大小序 / 定位】",   drag_nums_size_pos)
        txt_order_unpos = build_unpos_notebook_next(order_seqs, "【落球序 / 不定位】", drag_nums_order_unpos)
        txt_order_pos   = build_pos_notebook_next(order_seqs,   "【落球序 / 定位】",   drag_nums_order_pos)

        # ✅ 保證不空白
        txt_size_unpos  = _ensure_not_empty(txt_size_unpos,  "【大小序 / 不定位】", lag_list)
        txt_size_pos    = _ensure_not_empty(txt_size_pos,    "【大小序 / 定位】",   lag_list)
        txt_order_unpos = _ensure_not_empty(txt_order_unpos, "【落球序 / 不定位】", lag_list)
        txt_order_pos   = _ensure_not_empty(txt_order_pos,   "【落球序 / 定位】",   lag_list)

        drag_hint_nums_all = (drag_nums_size_unpos | drag_nums_size_pos | drag_nums_order_unpos | drag_nums_order_pos)

        t1 = time.time()
        print(f"版路拖牌計算完成（✅已改成預測多期 lag），耗時約 {t1 - t0:.2f} 秒。")
        return txt_size_unpos, txt_size_pos, txt_order_unpos, txt_order_pos, drag_hint_nums_all

    except Exception as e:
        err = f"版路拖牌計算發生例外：{type(e).__name__}: {e}\n"
        msg = "搜尋不到拖牌資訊（執行時發生例外）。\n" + err
        return msg, msg, msg, msg, set()

# ==============================
# ========== 小工具：推估下一次開獎日（僅估計；對獎用期數） ==========
# ==============================
def infer_next_draw_date(past_dates):
    if not past_dates:
        return (datetime.now().date() + timedelta(days=1))
    last = past_dates[-1]
    d = last + timedelta(days=1)
    if d.weekday() == 6:
        d += timedelta(days=1)
    s = set(past_dates)
    for _ in range(14):
        if d not in s and d.weekday() != 6:
            return d
        d += timedelta(days=1)
    return d


# ==============================
# ========== Walk-forward：統一訓練/預測封裝 ==========
# ==============================
def fit_model_for_i(model_name, X_bt, y_all, base_columns, start_idx, i, seed_offset=0):
    """
    用 [start_idx, i) 訓練；i 這期用來預測下一期
    """
    if model_name == "CRF":
        return fit_crf(X_bt, y_all, start_idx, i, seed_offset=seed_offset)

    y_cols = [f"y_{k:02d}" for k in range(1, 40)]
    X_train = fix_columns(X_bt.iloc[start_idx:i], base_columns)
    y_train = y_all[y_cols].iloc[start_idx:i]

    mo = make_multi_model(model_name, seed_offset=seed_offset)

    # ✅ 所有模型 fit 期間都包：避免任何雜訊噴出（stdout/stderr/warnings）
    with suppress_all_output(True):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # 比 filterwarnings 更乾淨：直接忽略所有 warning
            mo.fit(X_train, y_train)

    return mo


def predict_scores_for_i(model_name, model_obj, X_bt, base_columns, i):
    """
    用第 i 期特徵，輸出 39 個號碼機率（預測 i+1 期）
    """
    if model_name == "CRF":
        # CRF predict 也包一下，避免套件內部訊息
        with suppress_all_output(True):
            return crf_predict_proba(model_obj, X_bt, i)

    X_pred = fix_columns(X_bt.iloc[[i]], base_columns)
    with suppress_all_output(True):
        return proba_from_multioutput(model_obj, X_pred)


# ==============================
# ========== 主程式 ==========
# ==============================
def main():
    warnings.simplefilter(action='ignore', category=FutureWarning)
    warnings.simplefilter(action='ignore', category=pd.errors.SettingWithCopyWarning)

    n_cpu_all = multiprocessing.cpu_count()
    print(f"本機核心數：{n_cpu_all} 核心")

    try:
        backtest_N = int(input("請輸入欲回測最近期數："))
        BACKTEST_TOP_N = int(input("請輸入【回測】每期要選幾顆號碼（例如 10）："))
        PRED_TOP_N = int(input("請輸入【預測】每期要選幾顆號碼（例如 7）："))
        DISPLAY_LOGS_N = int(input("回測命中紀錄終端機要顯示最近幾期（例如 12）："))
    except Exception:
        print("輸入格式錯誤，請輸入整數。")
        return

    if BACKTEST_TOP_N <= 0 or BACKTEST_TOP_N > 39 or PRED_TOP_N <= 0 or PRED_TOP_N > 39:
        print("TOP_N 數量需介於 1 ~ 39 之間。")
        return

    if DISPLAY_LOGS_N <= 0:
        DISPLAY_LOGS_N = 12

    print("下載/載入官方資料...")
    try:
        df = load_tw539(force_download=True)
    except Exception as e:
        print(f"下載失敗：{e}")
        return

    nums = ["正序號碼1", "正序號碼2", "正序號碼3", "正序號碼4", "正序號碼5"]

    def pretty_print_row(row, title=""):
        print(f"{Fore.LIGHTYELLOW_EX}==== {title} ===={Style.RESET_ALL}")
        print(f"{Fore.LIGHTCYAN_EX}日期        {row['日期'].strftime('%Y-%m-%d')}{Style.RESET_ALL}")
        print(f"{Fore.LIGHTCYAN_EX}期數        {int(row['期數'])}{Style.RESET_ALL}")
        print(f"{Fore.LIGHTWHITE_EX}落序號碼1~5  ", end="")
        for i in range(1, 6):
            print(f"{int(row[f'落序號碼{i}']):02d} ", end="")
        print(Style.RESET_ALL)
        print(f"{Fore.LIGHTWHITE_EX}正序號碼1~5  ", end="")
        for i in range(1, 6):
            print(f"{int(row[f'正序號碼{i}']):02d} ", end="")
        print(Style.RESET_ALL)
        print()

    pretty_print_row(df.iloc[0], "第一筆資料")
    pretty_print_row(df.iloc[-1], "最後一筆資料")
    print(f"{Fore.LIGHTGREEN_EX}==== 全部期數：{len(df)} 筆 ===={Style.RESET_ALL}\n")

    # ✅ 先對獎（用期數優先）
    try_save_latest_prediction_result(df, nums)

    # 建特徵一次
    X_full, base_columns, _, _ = build_full_features_once(df)
    X_bt = X_full.reset_index(drop=True)
    df_bt = df.reset_index(drop=True)

    # y_all：用第 t 期特徵 → 預測 t+1 期是否包含 1..39
    future_open = df[nums].shift(-1)
    y_all = pd.DataFrame({
        f"y_{i:02d}": future_open.apply(lambda x, i=i: int(i in x.values), axis=1).astype(np.int8)
        for i in range(1, 40)
    }).reset_index(drop=True)

    # 最後一筆無法預測下一期（缺 y）
    num_rows = X_bt.shape[0]
    feature_row_count = num_rows - 1

    if feature_row_count < backtest_N or backtest_N <= 0:
        print(f"【錯誤】可回測期數 ({feature_row_count}) 必須大於回測期數 ({backtest_N})！")
        return

    backtest_indexes = list(range(feature_row_count - backtest_N, feature_row_count))

    backtest_eval_start_date = df_bt.iloc[backtest_indexes[0] + 1]["日期"].date()
    backtest_eval_end_date   = df_bt.iloc[backtest_indexes[-1] + 1]["日期"].date()
    print(f"==== 回測評估日期（實際被預測的開獎日）: {backtest_eval_start_date} ~ {backtest_eval_end_date} ====")

    latest_issue = int(df_bt.iloc[-1]["期數"])
    predict_next_issue = latest_issue + 1
    predict_next_date = infer_next_draw_date(df_bt["日期"].dt.date.tolist())

    print(f"==== 預測下一期（期數優先）：期數 {predict_next_issue} / 日期（估計）{predict_next_date} ====")
    print(f"==== Walk-forward 回測：每 {WALK_FORWARD_RETRAIN_INTERVAL} 期重訓一次（K={WALK_FORWARD_RETRAIN_INTERVAL}） ====")

    # ✅ 先跑版路拖牌（已改為：預測下一期/lag期後）
    if DRAG_ENABLE:
        txt_size_unpos, txt_size_pos, txt_order_unpos, txt_order_pos, drag_hint_nums_all = compute_drag_notebooks(
            df_bt, nums, predict_lags=DRAG_PREDICT_LAGS
        )
        files_and_txt = [
            ("版路拖牌_順球_未定位.txt",  txt_size_unpos),
            ("版路拖牌_順球_定位.txt",    txt_size_pos),
            ("版路拖牌_落球_未定位.txt", txt_order_unpos),
            ("版路拖牌_落球_定位.txt",   txt_order_pos),
        ]
        for fname, content in files_and_txt:
            with open(fname, "w", encoding="utf-8-sig") as f:
                f.write(content)
        print("版路拖牌筆記本已輸出為：版路拖牌_順球_未定位.txt / 版路拖牌_順球_定位.txt / 版路拖牌_落球_未定位.txt / 版路拖牌_落球_定位.txt")
        if drag_hint_nums_all:
            print("✅ 版路拖牌『預測下一期』推薦號碼（B union）：",
                  "[" + ", ".join(f"{n:02d}" for n in sorted(drag_hint_nums_all)) + "]")
    else:
        drag_hint_nums_all = set()

    model_names = ["XGBoost", "LightGBM", "CatBoost", "Logistic", "RandomForest", "CRF"]
    model_colors = {
        "XGBoost": Fore.LIGHTGREEN_EX,
        "LightGBM": Fore.LIGHTYELLOW_EX,
        "CatBoost": Fore.LIGHTMAGENTA_EX,
        "Logistic": Fore.LIGHTBLUE_EX,
        "RandomForest": Fore.LIGHTRED_EX,
        "CRF": Fore.LIGHTCYAN_EX,
    }

    def format_nums(nums_list):
        return "[" + ", ".join(f"{n:02d}" for n in sorted(nums_list)) + "]"

    def format_nums_with_star(nums_list, star_set):
        parts = []
        for n in nums_list:
            mark = "★" if n in star_set else ""
            parts.append(f"{n:02d}{mark}")
        return "[" + ", ".join(parts) + "]"

    total_start = time.time()
    error_logs = []
    model_hits = {}
    model_logs = {}
    model_prizes = {}

    # ✅ Walk-forward 回測（每 K 期重訓一次）
    def run_backtest_walkforward(model_name):
        hits, logs = [], []
        print(f"\n--- 開始模型 [{model_name}] Walk-forward 回測 ---")

        K = max(1, int(WALK_FORWARD_RETRAIN_INTERVAL))
        model_obj = None
        last_fit_i = None
        fit_count = 0

        for _, i in enumerate(tqdm(
            backtest_indexes,
            desc=f"回測WF [{model_name}]",
            ncols=80,
            dynamic_ncols=True,
            leave=True
        )):
            start_idx = 0 if (TRAIN_WINDOW is None) else max(0, i - TRAIN_WINDOW)

            need_refit = (model_obj is None) or (last_fit_i is None) or ((i - last_fit_i) >= K)
            if need_refit:
                try:
                    seed_offset = (fit_count * 1000) + (model_names.index(model_name) * 100)
                    model_obj = fit_model_for_i(model_name, X_bt, y_all, base_columns, start_idx, i, seed_offset=seed_offset)
                    last_fit_i = i
                    fit_count += 1
                except Exception as e:
                    err_msg = f"{model_name} | fit_i={i} | {str(e)}"
                    error_logs.append(err_msg)
                    model_obj = None

            try:
                if model_obj is None:
                    pred_scores = np.zeros(39, dtype=np.float32)
                else:
                    pred_scores = predict_scores_for_i(model_name, model_obj, X_bt, base_columns, i)
            except Exception as e:
                err_msg = f"{model_name} | pred_i={i} | {str(e)}"
                error_logs.append(err_msg)
                pred_scores = np.zeros(39, dtype=np.float32)

            top_idx = np.argsort(pred_scores)[-BACKTEST_TOP_N:]
            pred_top_n = [int(j + 1) for j in top_idx]

            real_idx = i + 1
            real_date = df_bt.iloc[real_idx]["日期"].date()
            real_numbers = df_bt.iloc[real_idx][nums].tolist()

            hit = len(set(pred_top_n) & set(real_numbers))
            hits.append(hit)

            logs.append(
                f"{model_name:<12} | {real_date} | 命中:{hit:<2d} | "
                f"預測號碼:{format_nums(pred_top_n)} | 開獎號碼:{format_nums(real_numbers)}"
            )

        prize_stat = calc_prize(hits, BACKTEST_TOP_N)
        print(f"--- 完成模型 [{model_name}] Walk-forward 回測 ---")
        return model_name, hits, logs, prize_stat

    for mn in model_names:
        k, hits, logs, prize_stat = run_backtest_walkforward(mn)
        model_hits[k] = hits
        model_logs[k] = logs
        model_prizes[k] = prize_stat

    total_end = time.time()
    print(f"\n回測全部模型總耗時：{total_end - total_start:.2f} 秒")
    if error_logs:
        with open("error_log.txt", "a", encoding="utf-8") as ef:
            for line in error_logs:
                ef.write(line + "\n")
        print(f"【已記錄異常於 error_log.txt】")

    nowtag = datetime.now().strftime('%Y%m%d_%H%M%S')

    backtest_log_file = f"backtest_all_models_WF_{backtest_N}N_K{WALK_FORWARD_RETRAIN_INTERVAL}_{nowtag}.txt"
    with open(backtest_log_file, "w", encoding="utf-8-sig") as f:
        f.write(f"Walk-forward 回測總表\n")
        f.write(f"日期範圍(被預測開獎日)：{backtest_eval_start_date} ~ {backtest_eval_end_date}\n")
        f.write(f"回測期數：{backtest_N}\n")
        f.write(f"K(每幾期重訓)：{WALK_FORWARD_RETRAIN_INTERVAL}\n")
        f.write(f"回測TOP_N：{BACKTEST_TOP_N}\n")
        f.write("="*80 + "\n\n")
        for mn in model_names:
            f.write(f"[{mn}] 回測命中紀錄（完整）\n")
            for line in model_logs.get(mn, []):
                f.write(line + "\n")
            f.write("\n")
    print(f"\n✅ 已輸出「全部模型完整回測紀錄」到檔案：{backtest_log_file}")

    history_fname = os.path.join(
        os.getcwd(),
        f"history_WF_{backtest_N}N_K{WALK_FORWARD_RETRAIN_INTERVAL}_bt{BACKTEST_TOP_N}_pred{PRED_TOP_N}_{nowtag}.pkl"
    )
    history_data = {
        "datetime": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "mode": "walk_forward",
        "K": int(WALK_FORWARD_RETRAIN_INTERVAL),
        "backtest_eval_start_date": str(backtest_eval_start_date),
        "backtest_eval_end_date": str(backtest_eval_end_date),
        "backtest_N": int(backtest_N),
        "BACKTEST_TOP_N": int(BACKTEST_TOP_N),
        "PRED_TOP_N": int(PRED_TOP_N),
        "TRAIN_WINDOW": int(TRAIN_WINDOW) if TRAIN_WINDOW is not None else None,
        "FAST_MODE": bool(FAST_MODE),
        "USE_DETERMINISTIC": bool(USE_DETERMINISTIC),
        "GLOBAL_SEED": int(GLOBAL_SEED),
        "model_hits": model_hits,
        "model_prizes": model_prizes,
    }
    with open(history_fname, "wb") as f:
        pickle.dump(history_data, f)
    print(f"已輸出回測統計檔：{history_fname}")

    print(f"\n==== 各模型回測命中紀錄（終端機僅顯示最近 {DISPLAY_LOGS_N} 期；完整見 {backtest_log_file}） ====")
    for k in model_names:
        color = model_colors.get(k, Fore.WHITE)
        print(f"\n{color}{k} 回測命中紀錄（最近 {DISPLAY_LOGS_N} 期）：{Style.RESET_ALL}")
        show_lines = model_logs.get(k, [])[-DISPLAY_LOGS_N:]
        for log in show_lines:
            print(f"{color}{log}{Style.RESET_ALL}")

        print(f"{color}{k}【回測獎金統計】(回測 TOP_{BACKTEST_TOP_N}){Style.RESET_ALL}")
        for star in ["2星", "3星", "4星"]:
            pz = model_prizes[k][star]
            print(
                f"  {star}  總中獎組數：{pz['win']}，總投注：{pz['bets']}，花費：{pz['cost']} 元，"
                f"彩金：{pz['prize']} 元，盈虧：{pz['profit']} 元"
            )

    print(f"\n==== 各模型平均命中總結（Walk-forward / 回測 TOP_{BACKTEST_TOP_N}） ====")
    for k, v in model_hits.items():
        print(
            f"{model_colors[k]}{k} 平均每期命中：{np.mean(v):.2f}（標準差：{np.std(v):.2f}）{Style.RESET_ALL}"
        )

    profit_rank = []
    for k in model_names:
        total_profit = sum(model_prizes[k][star]['profit'] for star in ["2星", "3星", "4星"])
        profit_rank.append((k, total_profit))
    profit_rank.sort(key=lambda x: x[1], reverse=True)

    print(f"\n==== 各模型總計盈虧排名（Walk-forward / 回測 TOP_{BACKTEST_TOP_N}） ====")
    for idx, (k, profit) in enumerate(profit_rank, 1):
        color = model_colors.get(k, Fore.WHITE)
        print(f"{color}{idx}. {k:<12} ➤【總計盈虧】：{profit} 元{Style.RESET_ALL}")

    # ==============================
    # ========== 預測下一期（只訓練一次，貼近真實） ==========
    # ==============================
    all_model_predictions_prob = {}
    all_model_predictions_size = {}
    all_model_pred_scores = {}

    def predict_one_model(model_name):
        i = len(X_bt) - 1
        start_idx = 0 if (TRAIN_WINDOW is None) else max(0, i - TRAIN_WINDOW)
        seed_offset = 999_000 + (model_names.index(model_name) * 100)

        try:
            model_obj = fit_model_for_i(model_name, X_bt, y_all, base_columns, start_idx, i, seed_offset=seed_offset)
            scores = predict_scores_for_i(model_name, model_obj, X_bt, base_columns, i)
        except Exception as e:
            print(f"\n{Fore.LIGHTRED_EX}{model_name} 預測例外: {e}{Style.RESET_ALL}")
            scores = np.zeros(39, dtype=np.float32)

        top_idx_prob = np.argsort(scores)[-PRED_TOP_N:][::-1]
        pred_prob = [int(j + 1) for j in top_idx_prob]
        pred_size = sorted(pred_prob)
        return pred_prob, pred_size, scores

    for mn in model_names:
        pred_prob, pred_size, scores = predict_one_model(mn)

        all_model_predictions_prob[mn] = pred_prob
        all_model_predictions_size[mn] = pred_size
        all_model_pred_scores[mn] = scores.tolist()

        color = model_colors.get(mn, Fore.WHITE)
        print(f"\n{color}{mn:<12}{Style.RESET_ALL} | 預測期數 {predict_next_issue} / 日期(估) {predict_next_date}")
        print("  落球序預測（機率大→小）："
              f"{format_nums_with_star(pred_prob, drag_hint_nums_all)}")
        print("  大小序預測（正序）    ："
              f"{format_nums_with_star(pred_size, drag_hint_nums_all)}")
        print("-" * 60)

        if SAVE_PROBA_CSV:
            out_csv = f"pred_scores_{mn}.csv"
            pd.DataFrame({
                "number": [f"{i:02d}" for i in range(1, 40)],
                "proba": scores
            }).to_csv(out_csv, index=False, encoding="utf-8-sig")

    print("\n==== 各模型單獨預測（純大小序列表；不含★） ====")
    for mn in model_names:
        color = model_colors.get(mn, Fore.WHITE)
        nums_str_plain = "[" + ", ".join(f"{n:02d}" for n in all_model_predictions_size[mn]) + "]"
        print(f"{color}{mn:<12}{Style.RESET_ALL} | 期數 {predict_next_issue} 預測號碼（TOP_{PRED_TOP_N}）：{nums_str_plain}")

    def save_latest_predict(predict_issue, predict_date, preds_dict):
        filename = "latest_predict.csv"
        with open(filename, mode="w", newline="", encoding="utf-8-sig") as f:
            writer = csv.writer(f)
            writer.writerow(["預測期數", "預測日期", "模型", "預測號碼"])
            for model_name, numbers in preds_dict.items():
                nums_str = ",".join([f"{n:02d}" for n in sorted(numbers)])
                writer.writerow([str(predict_issue), str(predict_date), model_name, nums_str])
        print(f"\n已儲存最新一期各模型預測號碼於 {filename}！（預測 TOP_{PRED_TOP_N}，採大小序；對獎優先用期數）")

    save_latest_predict(predict_next_issue, str(predict_next_date), all_model_predictions_size)

    # ==============================
    # ========== 綜合票選 ==========
    # ==============================
    score_sum_unweighted = np.zeros(39, dtype=np.float32)
    for _, s in all_model_pred_scores.items():
        score_sum_unweighted += np.array(s, dtype=np.float32)

    penalty_counts = load_penalty_from_real_results(result_file="latest_predict_result.csv")
    penalty = MISTAKE_ALPHA * penalty_counts

    combined_A = np.maximum(0.0, score_sum_unweighted - penalty)
    top_idx_A = np.argsort(combined_A)[-PRED_TOP_N:]
    top_common_A = [i + 1 for i in top_idx_A]
    print(
        f"\n{Fore.LIGHTWHITE_EX}所有模型【未加權+錯題本懲罰(真實對獎)】綜合票選（TOP_{PRED_TOP_N}）："
        f"{format_nums_with_star(sorted(top_common_A), drag_hint_nums_all)}{Style.RESET_ALL}"
    )

    model_weight = {k: max(1e-6, float(np.mean(v))) for k, v in model_hits.items()}
    score_sum_weighted = np.zeros(39, dtype=np.float32)
    for mn, s in all_model_pred_scores.items():
        score_sum_weighted += model_weight.get(mn, 1.0) * np.array(s, dtype=np.float32)

    combined_B = np.maximum(0.0, score_sum_weighted - penalty)
    top_idx_B = np.argsort(combined_B)[-PRED_TOP_N:]
    top_common_B = [i + 1 for i in top_idx_B]
    print(
        f"{Fore.LIGHTWHITE_EX}所有模型【表現加權(WF回測)+錯題本懲罰(真實對獎)】綜合票選（TOP_{PRED_TOP_N}）："
        f"{format_nums_with_star(sorted(top_common_B), drag_hint_nums_all)}{Style.RESET_ALL}"
    )

    all_nums_flat = [num for pred_list in all_model_predictions_size.values() for num in pred_list]
    cnt = Counter(all_nums_flat)

    overlap_min = 2
    overlap_nums = [n for n, v in cnt.items() if v >= overlap_min]
    if overlap_nums:
        print(
            f"{Fore.LIGHTWHITE_EX}所有模型【至少 {overlap_min} 個模型都推薦】號碼："
            f"{format_nums_with_star(sorted(overlap_nums), drag_hint_nums_all)}{Style.RESET_ALL}"
        )

    union_nums = sorted(set(all_nums_flat))
    print(
        f"{Fore.LIGHTWHITE_EX}所有模型【聯合集合】推薦號碼（TOP_{PRED_TOP_N} 之 union）："
        f"{format_nums_with_star(union_nums, drag_hint_nums_all)}{Style.RESET_ALL}"
    )

    print("\n==== 依歷史平均命中率排序的模型預測總表（由高→低 / Walk-forward 回測）====")
    summary_rows = []
    for mn in model_names:
        hits = model_hits.get(mn, [])
        avg_hit = float(np.mean(hits)) if hits else 0.0
        std_hit = float(np.std(hits)) if hits else 0.0
        pred_size = all_model_predictions_size.get(mn, [])
        summary_rows.append((avg_hit, std_hit, mn, pred_size))

    summary_rows.sort(key=lambda x: (-x[0], x[2]))

    for avg_hit, std_hit, mn, pred_size in summary_rows:
        color = model_colors.get(mn, Fore.WHITE)
        nums_str = " ".join(f"{n:02d}{('★' if n in drag_hint_nums_all else '')}" for n in pred_size)
        print(
            f"{color}{mn:<12}{Style.RESET_ALL} | 平均命中：{avg_hit:.2f}（σ={std_hit:.2f}） | "
            f"期數 {predict_next_issue} 大小序預測：[{nums_str}]"
        )

    print("\n（提醒）本版本回測已改為 Walk-forward（每 K 期重訓一次），命中率通常會比「每期重訓」更接近真實預測表現。")
    print("（提醒）對獎以『期數』為準，日期僅為估計，避免你以前因為日期推錯而誤判沒中。")
    print("（提醒）帶 ★ 的號碼為「版路拖牌（預測下一期）」看好之號碼，可搭配 4 份 drag_*.txt 交叉參考。")

if __name__ == "__main__":
    main()
# =================== 程式結束 ===================
