# feature_sel.py
# Time-series feature selection for news-aggregated daily panels.
# Python 3.10+ recommended

from __future__ import annotations

import re
import warnings
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import mutual_info_regression
from sklearn.inspection import permutation_importance
from sklearn.metrics import make_scorer, r2_score
from sklearn.model_selection import TimeSeriesSplit

# Optional LightGBM; we fallback to HistGradientBoostingRegressor
try:
    import lightgbm as lgb  # type: ignore
except Exception:  # pragma: no cover
    lgb = None

from sklearn.ensemble import HistGradientBoostingRegressor


@dataclass
class FScores:
    """Holds per-feature scores (NaN if step not used/available)."""
    mi: Dict[str, float]
    model_importance: Dict[str, float]
    perm_importance: Dict[str, float]
    combined: Dict[str, float]


@dataclass
class FReport:
    """Human-readable report of drops/keeps and scores."""
    dropped_constant: List[str]
    dropped_high_missing: List[str]
    dropped_duplicates: List[str]
    dropped_forbidden: List[str]
    dropped_corr_pruned: List[str]
    kept: List[str]
    scores: FScores
    params: Dict[str, Union[str, int, float, bool]]


class FeatureSelectorTS(BaseEstimator, TransformerMixin):
    """
    Time-series Feature Selector with two presets: 'boosting' and 'lstm'.

    Pipeline:
      1) Sort by date; cast dtypes (bool->int, keep numeric only).
      2) Drop constant/near-constant, high-missing, duplicates, forbidden regex.
      3) Compute mutual information (MI) with target.
      4) Model-based importance via LightGBM (if available) or HistGBR over rolling splits.
      5) Time-aware permutation importance averaged across splits.
      6) Combine scores (weighted), rank, cap by quantile and/or max_features.
      7) Correlation pruning on the ranked list (greedy) with Spearman/Pearson.

    Notes:
      - No target shifting is done here. Provide y already aligned to forecasting horizon (e.g., y_t+h).
      - Only numeric features are used. Keep your 'fecha' separately; this class returns X with selected columns.
      - Works well with your aggregated daily frame (~900+ cols, ~1500 rows).

    Parameters
    ----------
    date_col : str
        Name of date column (not used as a feature).
    target_col : Optional[str]
        If provided in fit(df), target is df[target_col]. Otherwise pass y to fit(X, y).
    mode : {'boosting','lstm'}
        Preset blending of scores and thresholds.
    missing_threshold : float
        Drop features with missing rate >= threshold (0..1).
    quasi_constant_tol : float
        Drop features where the most frequent value rate >= tol (e.g., 0.995).
    corr_threshold : float
        Absolute correlation threshold for pruning highly correlated features.
    corr_method : {'spearman','pearson'}
        Correlation method for pruning.
    n_splits : int
        Rolling TimeSeriesSplit folds.
    cv_gap : int
        Gap (number of rows) between train and test to mitigate leakage.
    random_state : int
        Random seed for model steps.
    max_features : Optional[int]
        Cap on the number of features kept after ranking (before corr pruning).
    score_quantile : Optional[float]
        Keep features with combined score >= this quantile (0..1). Applied before max_features and pruning.
    forbid_patterns : Optional[Sequence[str]]
        Regex patterns of columns to drop (e.g., proxies of the target).
    mi_discrete_features : Optional[Iterable[str]]
        Treat these features as discrete for MI.
    weights : Optional[Tuple[float,float,float]]
        Weights (w_mi, w_model, w_perm). If None, chosen by mode preset.
    scorer : Optional[callable]
        Sklearn scorer; default R^2 for regression.
    """

    def __init__(
        self,
        date_col: str = "fecha",
        target_col: Optional[str] = None,
        mode: str = "boosting",
        missing_threshold: float = 0.4,
        quasi_constant_tol: float = 0.995,
        corr_threshold: float = 0.95,
        corr_method: str = "spearman",
        n_splits: int = 5,
        cv_gap: int = 2,
        random_state: int = 42,
        max_features: Optional[int] = 256,
        score_quantile: Optional[float] = 0.50,
        forbid_patterns: Optional[Sequence[str]] = None,
        mi_discrete_features: Optional[Iterable[str]] = None,
        weights: Optional[Tuple[float, float, float]] = None,
        scorer=None,
    ):
        self.date_col = date_col
        self.target_col = target_col
        self.mode = mode
        self.missing_threshold = float(missing_threshold)
        self.quasi_constant_tol = float(quasi_constant_tol)
        self.corr_threshold = float(corr_threshold)
        self.corr_method = str(corr_method)
        self.n_splits = int(n_splits)
        self.cv_gap = int(cv_gap)
        self.random_state = int(random_state)
        self.max_features = max_features if (max_features is None) else int(max_features)
        self.score_quantile = score_quantile if (score_quantile is None) else float(score_quantile)
        self.forbid_patterns = list(forbid_patterns) if forbid_patterns else []
        self.mi_discrete_features = set(mi_discrete_features) if mi_discrete_features else set()
        self.weights = weights  # (w_mi, w_model, w_perm)
        self.scorer = scorer if scorer is not None else make_scorer(r2_score)

        # learned/produced
        self.selected_features_: List[str] = []
        self.kept_features_before_pruning_: List[str] = []
        self._report: Optional[FReport] = None
        self._scores_: Optional[FScores] = None
        self._fit_columns_: List[str] = []  # numeric feature columns seen at fit

    # ----------------------------- public API -----------------------------

    def fit(self, df: pd.DataFrame, y: Optional[pd.Series] = None):
        """Fit selector and determine the final feature set."""
        X, y_vec = self._prepare_xy(df, y)
        # Basic filters
        dropped_constant, mask_const = self._drop_constant_quasi_constant(X)
        dropped_missing, mask_missing = self._drop_high_missing(X)
        dropped_forbidden, mask_forbid = self._drop_forbidden(X)

        # Apply masks
        keep_mask = mask_const & mask_missing & mask_forbid
        X1 = X.loc[:, keep_mask]
        # Duplicates
        dropped_dups, X2 = self._drop_duplicates(X1)

        # Scores
        mi_scores = self._compute_mi(X2, y_vec)
        model_importance = self._compute_model_importance(X2, y_vec)
        perm_importance = self._compute_permutation_importance(X2, y_vec)

        # Combine
        comb_scores, kept_before_cap = self._combine_and_cap(
            X2.columns.tolist(), mi_scores, model_importance, perm_importance
        )

        # Correlation pruning (greedy keep-highest-score)
        kept_final, dropped_corr = self._corr_pruning(X2[kept_before_cap], comb_scores)

        self.selected_features_ = kept_final
        self.kept_features_before_pruning_ = kept_before_cap
        self._fit_columns_ = X.columns.tolist()

        self._scores_ = FScores(
            mi=mi_scores, model_importance=model_importance,
            perm_importance=perm_importance, combined=comb_scores
        )
        self._report = FReport(
            dropped_constant=dropped_constant,
            dropped_high_missing=dropped_missing,
            dropped_duplicates=dropped_dups,
            dropped_forbidden=dropped_forbidden,
            dropped_corr_pruned=dropped_corr,
            kept=self.selected_features_,
            scores=self._scores_,
            params=dict(
                mode=self.mode,
                missing_threshold=self.missing_threshold,
                quasi_constant_tol=self.quasi_constant_tol,
                corr_threshold=self.corr_threshold,
                corr_method=self.corr_method,
                n_splits=self.n_splits,
                cv_gap=self.cv_gap,
                random_state=self.random_state,
                max_features=self.max_features if self.max_features is not None else -1,
                score_quantile=self.score_quantile if self.score_quantile is not None else -1,
            ),
        )
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.selected_features_:
            raise RuntimeError("FeatureSelectorTS is not fitted. Call fit() first.")
        # Be permissive: select intersection (future frames may miss some columns)
        miss = [c for c in self.selected_features_ if c not in df.columns]
        if miss:
            warnings.warn(f"[FeatureSelectorTS] Missing columns in transform: {miss[:8]}{'...' if len(miss)>8 else ''}")
        feats = [c for c in self.selected_features_ if c in df.columns]
        return df.loc[:, feats]

    def fit_transform(self, df: pd.DataFrame, y: Optional[pd.Series] = None) -> pd.DataFrame:
        return self.fit(df, y).transform(df)

    def get_support(self) -> List[str]:
        return list(self.selected_features_)

    def get_report(self) -> FReport:
        if self._report is None:
            raise RuntimeError("No report available. Fit the selector first.")
        return self._report

    # ----------------------------- internals ------------------------------

    def _prepare_xy(self, df: pd.DataFrame, y: Optional[pd.Series]) -> Tuple[pd.DataFrame, np.ndarray]:
        if self.target_col is not None:
            if self.target_col not in df.columns:
                raise ValueError(f"target_col '{self.target_col}' not found in DataFrame.")
            y_vec = pd.to_numeric(df[self.target_col], errors="coerce").values
            X = df.drop(columns=[self.target_col]).copy()
        else:
            if y is None:
                raise ValueError("Provide y or set target_col in constructor.")
            y_vec = pd.to_numeric(y, errors="coerce").values
            X = df.copy()

        # sort by date if available
        if self.date_col in X.columns:
            X = X.sort_values(self.date_col)
            # drop date from features
            X = X.drop(columns=[self.date_col])

        # cast bool->int, keep numeric only
        for c in X.columns:
            if pd.api.types.is_bool_dtype(X[c]):
                X[c] = X[c].astype("int8")
        X = X.select_dtypes(include=[np.number]).copy()
        return X, y_vec

    def _drop_constant_quasi_constant(self, X: pd.DataFrame) -> Tuple[List[str], pd.Series]:
        dropped = []
        keep_mask = pd.Series(True, index=X.columns)
        # constant / near-constant by frequency
        for c in X.columns:
            s = X[c]
            if s.notna().sum() == 0:
                dropped.append(c)
                keep_mask[c] = False
                continue
            # constant
            if s.nunique(dropna=True) <= 1:
                dropped.append(c)
                keep_mask[c] = False
                continue
            # quasi-constant: dominant frequency
            top_freq = s.value_counts(dropna=True, normalize=True).iloc[0]
            if top_freq >= self.quasi_constant_tol:
                dropped.append(c)
                keep_mask[c] = False
        return dropped, keep_mask

    def _drop_high_missing(self, X: pd.DataFrame) -> Tuple[List[str], pd.Series]:
        miss_rate = X.isna().mean()
        dropped = miss_rate[miss_rate >= self.missing_threshold].index.tolist()
        keep_mask = ~X.columns.isin(dropped)
        keep_mask = pd.Series(keep_mask, index=X.columns)
        return dropped, keep_mask

    def _drop_forbidden(self, X: pd.DataFrame) -> Tuple[List[str], pd.Series]:
        if not self.forbid_patterns:
            return [], pd.Series(True, index=X.columns)
        pattern = re.compile("|".join(self.forbid_patterns), flags=re.IGNORECASE)
        mask = ~X.columns.to_series().str.contains(pattern)
        dropped = X.columns[~mask].tolist()
        return dropped, mask

    def _drop_duplicates(self, X: pd.DataFrame) -> Tuple[List[str], pd.DataFrame]:
        # hash columns to find exact duplicates efficiently
        hashed = X.apply(pd.util.hash_pandas_object, index=False).T
        _, idx = np.unique(hashed, axis=0, return_index=True)
        keep_cols = X.columns[np.sort(idx)]
        dropped = [c for c in X.columns if c not in keep_cols]
        return dropped, X.loc[:, keep_cols]

    def _compute_mi(self, X: pd.DataFrame, y: np.ndarray) -> Dict[str, float]:
        if X.shape[1] == 0:
            return {}
        disc_mask = np.array([c in self.mi_discrete_features for c in X.columns], dtype=bool)
        # fill NaN with column median for MI
        X_mi = X.copy()
        for c in X_mi.columns:
            if X_mi[c].isna().any():
                X_mi[c] = X_mi[c].fillna(X_mi[c].median())
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            mi_vals = mutual_info_regression(X_mi.values, y, discrete_features=disc_mask if disc_mask.any() else "auto",
                                             random_state=self.random_state)
        return {c: float(v) for c, v in zip(X.columns, mi_vals)}

    def _get_tree_model(self):
        if lgb is not None:
            # small, robust config for ranking importance
            return lgb.LGBMRegressor(
                n_estimators=400,
                learning_rate=0.05,
                max_depth=-1,
                num_leaves=63,
                colsample_bytree=0.8,
                subsample=0.8,
                subsample_freq=1,
                random_state=self.random_state,
                n_jobs=-1,
            )
        # fallback
        return HistGradientBoostingRegressor(
            max_depth=None, learning_rate=0.05, max_iter=400,
            random_state=self.random_state
        )

    def _rolling_splits(self, n_samples: int):
        # sklearn TimeSeriesSplit has no "gap" in <1.3; emulate simple gap by slicing
        tscv = TimeSeriesSplit(n_splits=self.n_splits)
        for train_idx, test_idx in tscv.split(np.arange(n_samples)):
            if self.cv_gap > 0:
                # move test start forward by gap; possibly shrink test
                test_start = test_idx[0] + self.cv_gap
                if test_start >= test_idx[-1]:
                    continue
                test_idx = np.arange(test_start, test_idx[-1] + 1)
            yield train_idx, test_idx

    def _compute_model_importance(self, X: pd.DataFrame, y: np.ndarray) -> Dict[str, float]:
        if X.shape[1] == 0:
            return {}
        model = self._get_tree_model()
        fi_accum = np.zeros(X.shape[1], dtype=float)
        denom = 0
        for tr, te in self._rolling_splits(X.shape[0]):
            if len(np.unique(y[tr])) < 2:
                continue
            Xtr = X.iloc[tr].copy()
            ytr = y[tr].copy()
            # fill NaNs per fold with medians to keep pipeline simple
            Xtr = Xtr.fillna(Xtr.median(numeric_only=True))
            model.fit(Xtr, ytr)
            if hasattr(model, "feature_importances_"):
                fi = model.feature_importances_
            else:
                # LightGBM always has it; HistGBR also has it
                fi = getattr(model, "feature_importances_", np.zeros(X.shape[1]))
            fi_accum += np.asarray(fi, dtype=float)
            denom += 1
        if denom == 0:
            return {c: 0.0 for c in X.columns}
        fi_mean = fi_accum / max(denom, 1)
        return {c: float(v) for c, v in zip(X.columns, fi_mean)}

    def _compute_permutation_importance(self, X: pd.DataFrame, y: np.ndarray) -> Dict[str, float]:
        if X.shape[1] == 0:
            return {}
        model = self._get_tree_model()
        imp_accum = np.zeros(X.shape[1], dtype=float)
        denom = 0
        for tr, te in self._rolling_splits(X.shape[0]):
            if len(np.unique(y[tr])) < 2:
                continue
            Xtr, Xte = X.iloc[tr].copy(), X.iloc[te].copy()
            ytr, yte = y[tr].copy(), y[te].copy()
            Xtr = Xtr.fillna(Xtr.median(numeric_only=True))
            Xte = Xte.fillna(Xtr.median(numeric_only=True))  # train medians only
            model.fit(Xtr, ytr)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                res = permutation_importance(
                    model, Xte, yte, n_repeats=5, random_state=self.random_state, scoring=self.scorer
                )
            imp_accum += np.maximum(res.importances_mean, 0.0)  # clamp negatives to 0 for stability
            denom += 1
        if denom == 0:
            return {c: 0.0 for c in X.columns}
        imp_mean = imp_accum / max(denom, 1)
        return {c: float(v) for c, v in zip(X.columns, imp_mean)}

    def _combine_and_cap(
        self,
        cols: List[str],
        mi_scores: Dict[str, float],
        model_importance: Dict[str, float],
        perm_importance: Dict[str, float],
    ) -> Tuple[Dict[str, float], List[str]]:
        # Choose weights
        if self.weights is not None:
            w_mi, w_model, w_perm = self.weights
        else:
            if self.mode.lower() == "lstm":
                w_mi, w_model, w_perm = (0.45, 0.10, 0.45)
            else:  # boosting
                w_mi, w_model, w_perm = (0.20, 0.55, 0.25)

        def _norm(d: Dict[str, float]) -> Dict[str, float]:
            vals = np.array([d.get(c, 0.0) for c in cols], dtype=float)
            if not np.isfinite(vals).any():
                return {c: 0.0 for c in cols}
            # min-max normalize (avoid zero-div)
            vmin, vmax = np.nanmin(vals), np.nanmax(vals)
            if vmax <= vmin:
                return {c: 0.0 for c in cols}
            normed = (vals - vmin) / (vmax - vmin)
            return {c: float(v) for c, v in zip(cols, normed)}

        mi_n = _norm(mi_scores)
        md_n = _norm(model_importance)
        pm_n = _norm(perm_importance)

        combined: Dict[str, float] = {}
        for c in cols:
            combined[c] = w_mi * mi_n.get(c, 0.0) + w_model * md_n.get(c, 0.0) + w_perm * pm_n.get(c, 0.0)

        # Rank by combined score
        ranked = sorted(cols, key=lambda c: combined[c], reverse=True)

        # Quantile filter
        if self.score_quantile is not None:
            thr = np.nanquantile([combined[c] for c in cols], self.score_quantile)
            ranked = [c for c in ranked if combined[c] >= thr]

        # Cap
        if self.max_features is not None and len(ranked) > self.max_features:
            ranked = ranked[: self.max_features]

        return combined, ranked

    def _corr_pruning(self, X: pd.DataFrame, scores: Dict[str, float]) -> Tuple[List[str], List[str]]:
        if X.shape[1] <= 1:
            return X.columns.tolist(), []

        # Spearman is robust for monotonic non-linear relations (good default for heterogeneous feats)
        method = "spearman" if self.corr_method.lower().startswith("spear") else "pearson"
        corr = X.corr(method=method).abs()
        # Greedy: iterate by descending score, keep feature if not highly correlated with any kept one
        order = sorted(X.columns, key=lambda c: scores.get(c, 0.0), reverse=True)
        kept, dropped = [], set()
        selected = set()
        for c in order:
            if c in dropped:
                continue
            # compare corr with already kept
            if not kept:
                kept.append(c)
                selected.add(c)
                continue
            high_corr = any(corr.loc[c, k] >= self.corr_threshold for k in kept if k in corr.columns)
            if high_corr:
                dropped.add(c)
            else:
                kept.append(c)
                selected.add(c)
        return kept, [c for c in order if c not in selected]

# ----------------------------- convenience -------------------------------

def run_feature_selection(
    df_daily: pd.DataFrame,
    target_col: str,
    *,
    date_col: str = "fecha",
    mode: str = "boosting",
    forbid_patterns: Optional[Sequence[str]] = None,
    max_features: Optional[int] = 256,
    score_quantile: Optional[float] = 0.50,
    corr_threshold: float = 0.95,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, FeatureSelectorTS, FReport]:
    """
    One-shot feature selection. Returns (X_selected, selector, report).

    Example
    -------
    X_sel, sel, rep = run_feature_selection(df_daily, target_col="merval_total_return_ars",
                                            mode="boosting", forbid_patterns=[r"^merval(_|$)"])
    """
    selector = FeatureSelectorTS(
        date_col=date_col,
        target_col=target_col,
        mode=mode,
        forbid_patterns=forbid_patterns or [],
        max_features=max_features,
        score_quantile=score_quantile,
        corr_threshold=corr_threshold,
        random_state=random_state,
    )
    X_sel = selector.fit_transform(df_daily)
    return X_sel, selector, selector.get_report()
