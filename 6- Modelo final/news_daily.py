# news_daily.py
# Time-series daily feature agg from news.
# Python 3.10+ recommended

import ast
import math
from typing import Iterable, Sequence, Optional, Dict, List, Union, Any
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class NewsDailyAggregator(BaseEstimator, TransformerMixin):
    """
    Daily feature aggregator for news-derived LLM features with robust imputations.

    Features
    --------
    - Groups by a date column (default: 'fecha')
    - Numeric columns: count, min, max, mean, median, std + configurable percentiles
      * Optional per-column numeric imputation (e.g., horizonte_dias)
    - Boolean columns: true-rate, true-count, false-count, any_true, all_true
    - Object columns (e.g., 'categoria_fuente'):
        nunique, notnull_rate, and optional top-k value rate features (learned on fit)
        * Optional object imputation (e.g., categoria_fuente='desconocido')
    - List-like columns (e.g., 'empresas_mencionadas_list', 'personas_mencionadas'):
        has_value_rate, items_count_{sum,mean,max}, unique_items_count (set union per day),
        optional top-k item rate features (learned on fit)
        * Coerces strings to lists (literal_eval or comma-split); nulls → empty list by default

    Notes
    -----
    - fit() freezes schema, imputers, and any top-k vocabularies for transform().
    - Outputs are numeric and modeling-ready; 'fecha' is kept as a column by default.
    """

    def __init__(
        self,
        date_col: str = "fecha",
        percentiles: Sequence[float] = (0.1, 0.25, 0.5, 0.75, 0.9),

        # --- Column role hints / handling ---
        list_like_cols: Optional[Sequence[str]] = (
            "empresas_mencionadas_list",
            "personas_mencionadas",
        ),

        # --- Imputation configs ---
        # e.g. {"horizonte_dias": 0, "score_fuente": "median"}
        numeric_impute_map: Optional[Dict[str, Union[str, float, int]]] = (
            {"horizonte_dias": 0}
        ),
        # e.g. {"categoria_fuente": "desconocido"}
        object_impute_map: Optional[Dict[str, Any]] = (
            {"categoria_fuente": "desconocido"}
        ),
        list_null_as_empty: bool = True,
        coerce_listlike: bool = True,  # try to parse strings into lists

        # --- Expansion knobs (kept off by default to avoid feature explosion) ---
        object_topk: int = 0,                       # top-k value rates for selected object cols
        object_topk_includes: Optional[Sequence[str]] = ("categoria_fuente",),
        object_topk_cardinality_max: int = 30,

        list_topk: int = 0,                         # top-k item rates for list-like cols
        list_topk_cardinality_max: int = 2000,      # skip if too many unique items

        add_count_false: bool = True,
        keep_date_as: str = "column",  # "column" or "index"
        datetime_infer: bool = True,
        observed: bool = True
    ):
        self.date_col = date_col
        self.percentiles = tuple(sorted(set([float(p) for p in percentiles if 0.0 < p < 1.0])))
        self.list_like_cols = tuple(list_like_cols) if list_like_cols is not None else tuple()

        self.numeric_impute_map = dict(numeric_impute_map) if numeric_impute_map else {}
        self.object_impute_map = dict(object_impute_map) if object_impute_map else {}

        self.list_null_as_empty = bool(list_null_as_empty)
        self.coerce_listlike = bool(coerce_listlike)

        self.object_topk = int(object_topk)
        self.object_topk_includes = tuple(object_topk_includes) if object_topk_includes else tuple()
        self.object_topk_cardinality_max = int(object_topk_cardinality_max)

        self.list_topk = int(list_topk)
        self.list_topk_cardinality_max = int(list_topk_cardinality_max)

        self.add_count_false = bool(add_count_false)
        self.keep_date_as = keep_date_as
        self.datetime_infer = bool(datetime_infer)
        self.observed = bool(observed)

        # learned on fit()
        self.fitted_: bool = False
        self.num_cols_: List[str] = []
        self.bool_cols_: List[str] = []
        self.obj_cols_: List[str] = []
        self.list_cols_: List[str] = []

        # learned imputers / vocabularies
        self._num_impute_values_: Dict[str, float] = {}
        self._obj_impute_values_: Dict[str, Any] = {}
        self._obj_top_values_: Dict[str, List[Any]] = {}
        self._list_top_items_: Dict[str, List[Any]] = {}

        self._feature_names_out_: List[str] = []

    # -------------------------- public API --------------------------

    def fit(self, df: pd.DataFrame, y=None):
        self._validate_input_df(df, require_date=True)

        # ----- role inference -----
        self.bool_cols_ = [c for c in df.columns
                           if c != self.date_col and pd.api.types.is_bool_dtype(df[c])]
        numeric_candidates = [c for c in df.columns
                              if c != self.date_col and pd.api.types.is_numeric_dtype(df[c])]
        self.num_cols_ = [c for c in numeric_candidates if c not in self.bool_cols_]
        self.obj_cols_ = [c for c in df.columns
                          if c != self.date_col and pd.api.types.is_object_dtype(df[c])]

        # list-like: prefer explicit config; otherwise detect
        if self.list_like_cols:
            self.list_cols_ = [c for c in self.list_like_cols if c in df.columns]
        else:
            self.list_cols_ = []
            for c in self.obj_cols_:
                sample = df[c].dropna().head(10).tolist()
                if any(isinstance(v, (list, tuple, set)) for v in sample):
                    self.list_cols_.append(c)

        # object columns excluding list-like
        self.obj_cols_ = [c for c in self.obj_cols_ if c not in self.list_cols_]

        # ----- learn imputers -----
        # numeric: resolve strategies into concrete values
        self._num_impute_values_ = {}
        for col, strat in self.numeric_impute_map.items():
            if col not in df.columns:
                continue
            if isinstance(strat, (int, float)):
                self._num_impute_values_[col] = float(strat)
            elif isinstance(strat, str):
                s = strat.lower()
                if s == "zero":
                    self._num_impute_values_[col] = 0.0
                elif s == "mean":
                    self._num_impute_values_[col] = float(pd.to_numeric(df[col], errors="coerce").mean())
                elif s == "median":
                    self._num_impute_values_[col] = float(pd.to_numeric(df[col], errors="coerce").median())
                elif s.startswith("constant:"):
                    v = float(s.split("constant:")[1].strip())
                    self._num_impute_values_[col] = v
                else:
                    raise ValueError(f"Unknown numeric impute strategy for {col}: {strat}")
        # object: fixed fill values
        self._obj_impute_values_ = {col: val for col, val in self.object_impute_map.items() if col in df.columns}

        # ----- learn top-k vocabularies -----
        # objects (e.g., categoria_fuente)
        self._obj_top_values_.clear()
        if self.object_topk > 0 and self.object_topk_includes:
            for col in self.object_topk_includes:
                if col in self.obj_cols_:
                    card = df[col].nunique(dropna=True)
                    if 0 < card <= self.object_topk_cardinality_max:
                        top_vals = (
                            df[col].dropna()
                            .astype(str)  # robust match on transform
                            .value_counts()
                            .head(self.object_topk)
                            .index.tolist()
                        )
                        self._obj_top_values_[col] = top_vals

        # list-like (e.g., empresas/personas)
        self._list_top_items_.clear()
        if self.list_topk > 0 and self.list_cols_:
            for col in self.list_cols_:
                series = self._coerce_list_column(df[col]) if self.coerce_listlike else df[col]
                exploded = (
                    series.dropna()
                    .apply(lambda x: list(x) if isinstance(x, (list, tuple, set)) else [])
                    .explode()
                )
                if exploded.notna().any():
                    card = exploded.nunique(dropna=True)
                    if 0 < card <= self.list_topk_cardinality_max:
                        top_items = (
                            exploded.dropna()
                            .astype(str)
                            .value_counts()
                            .head(self.list_topk)
                            .index.tolist()
                        )
                        self._list_top_items_[col] = top_items

        self.fitted_ = True
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        self._ensure_fitted()
        self._validate_input_df(df, require_date=True)

        X = df.copy()

        # date coercion
        if self.datetime_infer and not pd.api.types.is_datetime64_any_dtype(X[self.date_col]):
            X[self.date_col] = pd.to_datetime(X[self.date_col], errors="coerce")

        # ----- apply imputations BEFORE grouping -----
        # numeric
        for col, val in self._num_impute_values_.items():
            if col in X.columns:
                X[col] = pd.to_numeric(X[col], errors="coerce").fillna(val)

        # object
        for col, val in self._obj_impute_values_.items():
            if col in X.columns:
                X[col] = X[col].where(~X[col].isna(), val)

        # list-like coercion + null→empty
        for col in self.list_cols_:
            if col in X.columns:
                if self.coerce_listlike:
                    X[col] = self._coerce_list_column(X[col])
                if self.list_null_as_empty:
                    X[col] = X[col].apply(lambda x: [] if (x is None or (isinstance(x, float) and math.isnan(x))) else x)

        g = X.groupby(self.date_col, observed=self.observed, sort=True)

        parts: List[pd.DataFrame] = []

        # news count
        n_news = g.size().to_frame("n_news")

        # numeric stats
        if self.num_cols_:
            num_basic = g[self.num_cols_].agg(["count", "min", "max", "mean", "median", "std"])
            num_basic = self._flatten_cols(num_basic, level_order=("stat", "col"), rename_rule="{col}__{stat}")
            pct_frames = []
            for p in self.percentiles:
                qdf = g[self.num_cols_].quantile(p, interpolation="linear")
                qdf.columns = [f"{c}__p{int(round(p*100)):02d}" for c in qdf.columns]
                pct_frames.append(qdf)
            num_all = pd.concat([num_basic] + pct_frames, axis=1) if pct_frames else num_basic
            parts.append(num_all)

        # boolean stats
        if self.bool_cols_:
            # base aggregates
            rate_true = g[self.bool_cols_].mean()
            rate_true.columns = [f"{c}__rate_true" for c in self.bool_cols_]

            count_true = g[self.bool_cols_].sum().astype("int32")
            count_true.columns = [f"{c}__count_true" for c in self.bool_cols_]

            # reuse daily news count as a Series aligned by day
            n = n_news["n_news"].reindex(count_true.index)

            # count_false in one shot (broadcast across columns)
            count_false = count_true.rsub(n, axis=0)  # == n - count_true
            count_false.columns = [c.replace("__count_true", "__count_false") for c in count_true.columns]
            count_false = count_false.astype("int32")

            # any_true / all_true without extra groupby
            any_true = (count_true > 0).astype("int8")
            any_true.columns = [c.replace("__count_true", "__any_true") for c in count_true.columns]

            all_true = count_true.eq(n, axis=0).astype("int8")
            all_true.columns = [c.replace("__count_true", "__all_true") for c in count_true.columns]

            parts.append(pd.concat([rate_true, count_true, count_false, any_true, all_true], axis=1))

        # object stats (faster notnull_rate)
        if self.obj_cols_:
            obj_nunique = g[self.obj_cols_].nunique(dropna=True)
            obj_nunique.columns = [f"{c}__nunique" for c in self.obj_cols_]

            obj_notnull_cnt = g[self.obj_cols_].count()                    # non-null count per day
            obj_notnull_rate = obj_notnull_cnt.div(n_news["n_news"], axis=0)
            obj_notnull_rate.columns = [f"{c}__notnull_rate" for c in self.obj_cols_]

            obj_frames = [obj_nunique, obj_notnull_rate]
            # (keep your optional top-k code here)
            parts.append(pd.concat(obj_frames, axis=1))

        # list-like stats + optional top-k item rates
        if self.list_cols_:
            list_frames = []
            for c in self.list_cols_:
                # has any value?
                has_val = X[c].apply(lambda x: isinstance(x, (list, tuple, set)) and len(x) > 0)
                rate_has_val = has_val.groupby(X[self.date_col]).mean().to_frame(f"{c}__has_value_rate")

                # lengths
                lengths = X[c].apply(lambda x: len(x) if isinstance(x, (list, tuple, set)) else 0)
                glen = lengths.groupby(X[self.date_col])
                length_stats = pd.concat(
                    [
                        glen.sum().rename(f"{c}__items_count_sum"),
                        glen.mean().rename(f"{c}__items_count_mean"),
                        glen.max().rename(f"{c}__items_count_max"),
                    ],
                    axis=1,
                )

                # unique items in the day (set union)
                def _unique_items_count(series):
                    acc = set()
                    for item in series:
                        if isinstance(item, (list, tuple, set)):
                            for x in item:
                                try:
                                    acc.add(str(x))
                                except Exception:
                                    acc.add(repr(x))
                    return len(acc)

                unique_items = g[c].apply(_unique_items_count).to_frame(f"{c}__unique_items_count")

                pieces = [rate_has_val, length_stats, unique_items]

                # optional: top-k items learned on fit()
                top_items = self._list_top_items_.get(c, [])
                if top_items:
                    for it in top_items:
                        contains_it = X[c].apply(lambda lst: (isinstance(lst, (list, tuple, set)) and any(str(u) == str(it) for u in lst)))
                        rate_it = contains_it.groupby(X[self.date_col]).mean().to_frame(f"{c}__rate_item::{str(it)[:32]}")
                        pieces.append(rate_it)

                list_frames.append(pd.concat(pieces, axis=1))

            parts.append(pd.concat(list_frames, axis=1))

        # merge parts
        out = pd.concat([n_news] + parts if parts else [n_news], axis=1, copy=False).sort_index()
        out = out.loc[:, sorted(out.columns)]

        if self.keep_date_as == "column":
            out = out.reset_index().rename(columns={self.date_col: "fecha"})
        else:
            out.index.name = self.date_col

        self._feature_names_out_ = [c for c in out.columns if c != "fecha"]
        return out

    def fit_transform(self, df: pd.DataFrame, y=None) -> pd.DataFrame:
        return self.fit(df, y=y).transform(df)

    def get_feature_names_out(self) -> List[str]:
        self._ensure_fitted()
        return list(self._feature_names_out_)

    # -------------------------- helpers --------------------------

    def _coerce_list_column(self, s: pd.Series) -> pd.Series:
        """
        Convert series entries into lists:
        - If already list/tuple/set -> list(x)
        - If string representing Python list/tuple -> ast.literal_eval
        - Else if comma-separated string -> split by comma
        - NaNs/None -> [] (if list_null_as_empty=True) or leave as NaN
        """
        def _to_list(x):
            if isinstance(x, list):
                return x
            if isinstance(x, (tuple, set)):
                return list(x)
            if x is None or (isinstance(x, float) and math.isnan(x)):
                return [] if self.list_null_as_empty else np.nan
            if isinstance(x, str):
                xs = x.strip()
                # try literal_eval for "['a','b']" or "('a','b')"
                if (xs.startswith("[") and xs.endswith("]")) or (xs.startswith("(") and xs.endswith(")")):
                    try:
                        val = ast.literal_eval(xs)
                        if isinstance(val, (list, tuple, set)):
                            return list(val)
                    except Exception:
                        pass
                # fallback: comma-split
                if "," in xs:
                    return [t.strip() for t in xs.split(",") if t.strip()]
                # single token string -> treat as single-item list
                return [xs] if xs else ([] if self.list_null_as_empty else np.nan)
            # unknown type -> keep as empty or NaN
            return [] if self.list_null_as_empty else np.nan

        return s.apply(_to_list)

    def _flatten_cols(self, df: pd.DataFrame, level_order=("stat", "col"), rename_rule="{col}__{stat}") -> pd.DataFrame:
        if not isinstance(df.columns, pd.MultiIndex):
            return df
        try:
            df = df.copy()
            df.columns = df.columns.reorder_levels(level_order)
            new_cols = []
            for tup in df.columns:
                mapping = {"stat": tup[level_order.index("stat")], "col": tup[level_order.index("col")]}
                new_cols.append(rename_rule.format(**mapping))
            df.columns = new_cols
            return df
        except Exception:
            df = df.copy()
            df.columns = [f"{c[0]}__{c[1]}" for c in df.columns]
            return df

    def _validate_input_df(self, df: pd.DataFrame, require_date: bool = True):
        if not isinstance(df, pd.DataFrame):
            raise TypeError("Input must be a pandas DataFrame.")
        if require_date and self.date_col not in df.columns:
            raise ValueError(f"date_col '{self.date_col}' not found in DataFrame.")
        if require_date and self.datetime_infer:
            _ = pd.to_datetime(df[self.date_col], errors="coerce")

    def _ensure_fitted(self):
        if not self.fitted_:
            raise RuntimeError("Call fit() before transform().")
