
import pandas as pd
import re
import html
import os
from typing import Tuple


# ── helpers ────────────────────────────────────────────────────────────────────

URL_RE  = re.compile(r"http\S+")
AT_RE   = re.compile(r"@\w+")
WS_RE   = re.compile(r"\s+")
RT_RE   = re.compile(r"^RT ")

def _vectorized_clean(series: pd.Series) -> pd.Series:
    """Clean a text Series without row-wise apply."""
    s = series.fillna("")
    # html unescape has no vectorized form, but unescape is fast on strings
    # only unescape if any entity present — skip if clean
    if s.str.contains("&", regex=False).any():
        s = s.apply(html.unescape)
    s = s.str.replace(AT_RE,  "",  regex=True)
    s = s.str.replace(WS_RE,  " ", regex=True).str.strip()
    return s

def _set_time_index(df: pd.DataFrame, col: str) -> pd.DataFrame:
    return df.set_index(col, drop=False).sort_index()


# ── pipeline steps ─────────────────────────────────────────────────────────────

def load_data(posts_path=None, quotes_path=None, stock_path=None):
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if posts_path  is None: posts_path  = os.path.join(base, "data", "all_musk_posts.csv")
    if quotes_path is None: quotes_path = os.path.join(base, "data", "musk_quote_tweets.csv")
    if stock_path  is None: stock_path  = os.path.join(base, "data", "TSLA_1min_market_hours_2016_2025.csv")

    posts  = pd.read_csv(posts_path,  low_memory=False)
    quotes = pd.read_csv(quotes_path)
    stock  = pd.read_csv(stock_path)
    return posts, quotes, stock


def localize_timezones(posts, quotes, stock):
    posts  = posts.copy()
    quotes = quotes.copy()
    stock  = stock.copy()

    posts["createdAt"] = (
        pd.to_datetime(posts["createdAt"], utc=True)
        .dt.tz_convert("America/New_York")
    )
    for col in ("orig_tweet_created_at", "musk_quote_created_at"):
        quotes[col] = (
            pd.to_datetime(quotes[col], utc=True)
            .dt.tz_convert("America/New_York")
        )
    ts = pd.to_datetime(stock["timestamp"])
    if ts.dt.tz is None:
        stock["timestamp"] = ts.dt.tz_localize(
            "America/New_York", nonexistent="shift_forward", ambiguous="NaT"
        )
    else:
        stock["timestamp"] = ts.dt.tz_convert("America/New_York")

    return posts, quotes, stock


def filter_to_overlapping_date_range(posts, quotes, stock):
    date_min = stock["timestamp"].min()
    date_max = stock["timestamp"].max()
    posts  = posts[posts["createdAt"].between(date_min, date_max)]
    quotes = quotes[quotes["musk_quote_created_at"].between(date_min, date_max)]
    return posts, quotes, stock


def drop_replies_and_retweets(posts):
    return posts[
        (posts["isReply"]   != True) &
        (posts["isRetweet"] != True) &
        (~posts["fullText"].str.startswith("RT "))
    ]


def filter_to_market_hours_only(posts, quotes, stock):
    def between_market(df, col, start, end):
        idx = pd.DatetimeIndex(df[col]).tz_convert("America/New_York")
        mask = idx.indexer_between_time(start, end)
        return df.iloc[mask]

    stock  = between_market(stock,  "timestamp",            "09:30", "16:00")
    posts  = between_market(posts,  "createdAt",            "09:30", "15:45")
    quotes = between_market(quotes, "musk_quote_created_at","09:30", "15:45")

    return posts.reset_index(drop=True), quotes.reset_index(drop=True), stock.reset_index(drop=True)


def filter_urls(posts, quotes):
    # Vectorized URL detection — no row-wise apply
    def has_url(series):
        return series.str.contains(URL_RE, na=False)

    quotes = quotes[~has_url(quotes["musk_quote_tweet"]) & ~has_url(quotes["orig_tweet_text"])].copy()

    non_quote_mask  = (posts["isQuote"] != True) & ~has_url(posts["fullText"])
    quote_ids       = set(quotes["musk_tweet_id"])
    quote_mask      = (posts["isQuote"] == True) & posts["id"].isin(quote_ids)

    posts = (
        pd.concat([posts[non_quote_mask], posts[quote_mask]], ignore_index=True)
        .sort_values("createdAt")
        .drop_duplicates(subset="id")
    )
    return posts, quotes


def build_clean_text_and_drop_short(posts, quotes):
    quotes = quotes.copy()
    quotes["cleanMuskText"] = _vectorized_clean(quotes["musk_quote_tweet"])
    quotes["cleanOrigText"] = _vectorized_clean(quotes["orig_tweet_text"])
    quotes["combinedText"]  = (quotes["cleanMuskText"] + " " + quotes["cleanOrigText"]).str.strip()

    quote_text_map = quotes.set_index("musk_tweet_id")["combinedText"]

    posts = posts.copy()
    clean_full = _vectorized_clean(posts["fullText"])

    # Map quote tweets via merge instead of row-wise apply
    is_quote = posts["isQuote"] == True
    posts["cleanText"] = clean_full
    posts.loc[is_quote, "cleanText"] = (
        posts.loc[is_quote, "id"].map(quote_text_map).fillna(clean_full[is_quote])
    )

    posts = posts[posts["cleanText"].str.len() >= 15]
    return posts, quotes


def enforce_15m_tweet_isolation(posts, quotes):
    posts = posts.sort_values("createdAt").copy()
    t = posts["createdAt"]
    gap_before = t.diff().dt.total_seconds().div(60).fillna(float("inf"))
    gap_after  = t.diff(periods=-1).dt.total_seconds().div(-60).fillna(float("inf"))

    posts  = posts[(gap_before > 15.0) & (gap_after > 15.0)].copy()
    quotes = quotes[quotes["musk_tweet_id"].isin(posts["id"])].copy()
    return posts, quotes


def align_to_active_trading_days(posts, quotes, stock):
    # Compute all day keys once each
    stock_days       = pd.DatetimeIndex(stock["timestamp"]).normalize()
    post_days        = pd.DatetimeIndex(posts["createdAt"]).normalize()
    quote_days       = pd.DatetimeIndex(quotes["musk_quote_created_at"]).normalize()

    trading_day_set  = set(stock_days)
    active_tweet_set = set(post_days)

    posts   = posts[post_days.isin(trading_day_set)]
    quotes  = quotes[quote_days.isin(trading_day_set)]
    stock   = stock[stock_days.isin(active_tweet_set)]
    return posts, quotes, stock


# ── main ───────────────────────────────────────────────────────────────────────

def run_pipeline():
    posts, quotes, stock = load_data()
    print(posts[posts["createdAt"].str.contains("2023-07-12")]["createdAt"].values)
    posts, quotes, stock = localize_timezones(posts, quotes, stock)
    print(posts[posts["createdAt"].dt.strftime("%Y-%m-%d") == "2023-07-12"]["createdAt"].values)
    posts, quotes, stock = filter_to_overlapping_date_range(posts, quotes, stock)
    posts                = drop_replies_and_retweets(posts)
    posts, quotes, stock = filter_to_market_hours_only(posts, quotes, stock)
    posts, quotes        = filter_urls(posts, quotes)
    posts, quotes        = build_clean_text_and_drop_short(posts, quotes)
    posts, quotes        = enforce_15m_tweet_isolation(posts, quotes)
    posts, quotes, stock = align_to_active_trading_days(posts, quotes, stock)
    return posts, quotes, stock