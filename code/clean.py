import pandas as pd
import re
import html
import os


#  helpers 

URL_RE = re.compile(r"http\S+")
AT_RE  = re.compile(r"@\w+")          # removes every @handle including mid-text
WS_RE  = re.compile(r"\s+")
RT_RE  = re.compile(r"^RT\s+")        # strips leading "RT " marker
TESLA_RE = re.compile(r"\b(tesla|tsla)\b", re.IGNORECASE)

def _vectorized_clean(series: pd.Series) -> pd.Series:
    """
    Clean a text Series (vectorized):
      - html-unescape &amp; etc.
      - strip all URLs
      - strip all @handles (anywhere in the text)
      - strip leading 'RT ' retweet marker
      - collapse whitespace
    """
    s = series.fillna("")
    # html unescape only if needed
    if s.str.contains("&", regex=False).any():
        s = s.apply(html.unescape)
    s = s.str.replace(URL_RE, "", regex=True)
    s = s.str.replace(AT_RE,  "", regex=True)
    s = s.str.replace(RT_RE,  "", regex=True)
    s = s.str.replace(WS_RE,  " ", regex=True).str.strip()
    return s


def _to_naive_eastern(ts: pd.Series) -> pd.Series:
    """
    Normalize any timestamp Series to tz-naive America/New_York wall-clock time.
    This is used exclusively for the stock-lookup search so both sides of the
    searchsorted are in the same dtype (no tz vs tz mismatch).
    """
    if ts.dt.tz is not None:
        return ts.dt.tz_convert("America/New_York").dt.tz_localize(None)
    return ts


# pipeline steps 

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
    """
    Convert all timestamps to tz-aware America/New_York.
    Stock data comes in as tz-naive Eastern (Alpaca API) and is localized.
    Posts/quotes are stored as UTC strings and are converted.
    """
    posts  = posts.copy()
    quotes = quotes.copy()
    stock  = stock.copy()

    # Posts: UTC string -> tz-aware Eastern
    posts["createdAt"] = (
        pd.to_datetime(posts["createdAt"], utc=True)
        .dt.tz_convert("America/New_York")
    )

    # Quotes: both timestamp columns
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


def drop_bad_rows(posts, quotes):
    """
    Drop any post that is missing id or createdAt — these rows cannot be
    matched to stock data or used for downstream tasks.
    Also drop any quotes missing their musk_tweet_id or timestamp.
    """
    posts  = posts.dropna(subset=["id", "createdAt"]).copy()
    quotes = quotes.dropna(subset=["musk_tweet_id", "musk_quote_created_at"]).copy()
    # id must be a valid integer (no float NaN sneaking through)
    posts["id"] = posts["id"].astype("int64")
    quotes["musk_tweet_id"] = quotes["musk_tweet_id"].astype("int64")
    return posts, quotes


def filter_to_overlapping_date_range(posts, quotes, stock):
    date_min = stock["timestamp"].min()
    date_max = stock["timestamp"].max()
    posts  = posts[posts["createdAt"].between(date_min, date_max)]
    quotes = quotes[quotes["musk_quote_created_at"].between(date_min, date_max)]
    return posts, quotes, stock


def filter_to_market_hours_only(posts, quotes, stock):
    def between_market(df, col, start, end):
        idx  = pd.DatetimeIndex(df[col]).tz_convert("America/New_York")
        mask = idx.indexer_between_time(start, end)
        return df.iloc[mask]

    # Stock: full trading session kept for the 15-min window lookup
    stock  = between_market(stock,  "timestamp",             "09:30", "16:00")
    # Tweets: start at 09:45 (ensures ≥15 min pre-tweet baseline in full TSLA data)
    # and end at 15:45 so every tweet has at least 15 min left in session
    posts  = between_market(posts,  "createdAt",             "09:45", "15:45")
    quotes = between_market(quotes, "musk_quote_created_at", "09:45", "15:45")

    return (
        posts.reset_index(drop=True),
        quotes.reset_index(drop=True),
        stock.reset_index(drop=True),
    )


def filter_urls(posts, quotes):
    """
    Remove quote-tweet rows where both Musk's comment AND the original tweet
    are empty after cleaning (i.e. the tweet was purely a URL).
    Plain posts are all kept regardless.
    """
    quotes = quotes.copy()
    quotes["_clean_musk"] = _vectorized_clean(quotes["musk_quote_tweet"])
    quotes["_clean_orig"] = _vectorized_clean(quotes["orig_tweet_text"])

    quotes = quotes[
        (quotes["_clean_musk"].str.len() > 0) |
        (quotes["_clean_orig"].str.len() > 0)
    ].drop(columns=["_clean_musk", "_clean_orig"])

    quote_ids = set(quotes["musk_tweet_id"])
    non_quote_mask = posts["isQuote"] != True
    quote_mask     = (posts["isQuote"] == True) & posts["id"].isin(quote_ids)
    posts = (
        pd.concat([posts[non_quote_mask], posts[quote_mask]], ignore_index=True)
        .sort_values("createdAt")
        .drop_duplicates(subset="id")
    )
    return posts, quotes


def build_clean_text_and_drop_short(posts, quotes, k: int = 15):
    """
    Build cleanText per post:
      - Non-quote tweets: clean(fullText)
      - Quote tweets: clean(originalTweet) + " " + clean(muskComment)
      - Replies/retweets: prepend context text if available in the dataset

    Posts whose cleaned Musk text is shorter than k words are dropped.
    """
    quotes = quotes.copy()
    quotes["cleanMuskText"] = _vectorized_clean(quotes["musk_quote_tweet"])
    quotes["cleanOrigText"] = _vectorized_clean(quotes["orig_tweet_text"])
    quotes["combinedText"]  = (
        quotes["cleanOrigText"] + " " + quotes["cleanMuskText"]
    ).str.strip()
    quote_text_map = quotes.set_index("musk_tweet_id")["combinedText"]

    posts = posts.copy()
    clean_full = _vectorized_clean(posts["fullText"])
    posts["cleanText"] = clean_full

    # Quote tweets: replace with combined original + Musk comment
    is_quote = posts["isQuote"] == True
    posts.loc[is_quote, "cleanText"] = (
        posts.loc[is_quote, "id"]
            .map(quote_text_map)
            .fillna(clean_full[is_quote])
    )

    # Replies: prepend replied-to text if the column exists in the dataset
    if "inReplyToText" in posts.columns:
        is_reply = posts["isReply"] == True
        reply_ctx = _vectorized_clean(posts.loc[is_reply, "inReplyToText"])
        posts.loc[is_reply, "cleanText"] = (
            reply_ctx + " " + posts.loc[is_reply, "cleanText"]
        ).str.strip()

    # Retweets: prepend original tweet text if available
    if "retweetedText" in posts.columns:
        is_rt = posts["isRetweet"] == True
        rt_ctx = _vectorized_clean(posts.loc[is_rt, "retweetedText"])
        posts.loc[is_rt, "cleanText"] = (
            rt_ctx + " " + posts.loc[is_rt, "cleanText"]
        ).str.strip()

    # Drop short posts (word count based on Musk's own text only)
    word_counts = clean_full.str.split().str.len().fillna(0)
    posts = posts[word_counts >= k]

    return posts, quotes


def enforce_tweet_isolation(posts, quotes, gap_minutes=5):
    """
    Keep only tweets that have no other Musk tweet within a specified 
    minute window before or after. 
    
    A 5-minute gap allows for more data but may result in overlapping 
    stock price reactions if tweets are sent in a 'thread.'
    """
    posts = posts.sort_values("createdAt").copy()
    t = posts["createdAt"]
    
    # Calculate time difference to the previous and next tweet in minutes
    gap_before = t.diff().dt.total_seconds().div(60).fillna(float("inf"))
    gap_after  = t.diff(periods=-1).dt.total_seconds().div(-60).fillna(float("inf"))

    # Apply the new 5-minute filter
    posts  = posts[(gap_before > float(gap_minutes)) & (gap_after > float(gap_minutes))].copy()
    quotes = quotes[quotes["musk_tweet_id"].isin(posts["id"])].copy()
    
    return posts, quotes


def align_to_active_trading_days(posts, quotes, stock):
    stock_days       = pd.DatetimeIndex(stock["timestamp"]).normalize()
    post_days        = pd.DatetimeIndex(posts["createdAt"]).normalize()
    quote_days       = pd.DatetimeIndex(quotes["musk_quote_created_at"]).normalize()

    trading_day_set  = set(stock_days)
    active_tweet_set = set(post_days)

    posts   = posts[post_days.isin(trading_day_set)]
    quotes  = quotes[quote_days.isin(trading_day_set)]
    stock   = stock[stock_days.isin(active_tweet_set)]
    return posts, quotes, stock


# output builder 

# TSLA candle columns to include for each minute offset
_STOCK_COLS = ["close", "volume", "trade_count"]

# Update these constants at the top of your output builder section
_OFFSETS = [1, 2, 4] 

def build_output_csv(posts: pd.DataFrame, quotes: pd.DataFrame, stock: pd.DataFrame) -> pd.DataFrame:
    """
    Output one row per tweet with:
      - tweet metadata & mentions_tesla
      - t0: actual baseline close and volume
      - t1, t2, t4: binary labels (1 if > t0, else 0)
    """
    posts = posts.copy().reset_index(drop=True)
    stock_sorted = stock.sort_values("timestamp").reset_index(drop=True)
    
    # Pre-calculate naive timestamps for fast searching
    stock_ts_naive = _to_naive_eastern(stock_sorted["timestamp"])
    tweet_ts_naive = _to_naive_eastern(posts["createdAt"])

    stock_rows = []
    for tweet_time in tweet_ts_naive:
        tweet_min = tweet_time.floor("min")
        row_dict = {}

        # get t0 baseline the minute of the tweet 
        idx_0 = stock_ts_naive.searchsorted(tweet_min, side="left")
        
        if idx_0 < len(stock_sorted) and stock_ts_naive[idx_0] == tweet_min:
            t0_bar = stock_sorted.iloc[idx_0]
            base_price = t0_bar["close"]
            base_vol = t0_bar["volume"]
            
            row_dict["stock_t0_close"] = base_price
            row_dict["stock_t0_volume"] = base_vol
            
            # Calculate Binary Labels for Offsets
            for offset in _OFFSETS:
                target_time = tweet_min + pd.Timedelta(minutes=offset)
                idx_n = stock_ts_naive.searchsorted(target_time, side="left")
                
                if idx_n < len(stock_sorted) and stock_ts_naive[idx_n] == target_time:
                    tn_bar = stock_sorted.iloc[idx_n]
                    # Binary comparison: 1 if increased, 0 if decreased or flat
                    row_dict[f"stock_t{offset}_price_up"] = int(tn_bar["close"] > base_price)
                    row_dict[f"stock_t{offset}_volume_up"] = int(tn_bar["volume"] > base_vol)
                else:
                    row_dict[f"stock_t{offset}_price_up"] = float("nan")
                    row_dict[f"stock_t{offset}_volume_up"] = float("nan")
        else:

            row_dict["stock_t0_close"] = float("nan")
            row_dict["stock_t0_volume"] = float("nan")

        stock_rows.append(row_dict)

    # Combine metadata with the new stock features
    stock_df = pd.DataFrame(stock_rows, index=posts.index)
    
    out = pd.DataFrame({
        "row_id":          range(len(posts)),
        "tweet_id":        posts["id"].values,
        "tweet_timestamp": posts["createdAt"].values,
        "cleanText":       posts["cleanText"].values,
        "mentions_tesla":  (posts["cleanText"].str.contains(TESLA_RE, na=False)).astype(int),
    })

    result = pd.concat([out, stock_df.reset_index(drop=True)], axis=1)
    
    result = result.dropna().reset_index(drop=True)
    result["row_id"] = range(len(result))
    
    return result



def run_pipeline(k: int = 15, save_csv: bool = True):
    posts, quotes, stock = load_data()
    posts, quotes, stock = localize_timezones(posts, quotes, stock)
    posts, quotes        = drop_bad_rows(posts, quotes)
    posts, quotes, stock = filter_to_overlapping_date_range(posts, quotes, stock)
    posts, quotes, stock = filter_to_market_hours_only(posts, quotes, stock)
    posts, quotes        = filter_urls(posts, quotes)
    posts, quotes        = build_clean_text_and_drop_short(posts, quotes, k=k)
    posts, quotes        = enforce_tweet_isolation(posts, quotes)
    posts, quotes, stock = align_to_active_trading_days(posts, quotes, stock)

    n_samples = len(posts)
    print(f"Total samples (k={k}): {n_samples}")

    output = build_output_csv(posts, quotes, stock)

    if save_csv:
        base     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        out_dir  = os.path.join(base, "data", "cleaned")
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, "pipeline_output.csv")
        output.to_csv(out_path, index=False)
        print(f"Saved {len(output)} rows -> {out_path}")
        print(f"Columns ({len(output.columns)}): {list(output.columns)}")

    return posts, quotes, stock, output


if __name__ == "__main__":
    run_pipeline(k=10, save_csv=True)