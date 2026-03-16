# WIP.
# loads and cleans up the csvs into more usable data.
# Currently doesn't actually prune, just loads them into dataframes and prints the info of each df.

import pandas as pd

fpall = "../data/unzipped/all_musk_posts.csv"
fpqrt = "../data/unzipped/musk_quote_tweets.csv"
fptsla = "../data/unzipped/TSLA_1min_market_hours_2016_2025.csv"


# load csvs into dataframes
alltwt = pd.read_csv(fpall, low_memory=False)
qrts = pd.read_csv(fpqrt, low_memory=False)
stocks = pd.read_csv(fptsla, low_memory=False)

print("All tweets raw info: -------------------------------------")
alltwt.info()
print("ex: ===========")
print(alltwt.head)

print("quote tweets raw info: -----------------------------------")
qrts.info()
print("ex: ===========")
print(qrts.head)

print("stocks raw info: -----------------------------------------")
stocks.info()
print("ex: ===========")
print(stocks.head)

# cleaning tweets

# all: remove id, url, twitterurl, rtcount, reply count, like count, quote count, view count, bookmark count, in reply, inreplytoid, conversation id, holy shit
#      just keep full text id, is reply, inreplytousername, created at, is rt, is quote
#      remove all quotes, will use quote tweets for quote tweets
#      likes, etc are removed because i am assuming this will be used on tweets as they come out, so such numbers are redundant. or smth

# quote: keep tweet text, created at, orig tweet name, quote tweet, quote created at

# check for nulls based on stuff. good luck. ugh.

# stocks should be able to be kept, just prune the useless stuff. like the fact the symbol for all is tsla lmao