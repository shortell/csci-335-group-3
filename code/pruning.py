# loads and cleans tweets and stuff. preprocessing.
# resulting csv's to be saved in ../data/clean
#  @author: selene shen

# TODO: Process stocks dataframe

import pandas as pd
import preprocessfunctions as pre
from pathlib import Path


#results saved as in:
cleantwtfp = Path("../data/clean/cleantwt.csv")
cleanqrtfp = Path("../data/clean/cleanqrt.csv")

cleantwtfp.parent.mkdir(parents=True, exist_ok=True)
cleanqrtfp.parent.mkdir(parents=True, exist_ok=True)

# original file locations
fpall = "../data/unzipped/all_musk_posts.csv"
fpqrt = "../data/unzipped/musk_quote_tweets.csv"
fptsla = "../data/unzipped/TSLA_1min_market_hours_2016_2025.csv"



# load csvs into dataframes
alltwt = pd.read_csv(fpall, low_memory=False)
qrts = pd.read_csv(fpqrt, low_memory=False)
stocks = pd.read_csv(fptsla, low_memory=False)

# cleaning tweets ==========================================================================

# all:   just keep full text id, is reply, inreplytousername, created at, is rt, is quote
#        remove all quotes, will use quote tweets for quote tweets
#        likes, etc are removed because i am assuming this will be used on tweets as they come 
#        out, so such numbers are redundant. or smth
# quote: keep tweet text, created at, orig tweet name, quote tweet, quote created at
# stocks:

twtreduced = alltwt[['id', 'fullText', 'createdAt', 'isReply', 'inReplyToUsername', 'isQuote']] # id between quotes and in quotes db might be shared?
qrtreduced = qrts[['musk_tweet_id', 'musk_quote_tweet', 'musk_quote_created_at']] # why the fuck is thsi one snake while the others camel

# preprocess full texts

twtreduced['fullText'] = twtreduced['fullText'].apply(pre.preprocesstwt)
qrtreduced['musk_quote_tweet'] = qrtreduced['musk_quotetweet'].apply(pre.preprocesstwt)


# check for nulls based on stuff. good luck. ugh.

# TESTING PURPOSES!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# Print info and head of raw and reduced df's
pre.printinfo(alltwt, "raw tweets")
pre.printinfo(qrts, "raw quotes")
pre.printinfo(stocks, "raw stocks")
pre.printinfo(twtreduced, "reduced tweets")
pre.printinfo(qrtreduced, "reduced quotes")

# tweet and quotes to csv
twtreduced.to_csv(cleantwtfp)
qrtreduced.to_csv(cleanqrtfp)

# stocks should be able to be kept, just prune the useless stuff. like the fact the symbol for all is tsla lmao
