import sys
import os

# Add the code/ directory to the path so sibling modules resolve correctly
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from clean import run_pipeline
from tweet_sentiment import run_sentiment_pipeline

def main():
    # Pipeline configuration
    K_VALUE = 10
    INCLUDE_REPLIES = True
    
    print(f"============================================================")
    print(f" STARTING DATA PIPELINE (k={K_VALUE}, include_replies={INCLUDE_REPLIES})")
    print(f"============================================================\n")
    
    # ---------------------------------------------------------
    # 1. Cleaning & Feature Extraction (Includes temporal features)
    # ---------------------------------------------------------
    print(">>> STEP 1: Running Data Cleaning & Feature Extraction...")
    
    # run_pipeline will re-parse all tweets, filter them, and extract our new 
    # 'hour' and 'day_of_week' features, then save it into data/cleaned/
    run_pipeline(k=K_VALUE, include_replies=INCLUDE_REPLIES, save_csv=True)
    
    cleaned_file_path = f"data/cleaned/musk_events_k{K_VALUE}_replies_{INCLUDE_REPLIES}.csv"
    
    # ---------------------------------------------------------
    # 2. RoBERTa Sentiment Analysis
    # ---------------------------------------------------------
    print(f"\n>>> STEP 2: Running HuggingFace Sentiment Analysis...")
    print(f"Targeting cleaned file: {cleaned_file_path}")
    
    # run_sentiment_pipeline loads the cleaned dataset, scores it with RoBERTa in batches,
    # and then outputs the absolute final version into data/final/
    run_sentiment_pipeline(cleaned_file_path)
    
    print("\n============================================================")
    print(" PIPELINE COMPLETE!")
    print(f" Your final, fully-featured dataset is ready at:")
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    print(f" data/final/musk_events_k{K_VALUE}_replies_{INCLUDE_REPLIES}.csv (relative to {root})")
    print("============================================================")

if __name__ == "__main__":
    main()
