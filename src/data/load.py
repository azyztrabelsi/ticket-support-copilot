import pandas as pd

def load_bitext():
    print("Downloading Bitext dataset directly...")
    # Direct URL to the dataset parquet file on HuggingFace
    url = "hf://datasets/bitext/Bitext-customer-support-llm-chatbot-training-dataset@main/Bitext_Sample_Customer_Support_Training_Dataset_27K_responses-v11.csv"
    
    # Load it directly into pandas (requires huggingface_hub which is already installed)
    df = pd.read_csv(url)
    print(f"Loaded {len(df)} rows.")
    return df