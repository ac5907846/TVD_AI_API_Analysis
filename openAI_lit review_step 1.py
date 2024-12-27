import pandas as pd
import requests
import json
from datetime import datetime
import os
from dotenv import load_dotenv
from time import sleep

def setup_openai():
    load_dotenv()
    return {
        "url": "https://api.openai.com/v1/chat/completions",
        "headers": {
            "Authorization": f"Bearer {os.getenv('OPENAI_API_KEY')}",
            "Content-Type": "application/json"
        }
    }

def classify_paper(title, abstract, api_config):
    prompt = f"""Classify this academic paper into one of these categories:

1. Teaching Case Study: Papers SPECIFICALLY developed as classroom teaching tools that:
   - Present real scenarios with specific dilemmas for students to solve
   - Include learning objectives and discussion questions
   - Structure content for classroom use
   - Are designed for student engagement and learning
   Example: Harvard Business School teaching cases

2. Research Case Study: Papers that:
   - Document and analyze specific projects, events, or phenomena
   - Present findings from studying particular cases
   - Focus on describing what happened and lessons learned

3. Empirical Research:
   - Primary data collection and analysis
   - Surveys, experiments, or field studies
   - Statistical analysis or quantitative methods

4. Review/Theoretical:
   - Literature reviews or meta-analyses
   - Theoretical frameworks
   - Conceptual models or methodologies

Title: {title}
Abstract: {abstract}

Return only the category name exactly as written above.
"""
    
    try:
        response = requests.post(
            api_config["url"],
            headers=api_config["headers"],
            json={
                "model": "gpt-4o",
                "messages": [
                    {"role": "system", "content": "You are a research classifier. Return only the exact category name."},
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.5
            }
        )
        return response.json()["choices"][0]["message"]["content"].strip()
    except Exception as e:
        print(f"Error processing paper: {str(e)}")
        return "Error"

def process_papers(input_file, start_index=0):
    df = pd.read_csv(input_file, encoding='utf-8')
    api_config = setup_openai()
    
    if 'Classification' not in df.columns:
        df['Classification'] = ''
    
    try:
        for idx in range(start_index, len(df)):
            print(f"\nProcessing paper {idx + 1}/{len(df)}")
            print(f"Title: {df.iloc[idx]['Title'][:100]}...")
            
            if pd.isna(df.at[idx, 'Classification']) or df.at[idx, 'Classification'] == '':
                classification = classify_paper(
                    str(df.iloc[idx]['Title']) if pd.notna(df.iloc[idx]['Title']) else '',
                    str(df.iloc[idx]['Abstract']) if pd.notna(df.iloc[idx]['Abstract']) else '',
                    api_config
                )
                
                print(f"Classification: {classification}")
                df.at[idx, 'Classification'] = classification
                
                if idx % 10 == 0:  # Save every 10 papers
                    df.to_csv(input_file, index=False)
                    print(f"Progress saved at paper {idx + 1}")
                
                sleep(1)
    except Exception as e:
        print(f"Error at paper {idx + 1}: {str(e)}")
    finally:
        df.to_csv(input_file, index=False)
        print("\nClassification Summary:")
        print(df['Classification'].value_counts())
    
    return df, idx + 1

if __name__ == "__main__":
    results, last_index = process_papers('Abstracts.csv')
    print(f"Last processed index: {last_index}")