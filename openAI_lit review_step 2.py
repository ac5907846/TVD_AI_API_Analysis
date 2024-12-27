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

def extract_abstract_details(title, abstract, api_config):
    prompt = f"""
Analyze the following academic paper and classify it according to the following categories:

1. Case Study Type:
   - Case Study Research: An in-depth analysis of a particular individual, group, organization, event, or phenomenon within its real-world context, aimed at exploring or explaining complex issues. Note: If the paper describes the application of a case study research method in an educational setting, but it does not create a structured teaching case study, classify it as Case Study Research.
   - Teaching Case Study: A semi-fictionalized or real-world scenario designed to teach students specific concepts, skills, or decision-making strategies. These should include structured elements such as scenarios, learning objectives, and discussion prompts explicitly designed for educational purposes.

2. Does the paper describe a framework to develop a Teaching Case Study? (Yes/No)
3. If Yes, extract the framework described in the abstract.
4. Identify the field or discipline of the paper (e.g., Civil Engineering, Education, etc.).

Input:
- Title: {title}
- Abstract: {abstract}

Output Format:
- Case Study Type: [Research/Teaching/None]
- Framework for Teaching Case Study: [Yes/No]
- Framework Description: [Details if Yes, otherwise "No"]
- Field/Discipline: [Details]
"""
    try:
        response = requests.post(
            api_config["url"],
            headers=api_config["headers"],
            json={
                "model": "gpt-4o",
                "messages": [
                    {"role": "system", "content": "You are an academic research assistant. Return structured and concise information."},
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.5
            }
        )
        return response.json()["choices"][0]["message"]["content"].strip()
    except Exception as e:
        print(f"Error processing paper: {str(e)}")
        return "Error"

def process_papers(input_file, output_file, start_index=0):
    df = pd.read_csv(input_file, encoding='utf-8')
    api_config = setup_openai()

    # Add new columns for extracted information
    for col in ["Case Study Type", "Framework for Teaching Case Study", "Framework Description", "Field/Discipline"]:
        if col not in df.columns:
            df[col] = ''

    try:
        for idx in range(start_index, len(df)):
            print(f"\nProcessing paper {idx + 1}/{len(df)}")
            print(f"Title: {df.iloc[idx]['Title'][:100]}...")

            # Process only if columns are not filled
            if pd.isna(df.at[idx, 'Case Study Type']) or df.at[idx, 'Case Study Type'] == '':
                details = extract_abstract_details(
                    str(df.iloc[idx]['Title']) if pd.notna(df.iloc[idx]['Title']) else '',
                    str(df.iloc[idx]['Abstract']) if pd.notna(df.iloc[idx]['Abstract']) else '',
                    api_config
                )

                print(f"Details Extracted: {details}")

                # Parse and assign the extracted details to columns
                for line in details.split("\n"):
                    if line.startswith("- Case Study Type:"):
                        df.at[idx, 'Case Study Type'] = line.split(":", 1)[1].strip()
                    elif line.startswith("- Framework for Teaching Case Study:"):
                        df.at[idx, 'Framework for Teaching Case Study'] = line.split(":", 1)[1].strip()
                    elif line.startswith("- Framework Description:"):
                        df.at[idx, 'Framework Description'] = line.split(":", 1)[1].strip()
                    elif line.startswith("- Field/Discipline:"):
                        df.at[idx, 'Field/Discipline'] = line.split(":", 1)[1].strip()

                if idx % 10 == 0:  # Save progress every 10 papers
                    df.to_csv(output_file, index=False)
                    print(f"Progress saved at paper {idx + 1}")

                sleep(1)  # Avoid rate limiting

    except Exception as e:
        print(f"Error at paper {idx + 1}: {str(e)}")
    finally:
        df.to_csv(output_file, index=False)
        print("\nAnalysis Summary:")
        print(df["Case Study Type"].value_counts())

    return df, idx + 1

if __name__ == "__main__":
    results, last_index = process_papers('Abstracts.csv', 'Processed_Abstracts.csv')
    print(f"Last processed index: {last_index}")