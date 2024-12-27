import pandas as pd
import requests
from dotenv import load_dotenv
import os
import time

# Load environment variables from .env file
load_dotenv()

# Retrieve the OpenAI API key from the environment variables
API_KEY = os.getenv("OPENAI_API_KEY")

# Define the API endpoint
API_URL = "https://api.openai.com/v1/chat/completions"

# Define the headers for the request
headers = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}

def load_file(file_path):
    """Load content from a text file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read().strip()
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
        return None
    except Exception as e:
        print(f"Error reading file '{file_path}': {e}")
        return None

def load_questions(file_path):
    """Load the questions from a text file."""
    return load_file(file_path)

def create_prompt(case_study_content, rubric, objectives, examples, questions, student_responses):
    """Create a detailed prompt for the OpenAI API based on student responses."""
    prompt = f"""
You are grading student responses about a Target Value Design (TVD) case study. Please evaluate their understanding based on the case study context and their responses to the questions provided.

Case Study Context:
{case_study_content}

Learning Objectives:
{objectives}

Grading Rubric:
{rubric}

Examples of Good Responses:
{examples}

Questions:
{questions}

Student Responses:
Remember: {student_responses['remember']}
Key Takeaways: {student_responses['key_takeaways']}
Industry Challenges Addressed: {student_responses['address']}

Please rate each learning objective (LO1, LO2, LO3) with a score of 0, 1, or 2 based on the rubric.
Provide only the scores in this format: "LO1: X, LO2: Y, LO3: Z" where X, Y, and Z are scores.
"""
    return prompt

def get_openai_rating(prompt):
    """Send the prompt to OpenAI API and get the rating."""
    data = {
        "model": "gpt-4o",
        "messages": [
            {"role": "system", "content": "You are an educational assessment expert specializing in construction and engineering education."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.5
    }

    try:
        response = requests.post(API_URL, headers=headers, json=data)
        response.raise_for_status()
        result = response.json()
        return result["choices"][0]["message"]["content"]
    except requests.exceptions.RequestException as e:
        print(f"Error making API request: {e}")
        return None
    except (KeyError, IndexError) as e:
        print(f"Error parsing API response: {e}")
        return None

def parse_rating(rating_str):
    """Parse the rating string and return individual scores."""
    try:
        scores = {f'LO{i+1}': int(part.split(":")[1].strip()) for i, part in enumerate(rating_str.split(','))}
        return scores
    except Exception as e:
        print(f"Error parsing rating: {e}")
        return None

def process_student_responses(csv_file, case_study_file, rubric_file, objectives_file, examples_file, questions_file):
    """Process all student responses from the CSV file."""
    case_study_content = load_file(case_study_file)
    rubric = load_file(rubric_file)
    objectives = load_file(objectives_file)
    examples = load_file(examples_file)
    questions = load_questions(questions_file)
    
    if not (case_study_content and rubric and objectives and examples and questions):
        print("Failed to load one or more required files. Exiting.")
        return None

    try:
        # Read the CSV file
        df = pd.read_csv(csv_file)
        
        # Initialize new columns for AI ratings
        df['AI_LO1'] = None
        df['AI_LO2'] = None
        df['AI_LO3'] = None
        
        # Process each student's responses
        for index, row in df.iterrows():
            print(f"Processing student {index + 1}/{len(df)}")
            
            # Create prompt for this student's responses
            prompt = create_prompt(case_study_content, rubric, objectives, examples, questions, row)
            
            # Get rating from OpenAI
            rating_str = get_openai_rating(prompt)
            
            if rating_str:
                # Parse the rating
                scores = parse_rating(rating_str)
                
                if scores:
                    # Update the DataFrame with the scores
                    df.at[index, 'AI_LO1'] = scores.get('LO1')
                    df.at[index, 'AI_LO2'] = scores.get('LO2')
                    df.at[index, 'AI_LO3'] = scores.get('LO3')
            
            # Add a small delay to avoid rate limiting
            time.sleep(1)
            
        # Save final results
        df.to_csv('final_ratings.csv', index=False)
        print("Processing completed successfully!")
        return df
    
    except Exception as e:
        print(f"Error processing student responses: {e}")
        return None

if __name__ == "__main__":
    # File paths
    case_study_file = 'Case_Study_Content.txt'
    rubric_file = 'Grading_Rubric.txt'
    objectives_file = 'Learning_Objectives.txt'
    examples_file = 'Expert-Rated_Examples.txt'
    questions_file = 'Survey_Questions.txt'
    student_responses_file = 'Student_Responses.csv'
    
    # Process the student responses
    results = process_student_responses(
        student_responses_file, 
        case_study_file, 
        rubric_file, 
        objectives_file, 
        examples_file, 
        questions_file
    )
    
    if results is not None:
        print("\nProcessing completed!")
        print("Results saved to: final_ratings.csv")