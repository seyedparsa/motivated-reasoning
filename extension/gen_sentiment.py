import random
import json
import pandas as pd

subjects = {
    "people": ["Taylor Swift", "Barack Obama", "Albert Einstein", "Elon Musk", "Harry Potter"],
    "organizations": ["Google", "United Nations", "MIT", "Tesla", "Manchester United"],
    "countries": ["France", "China", "Brazil", "India", "Russia"],
    "concepts": ["capitalism", "veganism", "blockchain", "climate change", "freedom of speech"],
    "products": ["iPhone", "PlayStation", "Coca-Cola", "Nike shoes", "Roomba"],
    "entertainment": ["sushi", "Game of Thrones", "chess", "Call of Duty", "The Beatles"]
}

subject_list = []
for category, items in subjects.items():
    for item in items:
        subject_list.append({"category": category, "subject": item})


data = []
for entry in subject_list:
    subject = entry["subject"]
    category = entry["category"]
    
    data.append({        
        "prompt": f"Write a positive review about {subject}.",
        "sentiment": 1,
    })
    data.append({    
        "prompt": f"Write a negative review about {subject}.",
        "sentiment": -1,
    })

# Convert to DataFrame for display and export
df = pd.DataFrame(data)

csv_output_path = "sentiment_review_prompts.csv"

# Save CSV
df.to_csv(csv_output_path, index=False)