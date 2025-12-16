import google.generativeai as genai

# Load API key from secrets.toml
api_key = None
with open('.streamlit/secrets.toml', 'r') as f:
    for line in f:
        if 'GOOGLE_API_KEY' in line:
            api_key = line.split('=')[1].strip().strip('"')
            break

if api_key:
    genai.configure(api_key=api_key)
    print('Available models:')
    for model in genai.list_models():
        print(f'  - {model.name}')
else:
    print("No API key found")
