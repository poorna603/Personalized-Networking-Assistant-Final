from transformers import pipeline
classifier = pipeline(
    "zero-shot-classification",
    model="typeform/distilbert-base-uncased-mnli"
)
labels = [
    "Technology",
    "Business",
    "Healthcare",
    "Education",
    "Marketing",
    "Finance",
    "Startup",
    "Networking"
]
event_description = """
Join industry experts to discuss Artificial Intelligence,
Cloud Computing and Digital Transformation.
"""
result = classifier(
    event_description,
    candidate_labels=labels
)

print(result["labels"][0])
print(result["scores"][0])
generator = pipeline(
    "text-generation",
    model="gpt2"
)
prompt = """
Generate a professional networking conversation starter
for a Technology conference.
"""
response = generator(
    prompt,
    max_length=60,
    num_return_sequences=1
)

print(response[0]["generated_text"])
theme = classifier(
    event_description,
    candidate_labels=labels
)["labels"][0]

prompt = f"""
Generate a professional networking conversation starter
for a {theme} event.
"""

conversation = generator(
    prompt,
    max_length=60
)

print("Theme:", theme)
print(conversation[0]["generated_text"])
