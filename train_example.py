import openai
import time

# Set your OpenAI API key
openai.api_key = 'your-api-key'

# Upload the training file
training_file = openai.File.create(
  file=open('train_dataset.jsonl'),
  purpose='fine-tune'
)

# Get the file ID
training_file_id = training_file['id']

# Start the fine-tuning process
response = openai.FineTune.create(
  training_file=training_file_id,
  model="gpt-3.5-turbo",
  n_epochs=3,
  batch_size=1,
  learning_rate_multiplier=16,
)

# Print the fine-tuning response
print(response)

# Monitor the fine-tuning process
fine_tune_id = response['id']

while True:
    status = openai.FineTune.retrieve(fine_tune_id)['status']
    if status in ['succeeded', 'failed']:
        break
    print(f"Fine-tuning status: {status}")
    time.sleep(60)

print("Fine-tuning completed.")
