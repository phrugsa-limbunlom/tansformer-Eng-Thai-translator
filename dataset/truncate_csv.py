import pandas as pd

file_path = 'task_master_1.csv'
df = pd.read_csv(file_path)

df = df.head(10)

output_path = 'small_task_master_1.csv'
df.to_csv(output_path, index=False)

print(f"Processed file saved to: {output_path}")
