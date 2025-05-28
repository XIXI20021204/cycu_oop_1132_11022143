import pandas as pd

# Load CSV data
df = pd.read_csv('20250520/midterm_scores.csv')

subjects = ['Chinese', 'English', 'Math', 'History', 'Geography', 'Physics', 'Chemistry']

# 用於存儲篩選結果的列表
failed_students = []

print("Students failing 4 or more subjects:")
for idx, row in df.iterrows():
    failed_subjects_list = [subj for subj in subjects if row[subj] < 60]
    if len(failed_subjects_list) >= 4:  # 篩選條件：大於等於 4 科不及格
        print(f"{row['Name']} (ID: {row['StudentID']}), Failed subjects: {', '.join(failed_subjects_list)}")
        failed_students.append({
            'Name': row['Name'],
            'StudentID': row['StudentID'],
            'FailedCount': len(failed_subjects_list),
            'FailedSubjects': ', '.join(failed_subjects_list)
        })

# 將結果輸出為 CSV 檔案
failed_students_df = pd.DataFrame(failed_students)
failed_students_df.to_csv('20250520/failed_students.csv', index=False, encoding='utf-8-sig')

print("Results have been saved to '20250520/failed_students.csv'.")