import pandas as pd
import sqlite3

# Load the Excel file
excel_file = 'I:/VSCODE/Color-N-Caption/Colors.xlsx'  # Replace with your file path
df = pd.read_excel(excel_file)

# Connect to SQLite database (create if doesn't exist)
conn = sqlite3.connect('colors.db')
cursor = conn.cursor()

# Create the 'colors' table if it doesn't exist
cursor.execute('''
CREATE TABLE IF NOT EXISTS colors (
    Color_Name TEXT,
    Red INTEGER,
    Green INTEGER,
    Blue INTEGER
)
''')

# Clear the table before inserting new data
cursor.execute("DELETE FROM colors")

# Insert data into the colors table
for index, row in df.iterrows():
    cursor.execute('''
    INSERT INTO colors (Color_Name, Red, Green, Blue) VALUES (?, ?, ?, ?)
    ''', (row['Color Name'], row['Red'], row['Green'], row['Blue']))
'''cursor.execute("SELECT * FROM colors")
rows = cursor.fetchall()

# Print out the rows to verify the data
for row in rows:
    print(row)'''

conn.commit()
conn.close()
