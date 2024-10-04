import sqlite3
import pandas as pd
conn = sqlite3.connect("results.rpd")
df_top = pd.read_sql_query("SELECT * from top", conn)
conn.close()
print(df_top.head(10))
