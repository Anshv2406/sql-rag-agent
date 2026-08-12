import sqlite3

conn = sqlite3.connect('test.db')
conn.execute('CREATE TABLE employees (id INTEGER, name TEXT, dept TEXT)')
conn.execute("INSERT INTO employees VALUES (1, 'Asha', 'Engineering')")
conn.execute("INSERT INTO employees VALUES (2, 'Ravi', 'Sales')")
conn.commit()
conn.close()

print("test.db created successfully")