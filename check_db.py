import sqlite3

p = r"D:\cal\data\lab.db"
con = sqlite3.connect(p)
cur = con.cursor()

tables = [r[0] for r in cur.execute(
    "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
)]
print("DB:", p)
print("tables:", tables)

for t in tables:
    n = cur.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
    print(f"{t}: {n}")

con.close()
