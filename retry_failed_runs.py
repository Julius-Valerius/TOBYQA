"""
retry_failed_runs.py
====================
删除结果库中带 error 的行，使 run_benchmark.py 的断点续跑逻辑重新执行它们
（续跑按 (problem, dim, env, seed, solver) 跳过已存在的行，报错行同样算已存在，
不删则永远不会被重试）。

用法:
    python retry_failed_runs.py [db_path]      # 默认 results/full65.sqlite
"""
import sqlite3
import sys

db = sys.argv[1] if len(sys.argv) > 1 else 'results/full65.sqlite'
con = sqlite3.connect(db)
cond = "error IS NOT NULL AND error != ''"
rows = con.execute(f'SELECT solver, error, COUNT(*) FROM runs WHERE {cond} '
                   'GROUP BY solver, error').fetchall()
n = con.execute(f'SELECT COUNT(*) FROM runs WHERE {cond}').fetchone()[0]
for solver, err, cnt in rows:
    print(f'  {solver}: {cnt} x {err[:60]}')
con.execute(f'DELETE FROM runs WHERE {cond}')
con.commit()
print(f'{db}: 已删除 {n} 条报错行；重跑 run_benchmark.py 同一条命令即可补齐。')
