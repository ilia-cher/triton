import argparse
import sqlite3
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("-f","--file", help="DB file", type=str, required=True)
parser.add_argument("-w","--warmup", help="Skip warmup iterations", type=int, required=True)
parser.add_argument("-t","--topn", help='Show topn kernels', type=int, default=10)
args = parser.parse_args()

conn = sqlite3.connect(args.file)
cur = conn.cursor()
cur.execute("drop table if exists exec_times_id")
cur.execute("drop table if exists exec_times")
cur.execute("""
create table exec_times as
  select
    C.string as name,
    A.start as start,
    A.end as end,
    (A.end - A.start) / 1000.0 as duration_us
  from (
    select
      description_id as name_id,
      start,
      end
    from rocpd_op
    where
      description_id not in (select id from rocpd_string where string='')
  ) A
  join rocpd_string C
  on
    C.id = A.name_id
""")
cur.execute("""
create table exec_times_id as
select
  name,
  start,
  end,
  duration_us,
  row_number() over (partition by name order by start asc) as run_id
from
  exec_times
""")
conn.commit()

res = pd.read_sql_query(f"""
select
  name as Name,
  count(name) as TotalCalls,
  sum(duration_us) as TotalDurationUs,
  avg(duration_us) as AvgDurationUs
from
  exec_times_id
where
  run_id > {args.warmup}
group by name
order by AvgDurationUs desc
""", conn)
print(res.head(args.topn))

conn.close()
