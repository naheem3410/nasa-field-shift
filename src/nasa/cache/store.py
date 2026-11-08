from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from sqlalchemy import Column, MetaData, String, Table, Text, create_engine, delete, inspect, insert, select
from sqlalchemy.engine import make_url


class SqlStore:
    def __init__(self, url):
        self.url = url
        parsed = make_url(url)
        if parsed.get_backend_name() == "sqlite" and parsed.database not in (None, "", ":memory:"):
            Path(parsed.database).parent.mkdir(parents=True, exist_ok=True)
        self.engine = create_engine(url)
        meta = MetaData()
        self.meta = Table(
            "cache_meta", meta,
            Column("name", String(100), primary_key=True),
            Column("signature", Text),
            Column("note", Text),
            Column("built_at", String(40)),
        )
        meta.create_all(self.engine)

    def _row(self, name):
        query = select(self.meta.c.signature, self.meta.c.note).where(self.meta.c.name == name)
        with self.engine.connect() as conn:
            return conn.execute(query).first()

    def is_fresh(self, name, signature):
        row = self._row(name)
        return row is not None and row[0] == signature and inspect(self.engine).has_table(name)

    def note(self, name):
        row = self._row(name)
        return row[1] if row else None

    def read(self, name):
        return pd.read_sql_table(name, self.engine)

    def has_table(self, name):
        return inspect(self.engine).has_table(name)

    def read_where(self, name, **keys):
        if not self.has_table(name):
            return pd.DataFrame()
        table = Table(name, MetaData(), autoload_with=self.engine)
        query = select(table)
        for col, val in keys.items():
            many = isinstance(val, (list, tuple, set))
            query = query.where(table.c[col].in_(list(val)) if many else table.c[col] == val)
        with self.engine.connect() as conn:
            return pd.read_sql(query, conn)

    def replace_where(self, name, df, **keys):
        self.replace_many([(name, df, keys)])

    def replace_many(self, items):
        # one transaction so a crash never leaves a half written series
        with self.engine.begin() as conn:
            for name, df, keys in items:
                if inspect(conn).has_table(name):
                    table = Table(name, MetaData(), autoload_with=conn)
                    conn.execute(delete(table).where(*[table.c[c] == v for c, v in keys.items()]))
                df.to_sql(name, conn, if_exists="append", index=False, chunksize=2000)

    def write(self, name, df, signature, note=""):
        with self.engine.begin() as conn:
            conn.execute(delete(self.meta).where(self.meta.c.name == name))
        df.to_sql(name, self.engine, if_exists="replace", index=False, chunksize=2000)
        stamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
        with self.engine.begin() as conn:
            conn.execute(insert(self.meta).values(name=name, signature=signature, note=note, built_at=stamp))
