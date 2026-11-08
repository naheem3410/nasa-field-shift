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

    def write(self, name, df, signature, note=""):
        with self.engine.begin() as conn:
            conn.execute(delete(self.meta).where(self.meta.c.name == name))
        df.to_sql(name, self.engine, if_exists="replace", index=False, chunksize=2000)
        stamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
        with self.engine.begin() as conn:
            conn.execute(insert(self.meta).values(name=name, signature=signature, note=note, built_at=stamp))
