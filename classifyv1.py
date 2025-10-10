import os
import logging
import azure.functions as func
import pyodbc
from datetime import datetime, timezone

app = func.FunctionApp()

def _sql_conn_str():
    return (
        f"Driver={{ODBC Driver 18 for SQL Server}};"
        f"Server=tcp:{os.getenv('SQL_SERVER')},1433;"
        f"Database={os.getenv('SQL_DB')};"
        f"Uid={os.getenv('SQL_USER')};"
        f"Pwd={os.getenv('SQL_PASSWORD')};"
        f"Encrypt={os.getenv('SQL_ENCRYPT','yes')};"
        "TrustServerCertificate=no;Connection Timeout=30;"
    )

@app.service_bus_queue_trigger(
    arg_name="azservicebus",
    queue_name=os.getenv("SERVICEBUS_QUEUE_CLASSIFY","q-classify"),
    connection="SB_CONNECTION_STRING",
)
def classifyv1(azservicebus: func.ServiceBusMessage):
    body = azservicebus.get_body().decode("utf-8")
    msg_id = getattr(azservicebus, "message_id", None)
    seq_num = getattr(azservicebus, "sequence_number", None)
    enq_time = getattr(azservicebus, "enqueued_time_utc", None)

    schema = os.getenv("SQL_SCHEMA", "dbo")
    table = os.getenv("SQL_TABLE", "ClassifyAudit")

    logging.info("classifyv1: received msg_id=%s seq=%s body=%s", msg_id, seq_num, body)

    try:
        with pyodbc.connect(_sql_conn_str()) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    f"INSERT INTO {schema}.{table} "
                    "(MsgId, SequenceNumber, Body, EnqueuedUtc, ProcessedUtc) "
                    "VALUES (?, ?, ?, ?, ?)",
                    msg_id, seq_num, body, enq_time,
                    datetime.now(timezone.utc)
                )
                conn.commit()
        logging.info("classifyv1: inserted into %s.%s", schema, table)

    except Exception as e:
        logging.error("SQL insert failed: %s", str(e))
        raise








CREATE TABLE dbo.ClassifyAudit
(
  AuditId         UNIQUEIDENTIFIER NOT NULL DEFAULT NEWID(),
  MsgId           NVARCHAR(128)    NULL,
  SequenceNumber  BIGINT           NULL,
  Body            NVARCHAR(MAX)    NOT NULL,
  EnqueuedUtc     DATETIME2(3)     NULL,
  ProcessedUtc    DATETIME2(3)     NOT NULL DEFAULT SYSUTCDATETIME(),
  PRIMARY KEY (AuditId)
);


