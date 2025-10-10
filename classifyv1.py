import logging
import os
import json
import azure.functions as func
import pyodbc
from datetime import datetime, timezone

app = func.FunctionApp()

SQL_CONN_STR = os.getenv("SQL_CONN_STR")

def _insert_audit_row(conn_str: str, msg_id, seq_num, body_text, enqueued_utc):
    with pyodbc.connect(conn_str) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO dbo.ClassifyAudit (MsgId, SequenceNumber, Body, EnqueuedUtc, ProcessedUtc)
                VALUES (?, ?, ?, ?, ?)
                """,
                msg_id,
                seq_num,
                body_text,
                enqueued_utc,
                datetime.now(timezone.utc)
            )
            conn.commit()

@app.service_bus_queue_trigger(
    arg_name="azservicebus",
    queue_name="q-classify",
    connection="SB_CONNECTION_STRING",
)
def classifyv1(azservicebus: func.ServiceBusMessage):
    """
    Triggered by messages on q-classify. Stores each message into dbo.ClassifyAudit.
    """
    try:
        body = azservicebus.get_body().decode("utf-8")
        msg_id = getattr(azservicebus, "message_id", None)
        seq_num = getattr(azservicebus, "sequence_number", None)
        enq_time = getattr(azservicebus, "enqueued_time_utc", None)

        # Optional: validate JSON without failing the run
        try:
            _ = json.loads(body)
        except Exception:
            pass  # keep raw body

        logging.info("classifyv1: msg_id=%s seq=%s enq=%s", msg_id, seq_num, enq_time)

        _insert_audit_row(SQL_CONN_STR, msg_id, seq_num, body, enq_time)

        logging.info("classifyv1: wrote to dbo.ClassifyAudit for msg_id=%s seq=%s", msg_id, seq_num)

    except Exception as e:
        logging.exception("classifyv1 error: %s", e)
        # Let the Functions host handle retries / dead-lettering if enabled
        raise

