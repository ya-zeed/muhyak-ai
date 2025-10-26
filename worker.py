import os
import redis
from rq import Worker, Queue, Connection

redis_url = os.getenv("REDIS_URL", "redis://localhost:6379")
conn = redis.from_url(redis_url)

listen = ["default"]

if __name__ == "__main__":
    with Connection(conn):
        worker = Worker(map(Queue, listen))
        print("👷 Worker started. Waiting for jobs...")
        worker.work()
