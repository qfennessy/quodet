from protocols import ConnectionPool


async def run_job(pool: ConnectionPool, job: str) -> str:
    connection = await pool.acquire()
    result = await connection.execute(job)
    await pool.release(connection)
    return result
