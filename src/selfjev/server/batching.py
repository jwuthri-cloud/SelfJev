"""Cross-request batching: concurrent requests share forward passes. One worker takes what is queued (up to
max_batch_requests), waits at most max_wait_ms for more, runs the batch in a thread, and answers each request. If a batch
fails (e.g. one request is too long), its requests are retried one by one so only the faulty one gets the error.
"""

import asyncio


class Overloaded(Exception):
    """The queue is full: the client should retry with backoff (HTTP 529)."""


class Batcher:
    def __init__(self, fn, max_batch_requests=32, max_wait_ms=5.0, max_queue=256):
        self.fn, self.max_batch, self.max_wait, self.max_queue = fn, max_batch_requests, max_wait_ms / 1000, max_queue
        self.queue: asyncio.Queue | None = None
        self.task: asyncio.Task | None = None

    async def start(self):
        self.queue, self.task = asyncio.Queue(), asyncio.create_task(self._run())

    async def stop(self):
        if self.task:
            self.task.cancel()

    @property
    def depth(self) -> int:
        return self.queue.qsize() if self.queue else 0

    async def submit(self, item):
        if self.queue.qsize() >= self.max_queue:
            raise Overloaded(f"{self.queue.qsize()} requests queued")
        future = asyncio.get_running_loop().create_future()
        await self.queue.put((item, future))
        return await future

    async def _run(self):
        loop = asyncio.get_running_loop()
        while True:
            batch = [await self.queue.get()]
            deadline = loop.time() + self.max_wait
            while len(batch) < self.max_batch and (left := deadline - loop.time()) > 0:
                try:
                    batch.append(await asyncio.wait_for(self.queue.get(), left))
                except TimeoutError:
                    break
            await self._answer(loop, batch)

    async def _answer(self, loop, batch):
        try:
            results = await loop.run_in_executor(None, self.fn, [item for item, _ in batch])
        except Exception as e:
            if len(batch) == 1:
                if not batch[0][1].done():
                    batch[0][1].set_exception(e)
                return
            for one in batch:  # isolate the faulty request
                await self._answer(loop, [one])
            return
        for (_, future), result in zip(batch, results, strict=True):
            if not future.done():
                future.set_result(result)
