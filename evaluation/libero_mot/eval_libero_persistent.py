#!/usr/bin/env python3
"""Dynamic LIBERO dispatcher with one resident policy model per GPU."""

import argparse
import contextlib
import multiprocessing as mp
import os
import queue
import sys
import traceback
from pathlib import Path

from tqdm import tqdm

# Invoking this file directly makes Python add only evaluation/libero_mot to
# sys.path. Add the repository root so package imports also work from wrappers.
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _read_tasks(path: Path):
    tasks = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        suite, task_id = line.strip().split(",")
        tasks.append((suite, int(task_id)))
    return tasks


class _ActionClient:
    def __init__(self, client_id, request_queue, connection, retry_interval_s):
        self.client_id = client_id
        self.request_queue = request_queue
        self.connection = connection
        self.retry_interval_s = retry_interval_s
        self.next_request_id = 0
        self.pending_responses = {}

    def predict(self, **request):
        request_id = self.next_request_id
        self.next_request_id += 1
        message = ("request", self.client_id, request_id, request)
        # Observations include images, so use Queue's asynchronous feeder for
        # the large request and a dedicated Pipe for the small response.
        self.request_queue.put(message)
        retry_sent = False
        while True:
            if request_id in self.pending_responses:
                status, payload = self.pending_responses.pop(request_id)
                if status != "ok":
                    raise RuntimeError(payload)
                return payload
            if not self.connection.poll(self.retry_interval_s):
                if not retry_sent:
                    # One bounded replay covers a lost queue message. Do not
                    # keep appending retries when a server is genuinely stuck.
                    self.request_queue.put(message)
                    retry_sent = True
                continue
            try:
                status, response_id, payload = self.connection.recv()
            except EOFError as exc:
                raise RuntimeError("Policy server disconnected before sending an action") from exc
            # Retain server-side responses until this acknowledgement arrives;
            # a retry before that point returns the exact same action chunk.
            self.request_queue.put(("ack", self.client_id, response_id))
            if response_id < request_id:
                # A delayed duplicate from the previous RPC is no longer useful.
                continue
            self.pending_responses[response_id] = (status, payload)


def _env_client(gpu, worker_id, client_id, first_task, task_queue, request_queue, connection,
                status_queue, eval_args, cfg, rpc_retry_interval):
    log_path = Path(eval_args.out_dir) / "logs" / f"persistent_gpu{gpu}_client{client_id}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", buffering=1) as log, contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
        _run_env_client(gpu, worker_id, client_id, first_task, task_queue, request_queue,
                        connection, status_queue, eval_args, cfg, rpc_retry_interval)


def _run_env_client(gpu, worker_id, client_id, first_task, task_queue, request_queue, connection,
                    status_queue, eval_args, cfg, rpc_retry_interval):
    # Each client owns its MuJoCo/OpenGL context. It never initializes a model.
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    from evaluation.libero_mot import eval_libero
    runtime = {
        "cfg": cfg, "suite_cache": {}, "device": "cpu",
        "replan_steps": eval_args.replan_steps if eval_args.replan_steps > 0 else int(cfg.action_per_frame),
        "denormalize": None,
        "resolved_joint_future_denoising": (
            bool(getattr(cfg, "default_joint_future_denoising", False))
            if eval_args.joint_future_denoising is None else bool(eval_args.joint_future_denoising)
        ),
        "future_semantic_steps": eval_libero._resolve_future_semantic_steps(cfg),
        "vjepa_anchor_temporal_mode": eval_libero._resolve_vjepa_anchor_temporal_mode(
            cfg, eval_args.vjepa_anchor_temporal_mode
        ),
        "vjepa_anchor_prev_offset": eval_libero._resolve_vjepa_anchor_prev_offset(
            cfg, eval_args.vjepa_anchor_prev_offset
        ),
    }
    action_client = _ActionClient(client_id, request_queue, connection, rpc_retry_interval)
    item = first_task
    while True:
        if item is None:
            item = task_queue.get()
        if item is None:
            return
        suite, task_id = item
        try:
            result = eval_libero.run_persistent_task(
                runtime, eval_args, suite, task_id,
                f"worker{worker_id}_client{client_id}_task{task_id}",
                action_client=action_client,
            )
            status_queue.put(("done", worker_id, suite, task_id, {
                "skipped": bool(result.get("skipped")),
                "successes": int(result.get("successes", 0)),
                "total_episodes": int(result.get("total_episodes", 0)),
            }))
        except Exception:
            status_queue.put(("error", worker_id, suite, task_id, traceback.format_exc()))
            return
        item = None


def _worker(gpu, worker_id, task_queue, status_queue, eval_args, log_path, max_clients, initial_task, start_event):
    # Must happen before the first CUDA allocation in the child process.
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", buffering=1) as log, contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
        clients = []
        try:
            from evaluation.libero_mot import eval_libero
            runtime = eval_libero.create_persistent_runtime(eval_args)
            print(f"persistent worker={worker_id} physical_gpu={gpu} model loaded; env_clients={max_clients}")
            status_queue.put(("ready", worker_id, "", -1, False))
            start_event.wait()
            ctx = mp.get_context("spawn")
            request_queue = ctx.Queue()
            server_connections = []
            client_connections = []
            for _ in range(max_clients):
                server_connection, client_connection = ctx.Pipe(duplex=True)
                server_connections.append(server_connection)
                client_connections.append(client_connection)
            clients = [ctx.Process(
                target=_env_client,
                args=(gpu, worker_id, client_id, initial_task if client_id == 0 else None,
                      task_queue, request_queue, client_connections[client_id], status_queue, eval_args, runtime["cfg"],
                      eval_args.rpc_retry_interval),
                daemon=False,
            ) for client_id in range(max_clients)]
            for client in clients:
                client.start()
            for connection in client_connections:
                connection.close()
            response_cache = {}
            # The server is deliberately batch-1: one complete denoise chain per RPC.
            while any(client.is_alive() for client in clients):
                try:
                    message = request_queue.get(timeout=1)
                except queue.Empty:
                    continue
                if message[0] == "ack":
                    _, client_id, request_id = message
                    response_cache.pop((client_id, request_id), None)
                    continue
                _, client_id, request_id, request = message
                connection = server_connections[client_id]
                cache_key = (client_id, request_id)
                try:
                    if cache_key in response_cache:
                        status, payload = response_cache[cache_key]
                    else:
                        status = "ok"
                        payload = eval_libero.predict_persistent_action(runtime, eval_args, request)
                        response_cache[cache_key] = (status, payload)
                    connection.send((status, request_id, payload))
                except Exception:
                    traceback.print_exc()
                    try:
                        payload = traceback.format_exc()
                        response_cache[cache_key] = ("error", payload)
                        connection.send(("error", request_id, payload))
                    except (BrokenPipeError, EOFError):
                        pass
            for client in clients:
                client.join()
            for connection in server_connections:
                connection.close()
        except Exception:
            traceback.print_exc()
            status_queue.put(("startup_error", worker_id, "", -1, traceback.format_exc()))
        finally:
            for client in clients:
                if client.is_alive():
                    client.terminate()
            for client in clients:
                client.join()


def main():
    parser = argparse.ArgumentParser(description="Dynamic persistent LIBERO evaluation workers")
    parser.add_argument("--task-file", required=True, type=Path, help="Lines of suite,task_id")
    parser.add_argument("--gpu-ids", required=True, help="Comma-separated physical GPU IDs")
    parser.add_argument("--log-dir", required=True, type=Path)
    parser.add_argument("--max-tasks-per-gpu", type=int, default=1,
                        help="Maximum concurrent env clients per GPU; policy inference remains batch size 1.")
    parser.add_argument("--rpc-retry-interval", type=float, default=20.0,
                        help="Seconds before retrying an unacknowledged batch-1 policy request.")
    parser.add_argument("--per-task-output", action="store_true",
                        help="Keep the release's suite_taskN output directories.")
    master_args, eval_argv = parser.parse_known_args()
    if master_args.max_tasks_per_gpu <= 0:
        raise SystemExit("--max-tasks-per-gpu must be positive")
    if master_args.rpc_retry_interval <= 0:
        raise SystemExit("--rpc-retry-interval must be positive")
    gpu_ids = [gpu.strip() for gpu in master_args.gpu_ids.split(",") if gpu.strip()]
    if not gpu_ids:
        raise SystemExit("--gpu-ids cannot be empty")
    tasks = _read_tasks(master_args.task_file)
    if not tasks:
        print("No pending tasks.")
        return

    master_args.log_dir.mkdir(parents=True, exist_ok=True)
    with (master_args.log_dir / "dispatcher.log").open("a", buffering=1) as log, contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
        from evaluation.libero_mot import eval_libero
    eval_args = eval_libero.build_arg_parser().parse_args(eval_argv)
    eval_args.rpc_retry_interval = master_args.rpc_retry_interval
    if master_args.per_task_output:
        eval_args.task_output_root = eval_args.out_dir
    ctx = mp.get_context("spawn")
    task_queue = ctx.Queue()
    status_queue = ctx.Queue()
    start_event = ctx.Event()
    initial_tasks = tasks[:len(gpu_ids)]
    for task in tasks[len(initial_tasks):]:
        task_queue.put(task)
    for _ in range(len(gpu_ids) * master_args.max_tasks_per_gpu):
        task_queue.put(None)
    workers = [
        ctx.Process(
            target=_worker,
            args=(gpu, idx, task_queue, status_queue, eval_args, master_args.log_dir / f"persistent_gpu{gpu}.log", master_args.max_tasks_per_gpu, initial_tasks[idx] if idx < len(initial_tasks) else None, start_event),
            daemon=False,
        )
        for idx, gpu in enumerate(gpu_ids)
    ]
    for process in workers:
        process.start()

    completed = 0
    progress = tqdm(total=len(tasks), desc="LIBERO tasks", unit="task", dynamic_ncols=True)
    try:
        ready_workers = 0
        while ready_workers < len(workers):
            try:
                kind, worker_id, suite, task_id, detail = status_queue.get(timeout=10)
            except queue.Empty:
                failed = [p for p in workers if not p.is_alive()]
                if failed:
                    raise RuntimeError(f"persistent worker exited during startup: {[p.pid for p in failed]}")
                continue
            if kind in {"error", "startup_error"}:
                raise RuntimeError(f"worker {worker_id} failed on {suite}:{task_id}\n{detail}")
            if kind == "ready":
                ready_workers += 1
        start_event.set()
        while completed < len(tasks):
            try:
                status = status_queue.get(timeout=10)
            except queue.Empty:
                failed = [p for p in workers if not p.is_alive() and p.exitcode not in (None, 0)]
                if failed:
                    raise RuntimeError(f"persistent worker exited: {[p.pid for p in failed]}")
                continue
            kind, worker_id, suite, task_id, detail = status
            if kind in {"error", "startup_error"}:
                raise RuntimeError(f"worker {worker_id} failed on {suite}:{task_id}\n{detail}")
            if kind == "ready":
                continue
            completed += 1
            successes = detail["successes"]
            failures = detail["total_episodes"] - successes
            outcome = "skip" if detail["skipped"] else f"succ={successes} fail={failures}"
            progress.set_postfix_str(
                f"gpu={gpu_ids[worker_id]} {suite}:task{task_id} {outcome}", refresh=False
            )
            progress.update(1)
    except KeyboardInterrupt:
        print("Interrupted; stopping persistent workers.", file=sys.stderr)
        raise
    finally:
        progress.close()
        if completed == len(tasks):
            for process in workers:
                process.join(timeout=10)
        for process in workers:
            if process.is_alive():
                process.terminate()
        for process in workers:
            process.join()


if __name__ == "__main__":
    main()
