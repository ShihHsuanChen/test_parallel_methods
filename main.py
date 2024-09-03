import time
import multiprocessing as mp
from queue import Queue
from datetime import datetime
from threading import Thread
from typing import Optional, Literal, Callable, Union

import numpy as np
import timm
import torch
from PIL import Image


def load_model(device: str = 'cuda'):
    # model_name = 'mobilenetv4_conv_small.e2400_r224_in1k'
    model_name = 'mobilenetv4_hybrid_large.e600_r384_in1k'
    model = timm.create_model(model_name, pretrained=True)
    model = model.eval().to(device)

    # get model specific transforms (normalization, resize)
    data_config = timm.data.resolve_model_data_config(model)
    transforms = timm.data.create_transform(**data_config, is_training=False)
    return model, transforms


def log_timing(iterable, name: str = 'task', queue: Union[Queue, mp.JoinableQueue, None] = None):
    N = len(iterable)
    dtsum = 0
    for i, item in enumerate(iterable):
        st = datetime.now()
        print(f'{name} [{i+1}/{N}] start', st.strftime('%H:%M:%S.%f'))

        yield (i, item)

        et = datetime.now()
        dt = (et-st).microseconds
        print(f'{name} [{i+1}/{N}]   end', et.strftime('%H:%M:%S.%f'), dt, 'us')
        dtsum += dt

        if queue is not None:
            queue.put(i)
            print(f'put {i+1}')
    print(f'{name} dtsum {dtsum} us')


def run_iter(iterable):
    for _ in iterable:
        ...


def cpu_bound_task():
    Image.fromarray((np.random.random((3000, 3000, 3))*255).clip(0,255).astype(np.uint8))


@torch.no_grad()
def inference(queue: Union[Queue, mp.JoinableQueue], model, transforms, device: str = 'cuda', N: int = 10):
    images = [torch.rand((1, 3, 1024, 1024), dtype=torch.float32, device=device) for _ in range(N)]
    for i, image in log_timing(images, name='inference', queue=queue):
        output = model(image)
        yield i


def resize(queue: Union[Queue, mp.JoinableQueue], N: int = 10):
    for i, _ in log_timing(list(range(N)), name='resize', queue=queue):
        image = Image.fromarray((np.random.random((3000, 3000, 3))*255).clip(0,255).astype(np.uint8))
        image.resize((128, 128))
        yield i


def task_inference(queue: Union[Queue, mp.JoinableQueue], N: int, device: str = 'cpu'):
    model, transforms = load_model(device=device)
    run_iter(inference(queue, model, transforms, device=device, N=N))


def task_inference_cpu(queue: Union[Queue, mp.JoinableQueue], N: int):
    return task_inference(queue, N, device='cpu')


def task_inference_gpu(queue: Union[Queue, mp.JoinableQueue], N: int):
    return task_inference(queue, N, device='cuda')


def task_resize(queue: Union[Queue, mp.JoinableQueue], N: int):
    run_iter(resize(queue, N=N))


def print_results(queue, N: int, sleep: Optional[float] = None, block_by: Literal['cpu', 'io', None] = None):
    i = 0
    dtsum = 0
    while True:
        item = queue.get()
        print(f'recieve {i+1}')
        st = datetime.now()
        print(f'consume [{i+1}/{N}] start', st.strftime('%H:%M:%S.%f'))

        if sleep is not None:
            time.sleep(sleep)

        if block_by == 'cpu':
            cpu_bound_task()
        elif block_by == 'io':
            time.sleep(0.66)

        et = datetime.now()
        dt = (et-st).microseconds
        print(f'consume [{i+1}/{N}]   end', et.strftime('%H:%M:%S.%f'), dt, 'us')
        queue.task_done()
        dtsum += dt
        i += 1
        if i == N:
            break
    print('consume dtsum', dtsum, 'us')


def run_task_in_main_thread(
        task: Callable,
        N: int = 20,
        daemon: bool = True,
        sleep: Optional[float] = None,
        block_by: Literal['cpu', 'io', None] = None,
        ):
    queue = Queue()
    thread = Thread(
        target=print_results,
        args=(queue, N),
        kwargs=dict(sleep=sleep, block_by=block_by),
        daemon=daemon,
    )
    # start consumer thread
    thread.start()
    # start task in main thread
    task(queue, N)
    # waiting for queue empty
    queue.join()
    print('queue join')
    # waiting for thread complete
    thread.join()
    print('thread join')


def run_task_in_worker_thread(
        task: Callable,
        N: int = 20,
        daemon: bool = True,
        sleep: Optional[float] = None,
        block_by: Literal['cpu', 'io', None] = None,
        ):
    queue = Queue()
    thread = Thread(
        target=task,
        args=(queue, N),
        daemon=daemon,
    )
    # start task thread
    thread.start()
    # start consumer in main thread
    print_results(queue, N, sleep=sleep, block_by=block_by)
    # waiting for queue empty
    queue.join()
    print('queue join')
    # waiting for thread complete
    thread.join()
    print('thread join')


def run_task_in_main_process(
        task: Callable,
        N: int = 20,
        daemon: bool = True,
        sleep: Optional[float] = None,
        block_by: Literal['cpu', 'io', None] = None,
        start_method: Literal['spawn', 'fork'] = 'spawn',
        ):
    mp.set_start_method(start_method)
    queue = mp.JoinableQueue()
    process = mp.Process(
        target=print_results,
        args=(queue, N),
        kwargs=dict(sleep=sleep, block_by=block_by),
        daemon=daemon,
    )
    # start consumer process
    process.start()
    # start task in main process
    task(queue, N)
    # waiting for queue empty
    queue.join()
    print('queue join')
    # waiting for process complete
    process.join()
    print('process join')


def run_task_in_worker_process(
        task: Callable,
        N: int = 20,
        daemon: bool = True,
        sleep: Optional[float] = None,
        block_by: Literal['cpu', 'io', None] = None,
        start_method: Literal['spawn', 'fork'] = 'spawn',
        ):
    mp.set_start_method(start_method)
    queue = mp.JoinableQueue()
    process = mp.Process(
        target=task,
        args=(queue, N),
        daemon=daemon,
    )
    # start task process
    process.start()
    # start consumer in main process
    print_results(queue, N, sleep=sleep, block_by=block_by)
    # waiting for queue empty
    queue.join()
    print('queue join')
    # waiting for process complete
    process.join()
    print('process join')


if __name__ == '__main__':
    N = 20
    daemon = True
    sleep = None
    block_by = 'cpu'
    mp_start_method = 'spawn'

    # task = task_inference_gpu
    task=task_inference_cpu
    # task = task_resize

    st = time.time()
    # run_task_in_main_thread(task, N=N, daemon=daemon, sleep=sleep, block_by=block_by)
    # run_task_in_worker_thread(task, N=N, daemon=daemon, sleep=sleep, block_by=block_by)
    run_task_in_main_process(task, N=N, daemon=daemon, sleep=sleep, block_by=block_by, start_method=mp_start_method)
    # run_task_in_worker_process(task, N=N, daemon=daemon, sleep=sleep, block_by=block_by, start_method=mp_start_method)
    print('Total cost:', time.time() - st, 's')
