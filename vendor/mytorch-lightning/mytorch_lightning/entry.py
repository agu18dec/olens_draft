import multiprocessing as mp
import os
import socket

from mytorch_lightning.config import TrainingConfig
from mytorch_lightning.mydule import Mydule
from mytorch_lightning.trainer import Trainer


def do_train(
    config: TrainingConfig, local_rank: int, mydule: Mydule
) -> None:
    os.environ["MASTER_ADDR"] = config.master_addr
    os.environ["MASTER_PORT"] = str(config.master_port)
    os.environ["RANK"] = str(config.local_to_global_rank(local_rank))
    os.environ["LOCAL_RANK"] = str(local_rank)
    os.environ["WORLD_SIZE"] = str(config.world_size())

    trainer = Trainer(config=config, local_rank=local_rank)
    trainer.train(mydule)


def find_free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))  # Bind to a free port provided by the host.
        return s.getsockname()[1]  # Return the port number assigned.


def launch_procs_and_train(config: TrainingConfig, mydule: Mydule) -> None:
    procs: list[mp.Process] = []

    for local_rank in range(1, config.ngpu):
        p = mp.Process(target=do_train, args=(config, local_rank, mydule))
        p.start()
        procs.append(p)

    # Run rank 0 on main process
    do_train(config=config, local_rank=0, mydule=mydule)

    for p in procs:
        p.join()


def train(config: TrainingConfig, mydule: Mydule):
    if config.ngpu > 1 or config.nodes > 1:
        launch_procs_and_train(config, mydule)
    else:
        do_train(config, local_rank=0, mydule=mydule)
