import torch


def save(path, model, optim, epoch, all_step, **kwargs):
    torch.save(
        {
            "model": model,
            "optim": optim,
            "epoch": epoch,
            "all_step": all_step,
            "others": kwargs
        },
        path
    )


def load(path, return_dict=True):
    mp = torch.load(path)
    if return_dict:
        return mp
    return mp["model"], mp["optim"], mp["epoch"], mp['all_step'], mp["others"]
