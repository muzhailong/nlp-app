import torch


def save(path, model_state, optim_state, epoch, all_steps, **kwargs):
    torch.save(
        {
            "model_state": model_state,
            "optim_state": optim_state,
            "epoch": epoch,
            "all_steps": all_steps,
            "others": kwargs
        },
        path
    )


def load(path, return_dict=True):
    mp = torch.load(path)
    if return_dict:
        return mp
    return mp["model_state"], mp["optim_state"], mp["epoch"], mp['all_steps'], mp["others"]
