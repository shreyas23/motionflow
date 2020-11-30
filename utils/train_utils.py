import torch
from tqdm import tqdm


def step(args, data_dict, model, loss, augmentations, optimizer, gpu):
    # Get input and target tensor keys
    input_keys = list(filter(lambda x: "input" in x, data_dict.keys()))
    target_keys = list(filter(lambda x: "target" in x, data_dict.keys()))
    tensor_keys = input_keys + target_keys

    # transfer to cuda
    if args.cuda:
        for k, v in data_dict.items():
            if k in tensor_keys:
                data_dict[k] = v.cuda(device=gpu, non_blocking=True)

    if augmentations is not None:
        with torch.no_grad():
            data_dict = augmentations(data_dict)

    for k, t in data_dict.items():
        if k in input_keys:
            data_dict[k] = t.requires_grad_(True)
        if k in target_keys:
            data_dict[k] = t.requires_grad_(False)

    output_dict = model(data_dict)
    loss_dict = loss(output_dict, data_dict)

    return loss_dict, output_dict


def train_one_epoch(args, model, loss, dataloader, optimizer, augmentations, lr_scheduler, gpu):

    loss_dict_avg = None

    if gpu == 0:
        dataloader_iter = tqdm(dataloader)
    else:
        dataloader_iter = dataloader

    for i, data in enumerate(dataloader_iter):
        loss_dict, output_dict = step(
            args, data, model, loss, augmentations, optimizer, gpu)
        
        if loss_dict_avg is None:
            loss_dict_avg = {k:0 for k in loss_dict.keys()}

        # calculate gradients and then do Adam step
        optimizer.zero_grad()
        total_loss = loss_dict['total_loss']
        assert (not torch.isnan(total_loss)), f"training loss is nan, exiting...{loss_dict}"
        total_loss.backward()

        if args.grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm)
        if args.grad_clip_value > 0:
            torch.nn.utils.clip_grad_value_(model.parameters(), args.grad_clip_value)
        optimizer.step()

        for key in loss_dict.keys():
            loss_dict_avg[key] += loss_dict[key].detach()

    n = len(dataloader)
    for key in loss_dict_avg.keys():
        loss_dict_avg[key] /= n

    return loss_dict_avg, output_dict, data


def evaluate(args, model, loss, dataloader, augmentations):
    model.eval()
    loss_dict_avg = None

    for data_dict in tqdm(dataloader):
        with torch.no_grad():
            # Get input and target tensor keys
            input_keys = list(filter(lambda x: "input" in x, data_dict.keys()))
            target_keys = list(filter(lambda x: "target" in x, data_dict.keys()))
            tensor_keys = input_keys + target_keys

            # Possibly transfer to Cuda
            if args.cuda:
                for k, v in data_dict.items():
                    if k in tensor_keys:
                        data_dict[k] = v.cuda(non_blocking=True)
            data_dict = augmentations(data_dict)
            output_dict = model(data_dict)
            loss_dict = loss(output_dict, data_dict)

            if loss_dict_avg is None:
                loss_dict_avg = {k:0 for k in loss_dict.keys()}

            for key in loss_dict.keys():
                loss_dict_avg[key] += loss_dict[key].detach()

    n = len(dataloader)
    for key in loss_dict_avg.keys():
        loss_dict_avg[key] /= n

    return loss_dict_avg, data_dict, output_dict