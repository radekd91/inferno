

def test(model, batch):
    batch["image"] = batch["image"].cuda()
    if len(batch["image"].shape) == 3:
        batch["image"] = batch["image"].view(1,3,224,224)
    values = model(batch, training=False, validation=False)
    return values

