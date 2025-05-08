# test_forward_generic.py

import argparse
import torch
from torch.utils.data import DataLoader
from lava.lib.dl.slayer import obd
from torchinfo import summary

def register_hooks(model):
    """
    Registers forward hooks on all leaf modules to print input/output shapes.
    """
    hooks = []

    def hook_fn(name):
        def hook(module, inputs, output):
            # Collect tensor shapes for inputs
            in_shapes = [i.shape for i in inputs if isinstance(i, torch.Tensor)]
            # Determine output shape(s)
            if isinstance(output, torch.Tensor):
                out_shapes = output.shape
            else:
                out_shapes = [o.shape for o in output]
            print(f"[{name}] in: {in_shapes} -> out: {out_shapes}")
        return hook

    for name, module in model.named_modules():
        # Only attach to modules without children (leaf modules)
        if len(list(module.children())) == 0:
            hooks.append(module.register_forward_hook(hook_fn(name)))
    return hooks


def main():
    parser = argparse.ArgumentParser(
        description="Generic forward+loss/backward test for any SLAYER SDNN network"
    )
    parser.add_argument(
        "--model", type=str, required=True,
        help="Model name under `obd.models`, e.g. tiny_yolo_sdnn or yolo_kp"
    )
    parser.add_argument(
        "--path", type=str, default="data/bdd100k",
        help="Dataset root directory"
    )
    parser.add_argument(
        "--dataset", type=str, default="custom",
        choices=["custom", "BDD100K", "DSIAC"],
        help="Which dataset loader to use"
    )
    parser.add_argument(
        "--batch-size", type=int, default=1,
        help="Batch size for the DataLoader"
    )
    parser.add_argument(
        "--num-classes", type=int, default=11,
        help="Number of object classes"
    )
    parser.add_argument("--threshold", type=float, default=0.1)
    parser.add_argument("--tau-grad", type=float, default=0.1)
    parser.add_argument("--scale-grad", type=float, default=0.1)
    parser.add_argument("--clamp-max", type=float, default=5.0)
    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Dynamically load the network class
    try:
        model_cls = getattr(obd.models, args.model).Network
    except AttributeError:
        raise RuntimeError(f"Model `{args.model}` not found in `obd.models`")

    # Instantiate and initialize
    model = model_cls(
        num_classes=args.num_classes,
        threshold=args.threshold,
        tau_grad=args.tau_grad,
        scale_grad=args.scale_grad,
        clamp_max=args.clamp_max
    ).to(device)
    model.init_model((448, 448, 3))
    summary(model)
    model.train()

    # Prepare DataLoader
    yolo_target = obd.YOLOtarget(
        anchors=model.anchors,
        scales=model.scale,
        num_classes=model.num_classes,
        ignore_iou_thres=0.5
    )
    dataset = getattr(obd.dataset, args.dataset)(
        root=args.path,
        dataset="track",
        train=True,
        augment_prob=0.0,
        randomize_seq=False
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=yolo_target.collate_fn
    )

    # Fetch one batch
    inputs, targets, _ = next(iter(loader))
    inputs = inputs.to(device)

    # Register hooks to print internal shapes
    hooks = register_hooks(model)

    # Forward pass (hooks will print shapes)
    preds, counts = model(inputs)

    # Remove hooks
    for h in hooks:
        h.remove()

    # Print final outputs
    if isinstance(preds, list):
        for idx, p in enumerate(preds):
            print(f"[Output {idx}] shape: {p.shape}")
    else:
        print(f"[Output] shape: {preds.shape}")
    print(f"[Counts] shape: {counts.shape}")

    # Loss computation
    yolo_loss = obd.YOLOLoss(
        anchors=model.anchors,
        lambda_coord=1.0,
        lambda_noobj=2.0,
        lambda_obj=2.0,
        lambda_cls=4.0,
        lambda_iou=2.0,
        alpha_iou=0.8,
        label_smoothing=0.1
    ).to(device)
    loss, parts = yolo_loss(preds, targets)
    print("Loss:", loss.item(), "Breakdown:", [l.item() for l in parts])

    # Backward + optimizer step
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
    optimizer.step()
    print("Backward + optimizer step done.")


if __name__ == "__main__":
    main()
