import torch
from torch.utils.data import DataLoader
from lava.lib.dl.slayer import obd

def print_forward_details(model, x):
    # 1) Normalize
    if model.normalize_mean.device != x.device:
        model.normalize_mean = model.normalize_mean.to(x.device)
        model.normalize_std = model.normalize_std.to(x.device)
    x = (x - model.normalize_mean) / model.normalize_std
    print("=== After normalization ===", x.shape)

    # 2) Input blocks
    for i, block in enumerate(model.input_blocks):
        print(f"[Input Block {i}] before: {x.shape}")
        x = block(x)
        print(f"[Input Block {i}] after:  {x.shape}")

    # 3) Backbone + capture feature at block 5
    intermediate = x
    features = []
    for i, block in enumerate(model.blocks):
        print(f"[Backbone Block {i}] before: {intermediate.shape}")
        intermediate = block(intermediate)
        print(f"[Backbone Block {i}] after:  {intermediate.shape}")
        if i == 5:
            features.append(intermediate.clone())

    # 4) Large-object branch
    large = intermediate.clone()
    for i, block in enumerate(model.large_object_branch):
        print(f"[Large Branch {i}] before: {large.shape}")
        large = block(large)
        print(f"[Large Branch {i}] after:  {large.shape}")

    # 5) Small-object branch (use feature from block 5)
    small = features[0]
    for i, block in enumerate(model.small_object_branch):
        print(f"[Small Branch {i}] before: {small.shape}")
        small = block(small)
        print(f"[Small Branch {i}] after:  {small.shape}")

    return large, small

# --- Set up model ---
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = obd.models.yolo_kp.Network(
    threshold=0.1,
    tau_grad=0.1,
    scale_grad=0.1,
    num_classes=4,
    clamp_max=5.0
).to(device)
model.init_model((448, 448, 3))
model.eval()

# --- Create YOLO target for collate_fn ---
yolo_target = obd.YOLOtarget(
    anchors=model.anchors,
    scales=model.scale,
    num_classes=model.num_classes,
    ignore_iou_thres=0.5  # match your train script setting
)

# --- Load real custom dataset ---
dataset = obd.dataset.custom(
    root='/home/bcl/Documents/udacity',
    dataset='track',
    train=True,
    augment_prob=0.0,
    randomize_seq=False
)
loader = DataLoader(
    dataset,
    batch_size=1,
    shuffle=False,
    collate_fn=yolo_target.collate_fn
)

# --- Run one batch through print_forward_details ---
inputs, targets, bboxes = next(iter(loader))
inputs = inputs.to(device)

large_out, small_out = print_forward_details(model, inputs)

# --- Heads ---
print("=== Final Heads ===")
head_large = model.heads[1](model.heads[0](large_out))
print("Large head output shape:", head_large.shape)

small_in = small_out.reshape(small_out.size(0), -1, 1, 1, small_out.size(-1))
head_small = model.heads[3](model.heads[2](small_in))
print("Small head output shape:", head_small.shape)

# after your forward‐only test...

# 1) Create the loss module
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

# 2) Forward + compute loss on your real batch
predictions, counts = model(inputs)            
loss, loss_parts = yolo_loss(predictions, targets)

print("Loss:", loss.item())
print("Loss breakdown:", [l.item() for l in loss_parts])

# 3) Backward + optimizer step
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
optimizer.zero_grad()
loss.backward()
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
optimizer.step()
