# TuSimple, ResNet-18 baseline. Values come from the original repo's configs/tusimple.py
# plus constants that the original hardcoded in data/dataloader.py, data/constant.py and
# evaluation/eval_wrapper.py (marked "was hardcoded").

# DATA
dataset = 'Tusimple'
data_root = 'data/tusimple'
seed = 0
num_workers = 4

# was hardcoded: input resolution, (height, width)
input_size = (288, 800)
# was hardcoded: row anchors, defined in pixels of the 288-high network input.
# 64, 68, ..., 284 -> 56 rows (data/constant.py: tusimple_row_anchor)
row_anchor = list(range(64, 288, 4))
# was hardcoded: auxiliary segmentation mask size (height, width)
aux_size = (36, 100)
# was hardcoded: size of the raw TuSimple frames (height, width)
orig_size = (720, 1280)
# was hardcoded: y positions (in raw-frame pixels) the benchmark scores lanes at
test_h_samples = list(range(160, 720, 10))

# was hardcoded: training augmentation (data/dataloader.py)
rotate_deg = 6
ud_offset = 100
lr_offset = 200

# TRAIN
epoch = 100
batch_size = 32
optimizer = 'Adam'    # ['SGD', 'Adam']
learning_rate = 4e-4
weight_decay = 1e-4
momentum = 0.9

scheduler = 'cos'     # ['multi', 'cos']
steps = [50, 75]      # used by 'multi' only
gamma = 0.1
warmup = 'linear'
warmup_iters = 100

# NETWORK
backbone = '18'       # ['18', '34']
griding_num = 100
use_aux = True
num_lanes = 4

# LOSS
sim_loss_w = 1.0
shp_loss_w = 0.0      # the original disables the shape loss on TuSimple

# EXP
log_path = 'runs'     # tensorboard logs and checkpoints (gitignored via *.pth)
save_every = 10       # keep ep%03d.pth every N epochs (last.pth is always kept)
finetune = None
resume = None

# TEST
test_model = None
test_work_dir = 'runs/eval'
