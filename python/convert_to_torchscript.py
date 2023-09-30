import os

import torch

import data_processing_pytorch
import load_model
import modelconfigs
from model_pytorch import Model

BATCH_SIZE = 1
DESTINATION_PATH = "/nas/ucb/ttseng/go_attack/torch-script/traced-test-model.pt"
GPU_ID = 1
RANK = 0
TRAIN_DATA = [
        "/nas/ucb/k8/go-attack/victimplay/ttseng-cyclic-vs-b18-s6201m-20230517-130803/selfplay/t0-s9737216-d2331818/tdata/0B45CABDA2418864.npz"
]
WORLD_SIZE = 1

if torch.cuda.is_available():
    device = torch.device("cuda", GPU_ID)
else:
    print("No GPU, using CPU")
    device = torch.device("cpu")


# TODO argparse stuff instead of hardcoding

model, _, _ = load_model.load_model(
        checkpoint_file="/nas/ucb/ttseng/go_attack/victim-weights/kata1-b18c384nbt-s7619896320-d3691351602/model.ckpt",
        use_swa=False,
        device=device,
)
model.eval()

input_batch = next(data_processing_pytorch.read_npz_training_data(
    npz_files=TRAIN_DATA,
    batch_size=BATCH_SIZE,
    world_size=WORLD_SIZE,
    rank=RANK,
    pos_len=model.pos_len,
    device=device,
    randomize_symmetries=True,
    model_config=model.config,
))

# # Printing so that we can check whether the C++ version gives the same output
# torch.set_printoptions(profile="full")
# print("Input:")
# print("shape:", input_batch["binaryInputNCHW"].shape, input_batch["globalInputNC"].shape)
# print("binaryInputNCHW")
# print(input_batch["binaryInputNCHW"])
# print("globalInputNC")
# print(input_batch["globalInputNC"])
# print("Output:")
# output = model(input_batch["binaryInputNCHW"], input_batch["globalInputNC"])
# print("len:", len(output))
# for elem in output:
#     print("inner len:", len(elem))
#     for e in elem:
#         print("shape:", e.shape)
#         print(e)

traced_script_module = torch.jit.trace(
        func=model,
        example_inputs=(input_batch["binaryInputNCHW"], input_batch["globalInputNC"]),
)
traced_script_module.cpu()
traced_script_module.save(DESTINATION_PATH)
print("Model saved to", DESTINATION_PATH)
