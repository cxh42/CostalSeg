import torch
import segmentation_models_pytorch as smp
'''
This script convert trained model to cleaned weights model for deployment in huggingface.
'''
# 1. Create model architecture
model = smp.create_model(
    "DeepLabV3Plus",
    encoder_name="efficientnet-b6",
    in_channels=3,
    classes=7,
    encoder_weights=None
)

# 2. Load the weight dictionary
state_dict = torch.load("XXXXXXXXXX.pth", map_location="cpu")
new_state = {}
for k, v in state_dict.items():
    name = k
    if k.startswith("module."):
        name = k[7:]
    elif k.startswith("model."):
        name = k[6:]
    new_state[name] = v

# 3. Load weights into the model
model.load_state_dict(new_state)

# 4. Switch to evaluation mode
model.eval()

torch.save(model.state_dict(), "XXXXXXXXXX_weights.pth")
