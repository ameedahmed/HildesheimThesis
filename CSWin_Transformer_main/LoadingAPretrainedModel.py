import torch
from timm.models import create_model
from models import CSWin_64_12211_tiny_224  # Adjust if your model class is named differently
from PIL import Image
import torchvision.transforms as T

# 1. Create the model instance (do not use pretrained=True unless you want to load default weights)
model = CSWin_64_12211_tiny_224(pretrained=False)

# 2. Load the checkpoint
checkpoint = torch.load('cswin_tiny_224.pth', map_location='cpu')

# 3. Extract the state_dict (handles both plain and wrapped checkpoints)
if 'state_dict_ema' in checkpoint:
    state_dict = checkpoint['state_dict_ema']
elif 'state_dict' in checkpoint:
    state_dict = checkpoint['state_dict']
else:
    state_dict = checkpoint

# 4. Remove 'module.' prefix if present (for DataParallel checkpoints)
new_state_dict = {}
for k, v in state_dict.items():
    new_key = k.replace('module.', '') if k.startswith('module.') else k
    new_state_dict[new_key] = v

# 5. Load weights into the model
model.load_state_dict(new_state_dict, strict=False)  # strict=False ignores non-matching keys

missing, unexpected = model.load_state_dict(new_state_dict, strict=False)
print("Missing keys:", missing)
print("Unexpected keys:", unexpected)

print(list(state_dict.keys())[:20])  # Print first 20 keys
# 6. Set model to evaluation mode
model.eval()

# 5. (Optional) Move to GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

# Load and preprocess your image
img = Image.open('n01443537_goldfish.jpeg').convert('RGB')
transform = T.Compose([
    T.Resize(224),
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
input_tensor = transform(img).unsqueeze(0)  # Add batch dimension
input_tensor = input_tensor.to(device)
with torch.no_grad():
    output = model(input_tensor) # model inference mode
    pred = output.argmax(dim=1)
    print('Predicted class:', pred.item())