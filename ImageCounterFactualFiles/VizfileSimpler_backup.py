import sys
import os
import torch
import json
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms as T
from datetime import datetime
import gc


# Add base path
base_path = "/data/vifapi/ameed_ahmed_thesis/HildesheimThesis/CSWin_Transformer_main"
sys.path.append(base_path)

from models.cswinmodified import CSWin_96_24322_base_384

def avg_heads(cam, grad):
    """Enhanced avg_heads function with gradient scaling."""
    cam = cam.reshape(-1, cam.shape[-2], cam.shape[-1])
    grad = grad.reshape(-1, grad.shape[-2], grad.shape[-1])
    
    # Check if gradients are very small and scale them up
    grad_magnitude = torch.abs(grad).mean()
    if grad_magnitude < 1e-6:
        grad = grad * (1e-3 / grad_magnitude)
        print(f"Warning: Gradients were very small ({grad_magnitude:.2e}), scaled up")
    
    cam = grad * cam
    cam = cam.clamp(min=0).mean(dim=0)
    return cam

def apply_self_attention_rules(R_ss, cam_ss):
    R_ss_addition = torch.matmul(cam_ss, R_ss)
    return R_ss_addition

def generate_relevance(model, input, index=None):
    """Generate relevance maps focusing only on global attention (stage 4)."""

    result = model(input, register_hook=True, return_attentions=True)
    if len(result) == 5:
        output, global_attention, vertical_attention, horizontal_attention, overall_attention = result
    else:
        output, global_attention = result
    
    if index == None:
        index = np.argmax(output.cpu().data.numpy(), axis=-1)
    
    one_hot = np.zeros((1, output.size()[-1]), dtype=np.float32)
    one_hot[0, index] = 1
    one_hot = torch.from_numpy(one_hot).requires_grad_(True)
    one_hot = torch.sum(one_hot * output)
    model.zero_grad()
    one_hot.backward(retain_graph=True)
 
    # Get number of tokens from stage 4
    num_tokens = model.stage4[0].attns[0].get_attention_map().shape[-1]
    R_global = torch.eye(num_tokens, num_tokens)
    
    # Process both blocks in stage 4
    for blk_idx, blk in enumerate(model.stage4):
        grad_g = blk.attns[0].get_attn_gradients()
        cam_g = blk.attns[0].get_attention_map()
        
        if grad_g is not None and cam_g is not None:
            cam_g_processed = avg_heads(cam_g, grad_g)
            print(f"Stage 4 block {blk_idx + 1}: Using gradient-weighted attention")
            R_global += apply_self_attention_rules(R_global, cam_g_processed)
        else:
            print(f"grad empty line 72")
            break

    return R_global.sum(dim=0)

def show_cam_on_image(img, mask):
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    cam = heatmap + np.float32(img)
    cam = cam / np.max(cam)
    return cam

def generate_visualization(model, original_image, img_size, class_index=None):
    """Generate visualization focusing only on global attention."""
    R_global = generate_relevance(model, original_image.unsqueeze(0), index=class_index)
    
    def process_relevance(R):
        R = R.detach()
        numel = R.numel()
        # Find best rectangular shape
        best_h, best_w = 1, numel
        min_diff = numel
        for h in range(1, int(np.sqrt(numel)) + 1):
            if numel % h == 0:
                w = numel // h
                if abs(w - h) < min_diff:
                    best_h, best_w = h, w
                    min_diff = abs(w - h)
        
        if best_h * best_w != numel:
            next_square = int(np.ceil(np.sqrt(numel))) ** 2
            pad_len = next_square - numel
            R = torch.nn.functional.pad(R, (0, pad_len))
            best_h = best_w = int(np.sqrt(next_square))
        
        R = R.reshape(1, 1, best_h, best_w)
        R = torch.nn.functional.interpolate(R, size=(img_size, img_size), mode='bilinear')
        R = R.reshape(img_size, img_size).data.cpu().numpy()
        R = (R - R.min()) / (R.max() - R.min())
        return R

    global_map = process_relevance(R_global)
    image_np = original_image.permute(1, 2, 0).data.cpu().numpy()
    image_np = (image_np - image_np.min()) / (image_np.max() - image_np.min())

    vis_global = show_cam_on_image(image_np, global_map)
    vis_global = np.uint8(255 * vis_global)
    vis_global = cv2.cvtColor(np.array(vis_global), cv2.COLOR_RGB2BGR)

    return vis_global, global_map

def load_model():
    """Load the pretrained model."""
    # Model configuration
    model_path = f"{base_path}/outputSmoothingLossPLScheduler/NPortrait32B/finetune/20250713-125113-CSWin_96_24322_base_384-384/model_best.pth.tar"
    num_classes = 9
    img_size = 384
    
    device = torch.device('cpu')
    print(f"Using device: {device}")
    
    # Create model
    model = CSWin_96_24322_base_384(pretrained=False, num_classes=num_classes, img_size=img_size)
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    
    # Extract state_dict
    if 'state_dict_ema' in checkpoint:
        state_dict = checkpoint['state_dict_ema']
    elif 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
    
    # Remove 'module.' prefix if present
    new_state_dict = {}
    for k, v in state_dict.items():
        new_key = k.replace('module.', '') if k.startswith('module.') else k
        new_state_dict[new_key] = v
    
    # Load weights
    model.load_state_dict(new_state_dict, strict=False)
    model.eval()
    model = model.to(device)
    
    return model, device

def load_labels():
    """Load class labels."""
    labels_path = f"{base_path}/PortraitDataset/artist_class_index.json"
    with open(labels_path, 'r') as f:
        temp_label = json.load(f)
    
    labels = {}
    label_list = []
    for i in range(9):  # num_classes
        full_path = temp_label[str(i)][0]
        folder_name = os.path.basename(full_path)
        style_name = folder_name.replace("_resizeddataset", "")
        labels[style_name] = i
        label_list.append(style_name)
    
    return labels, label_list

def process_image(image_path, model, device, labels, label_list, output_dir):
    """Process a single image and generate attention analysis."""
    
    # Load and preprocess image
    img = Image.open(image_path).convert('RGB')
    transform = T.Compose([
        T.Resize(384),
        T.CenterCrop(384),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    input_tensor = transform(img).unsqueeze(0).to(device)
    
    # Get prediction
    with torch.no_grad():
        output = model(input_tensor)
        pred = output.argmax(dim=1)
        predicted_class_index = pred.item()
        confidence = torch.softmax(output, dim=1).max().item()
    
    # Determine true label from folder structure
    parent_folder = os.path.basename(os.path.dirname(image_path))
    true_label_idx = labels.get(parent_folder, 0)
    
    # Generate visualizations
    vis_global, raw_global = generate_visualization(
        model, transform(img), 384, class_index=predicted_class_index
    )
    
    # Save results
    base_filename = os.path.splitext(os.path.basename(image_path))[0]
    
    # Save raw attention map
    plt.figure(figsize=(10, 8))
    plt.imshow(raw_global, cmap='hot')
    plt.title('Raw Global Attention Map')
    plt.colorbar()
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'RawAttentionMap_{base_filename}.png'), 
                dpi=150, bbox_inches='tight')
    plt.close()
    
    # Save comparison plot
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    
    # Original image
    axs[0].imshow(img)
    axs[0].set_title(f'Original\nTrue: {label_list[true_label_idx]}\nPred: {label_list[predicted_class_index]}')
    axs[0].axis('off')
    
    # Global attention
    axs[1].imshow(vis_global)
    axs[1].set_title('Global Attention')
    axs[1].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'AttentionAnalysis_{base_filename}.png'), 
                dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Processed: {base_filename}")
    print(f"True: {label_list[true_label_idx]}, Predicted: {label_list[predicted_class_index]}")
    print(f"Confidence: {confidence:.3f}")
    print(f"Correct: {true_label_idx == predicted_class_index}")
    print("-" * 50)

def main():
    # Configuration
    test_data_dir = f"{base_path}/PortraitDataset/test/Aubry, Peter II_ogdataset"
    output_dir = f"{base_path}/ImageCounterfactualExplanations/NewAttentionMap"
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load model and labels
    print("Loading model...")
    model, device = load_model()
    
    print("Loading labels...")
    labels, label_list = load_labels()
    
    # Get all image files
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
    image_files = []
    
    for root, dirs, files in os.walk(test_data_dir):
        for file in files:
            if any(file.lower().endswith(ext) for ext in image_extensions):
                image_files.append(os.path.join(root, file))
    
    print(f"Found {len(image_files)} images to process")
    
    # Process each image
    for i, image_path in enumerate(image_files):
        try:
            print(f"Processing {i+1}/{len(image_files)}: {os.path.basename(image_path)}")
            process_image(image_path, model, device, labels, label_list, output_dir)
            gc.collect()
        except Exception as e:
            print(f"Error processing {image_path}: {str(e)}")
            gc.collect()
            break
    
    print("Processing complete!")

if __name__ == "__main__":
    main()