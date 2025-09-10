import gradio as gr
import torch
from PIL import Image
import numpy as np
from distillanydepth.modeling.archs.dam.dam import DepthAnything
from distillanydepth.depth_anything_v2.dpt import DepthAnythingV2
from distillanydepth.utils.image_util import chw2hwc, colorize_depth_maps
from distillanydepth.midas.transforms import Resize, NormalizeImage, PrepareForNet
from torchvision.transforms import Compose
import cv2
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

# Image processing function
def process_image(image, model, device):
    if model is None:
        return None
    
    # Preprocess the image
    # import ipdb;ipdb.set_trace()
    image_np = np.array(image)[..., ::-1] / 255
    
    transform = Compose([
        Resize(700, 700, resize_target=False, keep_aspect_ratio=False, ensure_multiple_of=14, resize_method='lower_bound', image_interpolation_method=cv2.INTER_CUBIC),
        NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        PrepareForNet()
    ])
    
    image_tensor = transform({'image': image_np})['image']
    image_tensor = torch.from_numpy(image_tensor).unsqueeze(0).to(device)
    
    with torch.no_grad():  # Disable autograd since we don't need gradients on CPU
        pred_disp, _ = model(image_tensor)
    torch.cuda.empty_cache()

    # Ensure the depth map is in the correct shape before colorization
    pred_disp_np = pred_disp.cpu().detach().numpy()[0, 0, :, :]  # Remove extra singleton dimensions
    
    # Normalize depth map
    pred_disp = (pred_disp_np - pred_disp_np.min()) / (pred_disp_np.max() - pred_disp_np.min())
    
    # Colorize depth map
    cmap = "Spectral_r"
    depth_colored = colorize_depth_maps(pred_disp[None, ..., None], 0, 1, cmap=cmap).squeeze()  # Ensure correct dimension
    
    # Convert to uint8 for image display
    depth_colored = (depth_colored * 255).astype(np.uint8)
    
    # Convert to HWC format (height, width, channels)
    depth_colored_hwc = chw2hwc(depth_colored)
    
    # Resize to match the original image dimensions (height, width)
    h, w = image_np.shape[:2]
    depth_colored_hwc = cv2.resize(depth_colored_hwc, (w, h), cv2.INTER_LINEAR)
    
    # Convert to a PIL image
    depth_image = Image.fromarray(depth_colored_hwc)
    return depth_image

# Gradio interface function
def gradio_interface(image, model_size):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 根据用户选择的模型大小加载不同的配置
    model_kwargs = {
        "large": dict(
            encoder="vitl", 
            features=256, 
            out_channels=[256, 512, 1024, 1024], 
            use_bn=False, 
            use_clstoken=False, 
            max_depth=150.0, 
            mode='disparity',
            pretrain_type='dinov2',
            del_mask_token=False
        ),
        "base": dict(
            encoder='vitb',
            features=128,
            out_channels=[96, 192, 384, 768],
        ),
        "small": dict(
            encoder='vits',
            features=64,
            out_channels=[48, 96, 192, 384],
        )
    }

    # 根据用户选择的模型大小加载对应的 checkpoint
    if model_size == "large":
        checkpoint_path = "models/large.safetensors"
    elif model_size == "base":
        checkpoint_path = "models/base.safetensors"
    elif model_size == "small":
        checkpoint_path = "models/small.safetensors"
    else:
        raise ValueError(f"Unknown model size: {model_size}")
    # 加载模型
    if model_size == "large":
        model = DepthAnything(**model_kwargs[model_size]).to(device)
    else:
        model = DepthAnythingV2(**model_kwargs[model_size]).to(device)
    model_weights = load_file(checkpoint_path)
    total_params = 0
    for param in model_weights.values():
        num_params = param.numel()
        total_params += num_params
        
    # 每个参数通常为 torch.float32 类型，占 4 个字节
    total_bytes = total_params * 4
    # 转换为 MB
    total_mb = total_bytes / (1024 * 1024)
    print(f"模型的总参数量为: {total_params}")
    print(f"模型的总参数量占用的存储空间约为: {total_mb:.2f} MB")
    model.load_state_dict(model_weights)
    model = model.to(device)
    
    if model is None:
        return None
    
    # 处理图像并返回结果
    depth_image = process_image(image, model, device)
    return depth_image

# 创建 Gradio 界面
iface = gr.Interface(
    fn=gradio_interface,
    inputs=[
        gr.Image(type="pil"),  # 图像输入
        gr.Dropdown(choices=["large", "base", "small"], label="Model Size", value="small")  # 模型大小选择
    ],
    outputs=gr.Image(type="pil"),  # 深度图输出
    title="Depth Estimation Demo",
    description="Upload an image and select a model size (large, base, or small) to see the depth estimation results."
)

# 启动 Gradio 界面
iface.launch()