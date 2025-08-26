import torch
import numpy as np
from PIL import Image, ImageOps
import cv2

class SnapsMaskFillNode:
    """
    A ComfyUI custom node that takes a base image with a white box and places
    a second image inside the white box area, fitting it to the box dimensions.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "base_image": ("IMAGE",),
                "fill_image": ("IMAGE",),
                "white_threshold": ("FLOAT", {
                    "default": 0.9,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "display": "slider"
                }),
                "fit_mode": (["fit_height", "fit_width", "stretch"], {
                    "default": "fit_height"
                }),
                "margin_size": ("INT", {
                    "default": 5,
                    "min": 0,
                    "max": 50,
                    "step": 1,
                    "display": "number"
                })
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("output_image",)
    FUNCTION = "process_images"
    CATEGORY = "image/processing"
    
    def tensor_to_pil(self, tensor):
        """Convert tensor to PIL Image"""
        # Handle batch dimension
        if len(tensor.shape) == 4:
            tensor = tensor.squeeze(0)
        
        # Ensure tensor is in [H, W, C] format
        if len(tensor.shape) == 3:
            # If tensor is [C, H, W], convert to [H, W, C]
            if tensor.shape[0] == 3 or tensor.shape[0] == 4:
                tensor = tensor.permute(1, 2, 0)
        
        # Ensure values are in [0, 1] range
        tensor = torch.clamp(tensor, 0, 1)
        
        # Convert to numpy and scale to [0, 255]
        np_image = (tensor.cpu().numpy() * 255).astype(np.uint8)
        
        return Image.fromarray(np_image)
    
    def tensor_batch_to_pil_list(self, tensor_batch):
        """Convert batch tensor to list of PIL Images"""
        pil_images = []
        
        # Handle single image case
        if len(tensor_batch.shape) == 3:
            tensor_batch = tensor_batch.unsqueeze(0)
        
        # Process each image in the batch
        for i in range(tensor_batch.shape[0]):
            single_tensor = tensor_batch[i]
            
            # Ensure values are in [0, 1] range
            single_tensor = torch.clamp(single_tensor, 0, 1)
            
            # Convert to numpy and scale to [0, 255]
            np_image = (single_tensor.cpu().numpy() * 255).astype(np.uint8)
            
            pil_images.append(Image.fromarray(np_image))
        
        return pil_images
    
    def pil_to_tensor(self, pil_image):
        """Convert PIL Image to tensor"""
        # Convert to RGB if needed
        if pil_image.mode != 'RGB':
            pil_image = pil_image.convert('RGB')
        
        # Convert to numpy array
        np_image = np.array(pil_image).astype(np.float32) / 255.0
        
        # Convert to tensor and add batch dimension
        tensor = torch.from_numpy(np_image).unsqueeze(0)
        
        return tensor
    
    def pil_list_to_tensor_batch(self, pil_images):
        """Convert list of PIL Images to batch tensor"""
        tensors = []
        
        for pil_image in pil_images:
            # Convert to RGB if needed
            if pil_image.mode != 'RGB':
                pil_image = pil_image.convert('RGB')
            
            # Convert to numpy array
            np_image = np.array(pil_image).astype(np.float32) / 255.0
            
            # Convert to tensor
            tensor = torch.from_numpy(np_image)
            tensors.append(tensor)
        
        # Stack tensors to create batch
        batch_tensor = torch.stack(tensors, dim=0)
        
        return batch_tensor
    
    def find_white_box(self, image, threshold=0.9):
        """Find the largest white rectangular area in the image"""
        # Convert PIL to numpy array
        img_array = np.array(image)
        
        # Convert to grayscale
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        
        # Create binary mask for white areas
        white_mask = (gray > threshold * 255).astype(np.uint8)
        
        # Find contours
        contours, _ = cv2.findContours(white_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            # If no white area found, return the entire image bounds
            return (0, 0, image.width, image.height)
        
        # Find the largest rectangular contour
        largest_area = 0
        best_rect = None
        
        for contour in contours:
            # Get bounding rectangle
            x, y, w, h = cv2.boundingRect(contour)
            area = w * h
            
            if area > largest_area:
                largest_area = area
                best_rect = (x, y, x + w, y + h)
        
        return best_rect if best_rect else (0, 0, image.width, image.height)
    
    def add_white_margin(self, image, margin_size):
        """Add white margin around the image"""
        if margin_size <= 0:
            return image
        
        # Calculate new dimensions with margin
        new_width = image.width + (margin_size * 2)
        new_height = image.height + (margin_size * 2)
        
        # Create new image with white background
        new_image = Image.new('RGB', (new_width, new_height), 'white')
        
        # Paste original image in the center
        new_image.paste(image, (margin_size, margin_size))
        
        return new_image
    
    def fit_image_to_box(self, image, box_width, box_height, fit_mode):
        """Fit image to the specified box dimensions"""
        img_width, img_height = image.size
        
        if fit_mode == "fit_height":
            # Scale to fit height, maintain aspect ratio
            scale = box_height / img_height
            new_width = int(img_width * scale)
            new_height = box_height
            
            # If scaled width is larger than box width, scale to fit width instead
            if new_width > box_width:
                scale = box_width / img_width
                new_width = box_width
                new_height = int(img_height * scale)
                
        elif fit_mode == "fit_width":
            # Scale to fit width, maintain aspect ratio
            scale = box_width / img_width
            new_width = box_width
            new_height = int(img_height * scale)
            
            # If scaled height is larger than box height, scale to fit height instead
            if new_height > box_height:
                scale = box_height / img_height
                new_width = int(img_width * scale)
                new_height = box_height
                
        else:  # stretch
            # Stretch to exact box dimensions
            new_width = box_width
            new_height = box_height
        
        # Resize the image
        resized_image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        return resized_image
    
    def process_single_image_pair(self, base_pil, fill_pil, white_threshold, fit_mode, margin_size):
        """Process a single pair of base and fill images"""
        # Add white margin to the fill image
        fill_with_margin = self.add_white_margin(fill_pil, margin_size)
        
        # Find the white box in the base image
        box_coords = self.find_white_box(base_pil, white_threshold)
        x1, y1, x2, y2 = box_coords
        box_width = x2 - x1
        box_height = y2 - y1
        
        # Fit the fill image (with margin) to the box dimensions
        fitted_image = self.fit_image_to_box(fill_with_margin, box_width, box_height, fit_mode)
        
        # Create a copy of the base image
        result_image = base_pil.copy()
        
        # Calculate position to center the fitted image in the box
        fitted_width, fitted_height = fitted_image.size
        paste_x = x1 + (box_width - fitted_width) // 2
        paste_y = y1 + (box_height - fitted_height) // 2
        
        # Paste the fitted image onto the base image
        result_image.paste(fitted_image, (paste_x, paste_y))
        
        return result_image
    
    def process_images(self, base_image, fill_image, white_threshold, fit_mode, margin_size):
        """Main processing function - handles both single images and batches"""
        # Convert tensors to PIL Images
        base_pil_list = self.tensor_batch_to_pil_list(base_image)
        fill_pil_list = self.tensor_batch_to_pil_list(fill_image)
        
        # Ensure both batches have the same size
        batch_size = max(len(base_pil_list), len(fill_pil_list))
        
        # If one batch is smaller, repeat the last image to match sizes
        if len(base_pil_list) < batch_size:
            base_pil_list.extend([base_pil_list[-1]] * (batch_size - len(base_pil_list)))
        if len(fill_pil_list) < batch_size:
            fill_pil_list.extend([fill_pil_list[-1]] * (batch_size - len(fill_pil_list)))
        
        # Process each image pair
        result_pil_list = []
        for i in range(batch_size):
            result_image = self.process_single_image_pair(
                base_pil_list[i], 
                fill_pil_list[i], 
                white_threshold, 
                fit_mode, 
                margin_size
            )
            result_pil_list.append(result_image)
        
        # Convert back to tensor batch
        result_tensor = self.pil_list_to_tensor_batch(result_pil_list)
        
        return (result_tensor,)

# Node mappings for ComfyUI
NODE_CLASS_MAPPINGS = {
    "SnapsMaskFillNode": SnapsMaskFillNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SnapsMaskFillNode": "Snaps Mask Fill"
}