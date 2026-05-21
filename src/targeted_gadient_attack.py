import os
import numpy as np
from PIL import Image

import torch
from torch import nn
from torchvision import models
from tkinter import filedialog, Tk


def get_imagenet_class_name(class_idx):
    try:
        categories = models.AlexNet_Weights.DEFAULT.meta["categories"]
        if 0 <= class_idx < len(categories):
            return categories[class_idx]
    except Exception:
        pass
    return f"class_{class_idx}"


def preprocess_image(pil_img):
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    pil_img = pil_img.convert("RGB")
    pil_img = pil_img.resize((224, 224), Image.Resampling.LANCZOS)

    img = torch.from_numpy(np.array(pil_img)).float() / 255.0
    img = img.permute(2, 0, 1)

    img = (img - mean) / std
    img = img.unsqueeze(0)

    return img


def deprocess_image(tensor):
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

    img = tensor.clone().detach()
    img = img * std + mean
    img = torch.clamp(img, 0, 1)

    img = img.squeeze(0).permute(1, 2, 0).cpu().numpy()
    img = (img * 255).astype(np.uint8)

    return Image.fromarray(img)


def get_params(img_path):
    file_name = os.path.splitext(os.path.basename(img_path))[0]

    original_image = Image.open(img_path).convert("RGB")
    prep_img = preprocess_image(original_image)

    weights = models.AlexNet_Weights.DEFAULT
    model = models.alexnet(weights=weights)
    model.eval()

    with torch.no_grad():
        out = model(prep_img)
        org_class = int(out.argmax(1).item())
        org_conf = nn.functional.softmax(out, dim=1)[0, org_class].item()

    return original_image, prep_img, org_class, org_conf, file_name, model


class FastGradientSignTargeted:
    def __init__(self, model, alpha=0.005, steps=40):
        self.model = model
        self.alpha = alpha
        self.steps = steps

        os.makedirs("../generated", exist_ok=True)

    def generate(self, original_image, org_class, target_class):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)
        self.model.eval()

        x = preprocess_image(original_image).to(device)
        x.requires_grad_(True)

        target = torch.tensor([target_class], device=device)
        loss_fn = nn.CrossEntropyLoss()

        for i in range(self.steps):
            self.model.zero_grad()

            out = self.model(x)
            loss = loss_fn(out, target)
            loss.backward()

            x = x - self.alpha * torch.sign(x.grad)
            x = x.detach().requires_grad_(True)

            pred = out.argmax(1).item()
            conf = nn.functional.softmax(out, dim=1)[0, pred].item()

            print(f"Step {i}:")
            print(f"[TENSOR] class={pred} ({get_imagenet_class_name(pred)}), conf={conf}")

            adv_img = deprocess_image(x.cpu())

            verify_tensor = preprocess_image(adv_img).to(device)

            with torch.no_grad():
                verify_out = self.model(verify_tensor)
                verify_pred = verify_out.argmax(1).item()
                verify_conf = nn.functional.softmax(verify_out, dim=1)[0, verify_pred].item()

            print(f"[IMAGE] class={verify_pred} ({get_imagenet_class_name(verify_pred)}), conf={verify_conf}")

            if verify_pred == target_class:
                print("\nAttack succeeded (verified after reconstruction)!")

                adv_path = f"../generated/adv_from_{org_class}_to_{target_class}.png"
                adv_img.save(adv_path)

                orig_resized = original_image.resize((224, 224))
                noise = np.array(adv_img).astype(int) - np.array(orig_resized).astype(int)
                noise = np.clip(noise + 128, 0, 255).astype(np.uint8)

                noise_img = Image.fromarray(noise)
                noise_path = f"../generated/noise_from_{org_class}_to_{target_class}.png"

                noise_img.save(noise_path)

                print(f"\nSaved adversarial image: {adv_path}")
                print(f"Saved noise image: {noise_path}")
                break

            elif i == self.steps - 1:
                print("\nAttack failed.")


if __name__ == "__main__":
    root = Tk()
    root.withdraw()

    img_path = filedialog.askopenfilename(
        title="Select an image",
        filetypes=[("Images", "*.jpg *.jpeg *.png *.bmp"), ("All files", "*.*")]
    )

    if not img_path:
        print("No image selected.")
        exit()

    original_image, prep_img, org_class, org_conf, _, model = get_params(img_path)

    print(f"Original class: {org_class} ({get_imagenet_class_name(org_class)})")
    print(f"Confidence: {org_conf}")

    try:
        target_class = int(input("Enter target class (0–999): "))
    except ValueError:
        print("Invalid input")
        exit()

    attack = FastGradientSignTargeted(model)
    attack.generate(original_image, org_class, target_class)
