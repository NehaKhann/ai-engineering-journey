# pip: transformers>=4.56 torch matplotlib scikit-learn pillow
# %% [markdown]
# # 📘 Module 10 — Multimodal & Diffusion Models
#
# **Generative AI Track • Intermediate**
#
# Everything so far was text. This module leaves text to see how models **understand images**
# (multimodal models) and **generate images** (diffusion models), with real experiments for both.
#
# **Part A, understanding:** CLIP maps images and text into the *same* embedding space (Module 04's
# idea, across two media). We test it on images we draw ourselves, so we know the right answers, and
# we design tests to find where it breaks.
#
# **Part B, generating:** we build a tiny **diffusion model from scratch** on 2D points. Real
# image diffusion models work on the same principle, and 2D data lets us see and measure every step.

# %%
import itertools
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image, ImageDraw
from sklearn.datasets import make_moons
from transformers import CLIPModel, CLIPProcessor

OUT = Path(__file__).parent / "assets" if "__file__" in globals() else Path("assets")
OUT.mkdir(exist_ok=True)
rng = np.random.default_rng(0)
torch.manual_seed(0)

# %% [markdown]
# # Part A: Multimodal understanding with CLIP
#
# ## 1. Images are just numbers
#
# An image is a grid of pixels, each with red, green, and blue values. We draw shapes with known
# colors and positions so we always know the correct answer.

# %%
COLORS = {"red": (220, 30, 30), "green": (30, 160, 60), "blue": (30, 70, 220), "yellow": (240, 200, 20)}
SHAPES = ["circle", "square", "triangle"]
SIZE = 224


def draw_shape(draw, shape, color, cx, cy, r):
    box = [cx - r, cy - r, cx + r, cy + r]
    if shape == "circle":
        draw.ellipse(box, fill=COLORS[color])
    elif shape == "square":
        draw.rectangle(box, fill=COLORS[color])
    else:
        draw.polygon([(cx, cy - r), (cx - r, cy + r), (cx + r, cy + r)], fill=COLORS[color])


def make_image(color, shape):
    """One shape somewhere near the middle, with random position and size."""
    image = Image.new("RGB", (SIZE, SIZE), "white")
    cx, cy = rng.integers(90, 135, 2)
    draw_shape(ImageDraw.Draw(image), shape, color, cx, cy, rng.integers(35, 60))
    return image


sample = make_image("red", "circle")
pixels = np.array(sample)
print("Image array shape:", pixels.shape, "(height, width, RGB channels)")
print("The center pixel:", pixels[112, 112], "-> red, green, blue values from 0 to 255")

combos = list(itertools.product(COLORS, SHAPES))
images = [(make_image(c, s), c, s) for c, s in combos for _ in range(5)]  # 12 combos x 5 images = 60 images

fig, axes = plt.subplots(2, 6, figsize=(12, 4.2))
for ax, (image, c, s) in zip(axes.ravel(), images[::5]):
    ax.imshow(image)
    ax.set_title(f"{c} {s}", fontsize=9)
    ax.axis("off")
plt.tight_layout()
plt.savefig(OUT / "sample_shapes.png", dpi=120)
plt.show()

# %% [markdown]
# ## 2. CLIP: one space for images and text
#
# **CLIP** was trained on hundreds of millions of image and caption pairs so that a picture and its
# description land **close together** in one embedding space. That gives it a superpower called
# **zero-shot classification**: to classify an image, write each candidate label as a sentence and
# pick the sentence whose embedding is closest to the image's. No training on our shapes is needed.

# %%
clip = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").eval()
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")


def clip_scores(pictures, sentences):
    """Probability that each picture matches each sentence (rows: pictures, columns: sentences)."""
    inputs = clip_processor(text=sentences, images=pictures, return_tensors="pt", padding=True)
    with torch.no_grad():
        logits = clip(**inputs).logits_per_image
    return torch.softmax(logits, dim=1).numpy()


labels = [f"a {c} {s}" for c, s in combos]
probabilities = clip_scores([img for img, _, _ in images], labels)
predicted = [combos[i] for i in probabilities.argmax(axis=1)]

color_ok = np.mean([p[0] == c for p, (_, c, _) in zip(predicted, images)])
shape_ok = np.mean([p[1] == s for p, (_, _, s) in zip(predicted, images)])
both_ok = np.mean([p == (c, s) for p, (_, c, s) in zip(predicted, images)])
print(f"Zero-shot on {len(images)} images, choosing among {len(labels)} descriptions (chance = {1 / len(labels):.0%}):")
print(f"  color correct:        {color_ok:.0%}")
print(f"  shape correct:        {shape_ok:.0%}")
print(f"  color AND shape:      {both_ok:.0%}")

# %% [markdown]
# CLIP has never seen these particular drawings and was never trained to name shapes, yet it does
# well. That is the value of a shared embedding space.
#
# ## 3. Where does it break?
#
# A model that does well on the easy test tells you little. Good evaluation hunts for **weaknesses**.
# Two things humans find trivial: **counting** and **spatial relations**.

# %%
def make_counting_image(n):
    image = Image.new("RGB", (SIZE, SIZE), "white")
    draw = ImageDraw.Draw(image)
    xs = rng.permutation(np.linspace(45, 180, 3))[:n] if n > 1 else [rng.integers(90, 135)]
    for x in xs:
        draw_shape(draw, "circle", "blue", int(x), int(rng.integers(90, 135)), 24)
    return image


def make_spatial_image(red_on_left):
    """A red circle and a blue square, side by side."""
    image = Image.new("RGB", (SIZE, SIZE), "white")
    draw = ImageDraw.Draw(image)
    left, right = (55, 170) if red_on_left else (170, 55)
    draw_shape(draw, "circle", "red", left, 112, 32)
    draw_shape(draw, "square", "blue", right, 112, 32)
    return image


counting = [(make_counting_image(n), n) for n in (1, 2, 3) for _ in range(8)]
count_prompts = ["one blue circle", "two blue circles", "three blue circles"]
count_pred = clip_scores([img for img, _ in counting], count_prompts).argmax(axis=1) + 1
counting_acc = np.mean([p == n for p, (_, n) in zip(count_pred, counting)])

spatial = [(make_spatial_image(flag), flag) for flag in (True, False) for _ in range(12)]
spatial_prompts = ["a red circle to the left of a blue square", "a red circle to the right of a blue square"]
spatial_pred = clip_scores([img for img, _ in spatial], spatial_prompts).argmax(axis=1)
spatial_acc = np.mean([(p == 0) == flag for p, (_, flag) in zip(spatial_pred, spatial)])

print(f"{'Test':<38}{'Accuracy':<11}{'Chance'}")
print("-" * 58)
print(f"{'Name the color and shape':<38}{both_ok:<11.0%}{1 / 12:.0%}")
print(f"{'Count circles (1, 2 or 3)':<38}{counting_acc:<11.0%}{1 / 3:.0%}")
print(f"{'Left vs right (red circle / blue square)':<38}{spatial_acc:<11.0%}{1 / 2:.0%}")

# %% [markdown]
# The same model that names colors and shapes well is much worse at **counting** and exactly at a
# coin flip on **left versus right**. Each of those two tests uses only 24 images, so treat the
# exact percentages as approximate. The pattern matches a well-known limitation of this style of
# model: its embedding seems to capture *what is in the picture* far better than *how it is
# arranged* (we did not test why). It matters when you build on top of it: a product that needs
# "the red car on the left" will misbehave.
#
# The lesson generalizes: **test the abilities your application depends on**, not just the ones the
# model is famous for.

# %% [markdown]
# # Part B: Generating with diffusion
#
# ## 4. The idea
#
# A **diffusion model** generates by **removing noise**. Training has two halves:
#
# 1. **Forward process (fixed, no learning):** take real data and add a little noise, again and again,
#    until nothing but static remains.
# 2. **Reverse process (learned):** train a network to **predict the noise** that was added. To
#    generate, start from pure static and repeatedly subtract the predicted noise.
#
# We do this on 2D points shaped like two crescent moons. The math is identical to image diffusion,
# but we can *see* and *measure* it.

# %%
T = 100
betas = torch.linspace(1e-3, 0.1, T)  # how much noise is added at each step
alphas = 1 - betas
alpha_bar = torch.cumprod(alphas, dim=0)  # how much of the original signal survives to step t


def real_data(n, seed):
    points, _ = make_moons(n, noise=0.06, random_state=seed)
    return torch.tensor((points - points.mean(0)) / 0.9, dtype=torch.float32)


data = real_data(4000, seed=0)


def add_noise(x0, t, noise):
    """Jump straight to step t: keep sqrt(alpha_bar) of the signal, add sqrt(1 - alpha_bar) of noise."""
    return alpha_bar[t].sqrt().unsqueeze(1) * x0 + (1 - alpha_bar[t]).sqrt().unsqueeze(1) * noise


fig, axes = plt.subplots(1, 4, figsize=(12, 3))
for ax, t in zip(axes, [0, 20, 50, 99]):
    noisy = add_noise(data[:1500], torch.full((1500,), t), torch.randn(1500, 2))
    ax.scatter(noisy[:, 0], noisy[:, 1], s=2, alpha=0.5)
    ax.set_title(f"step {t}: {alpha_bar[t]:.0%} signal left")
    ax.set_xlim(-3.5, 3.5)
    ax.set_ylim(-3.5, 3.5)
    ax.axis("off")
plt.tight_layout()
plt.savefig(OUT / "forward_noising.png", dpi=130)
plt.show()

# %% [markdown]
# By step 99 almost none of the original structure survives: it is essentially a fuzzy blob of
# Gaussian noise. The model's job is to learn to reverse this.
#
# ## 5. Training: predict the noise
#
# The network sees a noisy point and the step number, and predicts the noise that was mixed in.
# The loss is just the squared error between predicted and true noise.

# %%
class NoisePredictor(torch.nn.Module):
    def __init__(self, hidden=128):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(3, hidden), torch.nn.SiLU(),
            torch.nn.Linear(hidden, hidden), torch.nn.SiLU(),
            torch.nn.Linear(hidden, hidden), torch.nn.SiLU(),
            torch.nn.Linear(hidden, 2),
        )

    def forward(self, x, t):
        return self.net(torch.cat([x, (t.float() / T).unsqueeze(1)], dim=1))


net = NoisePredictor()
optimizer = torch.optim.Adam(net.parameters(), lr=2e-3)
losses = []
for step in range(4000):
    x0 = data[torch.randint(0, len(data), (256,))]
    t = torch.randint(0, T, (256,))
    noise = torch.randn_like(x0)
    loss = torch.nn.functional.mse_loss(net(add_noise(x0, t, noise), t), noise)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    losses.append(loss.item())
    if step % 1000 == 0:
        print(f"step {step:>4}  loss {np.mean(losses[-100:]):.3f}")
print(f"final      loss {np.mean(losses[-100:]):.3f}   (predicting no noise at all would score about 1.0)")

# %% [markdown]
# ## 6. Sampling, and the quality vs speed trade-off
#
# To generate, start from pure noise and step backward. **DDIM** sampling lets us take **fewer, bigger
# steps** by skipping steps, which is how real image models run in a handful of steps instead of
# hundreds. We measure how much quality that costs.
#
# We score quality with **MMD** (maximum mean discrepancy), a number that is near **0 when two sets of
# points come from the same distribution** and grows as they differ.

# %%
def ddim_sample(n, steps):
    """Generate n points using `steps` denoising steps (a deterministic DDIM sampler)."""
    schedule = np.linspace(T - 1, 0, steps).round().astype(int)
    x = torch.randn(n, 2)
    with torch.no_grad():
        for i, t in enumerate(schedule):
            t_batch = torch.full((n,), int(t))
            eps = net(x, t_batch)
            x0_guess = (x - (1 - alpha_bar[t]).sqrt() * eps) / alpha_bar[t].sqrt()
            if i == len(schedule) - 1:
                return x0_guess
            prev = alpha_bar[schedule[i + 1]]
            x = prev.sqrt() * x0_guess + (1 - prev).sqrt() * eps
    return x


def mmd(a, b, sigma=0.4):
    """Distance between two point clouds. 0 means indistinguishable."""
    def k(u, v):
        return torch.exp(-torch.cdist(u, v) ** 2 / (2 * sigma**2)).mean()
    return (k(a, a) + k(b, b) - 2 * k(a, b)).item()


reference = real_data(1000, seed=1)  # fresh real points the model never saw
gauss = torch.distributions.MultivariateNormal(data.mean(0), torch.cov(data.T)).sample((1000,))

results = {}
for steps in (2, 5, 10, 25, 100):
    results[steps] = mmd(ddim_sample(1000, steps), reference)

print(f"{'Sampling steps':<18}{'MMD (lower is better)'}")
print("-" * 40)
print(f"{'real vs real (floor)':<28}{mmd(real_data(1000, seed=2), reference):.4f}")
print(f"{'a plain Gaussian blob':<28}{mmd(gauss, reference):.4f}")
for steps, value in results.items():
    print(f"{f'diffusion, {steps} steps':<28}{value:.4f}")

fig, axes = plt.subplots(1, 4, figsize=(12, 3))
panels = [("real data", reference), ("2 steps", ddim_sample(1000, 2)), ("10 steps", ddim_sample(1000, 10)), ("100 steps", ddim_sample(1000, 100))]
for ax, (title, points) in zip(axes, panels):
    ax.scatter(points[:, 0], points[:, 1], s=2, alpha=0.5)
    ax.set_title(title)
    ax.set_xlim(-3, 3)
    ax.set_ylim(-3, 3)
    ax.axis("off")
plt.tight_layout()
plt.savefig(OUT / "diffusion_samples.png", dpi=130)
plt.show()

# %% [markdown]
# The model has learned the crescent shape from noise alone. In our run:
#
# - **2 and 5 steps were worse than a plain Gaussian blob.** The samples are smeared, because the
#   model is asked to jump from static to data in one or two giant leaps.
# - **Quality improved sharply up to about 25 steps, then flattened.** 25 steps and 100 steps scored
#   the same, so the last 75 steps bought nothing measurable.
# - **Even 100 steps stayed far above the "real vs real" floor.** Our tiny network is not perfect, and
#   that gap is model quality, which sampling steps cannot fix.
#
# That is the same trade-off real image models face: **more denoising steps give better images and
# cost more time, with diminishing returns**, which is why so much research goes into making
# sampling faster (fewer steps, better samplers, distilled models).
#
# ## 7. From this toy to Stable Diffusion
#
# | Toy model here | Real text-to-image model |
# |---|---|
# | 2D points | Images, encoded to a smaller **latent** space by a VAE (Module 01) |
# | Small MLP predicts noise | Large **U-Net** or Transformer predicts noise |
# | Step number as input | Step number **and the text prompt** as input |
# | 100 steps | Typically tens of steps, sometimes fewer |
#
# **How text steers the image:** a text encoder (a CLIP-style model, like the one in Part A) turns
# your prompt into embeddings that are fed to the noise predictor at every step, so the noise it
# removes is noise that moves the picture toward the description. **Classifier-free guidance** runs the
# model twice, with and without the prompt, and pushes the result further toward the prompt version.
#
# > We did not run a full Stable Diffusion model here: it needs a multi-gigabyte download and a GPU to
# > be practical. With a GPU (for example on Colab) the `diffusers` library runs one in a few lines:
# >
# > ```python
# > from diffusers import AutoPipelineForText2Image
# > pipe = AutoPipelineForText2Image.from_pretrained("stabilityai/sdxl-turbo").to("cuda")
# > image = pipe("a red circle above a blue square", num_inference_steps=1).images[0]
# > ```
# >
# > That snippet is shown for reference and was not run in this notebook.
#
# ## 🎯 Key Takeaways
#
# - An image is an array of numbers. **CLIP** puts images and text in one embedding space, enabling zero-shot classification.
# - Multimodal models are strong at *what is in the picture* and weak at **counting and spatial relations**. Test the abilities you depend on.
# - **Diffusion** generates by learning to **remove noise**: add noise in a fixed forward process, learn to predict it, then denoise from static.
# - More denoising steps means better quality but slower generation, with diminishing returns (here 25 steps matched 100).
# - Text prompts steer diffusion through **text embeddings**, and classifier-free guidance strengthens that steer.
#
# ## 🚀 What's Next?
#
# **Module 11 — Deployment & LLMOps**: serving a model behind an API, with streaming, caching, and cost tracking.
