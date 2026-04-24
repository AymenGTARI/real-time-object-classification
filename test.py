import sys, os
import torch
import cv2
import numpy as np
from torchvision import datasets, transforms, models
from PIL import Image
from collections import deque

# =====================
# CONFIG
# =====================
DATA_DIR        = "dataset"
MODEL_PATH      = "model.pth"
DEVICE          = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SMOOTH_N        = 5      # lissage sur N frames

# Seuils de détection
THRESHOLD_CONF  = 0.70   # confiance minimale pour accepter une classe
THRESHOLD_ENT   = 0.55   #  inconnu

# =====================
# VÉRIFICATIONS
# =====================
if not os.path.exists(MODEL_PATH):
    print("❌ model.pth introuvable — lancez train.py d'abord.")
    sys.exit(1)

# =====================
# CLASSES
# =====================
class_names = datasets.ImageFolder(DATA_DIR).classes
print(f"Classes : {class_names}")

# =====================
# MODÈLE
# =====================
model = models.resnet18(weights=None)
model.fc = torch.nn.Sequential(
    torch.nn.Dropout(0.4),
    torch.nn.Linear(model.fc.in_features, len(class_names))
)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval().to(DEVICE)
print("✅ Modèle chargé.")

# =====================
# TRANSFORM
# =====================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def normalized_entropy(probs):
    """
    Entropie normalisée entre 0 et 1.
    0 = le modèle est très sûr d'une classe
    1 = le modèle est totalement confus → objet inconnu
    """
    probs  = np.clip(probs, 1e-9, 1.0)
    ent    = -np.sum(probs * np.log(probs))
    max_ent = np.log(len(probs))   # entropie max possible
    return ent / max_ent

# =====================
# CAMÉRA
# =====================
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ Impossible d'ouvrir la caméra.")
    sys.exit(1)

cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
print("📷 Caméra active — ESC pour quitter")

prob_buffer = deque(maxlen=SMOOTH_N)

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    h, w = frame.shape[:2]

    # ── Zone de détection centrée ──
    size   = int(min(h, w) * 0.60)
    cx, cy = w // 2, h // 2
    x1, y1 = cx - size // 2, cy - size // 2
    x2, y2 = cx + size // 2, cy + size // 2
    roi    = frame[y1:y2, x1:x2]

    # ── Prédiction ──
    img = Image.fromarray(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))
    t   = transform(img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        probs = torch.softmax(model(t), dim=1).cpu().numpy()[0]

    prob_buffer.append(probs)
    avg_probs = np.mean(prob_buffer, axis=0)
    idx       = int(np.argmax(avg_probs))
    conf      = float(avg_probs[idx])
    entropy   = normalized_entropy(avg_probs)

    # ── Décision : connu ou inconnu ──
    # Inconnu si : confiance trop faible OU entropie trop haute
    if conf >= THRESHOLD_CONF and entropy <= THRESHOLD_ENT:
        detected = True
        label    = class_names[idx]
        color    = (0, 220, 80)
    else:
        detected = False
        label    = "Inconnu"
        color    = (60, 60, 220)   # bleu pour inconnu

    # ── Fond assombri hors ROI ──
    dark = (frame * 0.35).astype(np.uint8)
    dark[y1:y2, x1:x2] = frame[y1:y2, x1:x2]
    frame = dark

    # ── Coins de détection ──
    corner = size // 8
    pts  = [(x1,y1), (x2,y1), (x1,y2), (x2,y2)]
    dirs = [( 1, 1), (-1, 1), ( 1,-1), (-1,-1)]
    for (px, py), (dx, dy) in zip(pts, dirs):
        cv2.line(frame, (px, py), (px + dx*corner, py), color, 3)
        cv2.line(frame, (px, py), (px, py + dy*corner), color, 3)

    # ── Bandeau bas ──
    band_h  = 90
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, h - band_h), (w, h), (10, 10, 10), -1)
    cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)

    # ── Label ──
    text = f"{label}  {conf*100:.0f}%"
    (tw, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_DUPLEX, 1.2, 2)
    cv2.putText(frame, text, ((w - tw)//2, h - band_h + 38),
                cv2.FONT_HERSHEY_DUPLEX, 1.2, color, 2, cv2.LINE_AA)

    # ── Barre de confiance ──
    bar_w  = int(w * 0.6)
    bar_x  = (w - bar_w) // 2
    bar_y  = h - 28
    filled = int(bar_w * conf)
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + 10), (60,60,60), -1)
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + filled, bar_y + 10), color, -1)

    # ── Entropie affichée ──
    ent_text = f"incertitude: {entropy*100:.0f}%"
    (ew, _), _ = cv2.getTextSize(ent_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    cv2.putText(frame, ent_text, ((w - ew)//2, h - band_h + 62),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (160,160,160), 1, cv2.LINE_AA)

    # ── Guide si inconnu ──
    if not detected:
        guide = "Objet non reconnu — placez un objet connu dans le cadre"
        (gw, _), _ = cv2.getTextSize(guide, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        cv2.putText(frame, guide, ((w - gw)//2, y1 - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180,180,180), 1, cv2.LINE_AA)

    # ── Top-3 (coin haut gauche) ──
    top3 = np.argsort(avg_probs)[::-1][:3]
    for i, ti in enumerate(top3):
        c = (0, 220, 80) if (ti == idx and detected) else (140, 140, 140)
        cv2.putText(frame, f"{class_names[ti]}: {avg_probs[ti]*100:.0f}%",
                    (10, 25 + i*22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, c, 1, cv2.LINE_AA)

    cv2.imshow("Classificateur — ESC pour quitter", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
print("👋 Terminé.")