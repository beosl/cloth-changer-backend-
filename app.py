from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import FileResponse, JSONResponse
from PIL import Image
import io, os
import torch, numpy as np
from diffusers import StableDiffusionInpaintPipeline
import cv2, mediapipe as mp

CHAR_DIR = "characters"
os.makedirs(CHAR_DIR, exist_ok=True)

app = FastAPI(title="Advanced Cloth Changer API")

device = "cuda" if torch.cuda.is_available() else "cpu"
pipe = StableDiffusionInpaintPipeline.from_pretrained(
    "runwayml/stable-diffusion-inpainting",
    torch_dtype=torch.float16 if device=="cuda" else torch.float32
)
pipe = pipe.to(device)

mp_face = mp.solutions.face_mesh.FaceMesh(static_image_mode=True)

def get_face_mask(image):
    img = np.array(image)
    img_rgb = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    h, w, _ = img.shape
    mask = np.zeros((h, w), np.uint8)
    results = mp_face.process(img_rgb)
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            pts = [(int(lm.x*w), int(lm.y*h)) for lm in face_landmarks.landmark]
            hull = cv2.convexHull(np.array(pts))
            cv2.fillConvexPoly(mask, hull, 255)
    return Image.fromarray(mask)

def get_part_mask(image, part="upper"):
    img_np = np.array(image.convert("L"))
    mask = np.zeros_like(img_np)
    h, w = mask.shape
    if part=="upper": mask[:h//2,:] = 255
    elif part=="lower": mask[h//2:,:] = 255
    elif part=="jacket": mask[h//3:h//2,:] = 255
    elif part=="accessory": mask[:h//4,:] = 255
    else: mask[:,:] = 255
    return Image.fromarray(mask)

def cloth_changer(person_bytes, cloth_bytes, part, face_lock=True, bg_lock=True):
    person = Image.open(io.BytesIO(person_bytes)).convert("RGB")
    part_mask = get_part_mask(person, part)
    face_mask = get_face_mask(person) if face_lock else Image.new("L", person.size, 0)
    combined_mask = Image.fromarray(np.maximum(np.array(part_mask), np.array(face_mask))) if bg_lock else part_mask

    prompt = "Person wearing the clothing realistically, photorealistic, high quality, maintain original lighting"
    result = pipe(prompt=prompt, image=person, mask_image=combined_mask).images[0]
    out_bytes = io.BytesIO()
    result.save(out_bytes, format="PNG")
    out_bytes.seek(0)
    return out_bytes

# Cloth changer endpoint
@app.post("/tryon")
async def tryon_api(
    cloth: UploadFile = File(...),
    character: str = Form(...),
    part: str = Form(...),
    face_lock: bool = Form(True),
    bg_lock: bool = Form(True)
):
    person_path = os.path.join(CHAR_DIR, f"{character}.png")
    if not os.path.exists(person_path):
        return JSONResponse({"status": "error", "message": "Character not found"}, status_code=404)

    with open(person_path, "rb") as f:
        person_bytes = f.read()
    cloth_bytes = await cloth.read()
    result_bytes = cloth_changer(person_bytes, cloth_bytes, part, face_lock, bg_lock)
    return FileResponse(result_bytes, media_type="image/png", filename="result.png")

# Upload new character
@app.post("/upload-character")
async def upload_character(name: str = Form(...), file: UploadFile = File(...)):
    filename = f"{name}.png"
    path = os.path.join(CHAR_DIR, filename)
    img = Image.open(io.BytesIO(await file.read())).convert("RGB")
    img.save(path)
    return JSONResponse({"status":"success","message":f"Character '{name}' added."})

# List characters
@app.get("/list-characters")
def list_characters():
    chars = [f.split(".")[0] for f in os.listdir(CHAR_DIR) if f.endswith((".png",".jpg"))]
    return {"characters": chars}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)