from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

app = FastAPI()
from fastapi.staticfiles import StaticFiles

# Ajoute ceci après la ligne app = FastAPI()
app.mount("/static", StaticFiles(directory="app/static"), name="static")
# On explique à FastAPI où se trouve ton dossier de pages HTML
templates = Jinja2Templates(directory="app/templates")

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    # Cette ligne dit : "Quand on va sur le site, affiche index.html"
    return templates.TemplateResponse("index.html", {"request": request})