import torch

from config import model_config
from model import MusicVAE
from to_midi import get_midi_from_profile

def main():
    device = torch.device("cuda") if torch.cuda.is_available else torch.device("cpu")
    device = torch.device("cpu")
    model = MusicVAE(**model_config)
    ckpt_path = "checkpoint.ckpt"
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model = model.to(device)
    
    model.eval()
    with torch.no_grad():
        z = torch.randn([1, 512], device=device)
        generated = model(z=z, mode="generate")
    
    generated = generated[0, :, 0].detach().cpu().numpy().tolist()
    midi = get_midi_from_profile(generated)
    midi.write("gen.mid")

if __name__ == "__main__":
    main()