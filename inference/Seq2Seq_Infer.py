import yaml
import torch
from PIL import Image
from torchvision.transforms import ToTensor, Normalize

from pathlib import Path
import torch.nn.functional as F

from ..utils import Vocab
from ..models.Seq2Seq import Seq2Seq, CNN

# ~ 317 mỗi lần infer 
# nam@NamLaptop:~/Downloads$ python3 -m prelabeller_ocr.inference.seq_infer

class Seq2Seq__Independ_VietOCR(torch.nn.Module):
    def __init__(self, vocab_size, backbone_name, cnn_cfg, seq_cfg):
        super().__init__()
        
        # 1. Initialize CNN Backbone
        # We must pass the specific ss (strides) and ks (kernels) from config
        self.cnn = CNN(
            backbone=backbone_name,
            ss=cnn_cfg['ss'],
            ks=cnn_cfg['ks'],
            hidden=cnn_cfg['hidden']
        )
        
        # 2. Initialize Seq2Seq (LSTM) Head
        self.transformer = Seq2Seq(
            vocab_size=vocab_size,
            encoder_hidden=seq_cfg['encoder_hidden'],
            decoder_hidden=seq_cfg['decoder_hidden'],
            img_channel=seq_cfg['img_channel'],
            decoder_embedded=seq_cfg['decoder_embedded'],
            dropout=seq_cfg['dropout']
        )

    def forward(self, img):
        # If you were using this for training, you'd update this too:
        features = self.cnn(img)
        features = self.transformer(features) 
        return features

    # def forward(self, img):
    #     # Not used for inference loop
    #     pass

def process_image(image_path, height, device):
    img = Image.open(image_path).convert('RGB')
    
    # Calculate new width maintaining aspect ratio
    w_new = int(img.width * height / img.height)
    
    # Resize
    img = img.resize((w_new, height), Image.Resampling.LANCZOS)
    
    # ToTensor() automatically converts [0, 255] -> [0.0, 1.0]
    img_tensor = ToTensor()(img)
    
    # DELETE or COMMENT OUT this line:
    # img_tensor = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(img_tensor)
    
    # Add batch dimension
    return img_tensor.unsqueeze(0).to(device)

# Orignal - OK - decode 
def translate(img_tensor, model, vocab, device, max_seq_length=128):
    model.eval()
    with torch.no_grad():
        # 1. Extract features
        # The CNN in your implementation already returns (Width, Batch, Channels)
        # Shape: [36, 1, 256] - height, batch, size
        features = model.cnn(img_tensor)
        print(f"DEBUG: CNN Output Shape: {features.shape}")

        # 2. Reshape? NO.
        # The features are already in the correct shape: (Seq_Len, Batch, Input_Size)        if features.dim() == 4:
        # We just verify this for safety.
        if features.dim() == 4:
            # Only if it accidentally returns 4D, we fix it. 
            # But your logs show it returns 3D, so this won't run.
            features = features.squeeze(2).permute(2, 0, 1)

        print(f"DEBUG: LSTM Input Shape: {features.shape}")

        # 3. Encoder
        # Input: (36, 1, 256) -> Input_Size is 256. Perfect
        try:
            # encoder_outputs, hidden = model.seq2seq.encoder(features)
            encoder_outputs, hidden = model.transformer.encoder(features)
        except RuntimeError as e:
            print(f"DEBUG CRASH: The LSTM expected 256 channels but got {features.shape[-1]}")
            raise e

        # 4. Decoder Loop
        # decoder_input = torch.tensor([vocab.chars.index('sos')], device=device)
        # decoded_indices = []
        # FIX HERE: Use vocab.go instead of vocab.chars.index('sos')
        # decoder_input = torch.tensor([vocab.go], device=device)
        # decoder_input = torch.LongTensor([[vocab.go]]).to(device)
        decoder_input = torch.tensor([vocab.go], device=device) 
        
        decoded_indices = []

        for _ in range(max_seq_length):
            # The decoder expects:
            # decoder_input: [Batch] 
            # hidden: [1, Batch, Hidden]
            # encoder_outputs: [Seq_Len, Batch, Hidden]
            output, hidden, _ = model.transformer.decoder(
                decoder_input, 
                hidden, 
                encoder_outputs
            )
            
            # output shape: [Batch, Vocab_Size]
            topv, topi = output.data.topk(1)
            token_index = topi.item()
            
            # Stop condition
            if token_index == vocab.eos:
                break
                
            decoded_indices.append(token_index)
            
            # FIX 2: Prepare input for next time step
            # topi comes out as [Batch, 1]. We need [Batch] for the next input.
            decoder_input = topi.squeeze(1).detach()

    return vocab.decode(decoded_indices)


def main():
    # 1. Load Configuration
    BASE_DIR = Path(__file__).resolve().parent.parent 
    # Load Configs
    SEQ2SEQ_CONFIG_PATH = BASE_DIR / "config" / "seq2seq_vietocr_config.yaml"
    with open(SEQ2SEQ_CONFIG_PATH, "r") as f:
        cfg = yaml.safe_load(f)

    # Device setup
    device = torch.device(cfg['device'])
    print(f"Running on: {device}")

    # 2. Setup Vocabulary
    print (f"Len in YAML {len(cfg['vocab'])}")

    vocab = Vocab(cfg['vocab'])
    print(f"Vocab Length: {len(vocab)}")
    if len(vocab) != 233:
        raise ValueError(f"CRITICAL: Vocab length is {len(vocab)}, but checkpoint expects 233. Check your YAML config!")
    
    # Initialize Model with nested configs
    model = Seq2Seq__Independ_VietOCR(
        vocab_size=len(vocab),
        backbone_name=cfg['backbone'],
        cnn_cfg=cfg['cnn'],          # Pass the whole CNN dict
        seq_cfg=cfg['seq_modeling']  # Pass the whole Seq2Seq dict
    )
    model.to(device)

    
    try:
        # 4. Load Weights safely
        print(f"Loading weights from {cfg['weights']}...")
        checkpoint = torch.load(cfg['weights'], map_location=device)
        
        # Handle 'state_dict' wrapper if present
        if 'state_dict' in checkpoint:
            checkpoint = checkpoint['state_dict']
            
        # Load state dict (strict=False handles missing auxiliary keys common in VietOCR weights) - by pass
        model.load_state_dict(checkpoint, strict=True)
        print("SUCCESS: Model loaded with strict=True. Ready for inference.")
    except Exception as e:
        print(f"Error loading the model: {e}")
 

    # 5. Run Inference
    img_tensor = process_image(
        cfg['image_path'], 
        cfg['height'], 
        device
    )
    
    result = translate(img_tensor, model, vocab, device)
    
    print("------------------------------------------------")
    print(f"Predicted Text: {result}")
    print("------------------------------------------------")

if __name__ == "__main__":
    main()