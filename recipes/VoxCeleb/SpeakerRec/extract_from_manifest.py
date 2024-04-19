import torch
import torchaudio
#from speechbrain.pretrained.interfaces import Pretrained
from speechbrain.inference.speaker import EncoderClassifier
import sys
import pickle as pkl
from tqdm import tqdm
import os
from torch.utils.data import Dataset, DataLoader
from speechbrain.dataio.batch import PaddedBatch
import numpy as np
import json 

class CustomDataset(Dataset):
    def __init__(self, files, utterances):
        self.files = files
        self.utterances = utterances

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file_path = self.files[idx]
        utterance = self.utterances[idx]
        waveform, _ = torchaudio.load(file_path)
        return waveform, utterance, file_path


def compute_embeddings(model, wav_scp, outdir):
    """Compute speaker embeddings.

    Arguments
    ---------
    params: dict
        The parameter files storing info about model, data, etc
    wav_scp : str
        The wav.scp file in Kaldi, in the form of "$utt $wav_path"
    outdir: str
        The output directory where we store the embeddings in per-
        numpy manner.
    """

    files = []
    utterances = []
    with open(wav_scp, "r") as f:
        lines = f.readlines()

    out_dict = {}
    out_dict['concat_labels'] = []
    out_dict['concat_slices'] = []
    out_dict['concat_patchs'] = []
    out_dict['concat_features'] = []
    batch_size = 2048
    
    for index in range(0,len(lines),batch_size):
        print(index)
        wavs = []
        for i in range(index,index+batch_size,1):
            if i > len(lines)-1:
                continue    
            data = json.loads(lines[i].strip())
            batch_embs = []

            if 'patch' in data.keys():
                out_dict['concat_patchs'].append(data['patch'])
            else:
                out_dict['concat_patchs'].append(data['file_id'])

            out_dict['concat_labels'].append(data['label'])
            
            out_dict['concat_slices'].append(data['file_id'])
            wav = torchaudio.load(data['audio_filepath'])[0][0]#torch.tensor(model.load_audio(data['audio_filepath'])).unsqueeze(0)#

            wavs.append({"wav":wav})

        wavs = PaddedBatch(wavs)

        model.eval()
        #model.to(torch.device("cuda:0"))
        #model.to("cuda:0")
        
        with torch.no_grad():
    
            embeddings = model.encode_batch(*wavs["wav"])
            for embedding in embeddings:
                out_embedding = embedding.detach().cpu().numpy().squeeze(0)
                batch_embs.append(out_embedding)

            del embeddings, wavs

        if not os.path.exists(outdir):
            os.mkdir(outdir)
        
        out_file = "{}/{}_ecapa_embs.pkl".format(outdir, os.path.splitext(os.path.basename(wav_scp))[0])
        out_dict['concat_features'].append(np.asarray(batch_embs))
    
    out_dict['concat_features'] = np.concatenate(out_dict['concat_features'],axis=0)
    pkl.dump(out_dict, open(out_file, 'wb'))


if __name__ == "__main__":

    in_manifest = sys.argv[1]
    out_dir = sys.argv[2]
    classifier = EncoderClassifier.from_hparams(
        source="yangwang825/ecapa-tdnn-vox2", run_opts={"device":"cuda"}
        )

    compute_embeddings(classifier, in_manifest, out_dir)

    