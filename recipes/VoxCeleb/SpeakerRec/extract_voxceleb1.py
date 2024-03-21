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

    for line in lines:
        files.append(line.strip().split()[1])
        utterances.append(line.strip().split()[0])

    model.eval()
    #model.to(torch.device("cuda:0"))
    #model.to("cuda:0")
    with torch.no_grad():
    
        out_dict = {}
        out_dict['concat_labels'] = []
        out_dict['concat_slices'] = []
        out_dict['concat_patchs'] = []
        all_embs = []

        for (wav_path,utt) in tqdm(zip(files,utterances)):
            
            out_dict['concat_patchs'].append(utt)
            out_dict['concat_labels'].append(wav_path.split('/')[-3])
            if "_window" in wav_path:
                out_dict['concat_slices'].append(utt.split("_window")[0])
            else:
                out_dict['concat_slices'].append(utt)
            
            wav, _ = torchaudio.load(wav_path)
            wav = wav.to(torch.device("cuda:0"))
            embedding = model.encode_batch(wav)

            out_embedding = embedding.detach().cpu().numpy().squeeze(0).squeeze(0)
            all_embs.append(out_embedding)

            del out_embedding, wav

        if not os.path.exists(outdir):
            os.mkdir(outdir)
        
        out_file = "{}/{}_ecapa_embs.pkl".format(outdir, os.path.splitext(wav_scp)[0])
        out_dict['concat_features'] = np.asarray(all_embs)
        pkl.dump(out_dict, open(out_file, 'wb'))


if __name__ == "__main__":

    in_list = sys.argv[1]
    out_dir = sys.argv[2]
    classifier = EncoderClassifier.from_hparams(
        source="yangwang825/ecapa-tdnn-vox2", run_opts={"device":"cuda"}
        )

    compute_embeddings(classifier, in_list, out_dir)

    